//! Data-driven queries over private-company ability definitions.
//!
//! All engine consumers of private-company special powers go through this
//! module instead of keying on company syms (`co.sym == "CS"` etc.), so the
//! behavior is defined entirely by the title data
//! (`crate::title::g1830::companies()` — the Rust mirror of g1830.py's
//! `abilities:` arrays). Adding a new title's privates means writing data,
//! not engine code.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::title::{AbilityDef, AbilityWhen, OwnerType, ShareSource};

/// sym -> static ability list, memoized from the title data.
fn ability_map() -> &'static HashMap<String, &'static [AbilityDef]> {
    static MAP: OnceLock<HashMap<String, &'static [AbilityDef]>> = OnceLock::new();
    MAP.get_or_init(|| {
        crate::title::g1830::companies()
            .iter()
            .map(|c| (c.sym.to_string(), c.abilities))
            .collect()
    })
}

/// All abilities of the company `sym` (empty for unknown syms).
pub fn company_abilities(sym: &str) -> &'static [AbilityDef] {
    ability_map().get(sym).map(|&a| a).unwrap_or(&[])
}

/// The company's `blocks_hexes` entry: `(owner_type, hexes)`.
pub fn blocks_hexes(sym: &str) -> Option<(OwnerType, &'static [&'static str])> {
    company_abilities(sym).iter().find_map(|a| match a {
        AbilityDef::BlocksHexes { owner_type, hexes } => Some((*owner_type, *hexes)),
        _ => None,
    })
}

/// The company's bonus `tile_lay` entry: `(hexes, tiles, when, count)`.
/// 1830: CS (B20, yellow 3/4/58, owning_corp_or_turn, count 1).
pub fn tile_lay(
    sym: &str,
) -> Option<(
    &'static [&'static str],
    &'static [&'static str],
    AbilityWhen,
    u32,
)> {
    company_abilities(sym).iter().find_map(|a| match a {
        AbilityDef::TileLay {
            hexes,
            tiles,
            when,
            count,
            ..
        } => Some((*hexes, *tiles, *when, *count)),
        _ => None,
    })
}

/// The company's `teleport` entry: `(hexes, tiles)`. 1830: DH (F16, tile 57).
pub fn teleport(sym: &str) -> Option<(&'static [&'static str], &'static [&'static str])> {
    company_abilities(sym).iter().find_map(|a| match a {
        AbilityDef::Teleport { hexes, tiles, .. } => Some((*hexes, *tiles)),
        _ => None,
    })
}

/// The company's `exchange` entry: `(corporations, from)`. 1830: MH -> NYC.
pub fn exchange(sym: &str) -> Option<(&'static [&'static str], &'static [ShareSource])> {
    company_abilities(sym).iter().find_map(|a| match a {
        AbilityDef::Exchange {
            corporations, from, ..
        } => Some((*corporations, *from)),
        _ => None,
    })
}

/// Whether the company carries `no_buy` (1830: BO).
pub fn no_buy(sym: &str) -> bool {
    company_abilities(sym)
        .iter()
        .any(|a| matches!(a, AbilityDef::NoBuy))
}

/// The corporation whose PRESIDENT's certificate this company grants
/// (`shares` ability with index 0) — buying the company triggers a pending
/// par for that corporation. 1830: BO -> B&O.
pub fn par_trigger(sym: &str) -> Option<&'static str> {
    company_abilities(sym).iter().find_map(|a| match a {
        AbilityDef::Shares {
            corporation,
            share_index: 0,
        } => Some(*corporation),
        _ => None,
    })
}

/// The corporation of which this company grants a NORMAL share on purchase
/// (`shares` ability with index > 0). 1830: CA -> PRR.
pub fn share_grant(sym: &str) -> Option<&'static str> {
    company_abilities(sym).iter().find_map(|a| match a {
        AbilityDef::Shares {
            corporation,
            share_index,
        } if *share_index > 0 => Some(*corporation),
        _ => None,
    })
}

/// The company (if any) whose par is pending for `corp_sym` — the inverse of
/// [`par_trigger`], used to recover the triggering company from a pending par.
pub fn par_trigger_company_for(corp_sym: &str) -> Option<&'static str> {
    company_syms()
        .iter()
        .copied()
        .find(|sym| par_trigger(sym) == Some(corp_sym))
}

/// The company (if any) that closes when `corp_sym` buys its first train.
pub fn close_on_bought_train(corp_sym: &str) -> Option<&'static str> {
    company_syms().iter().copied().find(|sym| {
        company_abilities(sym).iter().any(|a| {
            matches!(
                a,
                AbilityDef::Close {
                    when: AbilityWhen::BoughtTrain,
                    corporation,
                } if *corporation == corp_sym
            )
        })
    })
}

/// Company syms in title order (memoized).
pub fn company_syms() -> &'static [&'static str] {
    static SYMS: OnceLock<Vec<&'static str>> = OnceLock::new();
    SYMS.get_or_init(|| crate::title::g1830::companies().iter().map(|c| c.sym).collect())
}

// ---------------------------------------------------------------------------
// Game-state-aware queries
// ---------------------------------------------------------------------------

impl BaseGame {
    /// Hexes currently blocked by private companies' `blocks_hexes` abilities:
    /// the union over open companies whose owner kind matches the ability's
    /// `owner_type`. The single source of the blocked-hex set (1830:
    /// SV->G15, CS->B20, DH->F16, MH->D18, CA->H18, BO->I13+I15, while
    /// player-owned).
    pub(crate) fn ability_blocked_hexes(&self) -> HashSet<&'static str> {
        let mut blocked = HashSet::new();
        for co in &self.companies {
            if co.closed {
                continue;
            }
            if let Some((owner_type, hexes)) = blocks_hexes(&co.sym) {
                let live = match owner_type {
                    OwnerType::Player => co.owner.is_player(),
                    OwnerType::Corporation => co.owner.corp_sym().is_some(),
                };
                if live {
                    blocked.extend(hexes.iter().copied());
                }
            }
        }
        blocked
    }

    /// Open companies owned by `corp_eid` whose special tile-lay ability
    /// (TileLay or Teleport) is still unused — the candidates for company
    /// LayTile actions during the corp's OR turn.
    pub(crate) fn special_lay_companies(&self, corp_eid: &EntityId) -> Vec<String> {
        self.companies
            .iter()
            .filter(|co| {
                !co.closed
                    && !co.ability_used
                    && (tile_lay(&co.sym).is_some() || teleport(&co.sym).is_some())
                    && &co.owner == corp_eid
            })
            .map(|co| co.sym.clone())
            .collect()
    }
}
