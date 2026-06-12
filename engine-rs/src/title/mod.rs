pub mod g1830;
pub mod g1867;
#[cfg(test)]
pub mod test_title;

use std::collections::HashMap;
use std::sync::Arc;

use crate::steps::{FinishedRound, RoundTransition, StepDesc};
use crate::tiles::TileDef;

// ---------------------------------------------------------------------------
// The GameTitle trait — THE title-dispatch funnel.
//
// Everything the shared rules engine needs from a title goes through this
// trait: the static game data (corporations, companies, trains, phases, map,
// tiles) and the per-title machinery descriptions (ordered step lists, the
// round-flow function). One impl per title; `resolve()` maps the game's
// `title` string to the impl. Engine code must not call `g1830::*` directly
// (the per-title AlphaZero bridge — action layout, encoder — is separate and
// still 1830-pinned; see the multi-title roadmap).
// ---------------------------------------------------------------------------

pub trait GameTitle: Sync {
    /// The title string stored on the game (`BaseGame.title`).
    fn name(&self) -> &'static str;

    // -- round/step machinery descriptions --
    fn auction_steps(&self) -> &'static [StepDesc];
    fn stock_steps(&self) -> &'static [StepDesc];
    fn operating_steps(&self) -> &'static [StepDesc];
    /// The title's round flow (Ruby/Python `next_round!`): the round that
    /// just finished decides what starts next.
    fn next_round(&self, finished: FinishedRound, phase_operating_rounds: u8) -> RoundTransition;

    // -- static game data --
    fn starting_cash(&self, num_players: u8) -> i32;
    fn cert_limit(&self, num_players: u8) -> u8;
    fn bank_cash(&self) -> i32;
    fn companies(&self) -> Vec<CompanyDef>;
    fn corporations(&self) -> Vec<CorporationDef>;
    fn trains(&self) -> Vec<TrainDef>;
    fn phases(&self) -> Vec<PhaseDef>;
    fn hex_definitions(&self) -> Vec<HexDef>;
    /// The preprinted tile DSL + color for a map hex, if any.
    fn preprinted_hex_dsl(&self, coord: &str) -> Option<(&'static str, &'static str)>;
    fn tile_counts(&self) -> Vec<(&'static str, u32)>;
    /// City slot counts for a catalog tile id (fallback when the catalog
    /// entry carries no city geometry).
    fn tile_cities(&self, tile_id: &str) -> Option<Vec<u8>>;
    fn tile_catalog(&self) -> Arc<HashMap<String, TileDef>>;
    /// The raw stock-market grid (rows of cells; None = empty cell).
    fn market_grid(&self) -> Vec<Vec<Option<MarketCell>>>;
    /// How share prices move on the market (Ruby's movement modules).
    fn market_movement(&self) -> MarketMovement;
    /// Max percent of one corporation the open market pool may hold
    /// (Ruby `market_share_limit`; 50 for 1830/1867).
    fn market_pool_limit(&self) -> u8 {
        50
    }
    /// The price grid step for auction bids (1830: bids move in $5 steps).
    /// Used by the MCTS price sampler's snap grid.
    fn bid_price_step(&self) -> i64 {
        5
    }
    /// Whether a corporation may be STARTED (parred / bid-founded) in the
    /// given phase (1867: the six green minors join from phase 3, majors
    /// from phase 4). Default: always startable (1830).
    fn corporation_startable(&self, _sym: &str, _phase_name: &str) -> bool {
        true
    }
    /// The per-OR-turn tile-lay allowance (Ruby `TILE_LAYS`), one slot per
    /// permitted lay in order. Default: 1830's single unrestricted lay.
    fn tile_lays(&self) -> &'static [TileLayDef] {
        const ONE: &[TileLayDef] = &[TileLayDef {
            lay: true,
            upgrade: UpgradeAllowance::Yes,
            cost: 0,
            cannot_reuse_same_hex: false,
        }];
        ONE
    }
    /// The map's hex orientation (Ruby `LAYOUT`). Edge numbers in tile DSL,
    /// borders and stubs are all relative to it. 1830: pointy-top;
    /// 1867: flat-top.
    fn hex_layout(&self) -> HexLayout {
        HexLayout::Pointy
    }

    // -- loans (Ruby InterestOnLoans titles; zero/off for 1830) --

    /// Face value of one loan; 0 = the title has no loans.
    fn loan_value(&self) -> i32 {
        0
    }
    /// Interest per loan per OR. Taking a loan nets `value - interest`
    /// (1867: $50 - $5 = $45); repaying costs the full value.
    fn loan_interest_rate(&self) -> i32 {
        0
    }
    /// Total loans in the bank pool (1867: 72).
    fn num_loans(&self) -> u32 {
        0
    }
    /// Max loans a corporation of the given class may hold.
    fn max_loans(&self, _corp_type: CorpType) -> u32 {
        0
    }

    // -- train-market rules --

    /// Whether a corp may buy trains from corps with a DIFFERENT president
    /// (Ruby `ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS`; 1830 false, 1867 true).
    fn train_buy_from_other_players(&self) -> bool {
        false
    }
    /// Whether the president may fund an emergency train buy from hand
    /// (1830's EMR). False for loans-only EMR titles (1867: no share
    /// selling, no president cash — loans are the emergency money).
    fn ebuy_president_may_contribute(&self) -> bool {
        true
    }
    /// Ruby `MUST_BUY_TRAIN`: `:route` (1830 — the obligation needs a
    /// runnable route) vs `:always` (1867 — a trainless corp must buy
    /// regardless of routes; true here).
    fn must_buy_train_always(&self) -> bool {
        false
    }

    // -- AlphaZero-bridge orders (action layout + encoder) --
    //
    // The flat action layout and the encoder need a pinned iteration order
    // for hexes and tiles. The defaults derive them from the title data
    // (map order / tile-catalog order); 1830 OVERRIDES both with frozen
    // historical orders (artifacts of the Python ActionMapper's original
    // construction) because trained checkpoints depend on the exact slots.

    /// Whether the AlphaZero bridge (flat action layout + encoder spec) can
    /// be derived for this title. A RULES title can register without it —
    /// 1867 plays/replays long before its action space and encoder land
    /// (the roadmap's per-title action-layout phase; the current slot
    /// layout cannot express 1867's three discounted trains, choose/merge
    /// actions, …). While false, `action_index::layout_for` /
    /// `encoder::spec_for` panic for this title instead of mis-deriving.
    fn alphazero_bridge_ready(&self) -> bool {
        true
    }

    /// Hex order for the flat action layout.
    fn action_hex_order(&self) -> Vec<&'static str> {
        self.hex_definitions().iter().map(|h| h.coord).collect()
    }
    /// Tile order for the flat action layout.
    fn action_tile_order(&self) -> Vec<&'static str> {
        self.tile_counts().iter().map(|(id, _)| *id).collect()
    }
}

/// One slot of a title's per-OR-turn tile-lay allowance (one Ruby
/// `TILE_LAYS` entry; 1867: `[{lay, upgrade}, {lay, upgrade:
/// :not_if_upgraded, cost: 20, cannot_reuse_same_hex: true}]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileLayDef {
    /// May this slot lay a NEW (yellow) tile?
    pub lay: bool,
    /// May this slot upgrade an existing tile?
    pub upgrade: UpgradeAllowance,
    /// Extra cost for using this slot (1867: $20 for the second lay).
    pub cost: i32,
    /// This slot may not target a hex already laid this turn (1867's
    /// second lay).
    pub cannot_reuse_same_hex: bool,
}

/// Whether a tile-lay slot may upgrade (Ruby's `upgrade:` values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpgradeAllowance {
    Yes,
    No,
    /// Allowed unless an EARLIER lay this turn was an upgrade (1867).
    NotIfUpgraded,
}

/// How share prices move on the stock market grid (Ruby's
/// `TwoDimensionalMovement` / `OneDimensionalMovement` modules).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarketMovement {
    /// 2-D grid (1830): right/up on gains, left/down on losses; an edge
    /// move falls back to the perpendicular direction (right→up, left→down).
    TwoDimensional,
    /// Single-row market (1867): prices only move along the row; there is
    /// no vertical movement and no edge fallback.
    OneDimensional,
}

/// 1830: Railways & Robber Barons.
pub struct G1830;

impl GameTitle for G1830 {
    fn name(&self) -> &'static str {
        "1830"
    }
    fn auction_steps(&self) -> &'static [StepDesc] {
        g1830::auction_steps()
    }
    fn stock_steps(&self) -> &'static [StepDesc] {
        g1830::stock_steps()
    }
    fn operating_steps(&self) -> &'static [StepDesc] {
        g1830::operating_steps()
    }
    fn next_round(&self, finished: FinishedRound, phase_operating_rounds: u8) -> RoundTransition {
        g1830::next_round(finished, phase_operating_rounds)
    }
    fn starting_cash(&self, num_players: u8) -> i32 {
        g1830::starting_cash(num_players)
    }
    fn cert_limit(&self, num_players: u8) -> u8 {
        g1830::cert_limit(num_players)
    }
    fn bank_cash(&self) -> i32 {
        g1830::BANK_CASH
    }
    fn companies(&self) -> Vec<CompanyDef> {
        g1830::companies()
    }
    fn corporations(&self) -> Vec<CorporationDef> {
        g1830::corporations()
    }
    fn trains(&self) -> Vec<TrainDef> {
        g1830::trains()
    }
    fn phases(&self) -> Vec<PhaseDef> {
        g1830::phases()
    }
    fn hex_definitions(&self) -> Vec<HexDef> {
        g1830::hex_definitions()
    }
    fn preprinted_hex_dsl(&self, coord: &str) -> Option<(&'static str, &'static str)> {
        g1830::preprinted_hex_dsl(coord)
    }
    fn tile_counts(&self) -> Vec<(&'static str, u32)> {
        g1830::tile_counts()
    }
    fn tile_cities(&self, tile_id: &str) -> Option<Vec<u8>> {
        g1830::tile_cities(tile_id)
    }
    fn tile_catalog(&self) -> Arc<HashMap<String, TileDef>> {
        crate::tiles::tile_catalog_1830()
    }
    fn market_grid(&self) -> Vec<Vec<Option<MarketCell>>> {
        g1830::market_grid()
    }
    fn market_movement(&self) -> MarketMovement {
        MarketMovement::TwoDimensional
    }
    fn action_hex_order(&self) -> Vec<&'static str> {
        g1830::action_hex_order()
    }
    fn action_tile_order(&self) -> Vec<&'static str> {
        g1830::action_tile_order()
    }
}

/// Every implemented title. New titles register here. Test builds also
/// register the synthetic `TEST-5SHARE` title (see `test_title.rs`) so the
/// per-title memoized maps (abilities, layouts, encoder specs) and the
/// frozen-1830 pins are exercised with MORE than one title.
pub fn all_titles() -> &'static [&'static dyn GameTitle] {
    #[cfg(test)]
    {
        static TITLES: [&'static dyn GameTitle; 3] =
            [&G1830, &g1867::G1867, &test_title::Test5Share];
        &TITLES
    }
    #[cfg(not(test))]
    {
        static TITLES: [&'static dyn GameTitle; 2] = [&G1830, &g1867::G1867];
        &TITLES
    }
}

/// The title impl for a game's `title` string. Unknown names panic — the
/// string is engine-internal (set from `GameTitle::name` at construction),
/// so a miss is a bug, not user input.
pub fn resolve(name: &str) -> &'static dyn GameTitle {
    all_titles()
        .iter()
        .copied()
        .find(|t| t.name() == name)
        .unwrap_or_else(|| panic!("unknown game title: {name}"))
}

// ---------------------------------------------------------------------------
// Shared hex-grid geometry (pointy-top, letter+number coordinates) — the
// coordinate scheme tobymao/18xx titles share. Moved out of g1830.rs because
// it is math over coordinates, not title data.
// ---------------------------------------------------------------------------

/// Parses a hex coordinate like "H12" into (row_letter_index, number).
/// Letter part → x: A=0..K=10.  Number part → y (raw number).
pub fn parse_coord(coord: &str) -> (i32, i32) {
    let bytes = coord.as_bytes();
    let letter = (bytes[0] - b'A') as i32;
    let number: i32 = coord[1..].parse().expect("invalid hex coordinate number");
    (letter, number)
}

/// A map's hex orientation (Ruby `Engine::Hex::DIRECTIONS` keys / a title's
/// `LAYOUT`). Determines which neighbor sits across each numbered edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HexLayout {
    /// Pointy-top hexes (1830).
    Pointy,
    /// Flat-top hexes (1867).
    Flat,
}

/// Pointy-top hex direction deltas in (d_letter, d_number) space.
///
/// The Python engine's pointy-top direction deltas are in
/// (dx, dy) = (d_number, d_letter) space:
///   0:(-1,1), 1:(-2,0), 2:(-1,-1), 3:(1,-1), 4:(2,0), 5:(1,1)
///
/// We store coordinates as (letter_index, number), so we swap to (dy, dx):
const HEX_DELTAS: [(i32, i32); 6] = [
    (1, -1),  // 0: upper-right  (Python dx=-1, dy=+1)
    (0, -2),  // 1: right        (Python dx=-2, dy= 0)
    (-1, -1), // 2: lower-right  (Python dx=-1, dy=-1)
    (-1, 1),  // 3: lower-left   (Python dx=+1, dy=-1)
    (0, 2),   // 4: left         (Python dx=+2, dy= 0)
    (1, 1),   // 5: upper-left   (Python dx=+1, dy=+1)
];

/// Flat-top hex direction deltas in (d_letter, d_number) space.
///
/// Ruby `Hex::DIRECTIONS[:flat]` maps `[other.x - x, other.y - y]` to the
/// edge number with x = letter index, y = number:
///   [0,2]=>0, [-1,1]=>1, [-1,-1]=>2, [0,-2]=>3, [1,-1]=>4, [1,1]=>5
/// (verified against the 1867 map: G15's `stub=edge:1` points at Toronto
/// F16; F16's city paths a:1 → E17 Hamilton and a:4 → G15 Peterborough;
/// D2 Timmins' a:1 pairs with C3's a:4.)
const HEX_DELTAS_FLAT: [(i32, i32); 6] = [
    (0, 2),   // 0
    (-1, 1),  // 1
    (-1, -1), // 2
    (0, -2),  // 3
    (1, -1),  // 4
    (1, 1),   // 5
];

/// Format (letter_index, number) back to a coordinate string like "H12".
fn format_coord(letter: i32, number: i32) -> String {
    let ch = (b'A' + letter as u8) as char;
    format!("{}{}", ch, number)
}

/// Compute hex adjacency from a set of hex coordinates.
/// Returns hex_id -> { direction -> neighbor_hex_id } for all valid neighbors.
pub fn compute_adjacency(coords: &[&str], layout: HexLayout) -> HashMap<String, HashMap<u8, String>> {
    let deltas = match layout {
        HexLayout::Pointy => &HEX_DELTAS,
        HexLayout::Flat => &HEX_DELTAS_FLAT,
    };
    let coord_set: std::collections::HashSet<String> =
        coords.iter().map(|c| c.to_string()).collect();

    let mut adjacency: HashMap<String, HashMap<u8, String>> = HashMap::new();

    for &coord in coords {
        let (letter, number) = parse_coord(coord);
        let mut neighbors = HashMap::new();

        for (dir, (dl, dn)) in deltas.iter().enumerate() {
            let nl = letter + dl;
            let nn = number + dn;
            if nl >= 0 {
                let neighbor = format_coord(nl, nn);
                if coord_set.contains(&neighbor) {
                    neighbors.insert(dir as u8, neighbor);
                }
            }
        }

        adjacency.insert(coord.to_string(), neighbors);
    }

    adjacency
}

// ---------------------------------------------------------------------------
// Title-agnostic definition structs — the shape every title's data module
// fills in (moved here from g1830.rs so future titles share them).
// ---------------------------------------------------------------------------

/// A corporation's class (Ruby `type:`). Drives per-type train limits,
/// operating order, dividend/par/merger rules in titles that mix classes
/// (1867: 16 minors + 8 majors + the CN national). Every 1830 corp is a
/// Major.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CorpType {
    #[default]
    Major,
    /// Single-cert (100%) player-owned company that operates like a small
    /// corporation (1867 minors).
    Minor,
    /// A non-operating national corporation (1867's CN).
    National,
}

/// How a corporation's treasury is funded (Ruby `capitalization:`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Capitalization {
    /// On float the corporation receives par × total shares from the bank
    /// (1830).
    #[default]
    Full,
    /// The corporation receives par × shares sold as each share is bought
    /// from the IPO; no lump sum on float (1867). Not yet implemented — no
    /// registered title sets it.
    Incremental,
}

pub struct CorporationDef {
    pub sym: &'static str,
    pub name: &'static str,
    pub token_prices: &'static [i32],
    pub home_hex: &'static str,
    pub home_city_index: u8,
    /// Whether the home hex tile is reserved for this corp (token placed after tile upgrade).
    pub reserved: bool,
    /// Certificate sizes in percent, president first (Ruby `shares:`;
    /// 1830: 20,10×8). The share UNIT is the smallest entry; a cert's unit
    /// count is `percent / unit` (1830 president = 2 units of 10%).
    pub shares: &'static [u8],
    /// Percent of shares that must leave the IPO for the corp to float
    /// (Ruby `float_percent:`; 1830: 60).
    pub float_percent: u8,
    pub capitalization: Capitalization,
    /// The corporation's class (1830: always Major).
    pub corp_type: CorpType,
    /// Max percent one player may hold (Ruby `max_ownership_percent`;
    /// 1830: 60. 1867 minors: 100).
    pub max_ownership_percent: u8,
    /// Treasury/IPO shares sell at market price instead of par (Ruby
    /// `always_market_price`; 1867: true).
    pub always_market_price: bool,
}

pub struct CompanyDef {
    pub sym: &'static str,
    pub name: &'static str,
    pub value: i32,
    pub revenue: i32,
    /// Special powers, transcribed from the title data's `abilities` arrays.
    /// Queried via `crate::abilities` — engine code must not key on company
    /// syms.
    pub abilities: &'static [AbilityDef],
    /// Auction price-floor discount (1867's dutch single-item auction
    /// lowers min_bid by $5/round via this; 0 = none).
    pub discount: i32,
    /// Whether the company is offered in the opening auction (false for
    /// 1867's hidden '3' phase blocker, which exists as an entity but is
    /// never auctioned or owned).
    pub auctionable: bool,
}

pub struct TrainDef {
    pub name: &'static str,
    pub distance: u32,
    pub price: i32,
    pub count: u32,
    pub rusts_on: Option<&'static str>,
    /// Revenue multiplier (1867's 2+2 and 5+5E double route revenue; 1).
    pub multiplier: u32,
    /// Phase on which this train becomes obsolete (runs once more, then
    /// removed — 1867's 6/7/8 vs the hard rust of `rusts_on`). None = never.
    pub obsolete_on: Option<&'static str>,
    /// Event types fired when this train is bought (Ruby/Python `events:`;
    /// 1830: `close_companies` on the first 5-train).
    pub events: &'static [&'static str],
    /// The phase name on which this train becomes purchasable from the depot
    /// even while it is not the head-of-queue train. Mirrors Python's
    /// `Train.available_on` (entities.py:806); in 1830 only the D-train sets
    /// it ("6").
    pub available_on: Option<&'static str>,
    /// Exchange discount: when the buyer trades in a train of the given name,
    /// the depot price drops by the given amount. Mirrors the D-train's
    /// `discount` map (g1830.py:589). Empty for trains without a discount.
    pub discount: &'static [(&'static str, i32)],
}

pub struct PhaseDef {
    pub name: &'static str,
    /// The train purchase that triggers this phase (Ruby/Python `on:`;
    /// None for the opening phase).
    pub on: Option<&'static str>,
    pub train_limit: u8,
    /// Train limit for Minor-class corps when it differs (Ruby's per-type
    /// `train_limit: {minor: 2, major: 4}`; None = same as `train_limit`).
    pub minor_train_limit: Option<u8>,
    pub tiles: &'static [&'static str],
    pub operating_rounds: u8,
    /// Status flags active during this phase (Ruby/Python `status:`;
    /// 1830: `can_buy_companies` in phases 3-4).
    pub status: &'static [&'static str],
}

pub struct MarketCell {
    pub price: i32,
    /// The cell's type flags. 1830 cells carry at most one; 1867 cells
    /// COMBINE them (e.g. `165zCm` = major par + convert range + minor
    /// price cap), which is why this is a list, not a single zone.
    pub zones: Vec<MarketZone>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarketZone {
    Normal,
    /// Par cell (1830; 1867's `p` = par for every corp class).
    Par,
    /// Minor-class par (1867 `x`, Ruby `par_1`).
    Par1,
    /// Major-class par (1867 `z`, Ruby `par_2`).
    Par2,
    /// Minor→major conversion price range (1867 `C`).
    ConvertRange,
    /// Price cap for minor pars (1867 `m`, Ruby `max_price`).
    MaxPrice,
    /// No-cert-limit zone (1830 yellow).
    Yellow,
    /// Unlimited-holdings zone (1830 orange).
    Orange,
    /// Multiple-buy zone (1830 brown).
    Brown,
}

impl MarketZone {
    /// The Ruby/Python type string (`SharePrice.types` entries).
    pub fn type_str(&self) -> &'static str {
        match self {
            MarketZone::Normal => "normal",
            MarketZone::Par => "par",
            MarketZone::Par1 => "par_1",
            MarketZone::Par2 => "par_2",
            MarketZone::ConvertRange => "convert_range",
            MarketZone::MaxPrice => "max_price",
            MarketZone::Yellow => "no_cert_limit",
            MarketZone::Orange => "unlimited",
            MarketZone::Brown => "multiple_buy",
        }
    }
}

pub struct HexDef {
    pub coord: &'static str,
    pub hex_type: HexType,
    pub terrain_cost: i32,
    /// Per-edge specialities (impassable borders, edge-crossing costs).
    /// Empty for hexes without them (all of 1830's blocks are ability- or
    /// geometry-derived; 1867 has impassable river edges + $80 waters).
    pub borders: &'static [BorderDef],
}

/// One hex edge's border speciality (Ruby `borders:` entries).
pub struct BorderDef {
    pub edge: u8,
    /// Cost to cross when laying track over this edge (None with
    /// `impassable: false` is unused; None with `impassable: true` is a
    /// hard wall).
    pub cost: Option<i32>,
    pub impassable: bool,
}

pub enum HexType {
    Blank,
    City {
        revenue: i32,
        slots: u8,
    },
    Town {
        revenue: i32,
    },
    DoubleCity {
        revenue: i32,
    },
    DoubleTown,
    Offboard {
        yellow_revenue: i32,
        brown_revenue: i32,
    },
    Path,
}

// ---------------------------------------------------------------------------
// Shared, title-agnostic private-company ability definitions.
//
// Mirrors the Ruby tobymao/18xx `abilities:` arrays (and the Python engine's
// title data, e.g. rl18xx/game/engine/game/title/g1830.py). Each title's
// `CompanyDef` carries a static list of these; engine code queries them via
// `crate::abilities` instead of keying on company syms, so adding a new
// title's privates means writing data, not engine code.
// ---------------------------------------------------------------------------

/// Who must own the company for the ability to be live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnerType {
    Player,
    Corporation,
}

/// Timing gate (mirrors Ruby/Python `when:`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbilityWhen {
    /// Usable at any step of the owning corporation's OR turn (CS tile lay).
    OwningCorpOrTurn,
    /// Usable at any time (MH exchange `when: "any"`).
    Any,
    /// Triggered when the named corporation buys its first train (BO close).
    BoughtTrain,
}

/// Share sources an exchange ability may draw from (`from: [ipo, market]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShareSource {
    Ipo,
    Market,
}

/// A single private-company ability, transcribed from the title data.
#[derive(Debug)]
pub enum AbilityDef {
    /// Tile lays / upgrades on `hexes` are forbidden while the company is
    /// owned by an entity of `owner_type` (1830: all player-owned blocks).
    BlocksHexes {
        owner_type: OwnerType,
        hexes: &'static [&'static str],
    },
    /// Bonus tile lay on `hexes` restricted to `tiles` (CS on B20).
    TileLay {
        owner_type: OwnerType,
        hexes: &'static [&'static str],
        tiles: &'static [&'static str],
        when: AbilityWhen,
        count: u32,
    },
    /// Lay a tile + station token on an unconnected hex (DH on F16).
    /// Like Ruby's Teleport, timing is implicitly the track step.
    Teleport {
        owner_type: OwnerType,
        hexes: &'static [&'static str],
        tiles: &'static [&'static str],
    },
    /// Exchange the company for a share of one of `corporations` (MH -> NYC).
    Exchange {
        owner_type: OwnerType,
        corporations: &'static [&'static str],
        from: &'static [ShareSource],
        when: AbilityWhen,
    },
    /// The company may not be bought by a corporation (BO).
    NoBuy,
    /// Buying the company grants the share `<corporation>_<share_index>`.
    /// Index 0 is the president's certificate — granting it triggers a
    /// pending par for the corporation (BO -> B&O). Index > 0 is a normal
    /// share granted outright (CA -> PRR_1).
    Shares {
        corporation: &'static str,
        share_index: u8,
    },
    /// +`amount` route revenue for routes visiting any of `hexes` while the
    /// company is owned by an entity of `owner_type` (1867's NFB/MB/QB/SCT).
    HexBonus {
        owner_type: OwnerType,
        hexes: &'static [&'static str],
        amount: i32,
    },
    /// The company closes when the trigger fires (BO closes when B&O buys
    /// its first train).
    Close {
        when: AbilityWhen,
        corporation: &'static str,
    },
}
