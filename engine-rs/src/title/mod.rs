pub mod g1830;

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
    /// The company closes when the trigger fires (BO closes when B&O buys
    /// its first train).
    Close {
        when: AbilityWhen,
        corporation: &'static str,
    },
}
