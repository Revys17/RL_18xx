//! What the site layer records as a game goes: the records Ruby keeps and the board shows, which the RL engine has
//! no use for. Compiled in every build (`BaseGame::site` holds it), kept only in site mode (`site_rules`): a search
//! clone of an RL game copies one null pointer.

use std::collections::BTreeMap;

/// One entry of a corporation's operating history (Ruby `OperatingInfo`, keyed by `[turn, round_num]`).
#[derive(Clone, Debug, PartialEq)]
pub struct OpInfo {
    pub turn: u32,
    pub round: u8,
    pub revenue: i32,
    /// Ruby's dividend kind: "payout", "withhold" (also the automatic withhold of a company that didn't run), "half"
    pub kind: String,
    /// The company didn't run (Ruby: the entry's routes are empty)
    pub no_run: bool,
}

/// Each player's value at the end of a round (Ruby `Player#history`, `store_player_info`).
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerInfo {
    /// The round's short name ("SR", "OR", "PR"), turn and round number
    pub round: String,
    pub turn: u32,
    pub round_num: u8,
    pub values: Vec<(u32, i32)>,
}

#[derive(Clone, Debug, Default)]
pub struct SiteRecords {
    pub op_history: BTreeMap<String, Vec<OpInfo>>,
    pub player_history: Vec<PlayerInfo>,
}
