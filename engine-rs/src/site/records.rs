//! What the site layer records as a game goes: the records Ruby keeps and the board shows, which the RL engine has
//! no use for. Compiled in every build (`BaseGame::site` holds it), kept only in site mode (`site_rules`): a search
//! clone of an RL game copies one null pointer.

use crate::game::BaseGame;
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

/// A round's short name as Ruby's `store_player_info` writes it (`round.short_name`): "PR" the private auction,
/// "SR", "OR", "MR".
pub fn round_short(round: &crate::rounds::Round) -> &'static str {
    match round {
        crate::rounds::Round::Auction(_) => "PR",
        crate::rounds::Round::Stock(_) => "SR",
        crate::rounds::Round::Operating(_) => "OR",
        crate::rounds::Round::Merger(_) => "MR",
    }
}

impl BaseGame {
    /// A corporation's dividend step ran (Ruby `process_dividend` writes `operating_history[[turn, round_num]]`):
    /// a dividend action, or the automatic withhold of a company that didn't run. Called only with `site_rules`.
    pub(crate) fn site_note_operated(&mut self, corp: &str, revenue: i32, kind: &str, no_run: bool) {
        let round = match &self.round {
            crate::rounds::Round::Operating(s) => s.round_num,
            _ => 1,
        };
        let turn = self.turn;
        let Some(rec) = self.site.as_mut() else { return };
        let list = rec.op_history.entry(corp.to_string()).or_default();
        // one entry per [turn, round] (Ruby's hash key): a later write replaces it
        list.retain(|e| !(e.turn == turn && e.round == round));
        list.push(OpInfo { turn, round, revenue, kind: kind.to_string(), no_run });
    }

    /// `site_note_operated` for an operating round named by the caller (the site layer records what the engine does
    /// without a hook: 1867's minors' automatic payout, docs/SITE_ENGINE_RS_1867.md §3.1).
    pub(crate) fn site_note_operated_at(&mut self, corp: &str, turn: u32, round: u8, revenue: i32, kind: &str, no_run: bool) {
        let Some(rec) = self.site.as_mut() else { return };
        let list = rec.op_history.entry(corp.to_string()).or_default();
        list.retain(|e| !(e.turn == turn && e.round == round));
        list.push(OpInfo { turn, round, revenue, kind: kind.to_string(), no_run });
    }

    /// A corporation starts afresh (Ruby `reset_corporation`: a new Corporation, its operating history empty).
    pub(crate) fn site_forget_operated(&mut self, corp: &str) {
        if let Some(rec) = self.site.as_mut() {
            rec.op_history.remove(corp);
        }
    }

    /// A round has finished (Ruby `transition_to_next_round!` / `end_game!` call `store_player_info`). Called only
    /// with `site_rules`.
    pub(crate) fn site_note_round_finished(&mut self) {
        if self.site.is_none() {
            return;
        }
        // Ruby's round_num: 1 outside operating and merger rounds
        let (short, num) = (round_short(&self.round), self.round.round_num().max(1));
        let results = self.calculate_results();
        let values: Vec<(u32, i32)> =
            self.players.iter().map(|p| (p.id, results.get(&p.id).copied().unwrap_or(0))).collect();
        let turn = self.turn;
        if let Some(rec) = self.site.as_mut() {
            rec.player_history.push(PlayerInfo { round: short.to_string(), turn, round_num: num, values });
        }
    }
}
