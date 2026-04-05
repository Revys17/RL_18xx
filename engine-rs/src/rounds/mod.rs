//! Round logic for the 1830 game engine.
//!
//! The game progresses through rounds: Auction → Stock → Operating (repeat).
//! Each round type has its own state and action processing logic.

pub mod auction;
pub mod operating;
pub mod stock;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Round enum — the current round and its associated state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Round {
    Auction(AuctionState),
    Stock(StockState),
    Operating(OperatingState),
}

impl Round {
    pub fn round_type_str(&self) -> &'static str {
        match self {
            Round::Auction(_) => "Auction",
            Round::Stock(_) => "Stock",
            Round::Operating(_) => "Operating",
        }
    }

    pub fn round_num(&self) -> u8 {
        match self {
            Round::Auction(_) => 0,
            Round::Stock(_) => 0,
            Round::Operating(s) => s.round_num,
        }
    }
}

// ---------------------------------------------------------------------------
// Auction round state (Waterfall Auction)
// ---------------------------------------------------------------------------

/// A single bid on a company.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bid {
    pub player_id: u32,
    pub price: i32,
}

/// The waterfall auction state, modelling the Python WaterfallAuction step.
///
/// Companies are sold in order (cheapest first). The cheapest can be purchased
/// outright (placement bid). Non-cheapest companies accumulate bids. When the
/// cheapest is sold, bids on the new cheapest may trigger an auction. If all
/// players pass, the cheapest company's min_bid drops by 5 (to 0 → free).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuctionState {
    /// Remaining company indices (into `game.companies`), sorted by value.
    pub remaining_companies: Vec<usize>,
    /// Current entity index in the round (whose turn it is in waterfall phase).
    pub entity_index: usize,
    /// Player order (player IDs in seating order).
    pub player_order: Vec<u32>,
    /// Bids per company: company_index -> list of bids.
    pub bids: HashMap<usize, Vec<Bid>>,
    /// Which company is currently being auctioned (active auction), if any.
    /// When set, only the lowest bidder can act (bid higher or pass).
    pub auctioning: Option<usize>,
    /// Player-level pass flags (true = passed this waterfall cycle).
    pub passed: Vec<bool>,
    /// Accumulated discount on the cheapest company (SV).
    pub discount: i32,
    /// The company index currently being offered for auction (Python's `self.cheapest`).
    /// When this company is sold, it stays as-is until `auction_entity` updates it.
    pub current_auction_company: Option<usize>,
    /// A corporation pending par: (corp_sym, player_id who must set the par).
    /// Set when a company with triggers_par ability is bought (e.g., BO → B&O).
    pub pending_par: Option<(String, u32)>,
    /// Whether the auction is complete.
    pub finished: bool,
}

impl AuctionState {
    pub fn new(player_ids: &[u32], company_count: usize) -> Self {
        let remaining: Vec<usize> = (0..company_count).collect();
        let passed = vec![false; player_ids.len()];
        let first = remaining.first().copied();
        AuctionState {
            remaining_companies: remaining,
            entity_index: 0,
            player_order: player_ids.to_vec(),
            bids: HashMap::new(),
            auctioning: None,
            passed,
            discount: 0,
            current_auction_company: first,
            pending_par: None,
            finished: false,
        }
    }

    /// The current player in the waterfall (non-auction) phase.
    pub fn current_player_id(&self) -> u32 {
        self.player_order[self.entity_index % self.player_order.len()]
    }

    /// The cheapest remaining company index.
    pub fn cheapest_company(&self) -> Option<usize> {
        self.remaining_companies.first().copied()
    }

    /// Whether there is an active auction (2+ bids on the cheapest company).
    pub fn active_auction(&self) -> Option<(usize, Vec<Bid>)> {
        if let Some(company_idx) = self.auctioning {
            let bids = self.bids.get(&company_idx).cloned().unwrap_or_default();
            if bids.len() > 1 {
                return Some((company_idx, bids));
            }
        }
        None
    }

    /// In an active auction, the player with the lowest bid goes next.
    pub fn active_auction_player(&self) -> Option<u32> {
        self.active_auction()
            .and_then(|(_, bids)| bids.iter().min_by_key(|b| b.price).map(|b| b.player_id))
    }

    /// Who should act right now?
    pub fn active_player_id(&self) -> u32 {
        // Pending par takes priority
        if let Some((_, player_id)) = &self.pending_par {
            return *player_id;
        }
        self.active_auction_player()
            .unwrap_or_else(|| self.current_player_id())
    }

    /// Can the cheapest company be purchased outright (no active auction)?
    pub fn may_purchase(&self, company_idx: usize) -> bool {
        self.auctioning.is_none() && Some(company_idx) == self.cheapest_company()
    }

    /// Get the minimum bid for a company.
    pub fn min_bid_for(&self, company_idx: usize, company_value: i32) -> i32 {
        if self.may_purchase(company_idx) {
            return (company_value - self.discount).max(0);
        }
        let highest = self
            .bids
            .get(&company_idx)
            .and_then(|bids| bids.iter().map(|b| b.price).max())
            .unwrap_or((company_value - self.discount).max(0));
        highest + 5 // MIN_BID_INCREMENT
    }

    /// Calculate committed cash for a player (sum of all outstanding bids).
    pub fn committed_cash(&self, player_id: u32) -> i32 {
        self.bids
            .values()
            .flat_map(|bids| bids.iter())
            .filter(|b| b.player_id == player_id)
            .map(|b| b.price)
            .sum()
    }

    /// Max a player can bid on a company (cash - committed + current bid on this company).
    pub fn max_bid(&self, player_id: u32, company_idx: usize, player_cash: i32) -> i32 {
        let committed = self.committed_cash(player_id);
        let current_on_this = self
            .bids
            .get(&company_idx)
            .and_then(|bids| bids.iter().find(|b| b.player_id == player_id))
            .map_or(0, |b| b.price);
        player_cash - committed + current_on_this
    }

    /// Advance entity_index to next player in waterfall order.
    pub fn advance_entity(&mut self) {
        self.entity_index = (self.entity_index + 1) % self.player_order.len();
    }

    /// Check if all players have passed.
    pub fn all_passed(&self) -> bool {
        self.passed.iter().all(|&p| p)
    }

    /// Clear all pass flags.
    pub fn unpass_all(&mut self) {
        for p in &mut self.passed {
            *p = false;
        }
    }

    /// Remove a company from the remaining list.
    pub fn remove_company(&mut self, company_idx: usize) {
        self.remaining_companies.retain(|&c| c != company_idx);
        self.bids.remove(&company_idx);
    }
}

// ---------------------------------------------------------------------------
// Stock round state
// ---------------------------------------------------------------------------

/// "now" = sold this turn, "prev" = sold in a previous turn this round.
/// Both block buying that corp. In 1830 MUST_SELL_IN_BLOCKS is false so
/// "now" vs "prev" doesn't matter for sell restrictions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SoldTiming {
    Now,
    Prev,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StockState {
    /// Index into player order for whose turn it is.
    pub current_player_index: usize,
    /// Player order (by priority deal).
    pub player_order: Vec<u32>,
    /// Corps sold by each player this round: player_id -> { corp_sym -> timing }.
    /// Persists across turns within the round. "now" → "prev" on turn advance.
    pub players_sold: HashMap<u32, HashMap<String, SoldTiming>>,
    /// Whether current player has bought a share this turn.
    pub bought_this_turn: bool,
    /// Corporation bought this turn (for buy_multiple check).
    pub bought_corp_this_turn: Option<String>,
    /// Whether the player bought from IPO this turn (for multiple_buy_only_from_market).
    pub bought_from_ipo: bool,
    /// Whether current player has parred this turn.
    pub parred_this_turn: bool,
    /// Whether current player has taken any action this turn (buy, sell, par).
    pub acted_this_turn: bool,
    /// Number of consecutive pure passes. Round ends when this reaches player count.
    pub consecutive_passes: u32,
    /// Per-player passed flag. A player is "passed" when they had no valid actions
    /// on their most recent visit (either explicit pass with no prior actions, or
    /// auto-skipped). Matches Python's `entity.passed` per-player tracking.
    /// The round is finished when all players are passed.
    pub player_passed: Vec<bool>,
    /// Priority deal player ID.
    pub priority_deal_player: u32,
    /// Whether the round is finished.
    pub finished: bool,
}

impl StockState {
    pub fn new(player_ids: &[u32], priority_deal_player: u32) -> Self {
        // Reorder players starting from priority deal
        let mut order = Vec::with_capacity(player_ids.len());
        let start_idx = player_ids
            .iter()
            .position(|&id| id == priority_deal_player)
            .unwrap_or(0);
        for i in 0..player_ids.len() {
            order.push(player_ids[(start_idx + i) % player_ids.len()]);
        }
        let num_players = order.len();
        StockState {
            current_player_index: 0,
            player_order: order,
            players_sold: HashMap::new(),
            bought_this_turn: false,
            bought_corp_this_turn: None,
            bought_from_ipo: false,
            parred_this_turn: false,
            acted_this_turn: false,
            consecutive_passes: 0,
            player_passed: vec![false; num_players],
            priority_deal_player,
            finished: false,
        }
    }

    pub fn current_player_id(&self) -> u32 {
        self.player_order[self.current_player_index]
    }

    /// Advance to the next player, transitioning "now" → "prev" for sold tracking.
    pub fn advance_to_next_player(&mut self) {
        // Transition all "now" entries to "prev" for the ending player
        for corps in self.players_sold.values_mut() {
            for timing in corps.values_mut() {
                if *timing == SoldTiming::Now {
                    *timing = SoldTiming::Prev;
                }
            }
        }

        self.current_player_index = (self.current_player_index + 1) % self.player_order.len();
        self.bought_this_turn = false;
        self.bought_corp_this_turn = None;
        self.bought_from_ipo = false;
        self.parred_this_turn = false;
        self.acted_this_turn = false;
    }

    /// Check if a player sold a specific corp at any point in this round.
    /// Both "now" and "prev" block buying.
    pub fn sold_corp_this_round(&self, player_id: u32, corp_sym: &str) -> bool {
        self.players_sold
            .get(&player_id)
            .and_then(|corps| corps.get(corp_sym))
            .is_some()
    }

    /// Record that a player sold a corp this turn.
    pub fn record_sell(&mut self, player_id: u32, corp_sym: &str) {
        self.players_sold
            .entry(player_id)
            .or_default()
            .insert(corp_sym.to_string(), SoldTiming::Now);
    }

    /// Mark the current player as passed (they had no valid actions).
    pub fn mark_current_player_passed(&mut self) {
        self.player_passed[self.current_player_index] = true;
    }

    /// Unpass the current player (they took an action, then passed — still active).
    pub fn unpass_current_player(&mut self) {
        self.player_passed[self.current_player_index] = false;
    }

    /// Check if all players are passed → round should finish.
    pub fn all_players_passed(&self) -> bool {
        self.player_passed.iter().all(|&p| p)
    }
}

// ---------------------------------------------------------------------------
// Operating round state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OperatingStep {
    /// Blocking steps in 1830 OR order:
    /// Track → Token → Route → Dividend → DiscardTrain → BuyTrain → BuyCompany(blocking)
    LayTile,
    PlaceToken,
    RunRoutes,
    Dividend,
    DiscardTrain,
    BuyTrain,
    BuyCompany,
    Done,
}

impl OperatingStep {
    /// Returns the next step in the operating turn sequence.
    pub fn next(&self) -> Self {
        match self {
            OperatingStep::LayTile => OperatingStep::PlaceToken,
            OperatingStep::PlaceToken => OperatingStep::RunRoutes,
            OperatingStep::RunRoutes => OperatingStep::Dividend,
            OperatingStep::Dividend => OperatingStep::DiscardTrain,
            OperatingStep::DiscardTrain => OperatingStep::BuyTrain,
            OperatingStep::BuyTrain => OperatingStep::BuyCompany,
            OperatingStep::BuyCompany => OperatingStep::Done,
            OperatingStep::Done => OperatingStep::Done,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperatingState {
    /// Which OR in the set (1-based: 1, 2, or 3).
    pub round_num: u8,
    /// Total ORs in this set (determined at Stock→OR transition, not mid-OR).
    pub total_ors: u8,
    /// Index into operating order (sorted corps by share price desc).
    pub entity_index: usize,
    /// Operating order: corp syms sorted by share price descending.
    pub operating_order: Vec<String>,
    /// Current step within the operating turn.
    pub step: OperatingStep,
    /// Number of tiles laid this turn (normally max 1).
    pub num_laid_track: u8,
    /// Number of tokens placed this turn (normally max 1).
    pub num_placed_token: u8,
    /// Routes selected for this turn.
    pub routes: Vec<crate::actions::RouteData>,
    /// Revenue from routes this turn.
    pub revenue: i32,
    /// Whether the round is finished.
    pub finished: bool,
    /// Corps that are over the train limit and must discard (set after phase change).
    pub crowded_corps: Vec<String>,
    /// Tokens displaced by OO tile upgrades, awaiting re-placement.
    /// Each entry: (corporation_sym, token_index).
    pub pending_tokens: Vec<(String, usize)>,
}

impl OperatingState {
    pub fn new(round_num: u8, total_ors: u8, operating_order: Vec<String>) -> Self {
        OperatingState {
            round_num,
            total_ors,
            entity_index: 0,
            operating_order,
            step: OperatingStep::LayTile,
            num_laid_track: 0,
            num_placed_token: 0,
            routes: Vec::new(),
            crowded_corps: Vec::new(),
            pending_tokens: Vec::new(),
            revenue: 0,
            finished: false,
        }
    }

    /// Get the current operating corporation's sym.
    pub fn current_corp_sym(&self) -> Option<&str> {
        self.operating_order
            .get(self.entity_index)
            .map(|s| s.as_str())
    }

    /// Advance to the next step in the operating turn.
    pub fn advance_step(&mut self) {
        self.step = self.step.next();
    }

    /// Reset state for the next corporation's turn.
    pub fn advance_to_next_corp(&mut self) {
        self.entity_index += 1;
        self.step = OperatingStep::LayTile;
        self.num_laid_track = 0;
        self.num_placed_token = 0;
        self.routes.clear();
        self.pending_tokens.clear();
        self.revenue = 0;

        if self.entity_index >= self.operating_order.len() {
            self.finished = true;
        }
    }
}
