//! DRAFT static game data for the 1867: Railways of Canada title.
//!
//! Transcribed VERBATIM from the vendored Ruby reference at
//! docs/ruby_reference/g_1867/ (tobymao/18xx lib/engine/game/g_1867/):
//!   entities.rb  — COMPANIES, CORPORATIONS (majors + 16 minors + CN national)
//!   game.rb      — TRAINS, PHASES, COLUMN_MARKET, GRID_MARKET, cash/cert/bank,
//!                  loans, round flow, events
//!   map.rb       — TILES, LOCATION_NAMES, HEXES
//!   stock_market.rb, round/*.rb, step/*.rb — referenced in comments
//!
//! This file is deliberately named `_draft` and is NOT declared in
//! `title/mod.rs` — nothing compiles it. It follows the g1830.rs idiom so it
//! is structurally close to compiling, but several 1867 concepts have no
//! existing definition struct/variant yet. Every such spot carries a
//! `TODO(...)` comment with the verbatim Ruby. See docs/g1867_port_notes.md
//! for the full engine-gap analysis.

#![allow(dead_code)]
#![allow(unused)]

use std::collections::HashMap;

use super::{
    AbilityDef, AbilityWhen, Capitalization, CompanyDef, CorporationDef, HexDef, HexType,
    MarketCell, MarketZone, OwnerType, PhaseDef, ShareSource, TrainDef,
};
use crate::steps::{FinishedRound, RoundStart, RoundTransition, StepDesc, StepKind};

// ---------------------------------------------------------------------------
// Round flow — Ruby G1867::Game#next_round! (game.rb:876-897), verbatim-ish
// pseudocode. The engine's RoundStart/FinishedRound enums (steps.rs) have no
// Merger variant yet — TODO(round-flow).
//
//   def next_round!
//     clear_interest_paid                       # InterestOnLoans bookkeeping
//     @round =
//       case @round
//       when Engine::Round::Stock
//         @operating_rounds = @final_operating_rounds || @phase.operating_rounds
//         reorder_players
//         new_operating_round                   # operating_round calls calculate_interest first
//       when Engine::Round::Operating
//         or_round_finished                     # (game.rb:858-864):
//                                               #   if @phase.status.include?('export_train')
//                                               #     depot.export!         # next train given to CN,
//                                               #                           # triggers phase change as if purchased
//                                               #     post_train_buy        # postevent_trainless_nationalization!
//                                               #                           # if flagged
//                                               #     game_end_check
//         if phase.name.to_i < 3 || phase.name.to_i >= 8
//           new_or!                             # no merger rounds in phases 2 and 8
//         else
//           new_merger_round                    # a MERGER ROUND follows EVERY OR in phases 3-7
//         end
//       when G1867::Round::Merger
//         new_or!                               # merger round hands to the next OR / SR
//       when init_round.class                   # the opening SingleItemAuction round
//         reorder_players
//         new_stock_round
//       end
//   end
//
//   def new_or!                                 # (game.rb:866-874)
//     if @round.round_num < @operating_rounds
//       new_operating_round(@round.round_num + 1)
//     else
//       @turn += 1
//       or_set_finished
//       new_stock_round
//     end
//   end
//
// So the full sequence per turn from phase 3 to 7 is:
//   SR -> OR1 -> MR -> OR2 -> MR -> SR -> ...
// and in phases 2 and 8:
//   SR -> OR1 -> OR2 -> SR -> ...
//
// CN / nationalization timing (no fixed round — event driven):
//   * trainless_nationalization event fires on the first 4, first 6, and
//     first 8 train (TRAINS events, game.rb:200-269). The flag is processed
//     after the train buy / export (post_train_buy -> game.rb:1030-1045
//     postevent_trainless_nationalization!): operating trainless MINORS are
//     nationalized immediately; operating trainless MAJORS are queued in
//     @trainless_major and each may choose to nationalize via the
//     MajorTrainless step (step/major_trainless.rb), which is the first step
//     of the Stock, Merger AND Operating rounds.
//   * minors_nationalized event (first 8 train) nationalizes ALL remaining
//     minors (game.rb:1015-1023 event_minors_nationalized!).
//   * A corporation that cannot pay loan interest in LoanOperations is
//     nationalized (step/loan_operations.rb:316-320 interest_unpaid!).
//   * A corporation left with no train after the BuyTrain step pass is
//     nationalized (step/buy_train.rb:136-139 pass!).
//   * nationalize! itself: game.rb:568-641 (repay loans while cash >= 50,
//     move share price left once + twice per remaining loan, transfer cash
//     to bank, pay every player share_price per share from the BANK, replace
//     its station tokens with CN tokens (highest-revenue city first,
//     respecting the L12 Montreal national reservation), then minors close /
//     majors reset).
//
// Game end (game.rb:315-316, 787-790):
//   GAME_END_CHECK = { bank: :current_or, final_phase: :one_more_full_or_set }
//   GAME_END_TIMING_PRIORITY = %i[one_more_full_or_set current_or]
//   game_end_set_final_turn!: @final_operating_rounds = 3; @final_turn ||= @turn + 1
//   (i.e. the final OR set after the 8-train has THREE ORs, not 2; bank
//   breaking ends at the end of the current OR.)
//   end_game! (game.rb:765-785): every remaining loan moves the corp's share
//   price left one step before scoring; player_value (game.rb:745-763)
//   anticipates the same movement for share valuation.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Round descriptions — DRAFT, comments only.
//
// The StepKind enum (steps.rs:55-74) has none of 1867's new steps yet
// (TODO(steps)): SingleItemAuction, MajorTrainless, HomeToken-in-SR,
// BuySellParShares-via-bid, RedeemShares, BuyCompanyPreloan, LoanOperations,
// Merge, PostMergerShares, ReduceTokens — so the step lists below are the
// verbatim Ruby class orders from game.rb:806-856 rather than StepDesc lists.
// ---------------------------------------------------------------------------

/// 1867's auction round (game.rb:806-810 `new_auction_round`):
///   Engine::Round::Auction.new(self, [
///     G1867::Step::SingleItemAuction,
///   ])
pub fn auction_steps() {
    // TODO(steps): StepKind::SingleItemAuction does not exist.
    // const STEPS: &[StepDesc] = &[StepDesc::blocking(StepKind::SingleItemAuction)];
}

/// 1867's stock round (game.rb:812-819 `stock_round`):
///   G1867::Round::Stock.new(self, [
///     G1867::Step::MajorTrainless,
///     Engine::Step::DiscardTrain,
///     G1867::Step::HomeToken,
///     G1867::Step::BuySellParShares,        # < BuySellParSharesViaBid
///   ])
/// (G1867::Round::Stock overrides sold_out? — only MAJORS get the sold-out
/// price bump, round/stock.rb.)
pub fn stock_steps() {
    // TODO(steps): MajorTrainless / HomeToken(stock) / BuySellParShares-via-bid
    // do not exist as StepKinds.
}

/// 1867's merger round (game.rb:821-837 `merger_round`):
///   # The order of steps in the Grand Trunk Games rules is incorrect
///   # (confirmed by Ian D Wilson https://github.com/tobymao/18xx/issues/9655).
///   # It has buying shares before removing tokens.
///   G1867::Round::Merger.new(self, [
///     G1867::Step::MajorTrainless,
///     G1867::Step::ReduceTokens,            # Step E
///     G1867::Step::PostMergerShares,        # Step C & D
///     Engine::Step::DiscardTrain,           # Step F
///     G1867::Step::Merge,
///   ], round_num: @round.round_num)
pub fn merger_steps() {
    // TODO(steps): an entire new round kind (RoundStart::Merger /
    // FinishedRound::Merger) plus 4 new StepKinds.
}

/// 1867's operating round (game.rb:839-856 `operating_round`):
///   calculate_interest                       # interest owed snapshot BEFORE loans taken this OR
///   G1867::Round::Operating.new(self, [
///     G1867::Step::MajorTrainless,
///     Engine::Step::BuyCompany,
///     G1867::Step::RedeemShares,
///     G1867::Step::Track,                   # includes AutomaticLoan
///     G1867::Step::Token,                   # token cost = price x hex distance
///     Engine::Step::Route,
///     G1867::Step::Dividend,                # payout / half / withhold
///     # The blocking buy company needs to be before loan operations
///     [G1867::Step::BuyCompanyPreloan, { blocks: true }],
///     G1867::Step::LoanOperations,          # auto: pay interest, repay loans
///     Engine::Step::DiscardTrain,
///     G1867::Step::BuyTrain,                # includes AutomaticLoan, EMR via loans only
///     [Engine::Step::BuyCompany, { blocks: true }],
///   ], round_num: round_num)
/// (G1867::Round::Operating skips entities not floated or no longer in
/// @corporations — round/operating.rb.)
pub fn operating_steps() {
    // TODO(steps): RedeemShares / BuyCompanyPreloan / LoanOperations /
    // 1867-Dividend / 1867-Token / 1867-BuyTrain variants do not exist.
}

// ---------------------------------------------------------------------------
// Misc rule constants from game.rb (verbatim)
// ---------------------------------------------------------------------------

// CURRENCY_FORMAT_STR = '$%s'                                  (game.rb:23)
// CAPITALIZATION = :incremental                                (game.rb:31)
// MUST_SELL_IN_BLOCKS = false                                  (game.rb:33)
// TILE_UPGRADES_MUST_USE_MAX_EXITS = %i[cities]                (game.rb:35)
// HOME_TOKEN_TIMING = :par                                     (game.rb:306)
// MUST_BID_INCREMENT_MULTIPLE = true                           (game.rb:307)
// MUST_BUY_TRAIN = :always  # mostly true, needs custom code   (game.rb:308)
// POOL_SHARE_DROP = :none                                      (game.rb:309)
// SELL_MOVEMENT = :down_block_pres                             (game.rb:310)
// ALL_COMPANIES_ASSIGNABLE = true                              (game.rb:311)
// SELL_AFTER = :operate                                        (game.rb:312)
// SELL_BUY_ORDER = :sell_buy                                   (game.rb:313)
// EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST = false                    (game.rb:314)
// GAME_END_CHECK = { bank: :current_or, final_phase: :one_more_full_or_set } (game.rb:315)
// GAME_END_TIMING_PRIORITY = %i[one_more_full_or_set current_or]             (game.rb:316)
// CERT_LIMIT_CHANGE_ON_BANKRUPTCY = true                       (game.rb:321)
// TILE_LAYS = [                                                (game.rb:330-333)
//   { lay: true, upgrade: true },
//   { lay: true, upgrade: :not_if_upgraded, cost: 20, cannot_reuse_same_hex: true },
// ]  # Two lays with one being an upgrade, second tile costs 20
// CORPORATION_SIZES = { 2 => :small, 10 => :large }            (game.rb:358)
// TRAINS_REMOVE_2_PLAYER = { '2' => 3, '3' => 2, '4' => 1, '5' => 1, '6' => 1, '7' => 1 } (game.rb:362)
// HOME tokens: corporations have NO preprinted home hex; the home token
// location is CHOSEN when the corporation pars (HOME_TOKEN_TIMING = :par,
// home_token_locations game.rb:448-469: minors may take any untokened city
// incl. a disconnected Toronto/Montreal slot; majors need a city with no
// normal tokens anywhere on the hex, preferring fully unconnected hexes).

/// Revenue bonus hexes: a route that includes one of Toronto/Montreal/Quebec
/// AND Timmins earns +$40 (x train multiplier). (game.rb:318-319, 673-680)
pub const BONUS_CAPITALS: &[&str] = &["F16", "L12", "O7"];
pub const BONUS_REVENUE: &str = "D2";

/// A token spot in Montreal is reserved for the CN national. (game.rb:360)
pub const NATIONAL_RESERVATIONS: &[&str] = &["L12"];

/// Minors only available once the first 3-train is bought
/// (`green_minors_available` event). (game.rb:361)
pub const GREEN_CORPORATIONS: &[&str] = &["BBG", "LPS", "QLS", "SLA", "TGB", "THB"];

/// Loans (game.rb:899-907 init_loans / loan_value; 376-378 interest_rate;
/// 444-446 maximum_loans):
///   @loan_value = 50; 72 loans total ("16 minors * 2, 8 majors * 5")
///   interest_rate = 5 (constant, per loan per OR)
///   maximum_loans: major => 5, minor => 2
///   take_loan (game.rb:499-508): entity receives loan.amount - 5 (i.e. $45)
///   repay_loan (game.rb:510-516): entity pays the full amount ($50)
///   buying_power(full) (game.rb:524-530):
///     cash + (maximum_loans - loans.size) * (@loan_value - 5)
pub const LOAN_VALUE: i32 = 50;
pub const NUM_LOANS: u32 = 72;
pub const INTEREST_RATE: i32 = 5;
pub const MAXIMUM_LOANS_MAJOR: u32 = 5;
pub const MAXIMUM_LOANS_MINOR: u32 = 2;

/// After a merger the surviving major keeps at most 2 tokens (in different
/// hexes); a $40 token is added back if it ends with only 2 (game.rb:335,
/// 794-798 fix_token_count!, step/reduce_tokens.rb).
pub const LIMIT_TOKENS_AFTER_MERGER: u32 = 2;

/// Minimum par/min_price for minors (game.rb:336, init_corporations).
pub const MINIMUM_MINOR_PRICE: i32 = 50;

/// Stock-round bid rules (step/buy_sell_par_shares.rb:32-34):
///   MIN_BID = 100; MAX_MINOR_PAR = 135; MAJOR_PHASE = 4
pub const MIN_BID: i32 = 100;
pub const MAX_MINOR_PAR: i32 = 135;
pub const MAJOR_PHASE: u8 = 4;

/// Merge limits (step/merge.rb:411-412): at most 10 minors per merger; one
/// player may contribute at most 6 (so they end at 60%).
pub const LIMIT_MERGE: u32 = 10;
pub const LIMIT_OWNED_BY_ONE_ENTITY: u32 = 6;

// ---------------------------------------------------------------------------
// Corporation data — the 8 majors (entities.rb CORPORATIONS, type: 'major')
// ---------------------------------------------------------------------------

/// entities.rb omits `shares:` for the majors, so the tobymao base default
/// applies: a 20% president's certificate + eight 10% certificates
/// (corporation.rb default `[20, 10, 10, 10, 10, 10, 10, 10, 10]`) — 1867
/// majors are 10-share corporations.
const SHARES_1867_MAJOR: &[u8] = &[20, 10, 10, 10, 10, 10, 10, 10, 10];

/// All 8 majors share these Ruby fields verbatim:
///   float_percent: 20, always_market_price: true, tokens: [0, 20, 40],
///   type: 'major'
/// TODO(corp): CorporationDef has no `always_market_price` (treasury shares
///   are bought at MARKET price, not par) and no `type` field
///   (major/minor/national); both are load-bearing for 1867.
/// TODO(home): CorporationDef.home_hex is mandatory, but 1867 majors have NO
///   preprinted home — the player chooses the home city when the corporation
///   pars (HOME_TOKEN_TIMING = :par; game.rb:448-469 home_token_locations).
///   `home_hex: ""` below is a placeholder.
pub fn corporations() -> Vec<CorporationDef> {
    vec![
        CorporationDef {
            sym: "CNR",
            name: "Canadian Northern Railway",
            token_prices: &[0, 20, 40],
            home_hex: "", // TODO(home): chosen at par
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1867_MAJOR,
            float_percent: 20,
            capitalization: Capitalization::Incremental,
        },
        CorporationDef {
            sym: "CPR",
            name: "Canadian Pacific Railway",
            token_prices: &[0, 20, 40],
            home_hex: "", // TODO(home): chosen at par
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1867_MAJOR,
            float_percent: 20,
            capitalization: Capitalization::Incremental,
        },
        CorporationDef {
            sym: "C&O",
            name: "Chesapeake and Ohio Railway",
            token_prices: &[0, 20, 40],
            home_hex: "", // TODO(home): chosen at par
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1867_MAJOR,
            float_percent: 20,
            capitalization: Capitalization::Incremental,
        },
        CorporationDef {
            sym: "GTR",
            name: "Grand Trunk Railway",
            token_prices: &[0, 20, 40],
            home_hex: "", // TODO(home): chosen at par
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1867_MAJOR,
            float_percent: 20,
            capitalization: Capitalization::Incremental,
        },
        CorporationDef {
            sym: "GWR",
            name: "Great Western Railway",
            token_prices: &[0, 20, 40],
            home_hex: "", // TODO(home): chosen at par
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1867_MAJOR,
            float_percent: 20,
            capitalization: Capitalization::Incremental,
        },
        CorporationDef {
            sym: "ICR",
            name: "Intercolonial Railway",
            token_prices: &[0, 20, 40],
            home_hex: "", // TODO(home): chosen at par
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1867_MAJOR,
            float_percent: 20,
            capitalization: Capitalization::Incremental,
        },
        CorporationDef {
            sym: "NTR",
            name: "National Transcontinental Railway",
            token_prices: &[0, 20, 40],
            home_hex: "", // TODO(home): chosen at par
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1867_MAJOR,
            float_percent: 20,
            capitalization: Capitalization::Incremental,
        },
        CorporationDef {
            sym: "NYC",
            name: "New York Central Railroad",
            token_prices: &[0, 20, 40],
            home_hex: "", // TODO(home): chosen at par
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1867_MAJOR,
            float_percent: 20,
            capitalization: Capitalization::Incremental,
        },
    ]
}

// ---------------------------------------------------------------------------
// Minor data — 16 minors (entities.rb CORPORATIONS, type: 'minor')
//
// NOTE: in this vendored Ruby there is no separate MINORS constant — minors
// are full Corporation entries with type: 'minor'. game.rb:368 comment:
// "Minors are done as corporations with a size of 2" (CORPORATION_SIZES
// game.rb:358 maps total_shares 2 => :small). Each minor verbatim has:
//   float_percent: 100, always_market_price: true, tokens: [0],
//   shares: [100], max_ownership_percent: 100, type: 'minor'
// Minors have NO home hex either — the owner places the home token on any
// untokened city when the minor is won at auction (HOME_TOKEN_TIMING = :par,
// G1867::Step::HomeToken in the stock round).
// They are started by BID (min $100, step/buy_sell_par_shares.rb): the
// winner pays the bid into the minor's treasury and it pars at
// min(bid / 2, 135) rounded down to a par_1/par cell.
// ---------------------------------------------------------------------------

/// DRAFT definition struct for 1867 minors — no MinorDef exists in
/// title/mod.rs yet (TODO(minor)).
pub struct MinorDef {
    pub sym: &'static str,
    pub name: &'static str,
    /// Ruby `float_percent: 100` — floats when its single share is bought.
    pub float_percent: u8,
    /// Ruby `tokens: [0]` — one free token (placed as the chosen home).
    pub token_prices: &'static [i32],
    /// Ruby `shares: [100]` — a single 100% certificate. The game still
    /// treats a minor as size 2 for dividends (owner gets half, treasury
    /// half) and conversion (owner gets a 20% president cert of the major).
    pub shares: &'static [u8],
    /// Ruby `max_ownership_percent: 100`.
    pub max_ownership_percent: u8,
    /// Member of GREEN_CORPORATIONS (game.rb:361): only becomes startable
    /// when the first 3-train triggers `green_minors_available`.
    pub green: bool,
    pub capitalization: Capitalization,
}

pub fn minors() -> Vec<MinorDef> {
    const T: &[i32] = &[0];
    const S: &[u8] = &[100];
    fn m(sym: &'static str, name: &'static str, green: bool) -> MinorDef {
        MinorDef {
            sym,
            name,
            float_percent: 100,
            token_prices: T,
            shares: S,
            max_ownership_percent: 100,
            green,
            capitalization: Capitalization::Incremental,
        }
    }
    vec![
        m("BBG", "Buffalo, Brantford, and Goderich", true),
        m("BO", "Brockville and Ottawa", false),
        m("CS", "Canada Southern", false),
        m("CV", "Credit Valley Railway", false),
        m("KP", "Kingston and Pembroke", false),
        m("LPS", "London and Port Stanley", true),
        m("OP", "Ottawa and Prescott", false),
        m("SLA", "St. Lawrence and Atlantic", true),
        m("TGB", "Toronto, Grey, and Bruce", true),
        m("TN", "Toronto and Nipissing", false),
        m("AE", "Algoma Eastern Railway", false),
        m("CA", "Canada Atlantic Railway", false),
        m("NO", "New York and Ottawa", false),
        m("PM", "Pere Marquette Railway", false),
        m("QLS", "Quebec and Lake St. John", true),
        m("THB", "Toronto, Hamilton and Buffalo", true),
    ]
}

// ---------------------------------------------------------------------------
// The CN national (entities.rb CORPORATIONS, type: 'national')
//
// Verbatim Ruby:
//   { sym: 'CN', name: 'Canadian National', logo: '1867/CN',
//     tokens: [0, 0, 0, 0, 0, 0, 0, 0], shares: [100],
//     hide_shares: true, type: 'national', color: '#ef4223' }
//
// TODO(national): no definition struct / entity kind exists. The CN never
// operates and holds no shares (game.rb:963-967 setup: ipoed = true, shares
// cleared, removed from @corporations). It exists to hold tokens:
//   * 8 free tokens, placed when corporations are nationalized
//     (game.rb:568-641 nationalize!),
//   * neutral (passable) green tokens on D2 (Timmins, first city) and L12
//     (Montreal, last city) plus its first real token on F16 Toronto at
//     setup (game.rb:913-930 add_neutral_tokens); the green tokens are
//     removed when the first 3-train is bought (event_green_minors_available!),
//   * a reserved slot on L12 Montreal (NATIONAL_RESERVATIONS), claimed when
//     gray tile 639 is laid (game.rb:651-660 place_639_token, hooked from
//     step/track.rb lay_tile).
// ---------------------------------------------------------------------------
pub struct NationalDef {
    pub sym: &'static str,
    pub name: &'static str,
    pub token_prices: &'static [i32],
    pub shares: &'static [u8],
}

pub fn national() -> NationalDef {
    NationalDef {
        sym: "CN",
        name: "Canadian National",
        token_prices: &[0, 0, 0, 0, 0, 0, 0, 0],
        shares: &[100], // hide_shares: true — never sold
    }
}

// ---------------------------------------------------------------------------
// Private company data (entities.rb COMPANIES, 6 companies)
//
// All five real privates carry a `discount:` equal to value - 20: in the
// opening SingleItemAuction nobody bidding reduces the price by $5 at a time
// (dutch auction, step/single_item_auction.rb enter_dutch: company.discount
// += 5), and `min_bid = value - discount` — so the listed discount caps the
// price floor at $20. TODO(company): CompanyDef has no `discount` field.
// CompanyPriceUpToFace (game.rb:365): corporations may buy a private for
// $1 up to its face value during phases with `can_buy_companies`.
// On the first 6-train, `nationalize_companies` closes every private, paying
// the owner face value from the bank (game.rb:1047-1062).
// ---------------------------------------------------------------------------

pub fn companies() -> Vec<CompanyDef> {
    vec![
        // Ruby: { name: 'rules until start of phase 3', sym: '3', value: 3,
        //   revenue: 0, desc: 'Hidden corporation',
        //   abilities: [{ type: 'blocks_hexes', hexes: ['M13'] }] }
        // The "hidden company" (game.rb:960-961): never auctioned, never
        // owned; blocks the Granby hex M13 until the first 3-train closes it
        // (event_green_minors_available! game.rb:980-991).
        CompanyDef {
            sym: "3",
            name: "rules until start of phase 3",
            value: 3,
            revenue: 0,
            // TODO(ability): Ruby ability has NO owner_type —
            //   { type: 'blocks_hexes', hexes: ['M13'] }
            // it blocks unconditionally while the (unowned, hidden) company
            // is open. AbilityDef::BlocksHexes forces an owner_type; Player
            // is the closest fit but the gating semantics differ.
            abilities: &[AbilityDef::BlocksHexes {
                owner_type: OwnerType::Player,
                hexes: &["M13"],
            }],
        },
        // Ruby: { name: 'Champlain & St. Lawrence', sym: 'C&SL', value: 30,
        //   revenue: 10, discount: 10, desc: 'No special abilities.' }
        CompanyDef {
            sym: "C&SL",
            name: "Champlain & St. Lawrence",
            value: 30,
            revenue: 10,
            // TODO(company): discount: 10
            abilities: &[],
        },
        // Ruby: { name: 'Niagara Falls Bridge', sym: 'NFB', value: 45,
        //   revenue: 15, discount: 15,
        //   desc: 'When owned by a corporation, they gain $10 extra revenue
        //          for each of their routes that include Buffalo',
        //   abilities: [{ type: 'hex_bonus', owner_type: 'corporation',
        //                 hexes: ['F18'], amount: 10 }] }
        CompanyDef {
            sym: "NFB",
            name: "Niagara Falls Bridge",
            value: 45,
            revenue: 15,
            // TODO(company): discount: 15
            // TODO(ability): no HexBonus variant in AbilityDef —
            //   { type: 'hex_bonus', owner_type: 'corporation',
            //     hexes: ['F18'], amount: 10 }
            // (route revenue hook, game.rb:667-670 revenue_for)
            abilities: &[],
        },
        // Ruby: { name: 'Montreal Bridge', sym: 'MB', value: 60, revenue: 20,
        //   discount: 20,
        //   desc: 'When owned by a corporation, they gain $10 extra revenue
        //          for each of their routes that include Montreal',
        //   abilities: [{ type: 'hex_bonus', owner_type: 'corporation',
        //                 hexes: ['L12'], amount: 10 }] }
        CompanyDef {
            sym: "MB",
            name: "Montreal Bridge",
            value: 60,
            revenue: 20,
            // TODO(company): discount: 20
            // TODO(ability): { type: 'hex_bonus', owner_type: 'corporation',
            //                  hexes: ['L12'], amount: 10 }
            abilities: &[],
        },
        // Ruby: { name: 'Quebec Bridge', sym: 'QB', value: 75, revenue: 25,
        //   discount: 25,
        //   desc: 'When owned by a corporation, they gain $10 extra revenue
        //          for each of their routes that include Quebec',
        //   abilities: [{ type: 'hex_bonus', owner_type: 'corporation',
        //                 hexes: ['O7'], amount: 10 }] }
        CompanyDef {
            sym: "QB",
            name: "Quebec Bridge",
            value: 75,
            revenue: 25,
            // TODO(company): discount: 25
            // TODO(ability): { type: 'hex_bonus', owner_type: 'corporation',
            //                  hexes: ['O7'], amount: 10 }
            abilities: &[],
        },
        // Ruby: { name: 'St. Clair Tunnel', sym: 'SCT', value: 90,
        //   revenue: 30, discount: 30,
        //   desc: 'When owned by a corporation, they gain $10 extra revenue
        //          for each of their routes that include Detroit',
        //   abilities: [{ type: 'hex_bonus', owner_type: 'corporation',
        //                 hexes: %w[A19 A17], amount: 10 }] }
        CompanyDef {
            sym: "SCT",
            name: "St. Clair Tunnel",
            value: 90,
            revenue: 30,
            // TODO(company): discount: 30
            // TODO(ability): { type: 'hex_bonus', owner_type: 'corporation',
            //                  hexes: ['A19', 'A17'], amount: 10 }
            abilities: &[],
        },
    ]
}

// ---------------------------------------------------------------------------
// Train data (game.rb TRAINS, 9 types)
//
// Every 1867 train distance is BUCKETED, e.g. for the 2-train:
//   distance: [{ 'nodes' => ['city', 'offboard'], 'pay' => 2, 'visit' => 2 },
//              { 'nodes' => ['town'], 'pay' => 0, 'visit' => 99 }]
// i.e. it pays/visits N cities-or-offboards and may run through ANY number
// of towns for free. TrainDef.distance is a scalar u32, so the city-count N
// is stored and the full spec kept in a comment per train.
// TODO(train): bucketed distances need a real representation for the router
// (compute_stops, game.rb:706-739, picks the best optional-stop combination
// and requires a tokened stop among the counted stops).
// TODO(train): no `multiplier` field (2+2 and 5+5E double their revenue).
// TODO(train): count u32 cannot express Ruby num: 'unlimited' (8, 2+2, 5+5E);
//   u32::MAX used as a stand-in.
// TODO(train): the discount map on 8/2+2/5+5E is only usable in phase 8
//   (step/buy_train.rb discountable_trains_allowed? == (phase == 8), enabled
//   narratively by the 'train_trade_allowed' event: "Trains can be traded in
//   for 50% towards Phase 8 trains").
// ---------------------------------------------------------------------------

/// Trade-in discounts on the three phase-8 trains (identical map on each):
///   discount: { '5' => 275, '6' => 325, '7' => 400, '8' => 500,
///               '2+2' => 300, '5+5E' => 750 }
const PHASE8_DISCOUNT: &[(&str, i32)] = &[
    ("5", 275),
    ("6", 325),
    ("7", 400),
    ("8", 500),
    ("2+2", 300),
    ("5+5E", 750),
];

pub fn trains() -> Vec<TrainDef> {
    vec![
        // distance: [{nodes: [city offboard], pay: 2, visit: 2},
        //            {nodes: [town], pay: 0, visit: 99}]
        TrainDef {
            name: "2",
            distance: 2,
            price: 100,
            count: 10,
            rusts_on: Some("4"),
            events: &[],
            available_on: None,
            discount: &[],
        },
        // distance: [{nodes: [city offboard], pay: 3, visit: 3},
        //            {nodes: [town], pay: 0, visit: 99}]
        TrainDef {
            name: "3",
            distance: 3,
            price: 225,
            count: 7,
            rusts_on: Some("6"),
            events: &["green_minors_available"],
            available_on: None,
            discount: &[],
        },
        // distance: [{nodes: [city offboard], pay: 4, visit: 4},
        //            {nodes: [town], pay: 0, visit: 99}]
        TrainDef {
            name: "4",
            distance: 4,
            price: 350,
            count: 4,
            rusts_on: Some("8"),
            events: &["majors_can_ipo", "trainless_nationalization"],
            available_on: None,
            discount: &[],
        },
        // distance: [{nodes: [city offboard], pay: 5, visit: 5},
        //            {nodes: [town], pay: 0, visit: 99}]
        TrainDef {
            name: "5",
            distance: 5,
            price: 550,
            count: 4,
            rusts_on: None,
            events: &["minors_cannot_start"],
            available_on: None,
            discount: &[],
        },
        // distance: [{nodes: [city offboard], pay: 6, visit: 6},
        //            {nodes: [town], pay: 0, visit: 99}]
        TrainDef {
            name: "6",
            distance: 6,
            price: 650,
            count: 2,
            rusts_on: None,
            events: &["nationalize_companies", "trainless_nationalization"],
            available_on: None,
            discount: &[],
        },
        // distance: [{nodes: [city offboard], pay: 7, visit: 7},
        //            {nodes: [town], pay: 0, visit: 99}]
        TrainDef {
            name: "7",
            distance: 7,
            price: 800,
            count: 2,
            rusts_on: None,
            events: &[],
            available_on: None,
            discount: &[],
        },
        // distance: [{nodes: [city offboard], pay: 8, visit: 8},
        //            {nodes: [town], pay: 0, visit: 99}]
        // The 8-train is the permanent end-game train: unlimited supply,
        // rusts the 4s, fires minors_nationalized (ALL remaining minors are
        // nationalized into CN), trainless_nationalization (trainless
        // operating majors may choose nationalization) and
        // train_trade_allowed (the discount map below becomes usable).
        TrainDef {
            name: "8",
            distance: 8,
            price: 1000,
            count: u32::MAX, // Ruby num: 'unlimited'
            rusts_on: None,
            events: &[
                "minors_nationalized",
                "trainless_nationalization",
                "train_trade_allowed",
            ],
            available_on: None,
            discount: PHASE8_DISCOUNT,
        },
        // distance: [{nodes: [city offboard], pay: 2, visit: 2},
        //            {nodes: [town], pay: 0, visit: 99}],  multiplier: 2
        TrainDef {
            name: "2+2",
            distance: 2, // TODO(train): multiplier: 2 (doubles revenue)
            price: 600,
            count: u32::MAX, // Ruby num: 'unlimited'
            rusts_on: None,
            events: &[],
            available_on: Some("8"),
            discount: PHASE8_DISCOUNT,
        },
        // distance: [{nodes: [offboard], pay: 5, visit: 5},
        //            {nodes: [city town], pay: 0, visit: 99}],  multiplier: 2
        // NOTE the inverted buckets: the 5+5E pays ONLY offboards (up to 5)
        // and runs through cities and towns for free; compute_stops
        // additionally requires at least one counted stop to be tokened by
        // the running corporation (game.rb:720, 729-730).
        TrainDef {
            name: "5+5E",
            distance: 5, // TODO(train): multiplier: 2; pay-bucket is OFFBOARDS only
            price: 1500,
            count: u32::MAX, // Ruby num: 'unlimited'
            rusts_on: None,
            events: &[],
            available_on: Some("8"),
            discount: PHASE8_DISCOUNT,
        },
    ]
}

// ---------------------------------------------------------------------------
// Phase data (game.rb PHASES, 7 phases)
//
// TODO(phase): Ruby train_limit is a PER-TYPE hash (minor/major);
// PhaseDef.train_limit is a single u8. The MAJOR limit is recorded in the
// field (for phase '2', which has no major limit because majors cannot
// exist yet, the minor limit is recorded); the verbatim hash is in the
// comment on each phase.
// NOTE: every 1867 phase has operating_rounds: 2 — but phases 3-7 interleave
// a Merger Round after each OR, and the final OR set is extended to 3 ORs by
// game_end_set_final_turn! (game.rb:787-790).
// 'export_train' status (phases 4-7): at the end of each OR the next
// available train is exported — given to the CN — triggering phase change as
// if purchased (STATUS_TEXT game.rb:323-327, or_round_finished game.rb:858).
// ---------------------------------------------------------------------------

pub fn phases() -> Vec<PhaseDef> {
    vec![
        PhaseDef {
            name: "2",
            on: None,
            // TODO(phase): train_limit: { minor: 2 } — no major limit
            // (majors cannot be started before phase 4).
            train_limit: 2,
            tiles: &["yellow"],
            operating_rounds: 2,
            status: &[],
        },
        PhaseDef {
            name: "3",
            on: Some("3"),
            // TODO(phase): train_limit: { minor: 2, major: 4 }
            train_limit: 4,
            tiles: &["yellow", "green"],
            operating_rounds: 2,
            status: &["can_buy_companies"],
        },
        PhaseDef {
            name: "4",
            on: Some("4"),
            // TODO(phase): train_limit: { minor: 1, major: 3 }
            train_limit: 3,
            tiles: &["yellow", "green"],
            operating_rounds: 2,
            status: &["can_buy_companies", "export_train"],
        },
        PhaseDef {
            name: "5",
            on: Some("5"),
            // TODO(phase): train_limit: { minor: 1, major: 3 }
            train_limit: 3,
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 2,
            status: &["can_buy_companies", "export_train"],
        },
        PhaseDef {
            name: "6",
            on: Some("6"),
            // TODO(phase): train_limit: { minor: 1, major: 2 }
            train_limit: 2,
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 2,
            status: &["export_train"],
        },
        PhaseDef {
            name: "7",
            on: Some("7"),
            // TODO(phase): train_limit: { minor: 1, major: 2 }
            train_limit: 2,
            tiles: &["yellow", "green", "brown", "gray"],
            operating_rounds: 2,
            status: &["export_train"],
        },
        PhaseDef {
            name: "8",
            on: Some("8"),
            // TODO(phase): train_limit: { major: 2 } — no minors remain
            // (minors_nationalized fires on the first 8-train).
            train_limit: 2,
            tiles: &["yellow", "green", "brown", "gray"],
            operating_rounds: 2,
            status: &[],
        },
    ]
}

// ---------------------------------------------------------------------------
// Starting cash, cert limits, bank cash (game.rb:25-29)
// ---------------------------------------------------------------------------

/// STARTING_CASH = { 2 => 420, 3 => 420, 4 => 315, 5 => 252, 6 => 210 }
pub fn starting_cash(num_players: u8) -> i32 {
    match num_players {
        2 => 420,
        3 => 420,
        4 => 315,
        5 => 252,
        6 => 210,
        _ => panic!("Invalid player count: {}", num_players),
    }
}

/// CERT_LIMIT = { 2 => 21, 3 => 21, 4 => 16, 5 => 13, 6 => 11 }
/// A flat per-player-count hash in 1867 (no per-corporation-count nesting,
/// unlike some titles). BUT CERT_LIMIT_CHANGE_ON_BANKRUPTCY = true
/// (game.rb:321): when a player goes bankrupt the limit is recomputed as if
/// the game had one fewer player — cert_limit cannot be a pure function of
/// the starting player count. TODO(cert-limit).
pub fn cert_limit(num_players: u8) -> u8 {
    match num_players {
        2 => 21,
        3 => 21,
        4 => 16,
        5 => 13,
        6 => 11,
        _ => panic!("Invalid player count: {}", num_players),
    }
}

pub const BANK_CASH: i32 = 15_000;

// ---------------------------------------------------------------------------
// Stock market — default 1-D COLUMN_MARKET (game.rb:37-66)
//
// 1867 uses Engine::StockMarket subclass G1867::StockMarket
// (stock_market.rb) with ONE-dimensional movement by default:
//   * up() on a 1-D market is right(); sell movement is left.
//   * right() for a MINOR sitting on a :max_price cell ('m') does NOT move
//     (minors are price-capped at 165); majors move normally.
// Cell suffix letters (tobymao share_price.rb TYPE_MAP + 1867 MARKET_TEXT
// game.rb:352-357):
//   x = :par_1         "Minor Corporation Par"          (orange)
//   z = :par_2         "Major Corporation Par"          (green)
//   p = :par           "Major/Minor Corporation Par"
//   C = :convert_range "Price range to convert minor to major" (blue)
//   m = :max_price     "Maximum price for a minor"
// Par-price selection (step/buy_sell_par_shares.rb get_all_par_prices):
//   majors par on %i[par_2 par] (z + p cells, i.e. $70-$200),
//   minors par on %i[par_1 par] (x + p cells, i.e. $50-$135 — capped at
//   MAX_MINOR_PAR 135).
//   Conversion / merge prices come from the C range ($100-$165) and z/p
//   cells (step/merge.rb finish_convert / finish_merge_to_major).
//
// TODO(market-zones): MarketZone { Normal, Par, Yellow, Orange, Brown }
// cannot represent these: the suffixes are COMBINABLE FLAGS on one cell
// (100pC, 150zC, 165zCm). MarketCell needs a set of type flags, e.g.
//   ParMinor (x), ParMajor (z), ParBoth (p), ConvertRange (C), MaxPrice (m)
// Mapping chosen for this draft: any par-capable cell (p/x/z) => Par,
// everything else => Normal; the verbatim Ruby code string is kept as a
// comment on every non-plain cell.
// ---------------------------------------------------------------------------

pub fn market_grid() -> Vec<Vec<Option<MarketCell>>> {
    fn mc(price: i32, zone: MarketZone) -> Option<MarketCell> {
        Some(MarketCell { price, zone })
    }

    use MarketZone::*;

    // COLUMN_MARKET: a single row, 28 cells.
    vec![vec![
        mc(35, Normal),
        mc(40, Normal),
        mc(45, Normal),
        mc(50, Par),  // "50x"  par_1 (minor par)
        mc(55, Par),  // "55x"  par_1
        mc(60, Par),  // "60x"  par_1
        mc(65, Par),  // "65x"  par_1
        mc(70, Par),  // "70p"  par (major + minor)
        mc(80, Par),  // "80p"  par
        mc(90, Par),  // "90p"  par
        mc(100, Par), // "100pC" par + convert_range
        mc(110, Par), // "110pC" par + convert_range
        mc(120, Par), // "120pC" par + convert_range
        mc(135, Par), // "135pC" par + convert_range (MAX_MINOR_PAR)
        mc(150, Par), // "150zC" par_2 (major par) + convert_range
        mc(165, Par), // "165zCm" par_2 + convert_range + max_price (minor cap)
        mc(180, Par), // "180z" par_2
        mc(200, Par), // "200z" par_2
        mc(220, Normal),
        mc(245, Normal),
        mc(270, Normal),
        mc(300, Normal),
        mc(330, Normal),
        mc(360, Normal),
        mc(400, Normal),
        mc(440, Normal),
        mc(490, Normal),
        mc(540, Normal),
    ]]
}

/// The OPTIONAL 2-D market variant (game.rb:68-142 GRID_MARKET; meta.rb
/// optional rule :grid_market "Play with the Grid (2D) Stock Market from
/// 1861 rather than the default Column (1D) Stock Market").
/// G1867::StockMarket adds 1861/1867 2-D quirks (stock_market.rb): up() at
/// the ceiling moves right-and-down-one for MAJORS (price unchanged);
/// right() for a minor on a :max_price cell becomes up().
/// Same TODO(market-zones) flag problem as the column market.
pub fn grid_market() -> Vec<Vec<Option<MarketCell>>> {
    fn mc(price: i32, zone: MarketZone) -> Option<MarketCell> {
        Some(MarketCell { price, zone })
    }

    use MarketZone::*;

    vec![
        // Row 0
        vec![
            None, // ''
            None, // ''
            None, // ''
            None, // ''
            mc(135, Normal),
            mc(150, Normal),
            mc(165, Normal), // "165mC" max_price + convert_range
            mc(180, Normal),
            mc(200, Par), // "200z" par_2
            mc(220, Normal),
            mc(245, Normal),
            mc(270, Normal),
            mc(300, Normal),
            mc(330, Normal),
            mc(360, Normal),
            mc(400, Normal),
            mc(440, Normal),
            mc(490, Normal),
            mc(540, Normal),
        ],
        // Row 1
        vec![
            None, // ''
            None, // ''
            None, // ''
            mc(110, Normal),
            mc(120, Normal),
            mc(135, Normal),
            mc(150, Normal), // "150mC" max_price + convert_range
            mc(165, Par),    // "165z" par_2
            mc(180, Par),    // "180z" par_2
            mc(200, Normal),
            mc(220, Normal),
            mc(245, Normal),
            mc(270, Normal),
            mc(300, Normal),
            mc(330, Normal),
            mc(360, Normal),
            mc(400, Normal),
            mc(440, Normal),
            mc(490, Normal),
        ],
        // Row 2
        vec![
            None, // ''
            None, // ''
            mc(90, Normal),
            mc(100, Normal),
            mc(110, Normal),
            mc(120, Normal),
            mc(135, Par), // "135pmC" par + max_price + convert_range
            mc(150, Par), // "150z" par_2
            mc(165, Normal),
            mc(180, Normal),
            mc(200, Normal),
            mc(220, Normal),
            mc(245, Normal),
            mc(270, Normal),
            mc(300, Normal),
            mc(330, Normal),
            mc(360, Normal),
            mc(400, Normal),
            mc(440, Normal),
        ],
        // Row 3
        vec![
            None, // ''
            mc(70, Normal),
            mc(80, Normal),
            mc(90, Normal),
            mc(100, Normal),
            mc(110, Par), // "110p" par
            mc(120, Par), // "120pmC" par + max_price + convert_range
            mc(135, Normal),
            mc(150, Normal),
            mc(165, Normal),
            mc(180, Normal),
            mc(200, Normal),
        ],
        // Row 4
        vec![
            mc(60, Normal),
            mc(65, Normal),
            mc(70, Normal),
            mc(80, Normal),
            mc(90, Par),     // "90p" par
            mc(100, Par),    // "100p" par
            mc(110, Normal), // "110mC" max_price + convert_range
            mc(120, Normal),
            mc(135, Normal),
            mc(150, Normal),
        ],
        // Row 5
        vec![
            mc(55, Normal),
            mc(60, Normal),
            mc(65, Normal),
            mc(70, Par),     // "70p" par
            mc(80, Par),     // "80p" par
            mc(90, Normal),
            mc(100, Normal), // "100mC" max_price + convert_range
            mc(110, Normal),
        ],
        // Row 6
        vec![
            mc(50, Normal),
            mc(55, Normal),
            mc(60, Par), // "60x" par_1
            mc(65, Par), // "65x" par_1
            mc(70, Normal),
            mc(80, Normal),
        ],
        // Row 7
        vec![
            mc(45, Normal),
            mc(50, Par), // "50x" par_1
            mc(55, Par), // "55x" par_1
            mc(60, Normal),
            mc(65, Normal),
        ],
        // Row 8
        vec![mc(40, Normal), mc(45, Normal), mc(50, Normal), mc(55, Normal)],
        // Row 9
        vec![mc(35, Normal), mc(40, Normal), mc(45, Normal)],
    ]
}

// ---------------------------------------------------------------------------
// Tile counts (map.rb TILES)
// ---------------------------------------------------------------------------

pub fn tile_counts() -> Vec<(&'static str, u32)> {
    vec![
        ("3", 2),
        ("4", 4),
        ("5", 2),
        ("6", 2),
        ("7", u32::MAX),  // Ruby: 'unlimited'
        ("8", u32::MAX),  // Ruby: 'unlimited'
        ("9", u32::MAX),  // Ruby: 'unlimited'
        ("14", 2),
        ("15", 4),
        ("16", 2),
        ("17", 2),
        ("18", 2),
        ("19", 2),
        ("20", 2),
        ("21", 2),
        ("22", 2),
        ("23", 5),
        ("24", 5),
        ("25", 4),
        ("26", 2),
        ("27", 2),
        ("28", 2),
        ("29", 2),
        ("30", 2),
        ("31", 2),
        ("39", 2),
        ("40", 2),
        ("41", 2),
        ("42", 2),
        ("43", 2),
        ("44", 2),
        ("45", 2),
        ("46", 2),
        ("47", 2),
        ("57", 2),
        ("58", 4),
        ("63", 3),
        ("70", 2),
        ("87", 2),
        ("88", 2),
        ("120", 1),
        ("122", 1),
        ("124", 1),
        ("201", 3),
        ("202", 3),
        ("204", 2),
        ("207", 5),
        ("208", 2),
        ("611", 3),
        ("619", 2),
        ("621", 2),
        ("622", 2),
        ("623", 3),
        ("624", 1),
        ("625", 1),
        ("626", 1),
        ("637", 1),
        ("639", 1),
        ("801", 2),
        ("911", 3),
        // Custom 1867 tiles — see custom_tiles() for the DSL.
        ("X1", 1),
        ("X2", 1),
        ("X3", 1),
        ("X4", 1),
        ("X5", 1),
        ("X6", 1),
        ("X7", 1),
        ("X8", 1),
    ]
}

/// Custom 1867 tile definitions (map.rb TILES, the X* entries) — verbatim
/// Ruby tile DSL. X1-X4 are the green Montreal tiles (label M), X5-X7 the
/// brown Montreal tiles, X8 the gray Ottawa tile (label O — Ottawa J12
/// carries future_label=label:O,color:gray).
/// TODO(tiles): the tile catalog (tiles.rs::tile_catalog_1830) has no 1867
/// entries; these DSL strings need to be parsed into TileDefs. Laying tile
/// '639' (the gray Montreal tile from the standard catalog) triggers the CN
/// reservation token placement (game.rb:651-660 place_639_token via
/// step/track.rb).
pub fn custom_tiles() -> Vec<(&'static str, &'static str, &'static str, u32)> {
    // (id, color, code, count)
    vec![
        (
            "X1",
            "green",
            "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;\
             path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4;path=a:2,b:_2;path=a:_2,b:5;label=M",
            1,
        ),
        (
            "X2",
            "green",
            "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;\
             path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:5;path=a:2,b:_2;\
             path=a:_2,b:4;label=M",
            1,
        ),
        (
            "X3",
            "green",
            "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;\
             path=a:_0,b:4;path=a:1,b:_1;path=a:_1,b:2;path=a:3,b:_2;\
             path=a:_2,b:5;label=M",
            1,
        ),
        (
            "X4",
            "green",
            "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;\
             path=a:1,b:_1;path=a:_1,b:2;path=a:4,b:_2;path=a:_2,b:5;label=M",
            1,
        ),
        (
            "X5",
            "brown",
            "city=revenue:70,slots:2;city=revenue:70;path=a:0,b:_1;path=a:1,b:_0;\
             path=a:2,b:_0;path=a:3,b:_1;path=a:4,b:_0;path=a:5,b:_0;label=M",
            1,
        ),
        (
            "X6",
            "brown",
            "city=revenue:70,slots:2;city=revenue:70;path=a:0,b:_0;path=a:3,b:_0;\
             path=a:4,b:_0;path=a:5,b:_0;path=a:1,b:_1;path=a:2,b:_1;label=M",
            1,
        ),
        (
            "X7",
            "brown",
            "city=revenue:70,slots:2;city=revenue:70;path=a:0,b:_0;path=a:1,b:_0;\
             path=a:2,b:_1;path=a:3,b:_0;path=a:4,b:_1;path=a:5,b:_0;label=M",
            1,
        ),
        (
            "X8",
            "gray",
            "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;\
             path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=O",
            1,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Location names (map.rb LOCATION_NAMES, verbatim)
// ---------------------------------------------------------------------------

pub fn location_names() -> HashMap<&'static str, &'static str> {
    [
        ("D2", "Timmins ($80 if includes T/M/Q)"),
        ("D8", "Sudbury"),
        ("F8", "North Bay"),
        ("E13", "Barrie"),
        ("E15", "Guelph"),
        ("E17", "Hamilton"),
        ("D16", "Berlin"),
        ("C17", "London"),
        ("G15", "Peterborough"),
        ("I15", "Kingston"),
        ("J12", "Ottawa"),
        ("M9", "Trois-Rivières"),
        ("O7", "Quebec"),
        ("N12", "Sherbrooke"),
        ("C15", "Goderich"),
        ("B18", "Sarnia"),
        ("H14", "Belleville"),
        ("H10", "Pembroke"),
        ("K13", "Cornwall"),
        ("L10", "St. Jerome"),
        ("M13", "Granby"),
        ("L12", "Montreal"),
        ("F16", "Toronto"),
        ("A7", "Sault Ste. Marie"),
        ("F18", "Buffalo"),
        ("M15", "New England"),
        ("O13", "Maine"),
        ("P8", "Maritime Provinces"),
        ("A19", "Detroit"),
    ]
    .iter()
    .copied()
    .collect()
}

// ---------------------------------------------------------------------------
// Hex grid definitions (map.rb HEXES)
//
// 94 hexes: 78 white, 5 gray, 2 yellow, 7 red, 2 blue.
// TODO(hex) summary for the whole map:
//   * impassable borders (D18, C9, D10, E11, C11, D12, C13 + the blue lakes'
//     edges) — HexDef has no border field; 1830's equivalents are hardcoded
//     per-hex in factored.rs (~line 1136).
//   * per-EDGE water border costs of $80 (N8, N10, M11, O9, M9, L10) —
//     terrain_cost is per-hex, and a border cost is paid when track CROSSES
//     that edge, which is a different mechanic.
//   * stubs (G15 stub:1, L10 stub:0, K13 stub:4) — preprinted track stubs
//     that new tiles must respect (game.rb includes StubsAreRestricted).
//   * labels Y / M / T / O and future_label (J12: gray O) gate upgrades.
//   * L12 Montreal is a TRIPLE city (three separate 1-slot cities) — no
//     HexType variant; DoubleCity is the closest fit.
//   * red offboards have FOUR revenue tiers (yellow|green|brown|gray) —
//     Offboard { yellow_revenue, brown_revenue } keeps yellow/brown only.
//   * A17+A19 are one grouped offboard (groups:Detroit, A17 hide:1).
//   * blue (lake port) hexes E19/H16 with flat revenue 10 — no HexType.
// ---------------------------------------------------------------------------

/// Plain white hexes (no DSL): map.rb HEXES white => ''
const PLAIN_WHITE_HEXES: &[&str] = &[
    "B6", "B8", "C5", "C7", "C19", "D4", "D6", "D14", "E3", "E5", "E7", "E9", "F2", "F4", "F6",
    "F10", "F12", "F14", "G3", "G5", "G7", "G9", "G11", "G13", "H4", "H6", "H8", "H12", "I5", "I7",
    "I9", "I11", "I13", "J6", "J8", "J10", "J14", "K5", "K7", "K9", "L6", "L8", "M5", "M7", "N6",
    "O11",
];

pub fn hex_definitions() -> Vec<HexDef> {
    use HexType::*;

    let mut hexes: Vec<HexDef> = PLAIN_WHITE_HEXES
        .iter()
        .map(|&coord| HexDef {
            coord,
            hex_type: Blank,
            terrain_cost: 0,
        })
        .collect();

    hexes.extend(vec![
        // --- White hexes with impassable borders ---
        // TODO(hex): 'border=edge:5,type:impassable'
        HexDef { coord: "D18", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:0,type:impassable;border=edge:5,type:impassable'
        HexDef { coord: "C9", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:2,type:impassable;border=edge:1,type:impassable;
        //             border=edge:0,type:impassable;border=edge:5,type:impassable'
        HexDef { coord: "D10", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:2,type:impassable;border=edge:1,type:impassable'
        HexDef { coord: "E11", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:0,type:impassable;border=edge:3,type:impassable;
        //             border=edge:4,type:impassable'
        HexDef { coord: "C11", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:3,type:impassable;border=edge:4,type:impassable'
        HexDef { coord: "D12", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:3,type:impassable'
        HexDef { coord: "C13", hex_type: Blank, terrain_cost: 0 },
        // --- White hexes with water costs ---
        // 'upgrade=cost:20,terrain:water' — a whole-hex lay cost, fits terrain_cost.
        HexDef { coord: "K11", hex_type: Blank, terrain_cost: 20 },
        // TODO(hex): per-edge water borders, NOT a hex lay cost:
        // 'border=edge:0,type:water,cost:80;border=edge:5,type:water,cost:80'
        HexDef { coord: "N8", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:2,type:water,cost:80;border=edge:3,type:water,cost:80'
        HexDef { coord: "N10", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:2,type:water,cost:80;border=edge:3,type:water,cost:80'
        HexDef { coord: "M11", hex_type: Blank, terrain_cost: 0 },
        // TODO(hex): 'border=edge:2,type:water,cost:80'
        HexDef { coord: "O9", hex_type: Blank, terrain_cost: 0 },
        // --- White city hexes ---
        // TODO(hex): M9 Trois-Rivières also has
        // 'border=edge:5,type:water,cost:80;border=edge:0,type:water,cost:80'
        HexDef { coord: "M9", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "D8", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "F8", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "E13", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "E15", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "C17", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "I15", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "N12", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        // TODO(hex): 'city=revenue:0;stub=edge:1' (StubsAreRestricted)
        HexDef { coord: "G15", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        // TODO(hex): label=Y on E17 / D16 / O7 ('city=revenue:0;label=Y')
        HexDef { coord: "E17", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "D16", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "O7", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0 },
        // TODO(hex): 'city=revenue:0;label=Y;future_label=label:O,color:gray;
        //             upgrade=cost:20,terrain:water' (Ottawa — gray X8 tile)
        HexDef { coord: "J12", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 20 },
        // --- White town hexes ---
        // TODO(hex): 'town=revenue:0;border=edge:5,type:water,cost:80;stub=edge:0'
        HexDef { coord: "L10", hex_type: Town { revenue: 0 }, terrain_cost: 0 },
        // TODO(hex): 'town=revenue:0;border=edge:0,type:impassable'
        HexDef { coord: "H14", hex_type: Town { revenue: 0 }, terrain_cost: 0 },
        HexDef { coord: "C15", hex_type: Town { revenue: 0 }, terrain_cost: 0 },
        HexDef { coord: "B18", hex_type: Town { revenue: 0 }, terrain_cost: 0 },
        HexDef { coord: "H10", hex_type: Town { revenue: 0 }, terrain_cost: 0 },
        HexDef { coord: "M13", hex_type: Town { revenue: 0 }, terrain_cost: 0 },
        // TODO(hex): 'town=revenue:0;stub=edge:4'
        HexDef { coord: "K13", hex_type: Town { revenue: 0 }, terrain_cost: 0 },
        // --- Gray hexes (preprinted) ---
        // Timmins: capital-bonus partner hex ($80 with T/M/Q via 40 base +40
        // bonus); holds a neutral CN green token until phase 3.
        // TODO(hex): 'border=edge:1;border=edge:4' (plain border decoration)
        HexDef { coord: "D2", hex_type: City { revenue: 40, slots: 1 }, terrain_cost: 0 },
        HexDef { coord: "C3", hex_type: Path, terrain_cost: 0 },
        HexDef { coord: "E1", hex_type: Path, terrain_cost: 0 },
        HexDef { coord: "B16", hex_type: Path, terrain_cost: 0 },
        HexDef { coord: "L14", hex_type: Path, terrain_cost: 0 },
        // --- Yellow hexes (preprinted) ---
        // TODO(hex): L12 Montreal is a TRIPLE city:
        // 'city=revenue:40;city=revenue:40;city=revenue:40,loc:5;path=a:1,b:_0;
        //  path=a:3,b:_1;label=M;upgrade=cost:20,terrain:water'
        // (third city holds a neutral CN token until phase 3 and the CN
        // national reservation). DoubleCity is the closest existing fit.
        HexDef { coord: "L12", hex_type: DoubleCity { revenue: 40 }, terrain_cost: 20 },
        // Toronto: 'city=revenue:30;city=revenue:30;path=a:1,b:_0;path=a:4,b:_1;label=T'
        HexDef { coord: "F16", hex_type: DoubleCity { revenue: 30 }, terrain_cost: 0 },
        // --- Red hexes (offboard) ---
        // TODO(hex): all 1867 offboards have FOUR revenue tiers
        // (yellow|green|brown|gray); Offboard keeps yellow/brown only.
        // 'offboard=revenue:yellow_20|green_30|brown_40|gray_40'
        HexDef { coord: "A7", hex_type: Offboard { yellow_revenue: 20, brown_revenue: 40 }, terrain_cost: 0 },
        // 'offboard=revenue:yellow_30|green_40|brown_50|gray_60'
        HexDef { coord: "F18", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 50 }, terrain_cost: 0 },
        // 'offboard=revenue:yellow_30|green_40|brown_50|gray_60'
        HexDef { coord: "M15", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 50 }, terrain_cost: 0 },
        // 'offboard=revenue:yellow_20|green_30|brown_40|gray_40'
        HexDef { coord: "O13", hex_type: Offboard { yellow_revenue: 20, brown_revenue: 40 }, terrain_cost: 0 },
        // 'offboard=revenue:yellow_30|green_30|brown_40|gray_40'
        HexDef { coord: "P8", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 40 }, terrain_cost: 0 },
        // TODO(hex): A17 + A19 form ONE grouped offboard (groups:Detroit,
        // A17 hide:1) — counts once for run/revenue purposes.
        // 'offboard=revenue:yellow_30|green_40|brown_50|gray_70,hide:1,groups:Detroit'
        HexDef { coord: "A17", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 50 }, terrain_cost: 0 },
        // 'offboard=revenue:yellow_30|green_40|brown_50|gray_70,groups:Detroit'
        HexDef { coord: "A19", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 50 }, terrain_cost: 0 },
        // --- Blue hexes (lake ports) ---
        // TODO(hex): no HexType for blue port hexes; flat revenue 10 at all
        // phases. 'offboard=revenue:10;path=a:3,b:_0;border=edge:2,type:impassable'
        HexDef { coord: "E19", hex_type: Offboard { yellow_revenue: 10, brown_revenue: 10 }, terrain_cost: 0 },
        // 'offboard=revenue:10;path=a:2,b:_0;path=a:4,b:_0;border=edge:3,type:impassable'
        HexDef { coord: "H16", hex_type: Offboard { yellow_revenue: 10, brown_revenue: 10 }, terrain_cost: 0 },
    ]);

    hexes
}

// ---------------------------------------------------------------------------
// Preprinted hex DSL strings (map.rb HEXES, verbatim)
//
// NOTE: unlike 1830, 1867's WHITE hexes also carry DSL (borders, stubs,
// labels, upgrade costs, untiled cities/towns). The 1830 loader treats a
// preprinted-DSL hex as a fixed-track hex, so white entries below need new
// handling (their DSL is hex ATTRIBUTES, not preprinted track) —
// TODO(hex-loader). They are included verbatim for completeness.
// ---------------------------------------------------------------------------

pub fn preprinted_hex_dsl(coord: &str) -> Option<(&'static str, &'static str)> {
    // Returns (dsl_string, color_str)
    match coord {
        // --- Gray hexes ---
        "D2" => Some((
            "city=revenue:40;path=a:0,b:_0;path=a:1,b:_0;path=a:4,b:_0;\
             path=a:5,b:_0;border=edge:1;border=edge:4",
            "gray",
        )),
        "C3" => Some(("path=a:0,b:4;border=edge:4", "gray")),
        "E1" => Some(("path=a:1,b:5;border=edge:1", "gray")),
        "B16" => Some(("path=a:0,b:5", "gray")),
        "L14" => Some(("path=a:2,b:3", "gray")),
        // --- Yellow hexes ---
        "L12" => Some((
            "city=revenue:40;city=revenue:40;city=revenue:40,loc:5;path=a:1,b:_0;\
             path=a:3,b:_1;label=M;upgrade=cost:20,terrain:water",
            "yellow",
        )),
        "F16" => Some((
            "city=revenue:30;city=revenue:30;path=a:1,b:_0;path=a:4,b:_1;label=T",
            "yellow",
        )),
        // --- Red offboard hexes ---
        "A7" => Some((
            "offboard=revenue:yellow_20|green_30|brown_40|gray_40;path=a:4,b:_0;path=a:5,b:_0",
            "red",
        )),
        "F18" => Some((
            "offboard=revenue:yellow_30|green_40|brown_50|gray_60;path=a:2,b:_0",
            "red",
        )),
        "M15" => Some((
            "offboard=revenue:yellow_30|green_40|brown_50|gray_60;path=a:3,b:_0",
            "red",
        )),
        "O13" => Some((
            "offboard=revenue:yellow_20|green_30|brown_40|gray_40;path=a:2,b:_0;path=a:3,b:_0",
            "red",
        )),
        "P8" => Some((
            "offboard=revenue:yellow_30|green_30|brown_40|gray_40;path=a:2,b:_0;path=a:1,b:_0",
            "red",
        )),
        "A17" => Some((
            "offboard=revenue:yellow_30|green_40|brown_50|gray_70,hide:1,groups:Detroit;\
             path=a:5,b:_0;border=edge:0",
            "red",
        )),
        "A19" => Some((
            "offboard=revenue:yellow_30|green_40|brown_50|gray_70,groups:Detroit;\
             path=a:4,b:_0;border=edge:3",
            "red",
        )),
        // --- Blue hexes ---
        "E19" => Some((
            "offboard=revenue:10;path=a:3,b:_0;border=edge:2,type:impassable",
            "blue",
        )),
        "H16" => Some((
            "offboard=revenue:10;path=a:2,b:_0;path=a:4,b:_0;border=edge:3,type:impassable",
            "blue",
        )),
        // --- White hexes with DSL (attributes, not preprinted track) ---
        "D18" => Some(("border=edge:5,type:impassable", "white")),
        "C9" => Some((
            "border=edge:0,type:impassable;border=edge:5,type:impassable",
            "white",
        )),
        "D10" => Some((
            "border=edge:2,type:impassable;border=edge:1,type:impassable;\
             border=edge:0,type:impassable;border=edge:5,type:impassable",
            "white",
        )),
        "E11" => Some((
            "border=edge:2,type:impassable;border=edge:1,type:impassable",
            "white",
        )),
        "C11" => Some((
            "border=edge:0,type:impassable;border=edge:3,type:impassable;\
             border=edge:4,type:impassable",
            "white",
        )),
        "D12" => Some((
            "border=edge:3,type:impassable;border=edge:4,type:impassable",
            "white",
        )),
        "C13" => Some(("border=edge:3,type:impassable", "white")),
        "K11" => Some(("upgrade=cost:20,terrain:water", "white")),
        "N8" => Some((
            "border=edge:0,type:water,cost:80;border=edge:5,type:water,cost:80",
            "white",
        )),
        "N10" | "M11" => Some((
            "border=edge:2,type:water,cost:80;border=edge:3,type:water,cost:80",
            "white",
        )),
        "O9" => Some(("border=edge:2,type:water,cost:80", "white")),
        "M9" => Some((
            "city=revenue:0;border=edge:5,type:water,cost:80;border=edge:0,type:water,cost:80",
            "white",
        )),
        "D8" | "F8" | "E13" | "E15" | "C17" | "I15" | "N12" => Some(("city=revenue:0", "white")),
        "G15" => Some(("city=revenue:0;stub=edge:1", "white")),
        "E17" | "D16" | "O7" => Some(("city=revenue:0;label=Y", "white")),
        "J12" => Some((
            "city=revenue:0;label=Y;future_label=label:O,color:gray;upgrade=cost:20,terrain:water",
            "white",
        )),
        "L10" => Some((
            "town=revenue:0;border=edge:5,type:water,cost:80;stub=edge:0",
            "white",
        )),
        "H14" => Some(("town=revenue:0;border=edge:0,type:impassable", "white")),
        "C15" | "B18" | "H10" | "M13" => Some(("town=revenue:0", "white")),
        "K13" => Some(("town=revenue:0;stub=edge:4", "white")),
        _ => None,
    }
}
