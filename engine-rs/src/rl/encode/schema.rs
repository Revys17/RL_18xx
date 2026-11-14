//! Feature schemas: the name of every column, per token type. The encoder
//! writes features in exactly these orders (a debug assertion checks each
//! token's count), `rl_encoder_schema()` exports them to Python and
//! `docs/rl_encoder.md` documents their normalizations.
//!
//! Money is normalized by the game's starting cash per player ("sc"),
//! prices on the share market by the title's highest market price, stop
//! revenues by the title's highest stop revenue, lay / border costs by the
//! title's highest such cost, train prices by the title's dearest train.

/// Hex token features (`hex_feat[..., k]`).
pub const HEX_FEATURES: &[&str] = &[
    // position (layout-aware, normalized to [0, 1] over the title's map)
    "pos_x",
    "pos_y",
    // tile color one-hot
    "color_white",
    "color_yellow",
    "color_green",
    "color_brown",
    "color_gray",
    "color_red",
    // tile state
    "tile_laid",
    "preprinted",
    "has_label",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "rot_4",
    "rot_5",
    // stops
    "cities",        // / 6
    "slots",         // / 6
    "free_slots",    // / 6
    "towns",         // / 2
    "offboard",      // 0/1
    "city_revenue",  // max city revenue / max stop revenue
    "town_revenue",  // max town revenue / max stop revenue
    "offboard_revenue", // current-phase revenue / max stop revenue
    "offboard_yellow",
    "offboard_green",
    "offboard_brown",
    "offboard_gray",
    "offboard_grouped",
    // costs
    "upgrade_cost", // / max lay cost
    "terrain_mountain",
    "terrain_water",
    "terrain_hill",
    "terrain_swamp",
    "terrain_forest",
    "terrain_river",
    "terrain_other",
    "wall_0",
    "wall_1",
    "wall_2",
    "wall_3",
    "wall_4",
    "wall_5",
    "border_cost_0", // unpaid crossing cost / max lay cost
    "border_cost_1",
    "border_cost_2",
    "border_cost_3",
    "border_cost_4",
    "border_cost_5",
    // track (49): see TRACK_FEATURES
    "track_0_1", "track_0_2", "track_0_3", "track_0_4", "track_0_5",
    "track_1_2", "track_1_3", "track_1_4", "track_1_5",
    "track_2_3", "track_2_4", "track_2_5",
    "track_3_4", "track_3_5",
    "track_4_5",
    "via_0_1", "via_0_2", "via_0_3", "via_0_4", "via_0_5",
    "via_1_2", "via_1_3", "via_1_4", "via_1_5",
    "via_2_3", "via_2_4", "via_2_5",
    "via_3_4", "via_3_5",
    "via_4_5",
    "edge0_city", "edge0_town", "edge0_offboard",
    "edge1_city", "edge1_town", "edge1_offboard",
    "edge2_city", "edge2_town", "edge2_offboard",
    "edge3_city", "edge3_town", "edge3_offboard",
    "edge4_city", "edge4_town", "edge4_offboard",
    "edge5_city", "edge5_town", "edge5_offboard",
    "junction",
    // state
    "blocked",          // a live private blocks lays here
    "hex_bonus",        // open companies' route bonus here / max stop revenue
    "tokens",           // tokens in slots / 6
    "neutral_tokens",   // / 6
    "extra_tokens",     // tokens outside slots / 2
    "reserved_slots",   // / 6
    "laid_this_turn",
    // -- version 2 (the 1822 family's map state; 0 for 1830 / 1867) --------
    "cubes",            // builder cubes on the hex's tile / 2 (1822PNW, 1822MX)
    "forest",           // 1822PNW forest marker (+$10 a route crossing it; no upgrade)
    "assigned",         // company assignments on the hex (1822PNW P15 / P17, 1822CA P13)
    "assignment_bonus", // route bonus of the hex's assignments / forest / max stop revenue
    "grain_elevator",   // 1822CA
    "port_west",        // 1822CA
    "port_east",        // 1822CA
    "dest_tokens",      // destination tokens in the hex's slots / 2
    "coal_tokens",      // 1822PNW coal tokens / 2
    "token_for_sale",   // a token here is up for auction (1822MX NdeM tokens): its price / sc
    "token_on_block",   // ... and is the one being bid on now
];

/// Tile features (`rl_spec_info(title)["tile_feat"]`): a catalog tile at
/// rotation 0, the stop and track part of the hex schema.
pub const TILE_FEATURES: &[&str] = &[
    "color_white",
    "color_yellow",
    "color_green",
    "color_brown",
    "color_gray",
    "color_red",
    "has_label",
    "cities",
    "slots",
    "towns",
    "offboard",
    "city_revenue",
    "town_revenue",
    "upgrade_cost",
    "track_0_1", "track_0_2", "track_0_3", "track_0_4", "track_0_5",
    "track_1_2", "track_1_3", "track_1_4", "track_1_5",
    "track_2_3", "track_2_4", "track_2_5",
    "track_3_4", "track_3_5",
    "track_4_5",
    "via_0_1", "via_0_2", "via_0_3", "via_0_4", "via_0_5",
    "via_1_2", "via_1_3", "via_1_4", "via_1_5",
    "via_2_3", "via_2_4", "via_2_5",
    "via_3_4", "via_3_5",
    "via_4_5",
    "edge0_city", "edge0_town", "edge0_offboard",
    "edge1_city", "edge1_town", "edge1_offboard",
    "edge2_city", "edge2_town", "edge2_offboard",
    "edge3_city", "edge3_town", "edge3_offboard",
    "edge4_city", "edge4_town", "edge4_offboard",
    "edge5_city", "edge5_town", "edge5_offboard",
    "junction",
    "supply",   // initial count / 10 (unlimited -> 1)
];

pub const GLOBAL_FEATURES: &[&str] = &[
    "bank_cash",        // / initial bank
    "bank_broken",
    "game_end_triggered",
    "final_turn_set",
    "finished",
    "turn",             // / 20
    "round_num",        // OR k of the set / ORs in the set (0 outside ORs)
    "total_ors",        // ORs in the set / 3
    "phase_index",      // / (phases - 1)
    "num_seats",        // / MAX_SEATS
    "cert_limit",       // / 30
    "starting_cash",    // sc / initial bank
    "loans_remaining",  // / title loans (0 without loans)
    "loan_value",       // / sc
    "loan_interest",    // / loan value
    "market_pool_limit", // percent / 100
    "tiles_laid_this_turn", // / 2
    "tokens_placed_this_turn",
    "route_revenue",    // this turn's route revenue / sc
    "auction_discount", // / sc
    "auction_active",   // a lot is being bid on
    "sold_this_turn",
    "bought_this_turn",
    "parred_this_turn",
    "consecutive_passes", // / seats
    "events_fired",     // / 8
    "trainless_majors", // / 4
    "pending_tokens",   // / 2
    "crowded_corps",    // / 2
    // -- version 2 (the 1822 family; 0 for 1830 / 1867) ---------------------
    "sr_idle",          // in a stock round, nothing sold, bought or bid yet this game (the first SR ending so ends the game)
    "debt_interest",    // player debt interest rate / 100
    "bid_tokens",       // bidding tokens per player / 10
    "bids_this_turn",   // bids placed this turn / bids allowed per turn
    "upgraded_this_turn", // the operator has upgraded a tile this turn
    "pullman_attached", // this turn's routes: the pullman is attached to a train
    "grain_attached",   // 1822CA: the grain train is attached
    "attach_skipped",   // 1822CA: pullman / grain attachment declined this turn, / 2
    // -- version 3 (1822's optional rules; 0 for every other title and for base rules) ---
    "rule_plus_expansion",     // 1822+ is in play (six more minors, three more privates, category bid boxes)
    "rule_plus_no_removals",   // 1822+ without the setup removal of three privates
    "rule_plus_single_stack",  // 1822+ privates in one stack filling the three boxes (not by category)
    "rule_tax_haven_multiple", // the tax haven (P16) may own several shares
];

/// Version 3's rule flags: (global feature, the optional rule, the rule it needs). A flag is 1 when the
/// game's optional rules hold the rule and the rule it needs (a modifier without 1822+ does nothing, as in
/// Ruby): 1822's options only, so every other title's flags are 0 (1822CA's built-in multi-share tax haven
/// is its title's rule, not an option).
pub const RULE_FLAGS: &[(&str, &str, Option<&str>)] = &[
    ("rule_plus_expansion", "plus_expansion", None),
    ("rule_plus_no_removals", "plus_expansion_no_removals", Some("plus_expansion")),
    ("rule_plus_single_stack", "plus_expansion_single_stack", Some("plus_expansion")),
    ("rule_tax_haven_multiple", "tax_haven_multiple", None),
];

pub const PLAYER_FEATURES: &[&str] = &[
    "present",
    "is_mover",
    "cash",            // / sc
    "certs",           // / cert limit
    "cert_room",       // (limit - certs) / limit
    "priority_deal",
    "order_pos",       // stock / auction round order from the priority deal, / (seats - 1)
    "committed_bids",  // / sc
    "net_worth",       // cash + shares at market + company face values, / sc / 4
    "share_value",     // / sc / 4
    "company_value",   // / sc
    "bankrupt",
    "passed",
    "presidencies",    // / 4
    "sold_corps",      // corporations sold this stock round / 4
    "debt",            // 1822 player debt / sc
    // -- version 2 (the 1822 family; 0 for 1830 / 1867) ---------------------
    "bid_tokens_left", // bidding tokens not on a company / tokens per player
    "certs_with_bids", // certificates + winning bids / cert limit (the limit bids check)
    "double_cash",     // chose double cash (P7 / P9) this round: cash counts twice for the order
    "in_queue",        // in the current sub-step's queue (1822MX: NdeM tile layers, token choosers, bidders)
    "queue_pos",       // position in that queue / (seats - 1)
    "tax_haven_value", // the tax haven this player owns: its cash + shares / sc
];

pub const CORPORATION_FEATURES: &[&str] = &[
    "started",
    "floated",
    "operated",
    "closed",
    "cash",            // / sc
    "share_price",     // / max market price
    "par_price",       // / max market price
    "market_row",      // / rows
    "market_col",      // / columns
    "zone_par",
    "zone_par_1",
    "zone_par_2",
    "zone_convert_range",
    "zone_max_price",
    "zone_no_cert_limit",
    "zone_unlimited",
    "zone_multiple_buy",
    "zone_endgame",
    "ipo_units",       // treasury / IPO share units / total units
    "market_units",    // / total units
    "other_units",     // held by others (another corporation, a tax haven) / total
    "total_units",     // / 10
    "president_units", // / total units
    "share_unit_pct",  // / 100
    "float_pct",       // / 100
    "incremental_cap",
    "tokens_total",    // / 8
    "tokens_placed",   // / total tokens
    "next_token_price", // / max token price
    "loans",           // / max loans for its class (0 without loans)
    "max_loans",       // / 10
    "trains",          // / 4
    "train_room",      // (limit - trains) / 4
    "train_limit",     // / 4
    "or_order_pos",    // position in this OR's order / (n - 1)
    "in_or_order",
    "operated_this_or",
    "is_acting",
    "has_president",
    "max_ownership",   // percent / 100
    "market_price_ipo", // IPO shares sell at market price
    "startable",       // may be started in this phase
    "home_reserved",   // its home reservation is still on the map
    "destination_pending", // 1822: destination token not yet placed
    "exchange_tokens", // 1822: / 4
    "pct_seat_0",
    "pct_seat_1",
    "pct_seat_2",
    "pct_seat_3",
    "pct_seat_4",
    "pct_seat_5",
    "pct_market",
    "pct_ipo",
    // -- version 2 (the 1822 family; 0 for 1830 / 1867) ---------------------
    "payouts",         // 1822PNW regionals: dividends paid / 2 (acquirable after two)
    "winding_down",    // 1822MX: the NdeM is being privatized
    "subject",         // the current decision is about it (merger, acquisition, conversion, auction)
    "token_abilities", // live corporation token abilities / 2 (1822: SECR's channel token, minor 14's)
    "pending_token",   // a home / float token waits to be placed
];

pub const COMPANY_FEATURES: &[&str] = &[
    "value",           // face value / sc
    "revenue",         // / max company revenue
    "closed",
    "owner_seat",
    "owner_corp",
    "unowned",
    "auctionable",
    "up_for_auction",  // the current lot, or in a bid box
    "min_bid",         // / sc
    "high_bid",        // / sc
    "num_bids",        // / seats
    "discount",        // / sc
    "no_buy",
    "ability_used",
    "ab_blocks_hexes",
    "ab_tile_lay",
    "ab_teleport",
    "ab_exchange",
    "ab_no_buy",
    "ab_shares",
    "ab_hex_bonus",
    "ab_close",
    "bid_seat_0",      // / sc
    "bid_seat_1",
    "bid_seat_2",
    "bid_seat_3",
    "bid_seat_4",
    "bid_seat_5",
    // -- version 2 (the 1822 family; 0 for 1830 / 1867) ---------------------
    "cash",            // the company's own money (tax haven, phase-revenue privates) / sc
    "holdings_value",  // shares it holds (the tax haven) at market / sc
    "bidbox_slot",     // its place in its bid-box row: (k + 1) / boxes, 0 = in none
    "bidbox_first",    // the first minor box (unbid: removed with a train; 1822MX: the NdeM's)
    "uses_left",       // remaining ability uses / 3 (an ability without a count: 1)
    "used_this_or",    // per-OR uses spent / allowed (1822CA P14-P18)
    "assigned",        // its assignment is on the map (`ent_hex` home = that hex)
    "subject",         // the current decision is about it (a company mid-use, the sawmill's choice)
    "holds_train",     // reserves a train for its buyer (`trains` relation to the type)
];

pub const TRAIN_FEATURES: &[&str] = &[
    "distance",        // / 8 (capped)
    "unlimited_distance", // distance >= 99 (1822's E / P+)
    "multiplier",      // / 2
    "price",           // / max train price
    "depot_count",     // remaining in the depot / base count (unlimited: 1 while any)
    "depot_log",       // log1p(depot count) / log1p(20)
    "pool_count",      // in the open market pool / 4
    "unlimited_supply",
    "base_count",      // / 10 (unlimited -> 1)
    "held_count",      // held by corporations / 8
    "rusted",          // its rusts-on phase has been reached
    "obsolete",        // its obsolete-on phase has been reached
    "available",       // buyable from the depot now (next up, or available_on reached)
    "next_up",         // the depot's next train is of this type
    "max_discount",    // best trade-in discount / price
    "events",          // purchase events / 3
    "variant",         // a variant without cards of its own (1822 '2', 'E')
];

pub const PHASE_FEATURES: &[&str] = &[
    "current",
    "past",
    "index",           // / (phases - 1)
    "train_limit",     // / 4
    "minor_train_limit", // / 4
    "operating_rounds", // / 3
    "tiles_yellow",
    "tiles_green",
    "tiles_brown",
    "tiles_gray",
    "st_can_buy_companies",
    "st_export_train",
    "st_can_buy_trains",
    "st_can_acquire_minor_bidbox",
    "st_can_par",
    "st_full_capitalisation",
    "st_minors_green_upgrade",
    "st_minor_float_phase1",
    "st_minor_float_phase2",
    "st_minor_float_phase3on",
    "st_can_convert_concessions",
    "st_other",
];

/// `ent_ptr` columns. Per type:
/// * global: acting entity, priority-deal seat, current phase, current lot
///   (company or corporation being auctioned; the 1822 family: the entity
///   the current decision is about, the corporation / company with the
///   `subject` feature first named -- 1822MX's NdeM in its token auction,
///   1822PNW's new major in a merger, an acquisition's selected minor, ...);
/// * player: -, -, -, -;
/// * corporation: president seat (owner), linked entity (1822PNW: the
///   associated major of a minor; 1822: the concession that founds a
///   major), -, -;
/// * company: owner (seat or corporation), linked entity (1822: the minor a
///   proxy stands for, the major a concession founds), -, -;
/// * train type: -, -, rusts-on phase, obsolete-on phase (else
///   available-on phase);
/// * phase: -, -, the train type whose purchase starts it, -.
pub const ENT_PTR_COLUMNS: &[&str] = &["owner", "link", "phase_a", "phase_b"];

/// `ent_hex` columns: home hex, destination hex (corporations), the first
/// hex of the company's map ability (companies; once an assignable company
/// is assigned -- 1822PNW's P15 / P17, 1822CA's P13 -- the hex it is
/// assigned to).
pub const ENT_HEX_COLUMNS: &[&str] = &["home", "destination"];

/// `meta` columns.
pub const META_COLUMNS: &[&str] = &[
    "title",        // TITLES id
    "round_kind",   // ROUND_KINDS id
    "step_kind",    // STEP_KINDS id
    "acting_entity", // entity index of the acting entity, -1 none
    "num_seats",
    "rotation",     // the mover's seat in the seating order (BaseGame.players): seat 0 = that player
    "finished",
    "round_num",    // OR k of the set (0 outside ORs)
];

/// Widths.
pub const F_HEX: usize = HEX_FEATURES.len();
pub const F_TILE: usize = TILE_FEATURES.len();

/// The packed entity feature width: the widest per-type schema.
pub const F_ENT: usize = max_len(&[
    GLOBAL_FEATURES.len(),
    PLAYER_FEATURES.len(),
    CORPORATION_FEATURES.len(),
    COMPANY_FEATURES.len(),
    TRAIN_FEATURES.len(),
    PHASE_FEATURES.len(),
]);

const fn max_len(xs: &[usize]) -> usize {
    let mut m = 0;
    let mut i = 0;
    while i < xs.len() {
        if xs[i] > m {
            m = xs[i];
        }
        i += 1;
    }
    m
}

/// The feature names of an entity type (`super::ENT_*`).
pub fn entity_features(ent_type: i32) -> &'static [&'static str] {
    use super::*;
    match ent_type {
        ENT_GLOBAL => GLOBAL_FEATURES,
        ENT_PLAYER => PLAYER_FEATURES,
        ENT_MAJOR | ENT_MINOR | ENT_NATIONAL => CORPORATION_FEATURES,
        ENT_PRIVATE | ENT_CONCESSION | ENT_MINOR_PROXY => COMPANY_FEATURES,
        ENT_TRAIN => TRAIN_FEATURES,
        ENT_PHASE => PHASE_FEATURES,
        _ => &[],
    }
}
