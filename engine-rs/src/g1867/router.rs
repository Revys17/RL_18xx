//! 1867 auto-router (bridge design T2): the revenue-maximising legal run
//! for a corporation's unoperated trains under Ruby G1867's route rules —
//! the engine's own `run_routes` for self-play.
//!
//! Legality is Ruby `Route#revenue` (route.rb) with G1867's overrides:
//! * at least two visited stops, one a city the corporation has tokened
//!   (`check_route_token`; `City#tokened_by?` counts any token of the
//!   corporation);
//! * each group at most once (`groups:Detroit`, A17/A19);
//! * no node visited twice (`check_cycles!`) and — G1867 `revenue_for`'s
//!   "Route visits same hex twice" (`route.hexes` = the visited nodes'
//!   hexes) — no two stops on one hex (Toronto's two cities, Montreal's
//!   three: one per route);
//! * no track used twice by the run's routes (`check_overlap`: one key per
//!   hexside and per node-to-node path; 1867 has no lanes and its towns
//!   never take the town-as-edge key);
//! * offboards and cities whose every slot holds another corporation's
//!   non-neutral token only end a route (`check_connected` → `blocks?`;
//!   the CN's neutral tokens on Timmins/Montreal never block);
//! * track never crosses an impassable border (`hex.neighbors`).
//!
//! Which trains run: Ruby `route_trains` = `runnable_trains`, the trains
//! that have not operated this OR (as `optimal_routes` for 1830).
//!
//! Distance and price are Ruby G1867 `check_distance` / `compute_stops` /
//! `revenue_for`: trains 2..8 and 2+2 count cities+offboards (`pay` of
//! them at most) and run through any number of towns, which fill spare
//! paying capacity best-first; the 5+5E counts offboards only and fills its
//! five paying slots with the best cities/towns, one of the chosen stops
//! tokened. Stops pay `revenue × multiplier` (2+2 and 5+5E: ×2); a
//! corporation's `hex_bonus` companies add their flat amount per chosen
//! stop hex; a capital (F16/L12/O7) and Timmins (D2) among the chosen
//! stops add 40 × multiplier. Offboards pay their phase tier.
//!
//! The search:
//! 1. **Walk** every legal route depth-first over tile paths from each node
//!    the corporation's tokens reach (each route kept from its lower-keyed
//!    end; the 1830 router's walk). A route is priced for every train kind
//!    on arrival and kept for a kind only when it pays more than both
//!    sub-routes one stop shorter: a sub-route uses a subset of the track,
//!    so a route that pays no more is never needed (the kept lists hold
//!    "tight" runs whose end stops pay). A walk stops extending a route
//!    whose start can no longer pay (`start_hopeless`). Prices come from
//!    per-prefix accumulators (`Acc`), not from scratch. The normal
//!    trains' walk (bounded by their counted stops) and the 5+5E's (cities
//!    and towns unlimited) both run to completion within a step budget
//!    that the corpus never reaches; a call whose budget runs out reports
//!    itself inexact. A kept route is stored as its track bitset and
//!    prices only; the chosen ones are found again by re-walking their
//!    start (the walk is deterministic).
//! 2. **Combine**: branch-and-bound over one route (or none) per train, no
//!    track shared — the 1830 router's search, with per-kind prices; long
//!    lists (a late 5+5E keeps up to ~800k routes) are searched through a
//!    per-track index of their positions, and go last.
//!
//! The chosen run's prices come from `revenue::price_bucketed_stops` (the
//! replay pricer); the walk's own pricing is checked against it.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;

use crate::actions::RouteData;
use crate::game::BaseGame;
use crate::revenue::{self, RouteBonuses, Stop, StopKind};
use crate::tiles::PathEndpoint;
use crate::title::g1867 as data;

/// Walk steps (stop arrivals) for the normal trains' walk: a safety net
/// (the corpus needs at most 1.7M, ~0.3 s).
const NORMAL_STEP_BUDGET: u64 = 20_000_000;
/// Walk steps for the 5+5E's walk (the corpus needs at most 5.8M, ~1 s).
const EXPRESS_STEP_BUDGET: u64 = 20_000_000;
/// Ruby `visit: 99` on the free buckets.
const FREE_VISITS: usize = 99;
/// Most distinct train kinds one call prices (1867 corporations hold at
/// most four trains).
const MAXK: usize = 8;

type Vals = [Option<i32>; MAXK];
const NO_VALS: Vals = [None; MAXK];

/// Chosen runs whose fast price disagreed with `price_bucketed_stops` (a
/// router bug; tests assert 0).
static PRICE_MISMATCHES: AtomicU64 = AtomicU64::new(0);
/// Calls where a walk budget ran out (the run may be short).
static INEXACT_CALLS: AtomicU64 = AtomicU64::new(0);
/// Experiment override of the 5+5E walk's step budget (0: the default).
static EXPRESS_BUDGET_OVERRIDE: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Train kinds and stop pricing
// ---------------------------------------------------------------------------

/// How a train counts and pays its stops (Ruby G1867 TRAINS distance).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TrainKind {
    /// 5+5E: offboards are the counted stops; cities and towns fill.
    express: bool,
    /// The paying bucket's `pay` (= its `visit`).
    pay: u32,
    multiplier: i32,
}

impl TrainKind {
    fn of(name: &str) -> Option<TrainKind> {
        let def = data::trains().into_iter().find(|t| t.name == name)?;
        Some(TrainKind {
            express: name == "5+5E",
            pay: def.distance,
            multiplier: def.multiplier as i32,
        })
    }

    fn counts(&self, kind: StopKind) -> bool {
        if self.express {
            kind == StopKind::Offboard
        } else {
            kind != StopKind::Town
        }
    }

    fn value(&self, s: &StopInfo) -> i32 {
        s.revenue * self.multiplier + s.bonus
    }
}

/// A revenue node as the pricer sees it.
#[derive(Clone, Copy, Debug)]
struct StopInfo {
    kind: StopKind,
    /// Base revenue in the current phase (offboards: their phase tier).
    revenue: i32,
    /// The corporation has a token in this city.
    tokened: bool,
    /// Flat `hex_bonus` total of the corporation's companies on this hex.
    bonus: i32,
    capital: bool,
    timmins: bool,
}

impl StopInfo {
    /// Neither tokened nor part of the capitals bonus: interchangeable with
    /// any optional stop worth as much.
    fn plain(&self) -> bool {
        !self.tokened && !self.capital && !self.timmins
    }
}

/// Ruby G1867 `compute_stops` + `revenue_for` for a route's visited stops
/// (route legality checked elsewhere): the best price of the stops `k`
/// may pay, or None when `k` cannot run them (too many counted stops, or no
/// tokened choice). `opt` is scratch.
///
/// Exact without trying every combination: hex bonuses are additive and
/// distinct per stop (no hex twice), so the only couplings are the token
/// requirement and the capital+Timmins bonus. Every optimal choice contains
/// some forced set — a tokened stop (when a token is needed) plus, when it
/// earns the capitals bonus, a capital and Timmins — and the best fill of
/// the remaining slots around a forced set is the top values; trying each
/// forced set and filling greedily finds the optimum.
fn price_stops(stops: &[StopInfo], k: TrainKind, opt: &mut Vec<(i32, bool, bool, bool)>) -> Option<i32> {
    let mut mand = 0i32;
    let mut n_mand = 0u32;
    let (mut tok, mut cap, mut tim) = (false, false, false);
    opt.clear();
    for s in stops {
        let v = k.value(s);
        if k.counts(s.kind) {
            mand += v;
            n_mand += 1;
            tok |= s.tokened;
            cap |= s.capital;
            tim |= s.timmins;
        } else {
            opt.push((v, s.tokened, s.capital, s.timmins));
        }
    }
    if n_mand > k.pay || opt.len() > FREE_VISITS {
        return None;
    }
    let capitals_bonus = 40 * k.multiplier;
    if n_mand == k.pay {
        return Some(mand + if cap && tim { capitals_bonus } else { 0 });
    }
    let remaining = (k.pay - n_mand) as usize;
    let need_token = !tok;
    if opt.len() <= remaining {
        // Ruby: `combinations = [optional_stops] if combinations.empty?`
        if need_token && !opt.iter().any(|o| o.1) {
            return None;
        }
        let c = cap || opt.iter().any(|o| o.2);
        let t = tim || opt.iter().any(|o| o.3);
        return Some(mand + opt.iter().map(|o| o.0).sum::<i32>() + if c && t { capitals_bonus } else { 0 });
    }
    opt.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let opt: &[(i32, bool, bool, bool)] = opt;
    // The top values alone, when they meet the token requirement and earn
    // the capitals bonus or no choice could: the optimum (the most any
    // choice adds).
    let bonus_possible = (cap || opt.iter().any(|o| o.2)) && (tim || opt.iter().any(|o| o.3));
    {
        let top = &opt[..remaining];
        let t = tok || top.iter().any(|o| o.1);
        let c = cap || top.iter().any(|o| o.2);
        let x = tim || top.iter().any(|o| o.3);
        if (!need_token || t) && (!bonus_possible || (c && x)) {
            return Some(mand + top.iter().map(|o| o.0).sum::<i32>() + if c && x { capitals_bonus } else { 0 });
        }
    }
    let eval = |forced: &[usize]| -> Option<i32> {
        if forced.len() > remaining {
            return None;
        }
        let (mut sum, mut t, mut c, mut x) = (0, tok, cap, tim);
        for &f in forced {
            let o = opt[f];
            sum += o.0;
            t |= o.1;
            c |= o.2;
            x |= o.3;
        }
        let mut need = remaining - forced.len();
        for (i, o) in opt.iter().enumerate() {
            if need == 0 {
                break;
            }
            if forced.contains(&i) {
                continue;
            }
            sum += o.0;
            t |= o.1;
            c |= o.2;
            x |= o.3;
            need -= 1;
        }
        if need_token && !t {
            return None;
        }
        Some(mand + sum + if c && x { capitals_bonus } else { 0 })
    };
    let mut tokens: [Option<usize>; 16] = [None; 16];
    let mut n_tokens = 0;
    if need_token {
        for (i, o) in opt.iter().enumerate() {
            if o.1 && n_tokens < tokens.len() {
                tokens[n_tokens] = Some(i);
                n_tokens += 1;
            }
        }
        if n_tokens == 0 {
            return None;
        }
    } else {
        n_tokens = 1;
    }
    // Capitals and Timmins among the optional stops, when the bonus is not
    // already the mandatory stops' (None: nothing to force on that side).
    let pick = |want: bool, flag: fn(&(i32, bool, bool, bool)) -> bool| -> Vec<Option<usize>> {
        if !want {
            return vec![None];
        }
        (0..opt.len()).filter(|&i| flag(&opt[i])).map(Some).collect()
    };
    let (caps, tims) = if bonus_possible {
        (pick(!cap, |o| o.2), pick(!tim, |o| o.3))
    } else {
        (Vec::new(), Vec::new())
    };
    let mut best: Option<i32> = None;
    let mut forced: Vec<usize> = Vec::with_capacity(3);
    for &t in &tokens[..n_tokens] {
        // without forcing the capitals bonus
        forced.clear();
        forced.extend(t);
        best = best.max(eval(&forced));
        // forcing a capital and Timmins (when both can be had)
        for &c in &caps {
            for &x in &tims {
                if c.is_none() && x.is_none() {
                    continue;
                }
                forced.clear();
                for i in [t, c, x].into_iter().flatten() {
                    if !forced.contains(&i) {
                        forced.push(i);
                    }
                }
                best = best.max(eval(&forced));
            }
        }
    }
    best
}

/// Prices of `stops` for each priced kind (None: not a route for it). A
/// route needs two stops and a tokened one.
fn price_slice(
    stops: &[StopInfo],
    kinds: &[TrainKind],
    pricing: &[bool],
    opt: &mut Vec<(i32, bool, bool, bool)>,
) -> Vals {
    let mut out = NO_VALS;
    if stops.len() < 2 || !stops.iter().any(|s| s.tokened) {
        return out;
    }
    for (k, kind) in kinds.iter().enumerate() {
        if pricing[k] {
            out[k] = price_stops(stops, *kind, opt);
        }
    }
    out
}

/// Optional stops an accumulator keeps by value (≥ the most paying slots
/// any 1867 train leaves for them: an 8-train's seven, the 5+5E's five).
const ACC_TOP: usize = 8;
/// Optional stops that are tokened, a capital or Timmins, kept apart for
/// the forced choices.
const ACC_SPECIAL: usize = 12;

/// An optional stop in an accumulator: value, position in the route, and
/// tokened / capital / Timmins.
#[derive(Clone, Copy, Debug, Default)]
struct Opt {
    v: i32,
    id: u16,
    tok: bool,
    cap: bool,
    tim: bool,
}

/// `price_stops`'s inputs, accumulated stop by stop (the walk prices every
/// prefix it reaches, and the route without its start): the counted
/// stops' sum and flags, the optional stops' sum, count and flags, their
/// best `ACC_TOP` values, and the special ones.
#[derive(Clone, Copy, Debug, Default)]
struct Acc {
    mand: i32,
    n_mand: u32,
    tok: bool,
    cap: bool,
    tim: bool,
    opt_sum: i32,
    n_opt: u32,
    opt_tok: bool,
    opt_cap: bool,
    opt_tim: bool,
    n_top: u8,
    top: [Opt; ACC_TOP],
    n_spec: u8,
    spec: [Opt; ACC_SPECIAL],
    /// More special stops than `ACC_SPECIAL`: price from scratch.
    overflow: bool,
}

impl Acc {
    fn push(&mut self, k: TrainKind, s: &StopInfo, id: u16) {
        let v = k.value(s);
        if k.counts(s.kind) {
            self.mand += v;
            self.n_mand += 1;
            self.tok |= s.tokened;
            self.cap |= s.capital;
            self.tim |= s.timmins;
            return;
        }
        self.opt_sum += v;
        self.n_opt += 1;
        self.opt_tok |= s.tokened;
        self.opt_cap |= s.capital;
        self.opt_tim |= s.timmins;
        let o = Opt {
            v,
            id,
            tok: s.tokened,
            cap: s.capital,
            tim: s.timmins,
        };
        let mut i = self.n_top as usize;
        if i < ACC_TOP {
            self.top[i] = o;
            self.n_top += 1;
        } else if v > self.top[ACC_TOP - 1].v {
            i = ACC_TOP - 1;
            self.top[i] = o;
        } else {
            i = usize::MAX;
        }
        if i != usize::MAX {
            while i > 0 && self.top[i - 1].v < self.top[i].v {
                self.top.swap(i - 1, i);
                i -= 1;
            }
        }
        if s.tokened || s.capital || s.timmins {
            if (self.n_spec as usize) < ACC_SPECIAL {
                self.spec[self.n_spec as usize] = o;
                self.n_spec += 1;
            } else {
                self.overflow = true;
            }
        }
    }

    /// `price_stops` of the accumulated stops; Err when only a from-scratch
    /// pricing can tell (never on the 1867 map: see ACC_TOP/ACC_SPECIAL).
    fn price(&self, k: TrainKind) -> Result<Option<i32>, ()> {
        if self.n_mand > k.pay || self.n_opt as usize > FREE_VISITS {
            return Ok(None);
        }
        let capitals_bonus = 40 * k.multiplier;
        if self.n_mand == k.pay {
            return Ok(Some(self.mand + if self.cap && self.tim { capitals_bonus } else { 0 }));
        }
        let remaining = (k.pay - self.n_mand) as usize;
        let need_token = !self.tok;
        if self.n_opt as usize <= remaining {
            if need_token && !self.opt_tok {
                return Ok(None);
            }
            let bonus = (self.cap || self.opt_cap) && (self.tim || self.opt_tim);
            return Ok(Some(self.mand + self.opt_sum + if bonus { capitals_bonus } else { 0 }));
        }
        if remaining > ACC_TOP {
            return Err(());
        }
        let bonus_possible = (self.cap || self.opt_cap) && (self.tim || self.opt_tim);
        let top = &self.top[..remaining];
        let t = self.tok || top.iter().any(|o| o.tok);
        let c = self.cap || top.iter().any(|o| o.cap);
        let x = self.tim || top.iter().any(|o| o.tim);
        if (!need_token || t) && (!bonus_possible || (c && x)) {
            return Ok(Some(
                self.mand + top.iter().map(|o| o.v).sum::<i32>() + if c && x { capitals_bonus } else { 0 },
            ));
        }
        if self.overflow {
            return Err(());
        }
        // Forced choices (see price_stops): a token, and a capital + Timmins
        // for the bonus; the rest of the slots take the best other values,
        // all within `top` (at most three of its entries are forced).
        let spec = &self.spec[..self.n_spec as usize];
        let eval = |forced: &[Opt]| -> Option<i32> {
            if forced.len() > remaining {
                return None;
            }
            let (mut sum, mut t, mut c, mut x) = (0, self.tok, self.cap, self.tim);
            for o in forced {
                sum += o.v;
                t |= o.tok;
                c |= o.cap;
                x |= o.tim;
            }
            let mut need = remaining - forced.len();
            for o in &self.top[..self.n_top as usize] {
                if need == 0 {
                    break;
                }
                if forced.iter().any(|f| f.id == o.id) {
                    continue;
                }
                sum += o.v;
                t |= o.tok;
                c |= o.cap;
                x |= o.tim;
                need -= 1;
            }
            if need > 0 || (need_token && !t) {
                return None;
            }
            Some(self.mand + sum + if c && x { capitals_bonus } else { 0 })
        };
        let tokens: Vec<Option<Opt>> = if need_token {
            spec.iter().filter(|o| o.tok).copied().map(Some).collect()
        } else {
            vec![None]
        };
        let side = |want: bool, flag: fn(&Opt) -> bool| -> Vec<Option<Opt>> {
            if !want {
                return vec![None];
            }
            spec.iter().filter(|o| flag(o)).copied().map(Some).collect()
        };
        let (caps, tims) = if bonus_possible {
            (side(!self.cap, |o| o.cap), side(!self.tim, |o| o.tim))
        } else {
            (Vec::new(), Vec::new())
        };
        let mut best: Option<i32> = None;
        let mut forced: Vec<Opt> = Vec::with_capacity(3);
        for &t in &tokens {
            forced.clear();
            forced.extend(t);
            best = best.max(eval(&forced));
            for &c in &caps {
                for &x in &tims {
                    if c.is_none() && x.is_none() {
                        continue;
                    }
                    forced.clear();
                    for o in [t, c, x].into_iter().flatten() {
                        if !forced.iter().any(|f| f.id == o.id) {
                            forced.push(o);
                        }
                    }
                    best = best.max(eval(&forced));
                }
            }
        }
        Ok(best)
    }
}

// ---------------------------------------------------------------------------
// Node keys
// ---------------------------------------------------------------------------

/// A revenue node packed as `hex << 8 | kind << 6 | index` (keys order by
/// hex, then kind, then index — as the 1830 router's).
type NodeKey = u32;

const KIND_CITY: u32 = 0;
const KIND_TOWN: u32 = 1;
const KIND_OFFBOARD: u32 = 2;

fn endpoint_key(hex: usize, ep: &PathEndpoint) -> Option<NodeKey> {
    let (kind, index) = match ep {
        PathEndpoint::City(i) => (KIND_CITY, *i),
        PathEndpoint::Town(i) => (KIND_TOWN, *i),
        PathEndpoint::Offboard(i) => (KIND_OFFBOARD, *i),
        PathEndpoint::Edge(_) | PathEndpoint::Junction => return None,
    };
    (index < 64).then(|| (hex as u32) << 8 | kind << 6 | index as u32)
}

fn key_hex(key: NodeKey) -> usize {
    (key >> 8) as usize
}

fn key_endpoint(key: NodeKey) -> PathEndpoint {
    let index = (key & 63) as usize;
    match (key >> 6) & 3 {
        KIND_CITY => PathEndpoint::City(index),
        KIND_TOWN => PathEndpoint::Town(index),
        _ => PathEndpoint::Offboard(index),
    }
}

/// Position of a node among its tile's nodes: cities, then towns, then
/// offboards — Ruby's part index (`Node#signature`) for every 1867 tile,
/// whose DSL lists its cities/towns/offboards first, in that order.
fn node_pos(tile: &crate::graph::Tile, ep: &PathEndpoint) -> Option<usize> {
    match ep {
        PathEndpoint::City(i) => (*i < tile.cities.len()).then_some(*i),
        PathEndpoint::Town(i) => (*i < tile.towns.len()).then_some(tile.cities.len() + i),
        PathEndpoint::Offboard(i) => (*i < tile.offboards.len()).then_some(tile.cities.len() + tile.towns.len() + i),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Stage 1: route enumeration
// ---------------------------------------------------------------------------

const NBR_NONE: u16 = u16::MAX;
const NO_TRACK: u16 = u16::MAX;

/// How to find a kept route again (only its track and prices are stored):
/// the pass that kept it, its start (index into `starts`), and how many
/// routes that start's walk had kept before it. The walk is deterministic,
/// so re-walking that start finds it (`Walk::route_of`).
#[derive(Clone, Copy, Debug)]
struct FoundRef {
    pass: u8,
    start: u32,
    ordinal: u32,
}

/// A route as walked: stops in order, each stop's index into `trail`, and
/// the hexes, one per tile path.
type Walked = (Vec<NodeKey>, Vec<u16>, Vec<u16>);

struct Walk<'a> {
    g: &'a BaseGame,
    corp: &'a str,
    /// Distinct train kinds (index = column of `found_vals`).
    kinds: Vec<TrainKind>,
    /// Which kinds this pass prices.
    pricing: Vec<bool>,
    /// This pass's bound on counted stops (cities+offboards): the normal
    /// trains' largest pay; unbounded for the 5+5E pass.
    max_counted: u32,
    /// The passes run so far: (pricing, max_counted).
    passes: Vec<(Vec<bool>, u32)>,
    /// Keep only tight routes and stop hopeless walks (false: keep every
    /// priced route — the reference the tests compare against).
    filter: bool,
    /// Per hex, per node position: the node as priced.
    info: Vec<Vec<StopInfo>>,
    /// Neighbor hex by (hex, edge); NBR_NONE off the map or across a wall.
    neighbors: Vec<[u16; 6]>,
    /// Track id of each `(hex, edge)` — both sides of a hexside share one —
    /// and of each node-to-node path (`check_overlap`'s intra-tile key;
    /// none on 1867's tiles).
    hexside_track: Vec<u16>,
    node_path_track: HashMap<(usize, usize), u16>,
    /// u64 words of a track bitset.
    words: usize,
    /// Whether the route being walked uses each track id, as a bitset.
    in_use: Vec<u64>,
    /// Sorted: every node a walk starts from.
    starts: Vec<NodeKey>,

    // the route being walked
    stops: Vec<NodeKey>,
    infos: Vec<StopInfo>,
    /// Index into `trail` of each stop's hex.
    stop_pos: Vec<u16>,
    /// The route's hexes, one per tile path (Route#chain_id).
    trail: Vec<u16>,
    junction_hexes: Vec<usize>,
    groups: Vec<&'a str>,
    /// Hexes holding one of the route's stops.
    hex_stop: Vec<bool>,
    counted: u32,
    offboards: u32,
    tokened: u32,
    /// Per kind: the start stop's value when it is a plain optional stop
    /// for the kind, and how many later optional stops of the route are
    /// worth at least as much.
    s0_val: Vals,
    beat: [u32; MAXK],
    /// Per stop depth: the prefix's price per kind.
    prefix_vals: Vec<Vals>,
    /// The kinds this pass prices, and per stop depth their accumulators
    /// of the route (`acc`) and of the route without its start
    /// (`acc_tail`), stride `priced.len()`.
    priced: Vec<usize>,
    acc: Vec<Acc>,
    acc_tail: Vec<Acc>,

    steps: u64,
    budget: u64,
    exhausted: bool,
    /// The start being walked (index into `starts`) and the routes its walk
    /// kept so far.
    start_index: u32,
    kept_here: u32,
    /// Re-walk mode: stop at the start's `capture`-th kept route.
    capture: Option<u32>,
    captured: Option<Walked>,
    halt: bool,

    /// Per kept route: its track bitset (`words` u64s), its price per kind
    /// (stride `kinds.len()`; 0 when the kind does not keep it) and where
    /// it was found.
    found_bits: Vec<u64>,
    found_vals: Vec<i32>,
    found_ref: Vec<FoundRef>,
    opt: Vec<(i32, bool, bool, bool)>,
}

impl<'a> Walk<'a> {
    fn new(g: &'a BaseGame, corp: &'a str, kinds: Vec<TrainKind>) -> Walk<'a> {
        assert!(kinds.len() <= MAXK);
        let n = g.hexes.len();
        // hex_bonus companies the corporation owns (as recorded_route_revenue)
        let corp_eid = crate::entities::EntityId::corporation(corp);
        let mut hex_bonus: HashMap<&str, i32> = HashMap::new();
        for company in &g.companies {
            if company.closed || company.owner != corp_eid {
                continue;
            }
            for (hexes, amount) in crate::abilities::hex_bonuses("1867", &company.sym) {
                for h in hexes {
                    *hex_bonus.entry(h).or_insert(0) += amount;
                }
            }
        }
        let phase = &g.phase.tiles;
        let mut info = Vec::with_capacity(n);
        for hex in &g.hexes {
            let tile = &hex.tile;
            let bonus = hex_bonus.get(hex.id.as_str()).copied().unwrap_or(0);
            let capital = data::BONUS_CAPITALS.contains(&hex.id.as_str());
            let timmins = hex.id == data::BONUS_REVENUE;
            let stop = |kind, revenue, tokened| StopInfo {
                kind,
                revenue,
                tokened,
                bonus,
                capital,
                timmins,
            };
            let mut v = Vec::new();
            for c in &tile.cities {
                let tokened = c.tokens.iter().flatten().any(|t| t.corporation_id == corp);
                v.push(stop(StopKind::City, c.revenue, tokened));
            }
            for t in &tile.towns {
                v.push(stop(StopKind::Town, t.revenue, false));
            }
            for o in &tile.offboards {
                v.push(stop(StopKind::Offboard, o.phase_revenue(phase), false));
            }
            info.push(v);
        }
        let mut neighbors = vec![[NBR_NONE; 6]; n];
        for (hi, hex) in g.hexes.iter().enumerate() {
            let walls: &[u8] = g.walls.get(&hex.id).map_or(&[], |v| v.as_slice());
            if let Some(adj) = g.hex_adjacency.get(&hex.id) {
                for (&e, nb) in adj {
                    if (e as usize) < 6 && !walls.contains(&e) {
                        if let Some(&ni) = g.hex_idx.get(nb) {
                            neighbors[hi][e as usize] = ni as u16;
                        }
                    }
                }
            }
        }
        // track ids: every passable hexside, then node-to-node paths
        let mut hexside_track = vec![NO_TRACK; n * 6];
        let mut n_track: u16 = 0;
        for hi in 0..n {
            for e in 0..6u8 {
                let nb = neighbors[hi][e as usize];
                if nb != NBR_NONE && hexside_track[hi * 6 + e as usize] == NO_TRACK {
                    hexside_track[hi * 6 + e as usize] = n_track;
                    hexside_track[nb as usize * 6 + ((e + 3) % 6) as usize] = n_track;
                    n_track += 1;
                }
            }
        }
        let mut node_path_track = HashMap::new();
        for (hi, hex) in g.hexes.iter().enumerate() {
            for (pi, p) in hex.tile.paths.iter().enumerate() {
                if endpoint_key(hi, &p.a).is_some() && endpoint_key(hi, &p.b).is_some() {
                    node_path_track.insert((hi, pi), n_track);
                    n_track += 1;
                }
            }
        }
        let words = (n_track as usize).div_ceil(64).max(1);
        let nk = kinds.len();
        Walk {
            g,
            corp,
            pricing: vec![false; nk],
            kinds,
            max_counted: 0,
            passes: Vec::new(),
            filter: true,
            info,
            neighbors,
            hexside_track,
            node_path_track,
            words,
            in_use: vec![0; words],
            starts: Vec::new(),
            stops: Vec::new(),
            infos: Vec::new(),
            stop_pos: Vec::new(),
            trail: Vec::new(),
            junction_hexes: Vec::new(),
            groups: Vec::new(),
            hex_stop: vec![false; n],
            counted: 0,
            offboards: 0,
            tokened: 0,
            s0_val: NO_VALS,
            beat: [0; MAXK],
            prefix_vals: Vec::new(),
            priced: Vec::new(),
            acc: Vec::new(),
            acc_tail: Vec::new(),
            steps: 0,
            budget: 0,
            exhausted: false,
            start_index: 0,
            kept_here: 0,
            capture: None,
            captured: None,
            halt: false,
            found_bits: Vec::new(),
            found_vals: Vec::new(),
            found_ref: Vec::new(),
            opt: Vec::new(),
        }
    }

    fn n_found(&self) -> usize {
        self.found_ref.len()
    }

    fn info_of(&self, key: NodeKey) -> StopInfo {
        let hex = key_hex(key);
        let pos = node_pos(&self.g.hexes[hex].tile, &key_endpoint(key)).expect("node exists");
        self.info[hex][pos]
    }

    /// Ruby `Node#blocks?`: offboards always; a city whose every slot holds
    /// another corporation's non-neutral token.
    fn blocks(&self, hex: usize, ep: &PathEndpoint) -> bool {
        let tile = &self.g.hexes[hex].tile;
        match ep {
            PathEndpoint::Offboard(_) => true,
            PathEndpoint::City(i) => tile.cities.get(*i).is_some_and(|city| {
                !city.tokens.iter().flatten().any(|t| t.corporation_id == self.corp)
                    && !city.tokens.iter().any(|t| t.is_none())
                    && !city.tokens.iter().flatten().any(|t| t.token_type == "neutral")
            }),
            _ => false,
        }
    }

    fn node_groups(&self, hex: usize, ep: &PathEndpoint) -> &'a [String] {
        let g: &'a BaseGame = self.g;
        match ep {
            PathEndpoint::Offboard(i) => g.hexes[hex].tile.offboards.get(*i).map_or(&[], |o| &o.groups),
            _ => &[],
        }
    }

    fn stopped(&self) -> bool {
        self.exhausted || self.halt
    }

    /// Take track id `id` for the route being walked: false when it is in
    /// use already.
    fn take(&mut self, id: u16) -> bool {
        let (w, b) = (id as usize / 64, 1u64 << (id % 64));
        if self.in_use[w] & b != 0 {
            return false;
        }
        self.in_use[w] |= b;
        true
    }

    fn give_back(&mut self, id: u16) {
        self.in_use[id as usize / 64] &= !(1u64 << (id % 64));
    }

    /// Every node reachable from the corporation's token cities along track,
    /// passing only nodes that do not block (a superset of every legal
    /// route's stops), plus the token cities.
    fn reachable(&self) -> Vec<NodeKey> {
        let hexes = &self.g.hexes;
        let mut out: HashSet<NodeKey> = HashSet::new();
        let mut stack: Vec<NodeKey> = Vec::new();
        for (hi, hex) in hexes.iter().enumerate() {
            for (ci, city) in hex.tile.cities.iter().enumerate() {
                if city.tokens.iter().flatten().any(|t| t.corporation_id == self.corp) {
                    if let Some(k) = endpoint_key(hi, &PathEndpoint::City(ci)) {
                        if out.insert(k) {
                            stack.push(k);
                        }
                    }
                }
            }
        }
        let mut seen_paths: HashSet<(usize, usize)> = HashSet::new();
        // (hex, path, the end it is entered by)
        let mut work: Vec<(usize, usize, PathEndpoint)> = Vec::new();
        while let Some(k) = stack.pop() {
            let (hex, ep) = (key_hex(k), key_endpoint(k));
            for (pi, p) in hexes[hex].tile.paths.iter().enumerate() {
                if p.a == ep || p.b == ep {
                    work.push((hex, pi, ep.clone()));
                }
            }
            while let Some((h, pi, entry)) = work.pop() {
                if !seen_paths.insert((h, pi)) {
                    continue;
                }
                let p = &hexes[h].tile.paths[pi];
                let other = if p.a == entry { &p.b } else { &p.a };
                match other {
                    PathEndpoint::Edge(e) => {
                        let nb = self.neighbors[h][*e as usize];
                        if nb == NBR_NONE {
                            continue;
                        }
                        let nb = nb as usize;
                        let enter = PathEndpoint::Edge((e + 3) % 6);
                        for (qi, q) in hexes[nb].tile.paths.iter().enumerate() {
                            if q.a == enter || q.b == enter {
                                work.push((nb, qi, enter.clone()));
                            }
                        }
                    }
                    PathEndpoint::Junction => {
                        for (qi, q) in hexes[h].tile.paths.iter().enumerate() {
                            if qi != pi && (q.a == PathEndpoint::Junction || q.b == PathEndpoint::Junction) {
                                work.push((h, qi, PathEndpoint::Junction));
                            }
                        }
                    }
                    node => {
                        let Some(m) = endpoint_key(h, node) else { continue };
                        if node_pos(&hexes[h].tile, node).is_none() {
                            continue;
                        }
                        if out.insert(m) && !self.blocks(h, node) {
                            stack.push(m);
                        }
                    }
                }
            }
        }
        let mut v: Vec<NodeKey> = out.into_iter().collect();
        v.sort_unstable();
        v
    }

    fn set_pass(&mut self, pricing: Vec<bool>, max_counted: u32) {
        self.priced = (0..self.kinds.len()).filter(|&k| pricing[k]).collect();
        self.pricing = pricing;
        self.max_counted = max_counted;
    }

    /// One pass of the walk: price the `pricing` kinds, counted stops
    /// bounded by `max_counted`.
    fn run_pass(&mut self, pricing: Vec<bool>, max_counted: u32, budget: u64) {
        self.passes.push((pricing.clone(), max_counted));
        self.set_pass(pricing, max_counted);
        self.steps = 0;
        self.budget = budget;
        self.exhausted = false;
        for i in 0..self.starts.len() {
            if self.exhausted {
                break;
            }
            self.start_index = i as u32;
            self.kept_here = 0;
            self.walk_routes_from(self.starts[i]);
        }
    }

    /// Re-walk to kept route `r`: its stops, their trail positions and its
    /// trail.
    fn route_of(&mut self, r: usize) -> Walked {
        let fr = self.found_ref[r];
        let (pricing, max_counted) = self.passes[fr.pass as usize].clone();
        self.set_pass(pricing, max_counted);
        self.steps = 0;
        self.budget = u64::MAX;
        self.exhausted = false;
        self.capture = Some(fr.ordinal);
        self.captured = None;
        self.halt = false;
        self.start_index = fr.start;
        self.kept_here = 0;
        self.walk_routes_from(self.starts[fr.start as usize]);
        self.capture = None;
        self.halt = false;
        self.captured.take().expect("the re-walk reaches the kept route")
    }

    fn walk_routes_from(&mut self, start: NodeKey) {
        let hex = key_hex(start);
        let ep = key_endpoint(start);
        let Some(pos) = node_pos(&self.g.hexes[hex].tile, &ep) else {
            return;
        };
        let si = self.info[hex][pos];
        let counted = u32::from(si.kind != StopKind::Town);
        if counted > self.max_counted {
            return;
        }
        let groups = self.node_groups(hex, &ep);
        self.stops.push(start);
        self.infos.push(si);
        self.stop_pos.push(0);
        self.trail.push(hex as u16);
        self.groups.extend(groups.iter().map(String::as_str));
        self.hex_stop[hex] = true;
        self.counted = counted;
        self.offboards = u32::from(si.kind == StopKind::Offboard);
        self.tokened = u32::from(si.tokened);
        for (k, kind) in self.kinds.iter().enumerate() {
            self.s0_val[k] = (self.pricing[k] && si.plain() && !kind.counts(si.kind)).then(|| kind.value(&si));
            self.beat[k] = 0;
        }
        self.prefix_vals.push(NO_VALS);
        self.acc.clear();
        self.acc_tail.clear();
        for &k in &self.priced {
            let mut a = Acc::default();
            a.push(self.kinds[k], &si, 0);
            self.acc.push(a);
            self.acc_tail.push(Acc::default());
        }
        // A route may start at a blocked city or an offboard; it just
        // can't pass through one.
        self.walk_from(hex, &ep);
        self.prefix_vals.clear();
        self.acc.clear();
        self.acc_tail.clear();
        self.hex_stop[hex] = false;
        self.stops.clear();
        self.infos.clear();
        self.stop_pos.clear();
        self.trail.clear();
        self.groups.clear();
    }

    fn walk_from(&mut self, hex: usize, node: &PathEndpoint) {
        let g: &'a BaseGame = self.g;
        for (pi, p) in g.hexes[hex].tile.paths.iter().enumerate() {
            let other = if p.a == *node {
                &p.b
            } else if p.b == *node {
                &p.a
            } else {
                continue;
            };
            // 1867 has no terminal paths; kept for safety (check_terminals!)
            let last = p.terminal && self.stops.len() > 1;
            self.follow(hex, pi, other, last);
            if self.stopped() {
                return;
            }
        }
    }

    fn follow(&mut self, hex: usize, path: usize, other: &PathEndpoint, last: bool) {
        match other {
            PathEndpoint::Edge(e) => {
                if !last {
                    self.cross(hex, *e);
                }
            }
            PathEndpoint::Junction => {
                if !last {
                    self.through_junction(hex, path);
                }
            }
            node => match self.node_path_track.get(&(hex, path)).copied() {
                // a node-to-node path inside the tile is its own track
                Some(id) => {
                    if self.take(id) {
                        self.arrive(hex, node, last);
                        self.give_back(id);
                    }
                }
                None => self.arrive(hex, node, last),
            },
        }
    }

    fn cross(&mut self, hex: usize, edge: u8) {
        let nb = self.neighbors[hex][edge as usize];
        if nb == NBR_NONE {
            return;
        }
        let nb = nb as usize;
        let id = self.hexside_track[hex * 6 + edge as usize];
        if !self.take(id) {
            return;
        }
        self.trail.push(nb as u16);
        let enter = PathEndpoint::Edge((edge + 3) % 6);
        let g: &'a BaseGame = self.g;
        for (qi, q) in g.hexes[nb].tile.paths.iter().enumerate() {
            let other = if q.a == enter {
                &q.b
            } else if q.b == enter {
                &q.a
            } else {
                continue;
            };
            self.follow(nb, qi, other, q.terminal);
            if self.stopped() {
                break;
            }
        }
        self.trail.pop();
        self.give_back(id);
    }

    fn through_junction(&mut self, hex: usize, path: usize) {
        if self.junction_hexes.contains(&hex) {
            return;
        }
        self.junction_hexes.push(hex);
        let g: &'a BaseGame = self.g;
        for (ji, j) in g.hexes[hex].tile.paths.iter().enumerate() {
            if ji == path {
                continue;
            }
            let other = if j.a == PathEndpoint::Junction {
                &j.b
            } else if j.b == PathEndpoint::Junction {
                &j.a
            } else {
                continue;
            };
            if *other == PathEndpoint::Junction {
                continue;
            }
            self.follow(hex, ji, other, j.terminal);
            if self.stopped() {
                break;
            }
        }
        self.junction_hexes.pop();
    }

    /// Stop at `node` on `hex`: price and maybe keep the route so far, then
    /// keep going unless the node (or a terminal path, `last`) ends it.
    fn arrive(&mut self, hex: usize, node: &PathEndpoint, last: bool) {
        let Some(key) = endpoint_key(hex, node) else { return };
        // no node twice, and no second stop on a hex ("Route visits same
        // hex twice")
        if self.hex_stop[hex] {
            return;
        }
        let Some(pos) = node_pos(&self.g.hexes[hex].tile, node) else {
            return;
        };
        let si = self.info[hex][pos];
        let counted = self.counted + u32::from(si.kind != StopKind::Town);
        if counted > self.max_counted {
            return;
        }
        let groups = self.node_groups(hex, node);
        if groups.iter().any(|g| self.groups.contains(&g.as_str())) {
            return;
        }
        self.steps += 1;
        if self.steps > self.budget {
            self.exhausted = true;
            return;
        }
        let saved = (self.counted, self.offboards, self.tokened);
        self.stops.push(key);
        self.infos.push(si);
        self.stop_pos.push((self.trail.len() - 1) as u16);
        self.groups.extend(groups.iter().map(String::as_str));
        self.hex_stop[hex] = true;
        self.counted = counted;
        self.offboards += u32::from(si.kind == StopKind::Offboard);
        self.tokened += u32::from(si.tokened);
        let mut beaten = 0u32;
        for (k, kind) in self.kinds.iter().enumerate() {
            if let Some(v0) = self.s0_val[k] {
                if !kind.counts(si.kind) && kind.value(&si) >= v0 {
                    self.beat[k] += 1;
                    beaten |= 1 << k;
                }
            }
        }
        let np = self.priced.len();
        let depth = (self.stops.len() - 1) as u16;
        let base = self.acc.len() - np;
        for j in 0..np {
            let k = self.kinds[self.priced[j]];
            let mut a = self.acc[base + j];
            a.push(k, &si, depth);
            self.acc.push(a);
            let mut t = self.acc_tail[base + j];
            t.push(k, &si, depth);
            self.acc_tail.push(t);
        }

        let vals = if self.tokened == 0 {
            NO_VALS
        } else if self.filter {
            self.price_acc(false)
        } else {
            price_slice(&self.infos, &self.kinds, &self.pricing, &mut self.opt)
        };
        self.record(&vals);
        self.prefix_vals.push(vals);
        if !self.halt && !last && !self.blocks(hex, node) && !(self.filter && self.start_hopeless()) {
            self.walk_from(hex, node);
        }
        self.prefix_vals.pop();
        self.acc.truncate(base + np);
        self.acc_tail.truncate(base + np);
        for k in 0..self.kinds.len() {
            if beaten & (1 << k) != 0 {
                self.beat[k] -= 1;
            }
        }

        (self.counted, self.offboards, self.tokened) = saved;
        self.hex_stop[hex] = false;
        self.groups.truncate(self.groups.len() - groups.len());
        self.stop_pos.pop();
        self.infos.pop();
        self.stops.pop();
    }

    /// The route's prices per priced kind from the accumulators (`tail`:
    /// the route without its start), as `price_slice` would give them.
    fn price_acc(&mut self, tail: bool) -> Vals {
        let mut out = NO_VALS;
        let (from, tokened) = if tail {
            (1, self.tokened - u32::from(self.infos[0].tokened))
        } else {
            (0, self.tokened)
        };
        if self.stops.len() - from < 2 || tokened == 0 {
            return out;
        }
        let np = self.priced.len();
        let base = self.acc.len() - np;
        for j in 0..np {
            let k = self.priced[j];
            let a = if tail {
                &self.acc_tail[base + j]
            } else {
                &self.acc[base + j]
            };
            out[k] = match a.price(self.kinds[k]) {
                Ok(v) => v,
                Err(()) => price_stops(&self.infos[from..], self.kinds[k], &mut self.opt),
            };
        }
        out
    }

    /// No extension of the route can be kept for any priced kind: its start
    /// is a plain optional stop (a town for normal trains; an untokened
    /// city/town other than a capital or Timmins for the 5+5E) and the
    /// route already holds as many optional stops worth at least as much
    /// as the paying slots any extension could leave them (counted stops
    /// only grow). Any extension then prices the same without its start —
    /// swap the start for an unchosen stop worth as much — so it is never
    /// tight (`record`'s sub-route test would drop it).
    fn start_hopeless(&self) -> bool {
        let mut any = false;
        for (k, kind) in self.kinds.iter().enumerate() {
            if !self.pricing[k] {
                continue;
            }
            if self.s0_val[k].is_none() {
                return false;
            }
            let counted = if kind.express { self.offboards } else { self.counted };
            if (self.beat[k] as i64) < kind.pay as i64 - counted as i64 {
                return false;
            }
            any = true;
        }
        any
    }

    /// Keep the route walked so far for each kind it is a tight run for.
    fn record(&mut self, vals: &Vals) {
        let nk = self.kinds.len();
        if self.stops.len() < 2 || vals[..nk].iter().all(|v| v.is_none()) {
            return;
        }
        // The walk from the route's other end finds it too; keep one.
        let (first, last) = (self.stops[0], self.stops[self.stops.len() - 1]);
        if last < first && self.starts.binary_search(&last).is_ok() {
            return;
        }
        let mut keep = [0i32; MAXK];
        if self.filter {
            // tight: pays more than the route without its last stop (the
            // prefix) and without its first
            let parent = *self.prefix_vals.last().expect("prefix of the start");
            if !(0..nk).any(|k| vals[k] > parent[k].max(Some(0))) {
                return;
            }
            let minus_first = self.price_acc(true);
            for k in 0..nk {
                if vals[k] > parent[k].max(Some(0)) && vals[k] > minus_first[k] {
                    keep[k] = vals[k].unwrap_or(0);
                }
            }
        } else {
            for k in 0..nk {
                keep[k] = vals[k].filter(|&v| v > 0).unwrap_or(0);
            }
        }
        if keep[..nk].iter().all(|&v| v == 0) {
            return;
        }
        let ordinal = self.kept_here;
        self.kept_here += 1;
        if let Some(target) = self.capture {
            if ordinal == target {
                self.captured = Some((self.stops.clone(), self.stop_pos.clone(), self.trail.clone()));
                self.halt = true;
            }
            return;
        }
        self.found_bits.extend_from_slice(&self.in_use);
        self.found_vals.extend_from_slice(&keep[..nk]);
        self.found_ref.push(FoundRef {
            pass: (self.passes.len() - 1) as u8,
            start: self.start_index,
            ordinal,
        });
    }

    /// Ruby `Node#signature`.
    fn signature(&self, key: NodeKey) -> String {
        let hex = key_hex(key);
        let pos = node_pos(&self.g.hexes[hex].tile, &key_endpoint(key)).unwrap_or(0);
        format!("{}-{}", self.g.hexes[hex].id, pos)
    }

    /// The run_routes wire entry for a walked route.
    fn route_data(&self, w: &Walked, train: &str, revenue: i32) -> RouteData {
        let (stops, pos, trail) = w;
        let hexes = &self.g.hexes;
        RouteData {
            train_name: train.to_string(),
            hexes: stops.iter().map(|&k| hexes[key_hex(k)].id.clone()).collect(),
            revenue: Some(revenue),
            // Route#chain_id per connection: one hex per tile path, from
            // one stop's hex to the next's
            connections: pos
                .windows(2)
                .map(|w| {
                    trail[w[0] as usize..=w[1] as usize]
                        .iter()
                        .map(|&h| hexes[h as usize].id.clone())
                        .collect()
                })
                .collect(),
            nodes: stops.iter().map(|&k| self.signature(k)).collect(),
        }
    }

    /// The replay pricer's price of a walked route run by kind `k`.
    fn shared_price(&self, stops: &[NodeKey], k: TrainKind, bonuses: &RouteBonuses) -> Option<i32> {
        let stops: Vec<Stop> = stops
            .iter()
            .map(|&key| {
                let s = self.info_of(key);
                Stop {
                    hex: self.g.hexes[key_hex(key)].id.clone(),
                    kind: s.kind,
                    revenue: s.revenue,
                    tokened: s.tokened,
                }
            })
            .collect();
        let counted: &[StopKind] = if k.express {
            &[StopKind::Offboard]
        } else {
            &[StopKind::City, StopKind::Offboard]
        };
        revenue::price_bucketed_stops(&stops, k.pay as usize, k.multiplier, counted, bonuses).ok()
    }
}

/// The bonuses `title::g1867::recorded_route_revenue` prices a route with.
fn route_bonuses(g: &BaseGame, corp: &str) -> RouteBonuses {
    let mut bonuses = RouteBonuses {
        capital_hexes: data::BONUS_CAPITALS.iter().map(|s| s.to_string()).collect(),
        capital_partner: Some(data::BONUS_REVENUE.to_string()),
        capital_amount: 40,
        ..Default::default()
    };
    let corp_eid = crate::entities::EntityId::corporation(corp);
    for company in &g.companies {
        if company.closed || company.owner != corp_eid {
            continue;
        }
        for (hexes, amount) in crate::abilities::hex_bonuses("1867", &company.sym) {
            bonuses
                .hex_bonuses
                .push((hexes.iter().map(|h| h.to_string()).collect(), amount));
        }
    }
    bonuses
}

// ---------------------------------------------------------------------------
// Stage 2: best combination (the 1830 router's branch-and-bound)
// ---------------------------------------------------------------------------

/// Lists longer than this are searched through a per-track index of the
/// list positions using it (the 1830 router's); shorter ones by scanning.
const INDEX_MIN: usize = 1024;

/// Branch-and-bound over one route (or none) per train, trains ordered so
/// identical ones are adjacent.
struct Combo<'a> {
    /// Track bitset of each kept route, `words` u64s apiece.
    bits: &'a [u64],
    words: usize,
    /// Per ordered train: eligible routes, best first, and their prices.
    lists: Vec<&'a [u32]>,
    vals: Vec<&'a [i32]>,
    /// Per ordered train with a long list: for each track id, the bitset
    /// of the list positions whose route uses it.
    uses: Vec<Option<&'a [u64]>>,
    /// Whether ordered train `t` is identical to train `t - 1`.
    same_as_prev: Vec<bool>,
    /// First ordered train after `t`'s run of identical trains.
    group_end: Vec<usize>,
    /// Track bitset of the routes chosen so far.
    used: Vec<u64>,
    blocked: Vec<Vec<u64>>,
    chosen: Vec<Option<u32>>,
    best: i32,
    best_chosen: Vec<Option<u32>>,
}

impl Combo<'_> {
    fn toggle(&mut self, route: usize) {
        let b = &self.bits[route * self.words..(route + 1) * self.words];
        for (u, x) in self.used.iter_mut().zip(b) {
            *u ^= x;
        }
    }

    fn clear(&self, route: usize) -> bool {
        let b = &self.bits[route * self.words..(route + 1) * self.words];
        self.used.iter().zip(b).all(|(u, x)| u & x == 0)
    }

    /// The most trains `t..` could add, ignoring overlap, when train `t`
    /// takes routes from list position `from` on: it and the identical
    /// trains after it take distinct routes, the rest their best.
    fn upper_bound(&self, t: usize, from: usize) -> i32 {
        let n = self.lists.len();
        let mut bound = 0;
        let (mut t, mut from) = (t, from);
        while t < n {
            let end = self.group_end[t];
            let vals = self.vals[t];
            bound += vals[from.min(vals.len())..].iter().take(end - t).sum::<i32>();
            (t, from) = (end, 0);
        }
        bound
    }

    /// Try list position `pos` of train `t`; false when nothing later in
    /// the list can win.
    fn try_pos(&mut self, t: usize, pos: usize, revenue: i32, next_same: bool, rest_bound: i32) -> bool {
        let r = self.vals[t][pos];
        // Both terms shrink as `pos` grows, so nothing later can win.
        let rest = if next_same {
            self.upper_bound(t + 1, pos + 1)
        } else {
            rest_bound
        };
        if revenue + r + rest <= self.best {
            return false;
        }
        let route = self.lists[t][pos] as usize;
        self.toggle(route);
        self.chosen[t] = Some(route as u32);
        self.search(t + 1, revenue + r, if next_same { pos + 1 } else { 0 });
        self.chosen[t] = None;
        self.toggle(route);
        true
    }

    /// Choose routes for trains `t..`, having earned `revenue` so far.
    /// `from`: identical trains take routes in list order, so this train
    /// starts at the list position after the previous train's.
    fn search(&mut self, t: usize, revenue: i32, from: usize) {
        if revenue > self.best {
            self.best = revenue;
            self.best_chosen.clone_from(&self.chosen);
        }
        let n = self.lists.len();
        if t == n || revenue + self.upper_bound(t, from) <= self.best {
            return;
        }
        let next_same = t + 1 < n && self.same_as_prev[t + 1];
        let len = self.lists[t].len();
        let rest_bound = if next_same { 0 } else { self.upper_bound(t + 1, 0) };
        if let Some(uses) = self.uses[t] {
            // Rule out, 64 positions at a time, every route sharing track
            // with the ones chosen so far.
            let pw = len.div_ceil(64);
            let mut blocked = std::mem::take(&mut self.blocked[t]);
            blocked.clear();
            blocked.resize(pw, 0);
            for (w, &word) in self.used.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let id = w * 64 + word.trailing_zeros() as usize;
                    word &= word - 1;
                    for (b, u) in blocked.iter_mut().zip(&uses[id * pw..(id + 1) * pw]) {
                        *b |= u;
                    }
                }
            }
            'scan: for w in from / 64..pw {
                let mut free = !blocked[w];
                if w == from / 64 {
                    free &= !0u64 << (from % 64);
                }
                while free != 0 {
                    let pos = w * 64 + free.trailing_zeros() as usize;
                    free &= free - 1;
                    if pos >= len || !self.try_pos(t, pos, revenue, next_same, rest_bound) {
                        break 'scan;
                    }
                }
            }
            self.blocked[t] = blocked;
        } else {
            for pos in from..len {
                let rest = if next_same {
                    self.upper_bound(t + 1, pos + 1)
                } else {
                    rest_bound
                };
                if revenue + self.vals[t][pos] + rest <= self.best {
                    break;
                }
                if self.clear(self.lists[t][pos] as usize) {
                    self.try_pos(t, pos, revenue, next_same, rest_bound);
                }
            }
        }
        // This train runs nothing — and then neither do identical ones
        // after it (any other arrangement is a permutation of one tried).
        self.search(self.group_end[t], revenue, 0);
    }
}

/// The best (train index, kept route) pairs and their total, for trains
/// given by kind index; lists longer than `index_min` searched through an
/// index.
fn best_combination(walk: &Walk<'_>, train_kinds: &[usize], index_min: usize) -> (Vec<(usize, usize)>, i32) {
    let n_found = walk.n_found();
    if n_found == 0 || train_kinds.is_empty() {
        return (Vec::new(), 0);
    }
    let nk = walk.kinds.len();
    let words = walk.words;
    let n_track = words * 64;
    // per kind: the routes kept for it, best first
    let mut kind_lists: Vec<Vec<u32>> = Vec::with_capacity(nk);
    let mut kind_vals: Vec<Vec<i32>> = Vec::with_capacity(nk);
    let mut kind_uses: Vec<Option<Vec<u64>>> = Vec::with_capacity(nk);
    for k in 0..nk {
        let val = |r: u32| walk.found_vals[r as usize * nk + k];
        let mut list: Vec<u32> = if train_kinds.contains(&k) {
            (0..n_found as u32).filter(|&r| val(r) > 0).collect()
        } else {
            Vec::new()
        };
        list.sort_by_key(|&r| std::cmp::Reverse(val(r)));
        let uses = (list.len() > index_min).then(|| {
            let pw = list.len().div_ceil(64);
            let mut uses = vec![0u64; n_track * pw];
            for (pos, &r) in list.iter().enumerate() {
                let b = &walk.found_bits[r as usize * words..(r as usize + 1) * words];
                for (w, &word) in b.iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        let id = w * 64 + word.trailing_zeros() as usize;
                        word &= word - 1;
                        uses[id * pw + pos / 64] |= 1u64 << (pos % 64);
                    }
                }
            }
            uses
        });
        kind_vals.push(list.iter().map(|&r| val(r)).collect());
        kind_lists.push(list);
        kind_uses.push(uses);
    }
    // Short lists first (the long ones are queried through their index);
    // identical trains adjacent.
    let mut order: Vec<usize> = (0..train_kinds.len()).collect();
    order.sort_by_key(|&i| {
        let k = train_kinds[i];
        (kind_lists[k].len(), k)
    });
    let same_as_prev: Vec<bool> = (0..order.len())
        .map(|t| t > 0 && train_kinds[order[t]] == train_kinds[order[t - 1]])
        .collect();
    let mut group_end = vec![order.len(); order.len()];
    for t in (0..order.len().saturating_sub(1)).rev() {
        group_end[t] = if same_as_prev[t + 1] { group_end[t + 1] } else { t + 1 };
    }
    let mut combo = Combo {
        bits: &walk.found_bits,
        words,
        lists: order.iter().map(|&i| kind_lists[train_kinds[i]].as_slice()).collect(),
        vals: order.iter().map(|&i| kind_vals[train_kinds[i]].as_slice()).collect(),
        uses: order.iter().map(|&i| kind_uses[train_kinds[i]].as_deref()).collect(),
        same_as_prev,
        group_end,
        used: vec![0u64; words],
        blocked: vec![Vec::new(); order.len()],
        chosen: vec![None; order.len()],
        best: 0,
        best_chosen: vec![None; order.len()],
    };
    combo.search(0, 0, 0);
    let mut chosen: Vec<(usize, usize)> = combo
        .best_chosen
        .iter()
        .enumerate()
        .filter_map(|(t, r)| r.map(|r| (order[t], r as usize)))
        .collect();
    chosen.sort_unstable();
    (chosen, combo.best)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Result of one routing call.
pub(crate) struct AutoRoutes {
    pub routes: Vec<RouteData>,
    pub revenue: i32,
    /// No walk budget ran out: the run is the optimum.
    pub exact: bool,
    /// Walk steps (stop arrivals) of the normal and the 5+5E pass.
    pub steps: (u64, u64),
    /// Diagnostics: routes kept, and microseconds in the walk / the
    /// combination search.
    pub found: usize,
    pub micros: (u64, u64),
}

impl BaseGame {
    /// The revenue-maximising legal routes for `corp`'s unoperated trains
    /// (one `RouteData` per running train: train instance id, stop hexes,
    /// node-to-node `connections`, `nodes` signatures, the engine's price
    /// as `revenue`) and their total. Empty when nothing can run.
    pub(crate) fn g1867_auto_routes(&mut self, corp: &str) -> (Vec<RouteData>, i32) {
        let r = self.g1867_route_search(corp, true);
        (r.routes, r.revenue)
    }

    /// The search behind `g1867_auto_routes`; `filter` false keeps every
    /// priced route (the tests' reference).
    pub(crate) fn g1867_route_search(&self, corp: &str, filter: bool) -> AutoRoutes {
        let t0 = std::time::Instant::now();
        let none = AutoRoutes {
            routes: Vec::new(),
            revenue: 0,
            exact: true,
            steps: (0, 0),
            found: 0,
            micros: (0, 0),
        };
        let Some(&ci) = self.corp_idx.get(corp) else {
            return none;
        };
        let mut trains: Vec<(String, TrainKind)> = Vec::new();
        for t in self.corporations[ci].trains.iter().filter(|t| !t.operated) {
            if let Some(k) = TrainKind::of(&t.name) {
                trains.push((t.id.clone(), k));
            }
        }
        let mut kinds: Vec<TrainKind> = Vec::new();
        for (_, k) in &trains {
            if !kinds.contains(k) && kinds.len() < MAXK {
                kinds.push(*k);
            }
        }
        trains.retain(|(_, k)| kinds.contains(k));
        if trains.is_empty() {
            return none;
        }
        let train_kinds: Vec<usize> = trains
            .iter()
            .map(|(_, k)| kinds.iter().position(|x| x == k).unwrap())
            .collect();

        let mut walk = Walk::new(self, corp, kinds.clone());
        walk.filter = filter;
        walk.starts = walk.reachable();
        if !walk.starts.iter().any(|&k| walk.info_of(k).tokened) {
            return none;
        }
        let normal: Vec<bool> = kinds.iter().map(|k| !k.express).collect();
        let express: Vec<bool> = kinds.iter().map(|k| k.express).collect();
        let mut exact = true;
        let mut steps = (0u64, 0u64);
        if normal.iter().any(|&b| b) {
            let max_pay = kinds.iter().filter(|k| !k.express).map(|k| k.pay).max().unwrap_or(0);
            walk.run_pass(normal, max_pay, NORMAL_STEP_BUDGET);
            steps.0 = walk.steps;
            exact &= !walk.exhausted;
        }
        if express.iter().any(|&b| b) {
            let budget = match EXPRESS_BUDGET_OVERRIDE.load(Ordering::Relaxed) {
                0 => EXPRESS_STEP_BUDGET,
                b => b,
            };
            walk.run_pass(express, u32::MAX, budget);
            steps.1 = walk.steps;
            exact &= !walk.exhausted;
        }
        if !exact {
            INEXACT_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        let t1 = std::time::Instant::now();
        let (chosen, best) = best_combination(&walk, &train_kinds, INDEX_MIN);
        let t2 = std::time::Instant::now();
        let bonuses = route_bonuses(self, corp);
        let mut routes = Vec::new();
        let mut total = 0;
        for (ti, f) in chosen {
            let fast = walk.found_vals[f * kinds.len() + train_kinds[ti]];
            let w = walk.route_of(f);
            let price = walk.shared_price(&w.0, trains[ti].1, &bonuses);
            if price != Some(fast) {
                PRICE_MISMATCHES.fetch_add(1, Ordering::Relaxed);
            }
            let price = price.unwrap_or(fast);
            total += price;
            routes.push(walk.route_data(&w, &trains[ti].0, price));
        }
        debug_assert_eq!(total, best, "the fast pricer disagrees with price_bucketed_stops");
        AutoRoutes {
            routes,
            revenue: total,
            exact,
            steps,
            found: walk.n_found(),
            micros: ((t1 - t0).as_micros() as u64, (t2 - t1).as_micros() as u64),
        }
    }
}

#[pymethods]
impl BaseGame {
    /// 1867 auto-router: the best run for `corp`'s unoperated trains as
    /// JSON `{"routes": [{"train", "hexes", "connections", "nodes",
    /// "revenue"}...], "revenue": total, "exact": bool, "steps": [normal,
    /// 5+5E]}` — the route dicts an 18xx.games `run_routes` action carries.
    /// `exact` is false when a walk budget ran out; `steps` are the walks'
    /// stop arrivals.
    fn g1867_auto_routes_json(&mut self, corp: &str) -> String {
        let r = self.g1867_route_search(corp, true);
        let routes: Vec<serde_json::Value> = r
            .routes
            .iter()
            .map(|x| {
                serde_json::json!({
                    "train": x.train_name,
                    "hexes": x.hexes,
                    "connections": x.connections,
                    "nodes": x.nodes,
                    "revenue": x.revenue,
                })
            })
            .collect();
        serde_json::json!({
            "routes": routes,
            "revenue": r.revenue,
            "exact": r.exact,
            "steps": [r.steps.0, r.steps.1],
            "found": r.found,
            "micros": [r.micros.0, r.micros.1],
        })
        .to_string()
    }

    /// Router counters for this process: (chosen runs whose fast price
    /// disagreed with the replay pricer, calls whose walk budget ran out).
    #[staticmethod]
    fn g1867_router_counters() -> (u64, u64) {
        (
            PRICE_MISMATCHES.load(Ordering::Relaxed),
            INEXACT_CALLS.load(Ordering::Relaxed),
        )
    }

    /// Experiments: override the 5+5E walk's step budget for this process
    /// (0 restores the default).
    #[staticmethod]
    fn g1867_router_set_express_budget(budget: u64) {
        EXPRESS_BUDGET_OVERRIDE.store(budget, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{EntityId, Token, Train};
    use crate::tiles::{parse_tile, TileColor};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const CORP: &str = "CPR";

    fn game() -> BaseGame {
        let players: HashMap<u32, String> = (1..=4).map(|i| (i, format!("Player {i}"))).collect();
        BaseGame::build_titled("1867", vec![1, 2, 3, 4], players)
    }

    /// Replace `hex`'s tile with one built from `dsl` (rotation 0).
    fn lay(g: &mut BaseGame, hex: &str, dsl: &str) {
        let def = parse_tile("test", dsl, TileColor::Green);
        let hi = g.hex_idx[hex];
        g.hexes[hi].tile = BaseGame::tile_from_def(&def, 0);
    }

    fn token_as(g: &mut BaseGame, hex: &str, city: usize, corp: &str, kind: &str) {
        let hi = g.hex_idx[hex];
        let c = &mut g.hexes[hi].tile.cities[city];
        let slot = c.tokens.iter().position(|t| t.is_none()).expect("a free slot");
        let mut t = Token::new(corp.to_string(), 0);
        t.used = true;
        t.token_type = kind.to_string();
        t.city_hex_id = hex.to_string();
        c.tokens[slot] = Some(t);
    }

    fn token(g: &mut BaseGame, hex: &str) {
        token_as(g, hex, 0, CORP, "normal");
    }

    fn give(g: &mut BaseGame, names: &[&str]) {
        let ci = g.corp_idx[CORP];
        let defs = data::trains();
        for (i, name) in names.iter().enumerate() {
            let d = defs.iter().find(|d| d.name == *name).expect("a 1867 train");
            let mut t = Train::new(name.to_string(), d.distance, d.price);
            t.id = format!("{name}-{i}");
            t.owner = EntityId::corporation(CORP);
            g.corporations[ci].trains.push(t);
        }
    }

    fn run(g: &BaseGame) -> AutoRoutes {
        let before = PRICE_MISMATCHES.load(Ordering::Relaxed);
        let r = g.g1867_route_search(CORP, true);
        assert_eq!(
            PRICE_MISMATCHES.load(Ordering::Relaxed),
            before,
            "fast price != replay price"
        );
        assert!(r.exact);
        r
    }

    fn revenue(g: &BaseGame) -> i32 {
        run(g).revenue
    }

    fn stops(r: &RouteData) -> Vec<&str> {
        r.hexes.iter().map(String::as_str).collect()
    }

    const OFF40: &str = "offboard=revenue:yellow_40|green_50|brown_60|gray_70";

    /// The G column (G3..G15, each hex's edge 0 facing the next): an
    /// offboard, five stops, an offboard.
    fn line(g: &mut BaseGame) {
        lay(g, "G3", &format!("{OFF40};path=a:0,b:_0"));
        lay(g, "G5", "city=revenue:20;path=a:3,b:_0;path=a:_0,b:0");
        lay(g, "G7", "city=revenue:50;path=a:3,b:_0;path=a:_0,b:0");
        lay(g, "G9", "town=revenue:10;path=a:3,b:_0;path=a:_0,b:0");
        lay(g, "G11", "city=revenue:30;path=a:3,b:_0;path=a:_0,b:0");
        lay(g, "G13", "city=revenue:60;path=a:3,b:_0;path=a:_0,b:0");
        lay(g, "G15", &format!("{OFF40};path=a:3,b:_0"));
        token(g, "G5");
    }

    // -- pricing ----------------------------------------------------------

    /// The fast pricer equals the replay pricer (`price_bucketed_stops`,
    /// which tries every combination like Ruby) on random stop sets.
    #[test]
    fn fast_price_matches_the_replay_pricer() {
        let kinds = [
            TrainKind::of("2").unwrap(),
            TrainKind::of("3").unwrap(),
            TrainKind::of("5").unwrap(),
            TrainKind::of("8").unwrap(),
            TrainKind::of("2+2").unwrap(),
            TrainKind::of("5+5E").unwrap(),
        ];
        let mut rng = StdRng::seed_from_u64(1867);
        let mut opt = Vec::new();
        let mut checked = 0;
        for _ in 0..20_000 {
            let n = rng.gen_range(2..=10);
            let mut infos = Vec::new();
            let mut stops = Vec::new();
            let mut bonuses = RouteBonuses {
                capital_hexes: vec!["CAP1".into(), "CAP2".into()],
                capital_partner: Some("TIM".into()),
                capital_amount: 40,
                ..Default::default()
            };
            for i in 0..n {
                let kind = match rng.gen_range(0..10) {
                    0..=4 => StopKind::City,
                    5..=7 => StopKind::Town,
                    _ => StopKind::Offboard,
                };
                let revenue = 10 * rng.gen_range(1..=8);
                let tokened = kind == StopKind::City && rng.gen_bool(0.3);
                let hex = match rng.gen_range(0..12) {
                    0 if kind == StopKind::City => "CAP1".to_string(),
                    1 if kind == StopKind::City => "CAP2".to_string(),
                    2 if kind == StopKind::City => "TIM".to_string(),
                    _ => format!("H{i}"),
                };
                if stops.iter().any(|s: &Stop| s.hex == hex) {
                    continue; // one stop per hex
                }
                let bonus = if rng.gen_bool(0.15) { 10 } else { 0 };
                if bonus > 0 {
                    bonuses.hex_bonuses.push((vec![hex.clone()], bonus));
                }
                infos.push(StopInfo {
                    kind,
                    revenue,
                    tokened,
                    bonus,
                    capital: hex.starts_with("CAP"),
                    timmins: hex == "TIM",
                });
                stops.push(Stop {
                    hex,
                    kind,
                    revenue,
                    tokened,
                });
            }
            for k in kinds {
                let counted: &[StopKind] = if k.express {
                    &[StopKind::Offboard]
                } else {
                    &[StopKind::City, StopKind::Offboard]
                };
                let reference =
                    revenue::price_bucketed_stops(&stops, k.pay as usize, k.multiplier, counted, &bonuses).ok();
                assert_eq!(price_stops(&infos, k, &mut opt), reference, "{k:?} {infos:?}");
                checked += 1;
            }
        }
        assert!(checked > 100_000);
    }

    /// The walk's incremental pricing equals `price_stops` on every prefix
    /// of random stop sequences (long ones: the 5+5E's routes).
    #[test]
    fn accumulated_price_matches_price_stops() {
        let kinds: Vec<TrainKind> = ["2", "4", "8", "2+2", "5+5E"]
            .iter()
            .map(|n| TrainKind::of(n).unwrap())
            .collect();
        let mut rng = StdRng::seed_from_u64(67);
        let mut opt = Vec::new();
        let mut slow = 0;
        for _ in 0..4000 {
            let n = rng.gen_range(1..=30);
            let infos: Vec<StopInfo> = (0..n)
                .map(|_| {
                    let kind = match rng.gen_range(0..10) {
                        0..=5 => StopKind::City,
                        6..=8 => StopKind::Town,
                        _ => StopKind::Offboard,
                    };
                    let city = kind == StopKind::City;
                    StopInfo {
                        kind,
                        revenue: 10 * rng.gen_range(1..=9),
                        tokened: city && rng.gen_bool(0.15),
                        bonus: if rng.gen_bool(0.1) { 10 } else { 0 },
                        capital: city && rng.gen_bool(0.08),
                        timmins: city && rng.gen_bool(0.05),
                    }
                })
                .collect();
            for &k in &kinds {
                let mut a = Acc::default();
                for (i, s) in infos.iter().enumerate() {
                    a.push(k, s, i as u16);
                    let want = price_stops(&infos[..=i], k, &mut opt);
                    match a.price(k) {
                        Ok(v) => assert_eq!(v, want, "{k:?} {:?}", &infos[..=i]),
                        Err(()) => slow += 1,
                    }
                }
            }
        }
        // overflow (more than ACC_SPECIAL special stops) is the only
        // from-scratch case, and these sequences rarely reach it
        assert!(slow < 200, "{slow} from-scratch pricings");
    }

    #[test]
    fn express_needs_a_tokened_choice_and_pays_double() {
        let k = TrainKind::of("5+5E").unwrap();
        let s = |kind, revenue, tokened| StopInfo {
            kind,
            revenue,
            tokened,
            bonus: 0,
            capital: false,
            timmins: false,
        };
        let stops = [
            s(StopKind::Offboard, 40, false),
            s(StopKind::City, 20, true),
            s(StopKind::City, 50, false),
            s(StopKind::City, 40, false),
            s(StopKind::City, 60, false),
            s(StopKind::Offboard, 40, false),
        ];
        // offboards + the token city + the two best others, all doubled
        assert_eq!(
            price_stops(&stops, k, &mut Vec::new()),
            Some((40 + 40 + 20 + 60 + 50) * 2)
        );
    }

    #[test]
    fn capitals_bonus_needs_a_capital_and_timmins_among_the_paid_stops() {
        let s = |kind, revenue, tokened, capital, timmins| StopInfo {
            kind,
            revenue,
            tokened,
            bonus: 0,
            capital,
            timmins,
        };
        let two = TrainKind::of("2").unwrap();
        let twotwo = TrainKind::of("2+2").unwrap();
        let stops = [
            s(StopKind::City, 40, true, true, false),
            s(StopKind::City, 40, false, false, true),
        ];
        assert_eq!(price_stops(&stops, two, &mut Vec::new()), Some(40 + 40 + 40));
        assert_eq!(price_stops(&stops, twotwo, &mut Vec::new()), Some((40 + 40) * 2 + 80));
        // the 5+5E pays a $30 Timmins over a $50 city for the bonus
        let e = TrainKind::of("5+5E").unwrap();
        let stops = [
            s(StopKind::Offboard, 40, false, false, false),
            s(StopKind::City, 60, true, true, false),
            s(StopKind::City, 50, false, false, false),
            s(StopKind::City, 50, false, false, false),
            s(StopKind::City, 50, false, false, false),
            s(StopKind::City, 30, false, false, true),
            s(StopKind::Offboard, 40, false, false, false),
        ];
        assert_eq!(
            price_stops(&stops, e, &mut Vec::new()),
            Some((40 + 40 + 60 + 50 + 30) * 2 + 80)
        );
    }

    // -- routes on the 1867 map --------------------------------------------

    #[test]
    fn towns_ride_free_and_fill_spare_capacity() {
        let mut g = game();
        line(&mut g);
        give(&mut g, &["8"]);
        // every counted stop (6 <= 8) plus the town in a spare slot
        assert_eq!(revenue(&g), 40 + 20 + 50 + 10 + 30 + 60 + 40);

        let build = |train: &str| {
            let mut g = game();
            lay(&mut g, "G5", "city=revenue:20;path=a:_0,b:0");
            lay(&mut g, "G7", "town=revenue:10;path=a:3,b:_0;path=a:_0,b:0");
            lay(&mut g, "G9", "city=revenue:30;path=a:3,b:_0;path=a:_0,b:0");
            lay(&mut g, "G11", "town=revenue:20;path=a:3,b:_0");
            token(&mut g, "G5");
            give(&mut g, &[train]);
            g
        };
        // two cities fill the 2-train: towns pass free and unpaid; the
        // route ends at its last paying stop
        let r = run(&build("2"));
        assert_eq!(r.revenue, 50);
        assert_eq!(stops(&r.routes[0]), vec!["G5", "G7", "G9"]);
        // a spare slot: the better town
        assert_eq!(revenue(&build("3")), 20 + 30 + 20);
    }

    #[test]
    fn one_stop_per_hex() {
        // G7 holds two cities: city 0 on the G5 side and F8's, city 1 on
        // G9's. A 4-train could run G5 - G7.0 - F8 - G9 - G7.1 (130) but
        // "Route visits same hex twice".
        let mut g = game();
        lay(&mut g, "G5", "city=revenue:20;path=a:_0,b:0");
        lay(
            &mut g,
            "G7",
            "city=revenue:30;city=revenue:30;path=a:3,b:_0;path=a:1,b:_0;path=a:0,b:_1",
        );
        lay(&mut g, "F8", "path=a:4,b:5");
        lay(&mut g, "G9", "city=revenue:50;path=a:2,b:_0;path=a:3,b:_0");
        token(&mut g, "G5");
        give(&mut g, &["4"]);
        let r = run(&g);
        assert_eq!(r.revenue, 20 + 30 + 50, "{:?}", r.routes);
    }

    #[test]
    fn detroit_counts_once() {
        // B18 between the two Detroit offboards (A17 edge 5, A19 edge 4;
        // preprinted, groups:Detroit, yellow $30).
        let mut g = game();
        lay(&mut g, "B18", "city=revenue:20;path=a:1,b:_0;path=a:2,b:_0");
        token(&mut g, "B18");
        give(&mut g, &["3"]);
        let r = run(&g);
        assert_eq!(r.revenue, 30 + 20, "{:?}", r.routes);
        assert_eq!(r.routes[0].hexes.len(), 2);
    }

    #[test]
    fn full_foreign_city_only_ends_a_route_and_neutral_tokens_never_block() {
        let build = |kind: &str| {
            let mut g = game();
            lay(&mut g, "G5", "city=revenue:20;path=a:_0,b:0");
            lay(&mut g, "G7", "city=revenue:30;path=a:3,b:_0;path=a:_0,b:0");
            lay(&mut g, "G9", "city=revenue:40;path=a:3,b:_0");
            token(&mut g, "G5");
            token_as(&mut g, "G7", 0, "CNR", kind);
            give(&mut g, &["3"]);
            g
        };
        assert_eq!(
            revenue(&build("normal")),
            20 + 30,
            "G7 is full of CNR: the route ends there"
        );
        assert_eq!(
            revenue(&build("neutral")),
            20 + 30 + 40,
            "a neutral token keeps G7 open"
        );
    }

    #[test]
    fn track_never_crosses_a_wall() {
        // H14 | H16 is impassable (H14 edge 0, H16 edge 3).
        let mut g = game();
        lay(&mut g, "H14", "city=revenue:20;path=a:0,b:_0");
        lay(&mut g, "H16", "offboard=revenue:10;path=a:3,b:_0");
        token(&mut g, "H14");
        give(&mut g, &["2"]);
        assert_eq!(revenue(&g), 0);
        g.walls = std::sync::Arc::new(HashMap::new());
        assert_eq!(revenue(&g), 30, "without the wall the track joins");
    }

    #[test]
    fn trains_share_no_track() {
        let build = |trains: &[&str]| {
            let mut g = game();
            lay(&mut g, "G5", "city=revenue:20;path=a:_0,b:0");
            lay(&mut g, "G7", "city=revenue:30;path=a:3,b:_0;path=a:_0,b:0");
            lay(&mut g, "G9", "city=revenue:40;path=a:3,b:_0");
            token_as(&mut g, "G7", 0, CORP, "normal");
            give(&mut g, trains);
            g
        };
        // each 2-train takes one side of G7
        let r = run(&build(&["2", "2"]));
        assert_eq!(r.revenue, (30 + 40) + (30 + 20));
        assert_eq!(r.routes.len(), 2);
        // a 3-train alone runs the whole line
        assert_eq!(revenue(&build(&["3"])), 20 + 30 + 40);
        // a 3-train and a 2-train: the line, or both halves
        assert_eq!(revenue(&build(&["3", "2"])), 120);
    }

    #[test]
    fn express_runs_offboard_to_offboard() {
        let mut g = game();
        line(&mut g);
        give(&mut g, &["5+5E"]);
        let r = run(&g);
        // offboards, the token city and the two best others, ×2 — not the
        // untokened top three (60, 50, 30); dropping an offboard end for
        // the fourth city would pay less (30 < 40)
        assert_eq!(r.revenue, (40 + 40 + 20 + 60 + 50) * 2);
        assert_eq!(stops(&r.routes[0]), vec!["G3", "G5", "G7", "G9", "G11", "G13", "G15"]);
        // a 2+2 pays two counted stops, doubled
        let mut g = game();
        line(&mut g);
        give(&mut g, &["2+2"]);
        assert_eq!(revenue(&g), (20 + 50) * 2);
    }

    #[test]
    fn hex_bonus_is_flat_and_needs_the_company() {
        // F16 (a capital) joined to F18 (NFB's hex): F16 edge 0 | F18 edge 3.
        let build = |owned: bool, train: &str| {
            let mut g = game();
            lay(&mut g, "F16", "city=revenue:30;path=a:0,b:_0");
            lay(
                &mut g,
                "F18",
                "offboard=revenue:yellow_30|green_40|brown_50|gray_60;path=a:3,b:_0",
            );
            token(&mut g, "F16");
            if owned {
                let ci = g.companies.iter().position(|c| c.sym == "NFB").unwrap();
                g.companies[ci].owner = EntityId::corporation(CORP);
            }
            give(&mut g, &[train]);
            g
        };
        assert_eq!(revenue(&build(false, "2")), 60);
        assert_eq!(revenue(&build(true, "2")), 70);
        assert_eq!(revenue(&build(true, "2+2")), 60 * 2 + 10);
    }

    #[test]
    fn operated_trains_do_not_run_and_the_wire_shape() {
        let mut g = game();
        line(&mut g);
        give(&mut g, &["2", "3"]);
        let ci = g.corp_idx[CORP];
        g.corporations[ci].trains[1].operated = true;
        let r = run(&g);
        assert_eq!(r.routes.len(), 1);
        let route = &r.routes[0];
        assert_eq!(route.train_name, "2-0");
        assert_eq!(r.revenue, 20 + 50);
        assert_eq!(route.hexes, vec!["G5", "G7"]);
        assert_eq!(route.connections, vec![vec!["G5".to_string(), "G7".to_string()]]);
        assert_eq!(route.nodes, vec!["G5-0", "G7-0"]);
        assert_eq!(route.revenue, Some(70));
        let v: serde_json::Value = serde_json::from_str(&g.g1867_auto_routes_json(CORP)).unwrap();
        assert_eq!(v["revenue"], 70);
        assert_eq!(v["routes"][0]["train"], "2-0");
        assert_eq!(v["routes"][0]["connections"][0][1], "G7");
        // the replay pricer prices the emitted connections the same
        let rd = RouteData {
            revenue: None,
            ..route.clone()
        };
        assert_eq!(data::recorded_route_revenue(&g, CORP, &rd).unwrap(), 70);
    }

    /// Exactness of the tight-route filter, the hopeless-start cut and the
    /// branch-and-bound: on random tile layouts the router's total equals
    /// a reference that keeps every priced route, and (with few routes)
    /// a brute force over all train/route assignments.
    #[test]
    fn random_maps_match_the_unfiltered_reference() {
        let catalog = crate::tiles::tile_catalog_1867();
        let mut names: Vec<&String> = catalog.keys().collect();
        names.sort();
        let region = [
            "E5", "E7", "E9", "F4", "F6", "F8", "F10", "G3", "G5", "G7", "G9", "G11", "H4", "H6", "H8", "H10",
        ];
        let pool = ["2", "3", "4", "5", "8", "2+2", "5+5E"];
        let mut rng = StdRng::seed_from_u64(7);
        let (mut compared, mut brute, mut paying, mut express) = (0, 0, 0, 0);
        for _ in 0..300 {
            let mut g = game();
            for h in region {
                if rng.gen_bool(0.85) {
                    let def = catalog[names[rng.gen_range(0..names.len())]].clone();
                    let hi = g.hex_idx[h];
                    g.hexes[hi].tile = BaseGame::tile_from_def(&def, rng.gen_range(0..6));
                }
            }
            // a few offboards on the edge of the region
            for h in ["G3", "E9"] {
                if rng.gen_bool(0.4) {
                    let rot = rng.gen_range(0..6);
                    let def = parse_tile("off", &format!("{OFF40};path=a:0,b:_0;path=a:1,b:_0"), TileColor::Red);
                    let hi = g.hex_idx[h];
                    g.hexes[hi].tile = BaseGame::tile_from_def(&def, rot);
                }
            }
            let cities: Vec<(String, usize)> = region
                .iter()
                .flat_map(|h| {
                    let hi = g.hex_idx[*h];
                    (0..g.hexes[hi].tile.cities.len()).map(move |c| (h.to_string(), c))
                })
                .collect();
            if cities.is_empty() {
                continue;
            }
            for _ in 0..rng.gen_range(1..=2) {
                let (h, c) = &cities[rng.gen_range(0..cities.len())];
                let hi = g.hex_idx[h.as_str()];
                if g.hexes[hi].tile.cities[*c].tokens.iter().any(|t| t.is_none()) {
                    token_as(&mut g, h, *c, CORP, "normal");
                }
            }
            for _ in 0..rng.gen_range(0..=3) {
                let (h, c) = &cities[rng.gen_range(0..cities.len())];
                let hi = g.hex_idx[h.as_str()];
                if g.hexes[hi].tile.cities[*c].tokens.iter().any(|t| t.is_none()) {
                    token_as(&mut g, h, *c, "CNR", "normal");
                }
            }
            let n = rng.gen_range(1..=3);
            let trains: Vec<&str> = (0..n).map(|_| pool[rng.gen_range(0..pool.len())]).collect();
            give(&mut g, &trains);
            g.phase.tiles = ["yellow", "green", "brown"][..rng.gen_range(1..=3)]
                .iter()
                .map(|s| s.to_string())
                .collect();

            let fast = g.g1867_route_search(CORP, true);
            let reference = g.g1867_route_search(CORP, false);
            assert!(fast.exact && reference.exact);
            assert_eq!(fast.revenue, reference.revenue, "trains {trains:?}");
            compared += 1;
            if fast.revenue > 0 {
                paying += 1;
                express += usize::from(trains.contains(&"5+5E"));
            }

            // brute force over the reference's routes when small
            let kinds: Vec<TrainKind> = trains.iter().map(|t| TrainKind::of(t).unwrap()).collect();
            let mut walk = Walk::new(&g, CORP, {
                let mut v: Vec<TrainKind> = Vec::new();
                for k in &kinds {
                    if !v.contains(k) {
                        v.push(*k);
                    }
                }
                v
            });
            walk.filter = false;
            walk.starts = walk.reachable();
            let nk = walk.kinds.len();
            let normal: Vec<bool> = walk.kinds.iter().map(|k| !k.express).collect();
            let express: Vec<bool> = walk.kinds.iter().map(|k| k.express).collect();
            let max_pay = walk
                .kinds
                .iter()
                .filter(|k| !k.express)
                .map(|k| k.pay)
                .max()
                .unwrap_or(0);
            walk.run_pass(normal, max_pay, u64::MAX);
            walk.run_pass(express, u32::MAX, u64::MAX);
            if walk.n_found() > 150 {
                continue;
            }
            let tk: Vec<usize> = kinds
                .iter()
                .map(|k| walk.kinds.iter().position(|x| x == k).unwrap())
                .collect();
            let words = walk.words;
            let track: Vec<Vec<u64>> = walk.found_bits.chunks(words).map(|c| c.to_vec()).collect();
            fn go(t: usize, tk: &[usize], nk: usize, vals: &[i32], track: &[Vec<u64>], used: &mut [u64]) -> i32 {
                if t == tk.len() {
                    return 0;
                }
                let mut best = go(t + 1, tk, nk, vals, track, used);
                for (r, tr) in track.iter().enumerate() {
                    let v = vals[r * nk + tk[t]];
                    if v > 0 && tr.iter().zip(used.iter()).all(|(a, b)| a & b == 0) {
                        for (u, a) in used.iter_mut().zip(tr) {
                            *u |= a;
                        }
                        best = best.max(v + go(t + 1, tk, nk, vals, track, used));
                        for (u, a) in used.iter_mut().zip(tr) {
                            *u &= !a;
                        }
                    }
                }
                best
            }
            let b = go(0, &tk, nk, &walk.found_vals, &track, &mut vec![0u64; words]);
            // both list searches of the branch-and-bound, on every route
            assert_eq!(
                best_combination(&walk, &tk, 0).1,
                b,
                "indexed search, trains {trains:?}"
            );
            assert_eq!(
                best_combination(&walk, &tk, usize::MAX).1,
                b,
                "scanning search, trains {trains:?}"
            );
            assert_eq!(fast.revenue, b, "brute force, trains {trains:?}");
            brute += 1;
        }
        assert!(
            compared > 200 && brute > 50 && paying > 100 && express > 20,
            "compared {compared}, brute {brute}, paying {paying}, with a 5+5E {express}"
        );
    }
}
