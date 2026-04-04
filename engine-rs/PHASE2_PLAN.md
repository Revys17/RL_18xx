# Phase 2: Game State Construction + Fast Clone

## Context

This is Phase 2 of migrating the 1830 game engine from Python to Rust. Phases 0 (scaffolding) and 1 (data structures) are complete. The Rust crate at `engine-rs/` has PyO3 bindings for all entity types (Player, Corporation, Company, Bank, Depot, Train, Token, Share, Hex, Tile, City, Town, etc.) in `src/core.rs`, `src/entities.rs`, `src/graph.rs`.

Phase 2 adds: game construction from player names, all read-only state queries the Python encoder needs, and a fast `clone_for_search()`.

## Goal

After Phase 2:
- `BaseGame({1: "P1", 2: "P2", 3: "P3", 4: "P4"})` constructs a complete initial 1830 game state in Rust
- All state queries return correct values (testable against the Python engine)
- `pickle_clone()` is ≥10x faster than Python (target 50-100x)
- We cannot yet process actions (that's Phase 3)

## Existing Rust Code (Phase 1)

Read these files to understand what's already implemented:
- `engine-rs/src/lib.rs` — PyO3 module, exports all types
- `engine-rs/src/core.rs` — `SharePrice` (price, row, column, types), `Phase` (name, operating_rounds, train_limit, tiles)
- `engine-rs/src/entities.rs` — `EntityId` (string-based: "player:1", "corp:PRR", etc.), `Train`, `Token`, `Share`, `Company`, `Corporation`, `Player`, `Bank`, `Depot`
- `engine-rs/src/graph.rs` — `City`, `Town`, `Offboard`, `Edge`, `Upgrade`, `Tile`, `Hex`
- `engine-rs/src/game.rs` — STUB only (title + finished fields)

Build command: `cd engine-rs && unset CONDA_PREFIX && uv run maturin develop --release`

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/title/mod.rs` | Create — `pub mod g1830;` |
| `src/title/g1830.rs` | Create — all static 1830 data as Rust constants/functions |
| `src/game.rs` | Rewrite — full BaseGame with constructor, queries, clone |
| `src/lib.rs` | Update — add `pub mod title;` |

---

## Step 1: Static Game Data (`src/title/g1830.rs`)

### Definition Structs

These are NOT pyclass — they're plain Rust structs used as templates:

```rust
pub struct CorporationDef {
    pub sym: &'static str,
    pub name: &'static str,
    pub token_prices: &'static [i32],
    pub home_hex: &'static str,
    pub home_city_index: u8,  // usually 0, but NYNH uses 1 for G19
}

pub struct CompanyDef {
    pub sym: &'static str,
    pub name: &'static str,
    pub value: i32,
    pub revenue: i32,
    pub blocked_hexes: &'static [&'static str],
}

pub struct TrainDef {
    pub name: &'static str,
    pub distance: u32,
    pub price: i32,
    pub count: u32,
    pub rusts_on: Option<&'static str>,
}

pub struct PhaseDef {
    pub name: &'static str,
    pub train_limit: u8,
    pub tiles: &'static [&'static str],
    pub operating_rounds: u8,
}

pub struct MarketCell {
    pub price: i32,
    pub zone: MarketZone,
}

pub enum MarketZone { Normal, Par, Yellow, Orange, Brown }

pub struct HexDef {
    pub coord: &'static str,
    pub hex_type: HexType,
    pub terrain_cost: i32,       // 0, 80 (water), or 120 (mountain)
}

pub enum HexType {
    Blank,
    City { revenue: i32, slots: u8 },
    Town { revenue: i32 },
    DoubleCity { revenue: i32 },     // OO cities (E5, D10, E11, H18, G19)
    DoubleTown,                       // Double towns (G7, G17, F20)
    Offboard { yellow_revenue: i32, brown_revenue: i32 },
    Path,                             // Gray paths (E9, A17, D24)
}
```

### Corporation Data (8 corporations)

```rust
pub fn corporations() -> Vec<CorporationDef> {
    vec![
        CorporationDef { sym: "PRR",  name: "Pennsylvania Railroad",                    token_prices: &[0, 40, 100, 100], home_hex: "H12", home_city_index: 0 },
        CorporationDef { sym: "NYC",  name: "New York Central Railroad",                token_prices: &[0, 40, 100, 100], home_hex: "E19", home_city_index: 0 },
        CorporationDef { sym: "CPR",  name: "Canadian Pacific Railroad",                token_prices: &[0, 40, 100, 100], home_hex: "A19", home_city_index: 0 },
        CorporationDef { sym: "B&O",  name: "Baltimore & Ohio Railroad",                token_prices: &[0, 40, 100],      home_hex: "I15", home_city_index: 0 },
        CorporationDef { sym: "C&O",  name: "Chesapeake & Ohio Railroad",               token_prices: &[0, 40, 100],      home_hex: "F6",  home_city_index: 0 },
        CorporationDef { sym: "ERIE", name: "Erie Railroad",                             token_prices: &[0, 40, 100],      home_hex: "E11", home_city_index: 0 },
        CorporationDef { sym: "NYNH", name: "New York, New Haven & Hartford Railroad",  token_prices: &[0, 40],           home_hex: "G19", home_city_index: 1 },
        CorporationDef { sym: "B&M",  name: "Boston & Maine Railroad",                  token_prices: &[0, 40],           home_hex: "E23", home_city_index: 0 },
    ]
}
```

Each corporation has 9 shares: 1x president (20%) + 8x normal (10%). float_percent = 60%.

### Private Company Data (6 companies)

```rust
pub fn companies() -> Vec<CompanyDef> {
    vec![
        CompanyDef { sym: "SV", name: "Schuylkill Valley",      value: 20,  revenue: 5,  blocked_hexes: &["G15"] },
        CompanyDef { sym: "CS", name: "Champlain & St.Lawrence", value: 40,  revenue: 10, blocked_hexes: &["B20"] },
        CompanyDef { sym: "DH", name: "Delaware & Hudson",       value: 70,  revenue: 15, blocked_hexes: &["F16"] },
        CompanyDef { sym: "MH", name: "Mohawk & Hudson",         value: 110, revenue: 20, blocked_hexes: &["D18"] },
        CompanyDef { sym: "CA", name: "Camden & Amboy",          value: 160, revenue: 25, blocked_hexes: &["H18"] },
        CompanyDef { sym: "BO", name: "Baltimore & Ohio",        value: 220, revenue: 30, blocked_hexes: &["I13", "I15"] },
    ]
}
```

### Train Data (6 types, 40 total)

```rust
pub fn trains() -> Vec<TrainDef> {
    vec![
        TrainDef { name: "2", distance: 2,   price: 80,   count: 6,  rusts_on: Some("4") },
        TrainDef { name: "3", distance: 3,   price: 180,  count: 5,  rusts_on: Some("6") },
        TrainDef { name: "4", distance: 4,   price: 300,  count: 4,  rusts_on: Some("D") },
        TrainDef { name: "5", distance: 5,   price: 450,  count: 3,  rusts_on: None },
        TrainDef { name: "6", distance: 6,   price: 630,  count: 2,  rusts_on: None },
        TrainDef { name: "D", distance: 999, price: 1100, count: 20, rusts_on: None },
    ]
}
```

### Phase Data (6 phases)

```rust
pub fn phases() -> Vec<PhaseDef> {
    vec![
        PhaseDef { name: "2", train_limit: 4, tiles: &["yellow"],                    operating_rounds: 1 },
        PhaseDef { name: "3", train_limit: 4, tiles: &["yellow", "green"],            operating_rounds: 2 },
        PhaseDef { name: "4", train_limit: 3, tiles: &["yellow", "green"],            operating_rounds: 2 },
        PhaseDef { name: "5", train_limit: 2, tiles: &["yellow", "green", "brown"],   operating_rounds: 3 },
        PhaseDef { name: "6", train_limit: 2, tiles: &["yellow", "green", "brown"],   operating_rounds: 3 },
        PhaseDef { name: "D", train_limit: 2, tiles: &["yellow", "green", "brown"],   operating_rounds: 3 },
    ]
}
```

### Starting Cash and Cert Limits

```rust
pub fn starting_cash(num_players: u8) -> i32 {
    match num_players {
        2 => 1200, 3 => 800, 4 => 600, 5 => 480, 6 => 400,
        _ => panic!("Invalid player count: {}", num_players),
    }
}

pub fn cert_limit(num_players: u8) -> u8 {
    match num_players {
        2 => 28, 3 => 20, 4 => 16, 5 => 13, 6 => 11,
        _ => panic!("Invalid player count: {}", num_players),
    }
}

pub const BANK_CASH: i32 = 12000;
```

### Stock Market Grid (11 rows)

```rust
/// Returns the stock market grid. Each inner vec is a row (left to right).
/// None = empty cell (below-market dead zone).
pub fn market_grid() -> Vec<Vec<Option<MarketCell>>> {
    // Row 0: 60y 67 71 76 82 90 100p 112 126 142 160 180 200 225 250 275 300 325 350
    // Row 1: 53y 60y 66 70 76 82 90p 100 112 126 142 160 180 200 220 240 260 280 300
    // Row 2: 46y 55y 60y 65 70 76 82p 90 100 111 125 140 155 170 185 200
    // Row 3: 39o 48y 54y 60y 66 71 76p 82 90 100 110 120 130
    // Row 4: 32o 41o 48y 55y 62 67 71p 76 82 90 100
    // Row 5: 25b 34o 42o 50y 58y 65 67p 71 75 80
    // Row 6: 18b 27b 36o 45o 54y 63 67 69 70
    // Row 7: 10b 20b 30b 40o 50y 60y 67 68
    // Row 8: _ 10b 20b 30b 40o 50y 60y
    // Row 9: _ _ 10b 20b 30b 40o 50y
    // Row 10: _ _ _ 10b 20b 30b 40o
    //
    // Suffixes: y=Yellow, o=Orange, b=Brown, p=Par, (none)=Normal
    // Use a helper: mc(price, zone) -> Some(MarketCell { price, zone })
    // Empty cells: None
    ...
}
```

Parse the raw Python MARKET array (shown in Appendix A below) to build this.

### Hex Grid Definitions

The hex grid uses **pointy-top** layout. Coordinates are letter+number (e.g. "H12").

**Coordinate → (x, y) mapping:**
- Letter part → x: A=0, B=1, C=2, ..., H=7, I=8, J=9, K=10
- Number part → y: the number minus 1 (so "12" → y=11)

**Adjacency (direction → delta):**
```rust
// Pointy-top hex directions:
// Direction 0: (-1, +1)  upper-right
// Direction 1: (-2,  0)  right
// Direction 2: (-1, -1)  lower-right
// Direction 3: (+1, -1)  lower-left
// Direction 4: (+2,  0)  left
// Direction 5: (+1, +1)  upper-left
// Inverse: (dir + 3) % 6
```

Return all hex definitions from a function. See Appendix B below for the complete hex list extracted from the Python HEXES dict.

### Tile Catalog (counts only for Phase 2)

For Phase 2, we only need tile counts (for the "unplaced tiles" encoder feature). Full tile definitions with edges/connectivity are Phase 3/4.

```rust
pub fn tile_counts() -> Vec<(&'static str, u32)> {
    vec![
        ("1",1), ("2",1), ("3",2), ("4",2), ("7",4), ("8",8), ("9",7),
        ("14",3), ("15",2), ("16",1), ("18",1), ("19",1), ("20",1),
        ("23",3), ("24",3), ("25",1), ("26",1), ("27",1), ("28",1), ("29",1),
        ("39",1), ("40",1), ("41",2), ("42",2), ("43",2), ("44",1), ("45",2), ("46",2),
        ("47",1), ("53",2), ("54",1), ("55",1), ("56",1), ("57",4), ("58",2),
        ("59",2), ("61",2), ("62",1), ("63",3), ("64",1), ("65",1), ("66",1),
        ("67",1), ("68",1), ("69",1), ("70",1),
    ]
}
```

---

## Step 2: Game Construction (`src/game.rs`)

### BaseGame struct

```rust
use std::sync::Arc;
use std::collections::HashMap;

#[pyclass]
pub struct BaseGame {
    // Mutable game state (cloned per search)
    pub(crate) players: Vec<Player>,
    pub(crate) corporations: Vec<Corporation>,
    pub(crate) companies: Vec<Company>,
    pub(crate) bank: Bank,
    pub(crate) depot: Depot,
    pub(crate) phase: Phase,
    pub(crate) round_state: RoundState,
    pub(crate) hexes: Vec<Hex>,
    pub(crate) tile_counts_remaining: HashMap<String, u32>,  // unplaced tile counts

    // Immutable static data (shared via Arc for fast clone)
    pub(crate) hex_adjacency: Arc<HashMap<String, HashMap<u8, String>>>,  // hex_id -> {dir -> neighbor_id}
    pub(crate) starting_cash: i32,
    pub(crate) cert_limit: u8,

    // Lookup caches (rebuilt from vecs)
    pub(crate) corp_idx: HashMap<String, usize>,
    pub(crate) company_idx: HashMap<String, usize>,
    pub(crate) hex_idx: HashMap<String, usize>,

    // Metadata
    pub(crate) title: String,
    pub(crate) finished: bool,
    pub(crate) move_number: usize,
}
```

### Constructor: `BaseGame::new(player_names: HashMap<u32, String>)`

1. Determine `num_players` and look up `starting_cash`, `cert_limit`
2. Create `Player` objects with cash = `starting_cash`
3. Create `Bank` with cash = 12000
4. Create `Company` objects from `g1830::companies()`
5. Create `Corporation` objects from `g1830::corporations()`:
   - For each corp: create tokens from token_prices, create 9 shares (1x20% president + 8x10%)
6. Create `Depot` with trains from `g1830::trains()`:
   - For each train def: create `count` Train objects
7. Create `Hex` objects from `g1830::hex_definitions()`:
   - Each hex gets an initial `Tile` based on its HexType (blank tile, city tile, etc.)
   - Preprinted tiles (gray, yellow, red) have revenue, cities, towns set
   - White blank hexes get empty tiles
8. Compute hex adjacency from coordinates using the pointy-top direction deltas
9. Set `phase` to the first phase ("2")
10. Set `round_state` to `RoundState { round_type: Auction, round_num: 0, active_entity_id: player_1 }`
11. Build lookup caches: `corp_idx`, `company_idx`, `hex_idx`
12. Initialize `tile_counts_remaining` from `g1830::tile_counts()`

### Round State (minimal for Phase 2)

```rust
#[derive(Clone)]
pub enum RoundType { Auction, Stock, Operating }

#[pyclass]
#[derive(Clone)]
pub struct RoundState {
    #[pyo3(get)]
    pub round_type: String,  // "Auction", "Stock", "Operating"
    #[pyo3(get)]
    pub round_num: u8,
    pub active_entity_id: EntityId,
}
```

The encoder reads `game.round` and checks `isinstance(game.round, Operating)` etc. For Phase 2, we expose a stub `round` object with these basic fields. The encoder will need minor adaptation to work with the Rust round representation — or we can make the Rust RoundState duck-type compatible.

---

## Step 3: PyO3 Getters and Methods

```rust
#[pymethods]
impl BaseGame {
    #[new]
    fn new(player_names: HashMap<u32, String>) -> Self { ... }

    #[getter] fn players(&self) -> Vec<Player> { self.players.clone() }
    #[getter] fn corporations(&self) -> Vec<Corporation> { self.corporations.clone() }
    #[getter] fn companies(&self) -> Vec<Company> { self.companies.clone() }
    #[getter] fn bank(&self) -> Bank { self.bank.clone() }
    #[getter] fn depot(&self) -> Depot { self.depot.clone() }
    #[getter] fn hexes(&self) -> Vec<Hex> { self.hexes.clone() }
    #[getter] fn phase(&self) -> Phase { self.phase.clone() }
    #[getter] fn finished(&self) -> bool { self.finished }
    #[getter] fn title(&self) -> String { self.title.clone() }
    #[getter] fn move_number(&self) -> usize { self.move_number }

    /// Unplaced tiles as a list (for encoder's depot_tiles feature)
    #[getter]
    fn tiles(&self) -> Vec<Tile> {
        // Return stub Tile objects representing unplaced tile counts
        // The encoder iterates game.tiles and counts by ID
        ...
    }

    fn corporation_by_id(&self, sym: &str) -> Option<Corporation> {
        self.corp_idx.get(sym).map(|&i| self.corporations[i].clone())
    }

    fn company_by_id(&self, sym: &str) -> Option<Company> {
        self.company_idx.get(sym).map(|&i| self.companies[i].clone())
    }

    fn hex_by_id(&self, coord: &str) -> Option<Hex> {
        self.hex_idx.get(coord).map(|&i| self.hexes[i].clone())
    }

    fn active_players(&self) -> Vec<Player> {
        // For Phase 2: return the player matching round_state.active_entity_id
        ...
    }

    fn priority_deal_player(&self) -> Player {
        // Initially: first player
        self.players[0].clone()
    }

    fn num_certs(&self, player_id: u32) -> u32 {
        // Count shares + companies owned by this player
        // For initial state: 0
        0
    }

    fn pickle_clone(&self) -> BaseGame {
        self.clone_for_search()
    }
}
```

---

## Step 4: Fast Clone

```rust
impl BaseGame {
    pub fn clone_for_search(&self) -> BaseGame {
        BaseGame {
            players: self.players.clone(),
            corporations: self.corporations.clone(),
            companies: self.companies.clone(),
            bank: self.bank.clone(),
            depot: self.depot.clone(),
            phase: self.phase.clone(),
            round_state: self.round_state.clone(),
            hexes: self.hexes.clone(),
            tile_counts_remaining: self.tile_counts_remaining.clone(),

            // Shared via Arc (no deep copy)
            hex_adjacency: Arc::clone(&self.hex_adjacency),
            starting_cash: self.starting_cash,
            cert_limit: self.cert_limit,

            // Rebuild caches (cheap)
            corp_idx: self.corp_idx.clone(),
            company_idx: self.company_idx.clone(),
            hex_idx: self.hex_idx.clone(),

            title: self.title.clone(),
            finished: false,
            move_number: 0,
        }
    }
}
```

---

## Validation

### Test 1: Construction
```python
from engine_rs import BaseGame
g = BaseGame({1: "P1", 2: "P2", 3: "P3", 4: "P4"})
assert len(g.players) == 4
assert g.players[0].cash == 600  # 4-player starting cash
assert g.bank.cash == 12000
assert len(g.corporations) == 8
assert len(g.companies) == 6
assert g.phase.name == "2"
assert g.phase._train_limit == 4
assert len(g.hexes) > 80
corp = g.corporation_by_id("PRR")
assert corp is not None
assert corp.sym == "PRR"
assert len(corp.tokens) == 4
assert len(corp.shares) == 9
```

### Test 2: Clone speed
```python
import time
N = 10000
start = time.time()
for _ in range(N):
    c = g.pickle_clone()
elapsed = time.time() - start
print(f"{N} clones in {elapsed:.3f}s ({N/elapsed:.0f}/sec)")
# Target: >100K clones/sec (Python does ~1-5K/sec)
```

### Test 3: All existing Python tests still pass
```bash
uv run pytest tests/agent/alphazero/ -x  # 42 tests
```

---

## Appendix A: Raw Stock Market Data

Python source (`g1830.py` lines 447-530):
```python
MARKET = [
    ["60y","67","71","76","82","90","100p","112","126","142","160","180","200","225","250","275","300","325","350"],
    ["53y","60y","66","70","76","82","90p","100","112","126","142","160","180","200","220","240","260","280","300"],
    ["46y","55y","60y","65","70","76","82p","90","100","111","125","140","155","170","185","200"],
    ["39o","48y","54y","60y","66","71","76p","82","90","100","110","120","130"],
    ["32o","41o","48y","55y","62","67","71p","76","82","90","100"],
    ["25b","34o","42o","50y","58y","65","67p","71","75","80"],
    ["18b","27b","36o","45o","54y","63","67","69","70"],
    ["10b","20b","30b","40o","50y","60y","67","68"],
    ["","10b","20b","30b","40o","50y","60y"],
    ["","","10b","20b","30b","40o","50y"],
    ["","","","10b","20b","30b","40o"],
]
```

Parse: strip suffix to get price (`int`), suffix determines zone:
- `p` → Par (also treated as normal zone), `y` → Yellow, `o` → Orange, `b` → Brown, no suffix → Normal, `""` → None (empty cell)

The par prices (cells with `p` suffix) are: 100, 90, 82, 76, 71, 67.

## Appendix B: Complete Hex List

Extracted from `g1830.py` HEXES dict. Format: `coord: type, details`

**Red hexes (offboard):**
- F2: offboard, yellow=40, brown=70
- I1: offboard, yellow=30, brown=60 (hidden, group=Gulf)
- J2: offboard, yellow=30, brown=60
- A9: offboard, yellow=30, brown=50 (hidden, group=Canada)
- A11: offboard, yellow=30, brown=50 (group=Canada)
- K13: offboard, yellow=30, brown=40
- B24: offboard, yellow=20, brown=30

**Gray hexes (preprinted, not upgradable):**
- D2: city, revenue=20
- F6: city, revenue=30
- E9: path only (no city/town)
- H12: city, revenue=10
- D14: city, revenue=20
- C15: town, revenue=10
- K15: city, revenue=20
- A17: path only
- A19: city, revenue=40
- I19: town, revenue=10
- F24: town, revenue=10
- D24: path only

**Yellow hexes (preprinted upgradable):**
- E5: double city (OO), water cost=80
- D10: double city (OO), water cost=80
- E11: double city (OO)
- H18: double city (OO)
- I15: city, revenue=30, label=B
- G19: double city, revenue=40 each, label=NY, water cost=80
- E23: city, revenue=30, label=B

**White hexes (empty, available for tile placement):**

Cities (revenue=0):
- F4, J14, F22: city, water cost=80
- B16: city (impassable border edge 5)
- E19, H4, B10, H10, H16: city
- F16: city, mountain cost=120

Towns (revenue=0):
- E7: town (impassable border edge 5)
- B20, D4, F10: town
- G7, G17, F20: double town

Blank (no city/town):
- I13, D18, B12, B14, B22, C7, C9, C23, D8, D16, D20, E3, E13, E15, F12, F14, F18, G3, G5, G9, G11, H2, H6, H8, H14, I3, I5, I7, I9, J4, J6, J8

Mountains (cost=120):
- G15, C21, D22, E17, E21, G13, I11, J10, J12
- C17: mountain + impassable border edge 2

Water (cost=80):
- D6, I17, B18, C19

Special borders (impassable):
- F8: border edge 2
- C11: border edge 5
- C13: border edge 0
- D12: borders edges 2, 3

## Appendix C: Location Names

```rust
pub fn location_names() -> HashMap<&'static str, &'static str> {
    [
        ("D2", "Lansing"), ("F2", "Chicago"), ("J2", "Gulf"), ("F4", "Toledo"),
        ("J14", "Washington"), ("F22", "Providence"), ("E5", "Detroit & Windsor"),
        ("D10", "Hamilton & Toronto"), ("F6", "Cleveland"), ("E7", "London"),
        ("A11", "Canadian West"), ("K13", "Deep South"), ("E11", "Dunkirk & Buffalo"),
        ("H12", "Altoona"), ("D14", "Rochester"), ("C15", "Kingston"),
        ("I15", "Baltimore"), ("K15", "Richmond"), ("B16", "Ottawa"),
        ("F16", "Scranton"), ("H18", "Philadelphia & Trenton"), ("A19", "Montreal"),
        ("E19", "Albany"), ("G19", "New York & Newark"), ("I19", "Atlantic City"),
        ("F24", "Mansfield"), ("B20", "Burlington"), ("E23", "Boston"),
        ("B24", "Maritime Provinces"), ("D4", "Flint"), ("F10", "Erie"),
        ("G7", "Akron & Canton"), ("G17", "Reading & Allentown"),
        ("F20", "New Haven & Hartford"), ("H4", "Columbus"), ("B10", "Barrie"),
        ("H10", "Pittsburgh"), ("H16", "Lancaster"),
    ].iter().copied().collect()
}
```
