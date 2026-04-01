# Rust Engine Migration Plan

## Context

The game engine (`rl18xx/game/engine/`) is a ~18K line Python port of a Ruby 1830 implementation. It's the primary bottleneck for MCTS self-play: `pickle_clone()` is called thousands of times per game during tree search, and the entire game loop (action processing, legal move generation, route calculation) runs in pure Python.

Migrating to Rust via PyO3 would give 10-100x speedup on the hot paths while keeping the ML code (encoder, model, training) in Python unchanged. The Python code accesses ~30 distinct methods/properties on the BaseGame object — this is the API surface that PyO3 bindings must expose.

---

## Architecture: Rust Core + PyO3 Bindings

```
rl18xx/
├── engine-rs/                    # New Rust crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs               # PyO3 module entry point
│   │   ├── game.rs              # BaseGame struct + impl
│   │   ├── entities.rs          # Player, Corporation, Company, Bank, Depot
│   │   ├── round.rs             # Round/Step state machine
│   │   ├── graph.rs             # Hex grid, tiles, routes
│   │   ├── actions.rs           # Action types + processing
│   │   ├── core.rs              # SharedValues, Phase, StockMarket
│   │   ├── abilities.rs         # Company abilities
│   │   ├── autorouter.rs        # Route calculation
│   │   └── title/
│   │       └── g1830.rs         # 1830-specific data + rules
│   └── tests/
├── game/engine/                  # Existing Python (kept for reference, then removed)
└── agent/alphazero/              # Unchanged — talks to Rust via PyO3
```

**Build integration:** Use `maturin` (via `pyproject.toml`) to build the Rust extension. The Python package imports `from rl18xx.engine_rs import BaseGame` with the same API as before.

---

## Phased Implementation

### Phase 0: Scaffolding (1 week)

**Goal:** Rust crate builds, PyO3 bindings compile, Python can import a stub.

**Tasks:**
1. Create `engine-rs/Cargo.toml` with dependencies: `pyo3`, `serde`, `serde_json`
2. Create `src/lib.rs` with `#[pymodule]` entry point
3. Create stub `BaseGame` pyclass that constructs and returns a dummy
4. Add `maturin` to `pyproject.toml` build system
5. Verify `import rl18xx.engine_rs` works from Python

**Verification:** `uv run python -c "from rl18xx.engine_rs import BaseGame; g = BaseGame()"`

---

### Phase 1: Core Data Structures (2-3 weeks)

**Goal:** All entity types exist in Rust with serde serialization. No game logic yet.

**Files:** `core.rs`, `entities.rs`, `graph.rs`

**Key structs to implement:**

```rust
// core.rs
pub struct Phase { name: String, operating_rounds: u8, train_limit: u8, tiles: Vec<String> }
pub struct SharePrice { price: u32, row: u8, column: u8, types: Vec<String> }
pub struct StockMarket { grid: Vec<Vec<Option<SharePrice>>> }

// entities.rs
pub struct Player { id: u32, name: String, cash: i32, shares: Vec<Share>, companies: Vec<CompanyId> }
pub struct Corporation { id: String, floated: bool, cash: i32, trains: Vec<Train>, tokens: Vec<Token>, share_price: Option<SharePrice> }
pub struct Company { id: String, sym: String, value: u32, revenue: u32, owner: Option<EntityId>, closed: bool }
pub struct Bank { cash: i32 }
pub struct Depot { trains: Vec<Train>, discarded: Vec<Train> }
pub struct Train { name: String, distance: u32, price: u32, owner: Option<EntityId> }

// graph.rs
pub struct Hex { id: String, tile: Tile, neighbors: HashMap<u8, HexId>, all_neighbors: HashMap<u8, HexId> }
pub struct Tile { id: String, name: String, rotation: u8, cities: Vec<City>, towns: Vec<Town>, edges: Vec<Edge>, upgrades: Vec<Upgrade> }
pub struct City { revenue: u32, slots: u8, tokens: Vec<Option<CorporationId>> }
```

**PyO3 bindings:** Expose read-only getters for all attributes the encoder/action_helper access. Use `#[getter]` macros.

**Validation:** Port `encoder_gnn_test.py` assertions — create a game in Rust, encode it in Python, verify same tensor values.

---

### Phase 2: Game State + Clone (3-4 weeks)

**Goal:** `BaseGame` constructs a full 1830 game, `clone()` works (replacing `pickle_clone()`), read-only state queries work.

**Files:** `game.rs`, `title/g1830.rs`

**Critical implementation: fast clone**

The current `pickle_clone()` serializes the entire game state with pickle, strips logs/graph/tile-catalog, deserializes, then rebuilds the stripped data. In Rust:

```rust
impl BaseGame {
    pub fn clone_for_search(&self) -> BaseGame {
        // Deep copy game state (entities, round, phase)
        // Share static data (tile catalog, hex grid topology) via Arc
        // Rebuild graph on demand (lazy)
        BaseGame {
            players: self.players.clone(),
            corporations: self.corporations.clone(),
            round: self.round.clone(),
            phase: self.phase.clone(),
            // Static data shared, not cloned:
            tile_catalog: Arc::clone(&self.tile_catalog),
            hex_topology: Arc::clone(&self.hex_topology),
            // Skipped (rebuilt on demand):
            graph: None,
            log: Vec::new(),
            actions: Vec::new(),
        }
    }
}
```

This should be 50-100x faster than pickle because:
- No serialization/deserialization
- Static data (tiles, hex topology) is shared via `Arc`, not copied
- Graph is lazily rebuilt only when needed

**PyO3 binding:**
```python
# Python sees same API:
clone = game.pickle_clone()  # Actually calls Rust clone_for_search()
```

**Key queries to implement:**
- `corporation_by_id(sym)`, `company_by_id(sym)`, `hex_by_id(coord)` — HashMap lookups
- `active_players()`, `priority_deal_player()`, `num_certs(player)` — derived from state
- `buying_power(entity)`, `can_par(corp, entity)` — game rule queries
- `result()` — final scores
- `finished` — game end detection

**Validation:** Load a human game JSON in both Python and Rust engines, verify identical state at each step.

---

### Phase 3: Action Processing + Round Logic (4-6 weeks)

**Goal:** `process_action()` works. Full game can be played from start to finish.

**Files:** `actions.rs`, `round.rs`, `abilities.rs`

This is the largest and most complex phase. The round system is a state machine:
```
Auction → Stock → Operating → Stock → Operating → ... → End
```

Each round contains steps. Each step defines legal actions and how to process them.

**Action types to implement (16 total):**
| Action | Complexity | Lines in Python |
|--------|-----------|-----------------|
| Pass | Low | ~20 |
| Bid | Medium | ~80 |
| Par | Medium | ~100 |
| BuyShares | High | ~200 |
| SellShares | High | ~200 |
| LayTile | High | ~300 (tile validation, connectivity) |
| PlaceToken | Medium | ~150 |
| RunRoutes | Very High | ~400 (autorouter, revenue calculation) |
| BuyTrain | High | ~300 (forced buy, president contribution) |
| Dividend | Medium | ~100 |
| BuyCompany | Medium | ~100 |
| Bankrupt | High | ~200 |
| DiscardTrain | Low | ~30 |
| CompanyLayTile | Medium | ~100 |
| CompanyPlaceToken | Low | ~50 |
| CompanyBuyShares | Medium | ~100 |

**Approach:** Port round.py (5,763 lines) action-by-action. Test each action type independently against the Python engine.

**Validation:** Replay all 250 human games through both engines, compare state at every step.

---

### Phase 4: Route Calculation (2-3 weeks)

**Goal:** `autorouter.rs` + graph route finding works.

**Files:** `autorouter.rs`, graph route methods in `graph.rs`

Route calculation is the most algorithmically complex part:
- Find all valid routes for a corporation's trains
- Maximize total revenue across all routes
- Routes cannot share track segments
- Different train types reach different distances

This is also a performance hotspot — route calculation happens during every operating round for every corporation.

**Approach:** Port `autorouter.py` (345 lines) and the route-finding methods in `graph.py`.

---

### Phase 5: Integration + Cutover (2-3 weeks)

**Goal:** Python ML code uses Rust engine exclusively. Python engine removed.

**Tasks:**
1. Update `encoder.py` imports: `from rl18xx.engine_rs import BaseGame`
2. Update `action_helper.py` imports
3. Update `self_play.py` imports
4. Update `action_mapper.py` imports
5. Run full test suite against Rust engine
6. Benchmark: measure self-play games/hour improvement
7. Remove Python engine directory

**Validation:**
- All 42 existing tests pass
- Autoresearch experiments produce same metrics (±0.01)
- Self-play game generation ≥10x faster

---

### Phase 6: Optimization (1-2 weeks, ongoing)

**Goal:** Squeeze maximum performance from Rust engine.

**Opportunities:**
- Arena allocator for game clones (avoid heap allocation per clone)
- Incremental graph updates instead of full rebuild
- SIMD for route revenue calculation
- Parallel MCTS with Rust threads (bypass Python GIL)
- Move action_helper into Rust (eliminate Python↔Rust boundary per legal move query)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Ruby→Python port has subtle bugs | Validate against human game corpus at every phase |
| Rust type system friction with dynamic Python patterns | Use enums for entity types, trait objects sparingly |
| PyO3 overhead on frequent small calls | Batch operations where possible (e.g., `encode()` calls all getters in one Rust function) |
| Game rule edge cases | Port Python tests to Rust, run both engines in parallel during development |
| Long development time | Phased approach — each phase delivers value independently |

## Effort Estimate

| Phase | Duration | Cumulative |
|-------|----------|-----------|
| 0: Scaffolding | 1 week | 1 week |
| 1: Data Structures | 2-3 weeks | 3-4 weeks |
| 2: Game State + Clone | 3-4 weeks | 6-8 weeks |
| 3: Action Processing | 4-6 weeks | 10-14 weeks |
| 4: Route Calculation | 2-3 weeks | 12-17 weeks |
| 5: Integration | 2-3 weeks | 14-20 weeks |
| 6: Optimization | 1-2 weeks | 15-22 weeks |

**Expected speedup:** 10-100x on `pickle_clone()`, 5-20x on full self-play game generation.

## Verification

Each phase has its own validation gate:
- **Phase 0:** `import rl18xx.engine_rs` works
- **Phase 1:** Encoder produces same tensors from Rust entities
- **Phase 2:** Game clone + state queries match Python
- **Phase 3:** Full game replay matches Python at every step
- **Phase 4:** Route revenues match Python
- **Phase 5:** All 42 tests pass, autoresearch metrics unchanged
- **Phase 6:** Benchmark shows target speedup
