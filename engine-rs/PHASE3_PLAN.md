# Phase 3: Action Processing + Round Logic

## Context

Phase 2 is complete: the Rust engine can construct a full 1830 game state and clone it 450x faster than Python. Phase 3 adds the ability to **process actions** — the core game loop that transforms state via player decisions.

After Phase 3, a complete 1830 game can be played from start to finish in Rust, enabling the MCTS self-play pipeline to use the Rust engine end-to-end.

## Architecture Overview

The Python engine uses a layered dispatch:
```
game.process_action(action)
  → game.round.process_action(action)
    → step.process_{action_type}(action)
      → mutates game state
    → step checks if done → round checks if done → game transitions
```

In Rust, this becomes:
```
game.process_action(action_dict)
  → parse action from dict
  → match round_type:
      Auction → auction_step.process(action)
      Stock → stock_step.process(action)
      Operating → operating_step.process(action)
  → check round/game transitions
```

---

## Key Design Decisions

### 1. Action Representation

Use a Rust enum rather than dynamic class dispatch:

```rust
pub enum Action {
    Pass { entity_id: EntityId },
    Bid { entity_id: EntityId, company_sym: String, price: i32 },
    Par { entity_id: EntityId, corporation_sym: String, share_price: i32 },
    BuyShares { entity_id: EntityId, corporation_sym: String, shares: Vec<ShareRef>, percent: u8 },
    SellShares { entity_id: EntityId, corporation_sym: String, shares: Vec<ShareRef>, percent: u8 },
    LayTile { entity_id: EntityId, hex_id: String, tile_id: String, rotation: u8 },
    PlaceToken { entity_id: EntityId, hex_id: String, city_index: u8 },
    RunRoutes { entity_id: EntityId, routes: Vec<RouteData> },
    Dividend { entity_id: EntityId, kind: DividendKind },  // Payout or Withhold
    BuyTrain { entity_id: EntityId, train_id: String, price: i32, from: EntityId },
    DiscardTrain { entity_id: EntityId, train_id: String },
    BuyCompany { entity_id: EntityId, company_sym: String, price: i32 },
    Bankrupt { entity_id: EntityId },
    // Company-specific actions
    CompanyLayTile { entity_id: EntityId, company_sym: String, hex_id: String, tile_id: String, rotation: u8 },
    CompanyPlaceToken { entity_id: EntityId, company_sym: String, hex_id: String },
    CompanyBuyShares { entity_id: EntityId, company_sym: String, corporation_sym: String },
}

pub enum DividendKind { Payout, Withhold }
```

### 2. Round as Enum + State

Rather than a class hierarchy, use a round enum with associated state:

```rust
pub enum Round {
    Auction(AuctionState),
    Stock(StockState),
    Operating(OperatingState),
}

pub struct AuctionState {
    pub entity_index: usize,
    pub companies_for_auction: Vec<String>,  // remaining private company syms
    pub current_bids: HashMap<String, Vec<(u32, i32)>>,  // company_sym -> [(player_id, bid)]
    pub min_bids: HashMap<String, i32>,  // company_sym -> minimum bid
}

pub struct StockState {
    pub entity_index: usize,
    pub players_sold: HashMap<u32, HashMap<String, bool>>,  // player -> corp -> sold_this_turn
    pub current_actions: Vec<String>,  // action history this turn
    pub bought_from_ipo: bool,
}

pub struct OperatingState {
    pub entity_index: usize,
    pub round_num: u8,           // which OR in the set (1, 2, or 3)
    pub step: OperatingStep,     // which step within the OR
    pub num_laid_track: u8,
    pub routes: Vec<RouteData>,
    pub revenue: i32,
}

pub enum OperatingStep {
    LayTile,
    PlaceToken,
    RunRoutes,
    Dividend,
    BuyTrain,
    Done,
}
```

### 3. Action Parsing from Dict

The Python encoder/self-play passes actions as dicts. Implement:

```rust
impl Action {
    pub fn from_dict(data: &HashMap<String, PyObject>, game: &BaseGame) -> Result<Action, GameError> {
        let action_type = data["type"].as_str();
        match action_type {
            "pass" => Ok(Action::Pass { ... }),
            "bid" => Ok(Action::Bid { ... }),
            "par" => Ok(Action::Par { ... }),
            "buy_shares" => Ok(Action::BuyShares { ... }),
            // ... etc
        }
    }
}
```

---

## Implementation Steps

### Step 1: Action Types (`src/actions.rs`) — NEW FILE

Define the `Action` enum and `from_dict()` / `to_dict()` parsing.

**Fields per action type (from Python source):**

| Action | Fields |
|--------|--------|
| Pass | entity |
| Bid | entity, company (Company obj), price (int) |
| Par | entity, corporation (Corp obj), share_price (SharePrice obj) |
| BuyShares | entity, shares (list of Share), swap (optional Share) |
| SellShares | entity, shares (list of Share), swap (optional Share), percent (int) |
| LayTile | entity, hex (Hex obj), tile (Tile obj), rotation (int) |
| PlaceToken | entity, city (City obj), token (Token obj) |
| RunRoutes | entity, routes (list of Route), extra_revenue (int) |
| Dividend | entity, kind ("payout" or "withhold") |
| BuyTrain | entity, train (Train obj), price (int), variant (str), exchange (Train) |
| DiscardTrain | entity, train (Train obj) |
| BuyCompany | entity, company (Company obj), price (int) |
| Bankrupt | entity |
| CompanyLayTile | same as LayTile but entity is Company's corp |
| CompanyPlaceToken | same as PlaceToken |
| CompanyBuyShares | entity, company, shares |

### Step 2: Auction Round (`src/rounds/auction.rs`) — NEW FILE

The initial WaterfallAuction:

```
While companies remain:
  Current company = first unsold private
  Each player in order:
    Can Bid (>= min_bid) or Pass
  If only one bidder: they buy at bid price
  If all pass on cheapest company: reduce price by 5, loop
  When a company is bought: next company becomes available
  When all sold or bought: auction ends → Stock Round
```

Key methods:
- `process_bid(action)` — validate price >= min_bid, record bid
- `process_pass(action)` — mark player as passed for this company
- `resolve_auction()` — when all but one pass, sell to winner
- `finished()` — all companies sold
- `legal_actions(player_id)` — [Bid, Pass] or [] if not their turn

### Step 3: Stock Round (`src/rounds/stock.rs`) — NEW FILE

Player turn structure: sell-buy-sell (1830 specific)

```
For each player (starting from priority deal):
  Can: SellShares, BuyShares, Par, BuyCompany, Pass
  Rules:
    - Can't sell and buy same corp in same turn
    - Can sell multiple bundles
    - Can buy one bundle (IPO or market)
    - Can par one corp (sets initial price, buys president share)
    - Pass ends turn
  When all players pass consecutively: Stock Round ends
```

Key state mutations:
- `process_par(action)` — set share price, move shares from IPO to player, check float
- `process_buy_shares(action)` — transfer shares, pay price, check float
- `process_sell_shares(action)` — transfer shares to market, receive money, drop share price
- `process_pass(action)` — end player's turn, advance to next player

Share price movements:
- Sell: move left one column (price drops)
- Sold out (no shares in IPO or market): move right one column (price rises)
- After round: withhold → move down

### Step 4: Operating Round (`src/rounds/operating.rs`) — NEW FILE

Each corporation operates in stock price order (highest first):

```
For each operating corporation:
  1. LayTile (optional, 1 per turn normally)
  2. PlaceToken (optional, 1 per turn)
  3. RunRoutes (mandatory if trains exist)
  4. Dividend (payout or withhold)
  5. BuyTrain (optional, or mandatory if routes exist but no trains)
  At each step: can also BuyCompany from president
```

Key state mutations:

**LayTile:**
- Validate hex is reachable by corporation's track network
- Validate tile is legal upgrade for hex
- Validate rotation produces valid connections
- Pay terrain cost (water $80, mountain $120)
- Replace hex's tile with new tile

**PlaceToken:**
- Validate city is reachable and has open slot
- Pay token cost
- Place token in city

**RunRoutes:**
- For Phase 3: accept routes from action dict (route calculation is Phase 4)
- Record revenue

**Dividend:**
- Payout: distribute revenue proportionally to shareholders, move price right
- Withhold: add revenue to corp treasury, move price left

**BuyTrain:**
- Buy from depot (fixed price) or from another corp (negotiated price)
- Check forced buy if corp has routes but no trains
- President contributes personal funds if corp can't afford
- Trigger phase change if buying certain trains (3→phase3, 4→phase4, etc.)
- Rust old trains when phase changes

### Step 5: Round Transitions (`src/game.rs`)

```rust
impl BaseGame {
    fn transition_to_next_round(&mut self) {
        match &self.round {
            Round::Auction(_) => {
                // → Stock Round
                self.round = Round::Stock(StockState::new(&self.players));
            }
            Round::Stock(_) => {
                // → Operating Round (first in set)
                let or_count = self.phase.operating_rounds;
                self.round = Round::Operating(OperatingState::new(1, &self.corporations));
            }
            Round::Operating(state) => {
                if state.round_num < self.phase.operating_rounds {
                    // → Next Operating Round in set
                    self.round = Round::Operating(OperatingState::new(
                        state.round_num + 1, &self.corporations
                    ));
                } else {
                    // → Stock Round
                    self.round = Round::Stock(StockState::new(&self.players));
                }
            }
        }
    }
}
```

### Step 6: Game End Detection

```rust
fn check_game_end(&mut self) {
    // 1830 end conditions:
    // - Bank breaks (cash <= 0) → end after current round
    // - Player bankrupts → immediate end (BANKRUPTCY_ENDS_GAME_AFTER = "one")

    if self.bank.cash <= 0 && !self.game_end_triggered {
        self.game_end_triggered = true;
        self.game_end_after = GameEndAfter::CurrentRound;
    }
}

fn end_game(&mut self) {
    self.finished = true;
    // Calculate final scores: player cash + share values
}

fn result(&self) -> HashMap<u32, i32> {
    // For each player: cash + sum(shares * current_price)
}
```

### Step 7: Phase Transitions

When certain trains are bought, the game phase advances:

```rust
fn advance_phase(&mut self, train_name: &str) {
    // Phase triggers: buying first 3-train → phase "3", etc.
    let new_phase_name = match train_name {
        "3" => Some("3"),
        "4" => Some("4"),
        "5" => Some("5"),  // Also closes all private companies
        "6" => Some("6"),
        "D" => Some("D"),
        _ => None,
    };

    if let Some(name) = new_phase_name {
        self.phase = find_phase(name);

        // Rust trains that rust on this phase
        match name {
            "4" => self.rust_trains("2"),   // All 2-trains rust
            "6" => self.rust_trains("3"),   // All 3-trains rust
            "D" => self.rust_trains("4"),   // All 4-trains rust
            _ => {}
        }

        if name == "5" {
            self.close_all_companies();
        }
    }
}
```

---

## File Structure

```
engine-rs/src/
├── lib.rs          (update: add actions, rounds modules)
├── game.rs         (update: add process_action, transitions, game end)
├── actions.rs      (NEW: Action enum, from_dict/to_dict)
├── rounds/
│   ├── mod.rs      (NEW: Round enum, shared traits)
│   ├── auction.rs  (NEW: WaterfallAuction logic)
│   ├── stock.rs    (NEW: Stock round logic)
│   └── operating.rs(NEW: Operating round logic)
├── core.rs         (update: add StockMarket movement methods)
├── entities.rs     (update: add share transfer, float detection)
├── graph.rs        (minimal changes)
└── title/
    └── g1830.rs    (update: add par prices, operating order)
```

---

## Validation

### Test 1: Replay human games

The strongest validation: replay all 250 human games through both Python and Rust engines, comparing state at every action.

```python
import json
from engine_rs import BaseGame as RustGame
from rl18xx.game.engine.game import BaseGame as PyGame

for game_file in human_games:
    data = json.load(open(game_file))
    py_game = PyGame.load(data)
    rust_game = RustGame.load(data)  # New: load from JSON

    for i, action in enumerate(data["actions"]):
        py_game.process_action(action)
        rust_game.process_action(action)

        # Compare key state
        assert rust_game.bank.cash == py_game.bank.cash, f"Bank mismatch at action {i}"
        for rc, pc in zip(rust_game.corporations, py_game.corporations):
            assert rc.cash == pc.cash, f"{rc.sym} cash mismatch at action {i}"
            assert rc.floated == pc.floated(), f"{rc.sym} float mismatch at action {i}"
```

### Test 2: Legal action generation

At each position, verify the Rust engine produces the same legal actions as Python:

```python
from rl18xx.game.action_helper import ActionHelper
helper = ActionHelper()

# At each step of a replayed game:
py_legal = set(a.to_dict()["type"] for a in helper.get_all_choices(py_game))
rust_legal = set(rust_game.legal_action_types())
assert py_legal == rust_legal
```

### Test 3: Full game completion

Play a complete game with random legal actions:

```python
rust_game = RustGame({1: "P1", 2: "P2", 3: "P3", 4: "P4"})
while not rust_game.finished:
    actions = rust_game.legal_actions()
    action = random.choice(actions)
    rust_game.process_action(action)
assert rust_game.move_number > 100  # Game completed
result = rust_game.result()
assert len(result) == 4  # All players have scores
```

---

## Effort Estimate

| Component | Effort | Lines (est.) |
|-----------|--------|-------------|
| actions.rs (enum + parsing) | 2-3 days | ~400 |
| rounds/auction.rs | 2-3 days | ~300 |
| rounds/stock.rs | 3-4 days | ~500 |
| rounds/operating.rs | 4-5 days | ~800 |
| game.rs (transitions, end) | 2-3 days | ~300 |
| core.rs (market movements) | 1-2 days | ~200 |
| entities.rs (share transfers) | 1-2 days | ~200 |
| Validation + debugging | 3-5 days | — |
| **Total** | **~4-6 weeks** | **~2,700** |

## Priority Order

Implement in dependency order:

1. **actions.rs** — everything else depends on Action type
2. **rounds/auction.rs** — simplest round, good test of the pattern
3. **entities.rs updates** — share transfers needed for stock round
4. **core.rs updates** — market movement needed for stock round
5. **rounds/stock.rs** — Par, BuyShares, SellShares
6. **rounds/operating.rs** — LayTile, Token, RunRoutes, Dividend, BuyTrain
7. **game.rs transitions** — round transitions, phase changes, game end
8. **Validation** — replay human games through both engines

The auction round is the best starting point: it's self-contained (no share transfers, no map changes), tests the basic action dispatch pattern, and once it works, the same pattern extends to Stock and Operating rounds.
