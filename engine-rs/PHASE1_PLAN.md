# Phase 1: Core Data Structures

## Goal
Implement all entity types in Rust with PyO3 getters that the encoder can read.
No game logic yet — just data containers with the right shape.

## Strategy
Work backwards from the encoder: implement exactly the structs/fields it accesses.
Don't port the entire Python class hierarchy — only what the ML code reads.

## Encoder reads these on each entity type:

### Player
- `id: int`
- `name: str`
- `cash: int`
- `companies: List[Company]` (private companies owned)
- `shares_by_corporation: Dict[str, List[Share]]` (used via ShareHolder mixin)
- `percent_of(corp) -> float` (share percentage)
- `num_shares_of(corp) -> int`

### Corporation
- `id: str` (the symbol, e.g. "PRR")
- `sym: str` (same as id)
- `floated() -> bool`
- `cash: int`
- `trains: List[Train]`
- `tokens: List[Token]`
- `share_price: SharePrice` (current market price)
- `ipo_price: SharePrice` (IPO price, may differ)
- `shares: List[Share]` (all shares including IPO)
- `owner: Player` (president)

### Company (private)
- `id: str` (symbol)
- `sym: str`
- `value: int`
- `revenue: int`
- `owner: Option<Entity>` (player or corp)
- `closed: bool`

### Bank
- `cash: int`

### Depot
- `trains: List[Train]`
- `discarded: List[Train]`

### Train
- `name: str` (e.g. "2", "3", "D")
- `owner: Option<Entity>`
- `operated: bool`

### Token
- `city: Option<City>`
- `used: bool`
- `corporation: Corporation`
- `price: int`
- `type: str`

### SharePrice
- `price: int`
- `types: List[str]` (market zone colors)

### Share
- `corporation() -> Corporation`
- `owner: Entity`
- `percent: int`
- `president: bool`

### Phase
- `name: str`
- `operating_rounds: int`
- `_train_limit: int`
- `tiles: List[str]` (tile color names available)

### Hex
- `id: str` (coordinate like "E11")
- `tile: Tile`
- `all_neighbors: Dict[int, Hex]`

### Tile
- `id: str`
- `name: str`
- `rotation: int`
- `cities: List[City]`
- `towns: List[Town]`
- `edges: List[Edge]`
- `offboards: List[Offboard]`
- `upgrades: List[Upgrade]`
- `paths: List[Path]`

### City
- `revenue: int`
- `slots: int`
- `tokens: List[Option<Token>]`

### Town
- `revenue: int`

### Edge
- `num: int` (direction 0-5)

### Upgrade
- `cost: int`

## Files to create

### `src/core.rs`
- `Phase`, `SharePrice`, `StockMarket`

### `src/entities.rs`
- `Player`, `Corporation`, `Company`, `Bank`, `Depot`, `Train`, `Token`, `Share`, `ShareBundle`

### `src/graph.rs`
- `Hex`, `Tile`, `City`, `Town`, `Edge`, `Upgrade`, `Offboard`, `Path`

## PyO3 approach

Use `#[pyclass]` on each struct. For nested objects (e.g. `Corporation.trains`),
return Python lists of Rust pyclass objects via `#[getter]`.

For entity references (e.g. `Token.corporation`), use IDs rather than Rust
references to avoid lifetime complexity. The Python side already accesses
these by ID most of the time.

## Validation

After Phase 1, we should be able to:
1. Construct all entity types from Rust
2. Access all fields from Python
3. The structs should be `Clone` for Phase 2's fast clone

We do NOT need to:
- Load a real game (that's Phase 2)
- Process actions (that's Phase 3)
- Calculate routes (that's Phase 4)
