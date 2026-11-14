"""Per-title settings of the RL pipeline, and one factory for new games (design step S1).

A :class:`TitleSpec` says what the pipeline needs to know about a title: the player counts it
trains and evaluates at, the optional rules it imports and plays with, where its data lives, and
its eval defaults. :func:`new_game` builds an empty game of any registered title:

* **1830** delegates to :func:`start_positions.new_game` -- the Rust engine wrapped in a
  ``RustGameAdapter``, players "Player 1".."Player N" (ids 1..N), the rule against train shuffling
  on -- so 1830 games are built exactly as before;
* **1867 and the 1822 family** (1822, 1822PNW, 1822MX, 1822CA) get the raw ``engine_rs.BaseGame``
  (``BaseGame.new_titled``): new titles bypass the adapter, whose proxies are 1830's (design §5.3).
  Their legal moves come from the title's own generator (``g1867_legal_moves_json`` /
  ``g1822_legal_moves_json``), and have indices in the title's RL layout (``1867-v1``; ``1822-v1``,
  ``1822PNW-v1``, ``1822MX-v1``, ``1822CA-v1``; layout kind "generator": ``rl_legal_moves_json``,
  ``rl_legal_indices``, ``rl_decode_index``, ``rl_apply_index``, ``rl_price_range``;
  ``docs/rl_descriptors.md``). The 1830 index path (``factored_legal_indices``,
  ``apply_action_index``, the Rust MCTS) refuses them (:meth:`TitleSpec.layout_ready`,
  :func:`layout_metadata`).

Player counts follow the user's decisions (``docs/rl_bridge_design.md``, 2026-10-16 / 2026-11-13):
3-6 for every new title, 2-6 for 1830 (its 2-player game is trained since 2026-11-07).

The generic pipeline (package C, ``docs/rl_bridge_work_packages.md``) reads it: player ranges and start-file
templates in ``generic_pg`` / ``generic_play`` / ``scripts/eval_generic.py`` /
``scripts/build_generic_start_positions.py``, new games in ``generic_play.new_title_game``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence, Tuple

# 1830's optional rules that both engines implement (= ``pretraining.ENGINE_OPTIONAL_RULES``, which
# filters a recorded game's rules to these; a test keeps the two equal).
_1830_OPTIONAL_RULES = ("multiple_brown_from_ipo", "optional_6_train")

# New titles train at 3-6 players (the user's decision), within each title's own Ruby range: the
# engine accepts 1867 at 2-6 and 1822 / 1822CA at 2-7 (the pipeline trains 3-6), but 1822PNW
# (Ruby 3-5) and 1822MX (Ruby 2-5) have no 6-player game, so theirs is 3-5.
NEW_TITLE_PLAYERS = (3, 6)
MAX_SEATS = 6


@dataclass(frozen=True)
class TitleSpec:
    """What the RL pipeline needs to know about one title."""

    title: str
    #: The title family (shared rules lineage): "1830", "1867", "1822". Family pooling (design §4.2)
    #: groups titles by it.
    family: str
    min_players: int
    max_players: int
    #: Optional rules the pipeline imports human games with and may play with (a subset of what the
    #: engine implements; ``BaseGame`` raises ValueError for any other). New titles: base rules only.
    optional_rules: Tuple[str, ...] = ()
    #: Seat ids of a new game start here ("Player 1" gets this id). The 1822 family's minors have
    #: numeric syms ("1".."30"), so its seats start above them.
    first_player_id: int = 1
    #: Raw human-game JSON directories (18xx.games exports), relative to the repo root.
    human_games_dirs: Tuple[str, ...] = ()
    #: Supervised-data LMDB (None until one is converted).
    lmdb: Optional[str] = None
    #: ``convert-generic``'s default for ``finished_only``: keep only human games whose record replays to the
    #: game's end in our engine (1867: the user's decision of 2026-11-13, "just for correctness"; 1830 keeps its
    #: unfinished games, as SL checkpoint 10's data did).
    supervised_finished_only: bool = False
    #: Start files per player count (``{n}``): post-auction human positions, mid-game cuts. New
    #: titles' files are not built yet (package C).
    start_positions_template: Optional[str] = None
    midgame_positions_template: Optional[str] = None
    #: Eval defaults: the player count head-to-heads run at, and side A's seat counts at that table
    #: (4 players: 2v2 and 1v3 -- table composition changes results).
    eval_players: int = 4
    eval_a_seats: Tuple[int, ...] = (2, 1)
    #: Every player count the title trains and evaluates at.
    player_counts: Tuple[int, ...] = field(init=False)

    def __post_init__(self):
        if not 2 <= self.min_players <= self.max_players <= MAX_SEATS:
            raise ValueError(f"{self.title}: player range {self.min_players}-{self.max_players} outside 2-{MAX_SEATS}")
        if not self.min_players <= self.eval_players <= self.max_players:
            raise ValueError(f"{self.title}: eval_players {self.eval_players} outside the player range")
        object.__setattr__(self, "player_counts", tuple(range(self.min_players, self.max_players + 1)))

    def check_players(self, num_players: int) -> int:
        n = int(num_players)
        if not self.min_players <= n <= self.max_players:
            raise ValueError(
                f"{self.title}: {n} players is outside the pipeline's range {self.min_players}-{self.max_players}"
            )
        return n

    def start_positions(self, num_players: int) -> Optional[str]:
        """The post-auction start file of a ``num_players`` game (None: no template)."""
        n = self.check_players(num_players)
        return self.start_positions_template.format(n=n) if self.start_positions_template else None

    def midgame_positions(self, num_players: int) -> Optional[str]:
        """The mid-game start file of a ``num_players`` game (None: no template)."""
        n = self.check_players(num_players)
        return self.midgame_positions_template.format(n=n) if self.midgame_positions_template else None

    def player_names(self, num_players: int) -> dict:
        """``{seat id: "Player k"}`` in seating order (self-play's naming)."""
        n = self.check_players(num_players)
        return {self.first_player_id + i: f"Player {i + 1}" for i in range(n)}

    def status(self) -> dict:
        """The engine's RL readiness for the title: ``layout_ready``, ``layout_kind``, ``encoder_ready``."""
        import engine_rs

        return dict(engine_rs.rl_title_status(self.title))

    def layout_ready(self) -> bool:
        """Whether the title has an RL action layout (index space + descriptors)."""
        return bool(self.status()["layout_ready"])


def _scrape(title: str) -> str:
    return f"human_games/scrape_2026_10/{title}"


TITLE_SPECS = {
    spec.title: spec
    for spec in (
        TitleSpec(
            title="1830",
            family="1830",
            min_players=2,
            max_players=6,
            optional_rules=_1830_OPTIONAL_RULES,
            human_games_dirs=("human_games/1830",),
            lmdb="human_games/lmdb_v6k",
            start_positions_template="human_games/start_positions_1830_{n}p_v2.jsonl",
            midgame_positions_template="human_games/midgame_positions_1830_{n}p_v2.jsonl",
        ),
        TitleSpec(
            title="1867",
            family="1867",
            min_players=NEW_TITLE_PLAYERS[0],
            max_players=NEW_TITLE_PLAYERS[1],
            human_games_dirs=("human_games/1867", _scrape("1867")),
            start_positions_template="human_games/start_positions_1867_{n}p.jsonl",
            midgame_positions_template="human_games/midgame_positions_1867_{n}p.jsonl",
            supervised_finished_only=True,
        ),
        TitleSpec(
            title="1822",
            family="1822",
            min_players=NEW_TITLE_PLAYERS[0],
            max_players=NEW_TITLE_PLAYERS[1],
            first_player_id=1001,
            human_games_dirs=("human_games/1822", _scrape("1822")),
            start_positions_template="human_games/start_positions_1822_{n}p.jsonl",
            midgame_positions_template="human_games/midgame_positions_1822_{n}p.jsonl",
        ),
        TitleSpec(
            title="1822PNW",
            family="1822",
            min_players=NEW_TITLE_PLAYERS[0],
            max_players=5,  # Ruby's range ends at 5
            first_player_id=1001,
            human_games_dirs=(_scrape("1822PNW"),),
            start_positions_template="human_games/start_positions_1822PNW_{n}p.jsonl",
            midgame_positions_template="human_games/midgame_positions_1822PNW_{n}p.jsonl",
        ),
        TitleSpec(
            title="1822MX",
            family="1822",
            min_players=NEW_TITLE_PLAYERS[0],
            max_players=5,  # Ruby's range ends at 5
            first_player_id=1001,
            human_games_dirs=(_scrape("1822MX"),),
            start_positions_template="human_games/start_positions_1822MX_{n}p.jsonl",
            midgame_positions_template="human_games/midgame_positions_1822MX_{n}p.jsonl",
        ),
        TitleSpec(
            title="1822CA",
            family="1822",
            min_players=NEW_TITLE_PLAYERS[0],
            max_players=NEW_TITLE_PLAYERS[1],
            first_player_id=1001,
            human_games_dirs=(_scrape("1822CA"),),
            start_positions_template="human_games/start_positions_1822CA_{n}p.jsonl",
            midgame_positions_template="human_games/midgame_positions_1822CA_{n}p.jsonl",
        ),
    )
}


@lru_cache(maxsize=None)
def _layout_metadata(title: str) -> Optional[Tuple[Tuple[str, str], ...]]:
    import engine_rs

    status = engine_rs.rl_title_status(title)
    if not status["layout_ready"]:
        return None
    from rl18xx.agent.alphazero.action_descriptors import action_descriptors

    d = action_descriptors(title)
    return (
        ("title", title),
        ("layout_kind", str(status["layout_kind"])),
        ("layout_version", d.version),
        ("layout_fingerprint", d.fingerprint),
    )


def layout_metadata(title: str) -> Optional[dict]:
    """The RL action layout a title's games, data and checkpoints are indexed with:
    ``{"title", "layout_kind", "layout_version", "layout_fingerprint"}`` (None for a title without a
    layout). Design step S1: written into records so a reader can tell which index space they use, not
    yet enforced by any reader."""
    title_spec(title)  # ValueError for an unknown title
    meta = _layout_metadata(str(title))
    return dict(meta) if meta is not None else None


def titles() -> Tuple[str, ...]:
    """Every title the pipeline knows, 1830 first."""
    return tuple(TITLE_SPECS)


def title_spec(title: str) -> TitleSpec:
    """The spec of ``title``; ValueError for a title the pipeline does not know."""
    try:
        return TITLE_SPECS[str(title)]
    except KeyError:
        raise ValueError(f"unknown title {title!r}; known: {', '.join(TITLE_SPECS)}") from None


def new_game(
    title: str,
    num_players: int,
    *,
    optional_rules: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    auction_unlock: bool = False,
    no_buyback_rule: Optional[bool] = None,
):
    """An empty game of ``title`` with ``num_players`` players named "Player 1".."Player N".

    1830: :func:`start_positions.new_game` (a ``RustGameAdapter``; the rule against train shuffling
    on unless ``no_buyback_rule`` is False; ``auction_unlock`` as given), with ``optional_rules`` (a
    subset of :attr:`TitleSpec.optional_rules`) when any are asked for. Other titles: the raw
    ``engine_rs.BaseGame`` from ``BaseGame.new_titled`` (``seed`` reaches the engine's RNG; the
    1830-only ``auction_unlock`` / ``no_buyback_rule`` raise ValueError if set).
    """
    spec = title_spec(title)
    n = spec.check_players(num_players)
    rules = tuple(optional_rules or ())
    unknown = [r for r in rules if r not in spec.optional_rules]
    if unknown:
        raise ValueError(f"{spec.title}: optional rules {unknown} are not used by the pipeline ({spec.optional_rules})")

    if spec.title == "1830":
        from rl18xx.agent.alphazero.start_positions import new_game as new_1830_game

        buyback = True if no_buyback_rule is None else bool(no_buyback_rule)
        if not rules:
            return new_1830_game(n, auction_unlock=auction_unlock, no_buyback_rule=buyback)
        # Same construction as start_positions.new_game, with the rules.
        from engine_rs import BaseGame as RustGame

        from rl18xx.rust_adapter import RustGameAdapter

        game = RustGame(spec.player_names(n), list(rules))
        game.set_auction_unlock(auction_unlock)
        game.set_no_buyback_rule(buyback)
        return RustGameAdapter(game)

    if auction_unlock or no_buyback_rule:
        raise ValueError(f"{spec.title}: auction_unlock / no_buyback_rule are 1830 rule variants")
    import engine_rs

    return engine_rs.BaseGame.new_titled(spec.title, spec.player_names(n), list(rules) or None, seed)
