"""The bots: the model playing bot seats on several sites at once (``main.py advisor --bots``).

Each site has a connector -- the local copy of 18xx.games (:mod:`site_bots`) and 18xx.games itself
(:mod:`public_bots`) -- running on its own thread while it is switched on. The hub holds what they
share: the settings (a JSON file, changed from the admin page, :mod:`admin`), the games the bots are
in, the event log, and the step every connector takes for a bot seat's move (:meth:`BotHub.search_move`:
the advisor's search, as the extension's autoplay runs it). The advisor searches one position at a
time, so the connectors take turns with each other and with the extension.

Games of another title the hub has a policy for (``titled``: ``main.py advisor --title-policy
1867=<checkpoint>``, :mod:`titled`) are followed and played by that policy alone or, with ``--title-search
1867=<critic>``, by a search of it with that critic (in the calling connector's thread), on the local site
only; 18xx.games stays 1830-only.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from rl18xx.agent.advisor.advisor import MAX_READOUTS, SearchBusy
from rl18xx.agent.advisor.autoplay import AutoplayError, SiteView
from rl18xx.agent.advisor.live_game import LiveGame, UnsupportedGame, support_problem

LOGGER = logging.getLogger(__name__)

LOCAL = "local"
PUBLIC = "18xx.games"
DEFAULT_SETTINGS = {
    LOCAL: {
        "enabled": True,
        "url": "http://127.0.0.1:9292",
        "token_file": "~/.config/18xx/local_site_bot_token",
        "readouts": 400,
    },
    PUBLIC: {
        "enabled": False,
        "url": "https://18xx.games",
        "accounts_dir": "~/.config/18xx/bots",
        "join_file": "~/.config/18xx/bot_join",
        "readouts": 400,
        "min_interval": 2.0,  # seconds between requests: the site allows 2 a second per IP, shared
        "poll_seconds": 30.0,  # between looks at the bot accounts' game lists
    },
}
# What the admin page may change, and the range it is kept to.
EDITABLE = {
    "enabled": (bool, None),
    "readouts": (int, (1, MAX_READOUTS)),
    "min_interval": (float, (1.0, 60.0)),
    "poll_seconds": (float, (10.0, 3600.0)),
}
JOB_POLL_SECONDS = 0.2
MAX_EVENTS = 300
FINISHED_KEPT = 10  # finished games still shown, per site


class CannotMove(Exception):
    """The bots can't move here (the model can't follow the game, the site refused the move, ...)."""


@dataclass
class GameState:
    """A game the bots are in, as the admin page shows it."""

    site: str
    game_id: object
    url: str
    title: str = ""
    status: str = ""  # new / active / finished
    round: str = ""
    players: list = field(default_factory=list)  # [{"name", "bot"}]
    waiting_on: str = ""  # who must act
    bot_turn: bool = False
    last_move: Optional[dict] = None  # {"at", "actor", "description"}
    stuck: Optional[str] = None  # why the bots gave up at action ``stuck_at``
    stuck_at: Optional[int] = None
    updated: float = 0.0


class Connector:
    """One site's bots: :meth:`tick` runs on the connector's thread while it is switched on."""

    site = ""

    def __init__(self, hub: "BotHub"):
        self.hub = hub
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.state = "off"  # off / running / paused / error
        self.detail = ""
        self.last_look: Optional[float] = None

    @property
    def settings(self) -> dict:
        return self.hub.settings[self.site]

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def start(self) -> None:
        if self.running:
            return
        self._stop = threading.Event()
        self.state, self.detail = "running", ""
        self._thread = threading.Thread(target=self._run, name=f"bots-{self.site}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.state, self.detail = "off", ""

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def wait(self, seconds: float) -> None:
        self._stop.wait(max(0.0, seconds))

    def _run(self) -> None:
        self.hub.event(self.site, "info", "switched on")
        while not self._stop.is_set():
            try:
                delay = self.tick()
                if not self._stop.is_set() and self.state == "error":
                    self.state, self.detail = "running", ""
            except Exception as e:  # a bug, or the site down: say so and keep going
                if self.state != "error" or self.detail != str(e):
                    self.hub.event(self.site, "error", str(e))
                    if not isinstance(e, OSError) and not hasattr(e, "status"):  # not a site's answer
                        LOGGER.exception("bots on %s", self.site)
                self.state, self.detail = "error", str(e)
                delay = 10.0
            self.wait(delay)
        self.hub.event(self.site, "info", "switched off")

    def tick(self) -> float:
        """One look at the site, and the bots' moves there: the seconds until the next look."""
        raise NotImplementedError

    def status(self) -> dict:
        return {"state": self.state if self.running or self.state == "off" else "off", "detail": self.detail,
                "last_look": self.last_look}  # fmt: skip


class BotHub:
    def __init__(self, advisor, settings_path, connectors: Optional[dict] = None, titled: Optional[dict] = None):
        self.advisor = advisor
        # title -> titled.TitledPolicy: the titles other than 1830 the bots play (on the local site)
        self.titled = dict(titled or {})
        self.settings_path = Path(settings_path).expanduser()
        self.settings = self._load_settings()
        self.started = time.time()
        self.events: deque = deque(maxlen=MAX_EVENTS)
        self.games: dict = {}  # (site, game id) -> GameState
        self.live: dict = {}  # (site, game id) -> LiveGame
        self.lock = threading.RLock()
        if connectors is None:
            from rl18xx.agent.advisor.public_bots import PublicSite
            from rl18xx.agent.advisor.site_bots import LocalSite

            connectors = {LOCAL: LocalSite, PUBLIC: PublicSite}
        self.connectors = {site: cls(self) for site, cls in connectors.items()}

    # ---------------------------------------------------------------- settings
    def _load_settings(self) -> dict:
        settings = copy.deepcopy(DEFAULT_SETTINGS)
        if self.settings_path.exists():
            saved = json.loads(self.settings_path.read_text())
            for site, values in saved.items():
                if site in settings and isinstance(values, dict):
                    settings[site].update(values)
        return settings

    def _save_settings(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.settings_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.settings, indent=2) + "\n")
        tmp.replace(self.settings_path)

    def change(self, site: str, changes: dict) -> dict:
        """Change a site's settings from the admin page; switching it on or off starts or stops its
        connector. Unknown or out-of-range values raise ValueError."""
        if site not in self.settings:
            raise ValueError(f"no site {site!r}")
        clean = {}
        for key, value in changes.items():
            if key not in EDITABLE or key not in self.settings[site]:
                raise ValueError(f"{key!r} can't be changed")
            kind, bounds = EDITABLE[key]
            if kind is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be true or false")
            else:
                value = kind(value)
                if not bounds[0] <= value <= bounds[1]:
                    raise ValueError(f"{key} must be within {bounds[0]}-{bounds[1]}")
            clean[key] = value
        with self.lock:
            self.settings[site].update(clean)
            self._save_settings()
        if clean:
            self.event(site, "info", "settings: " + ", ".join(f"{k} = {v}" for k, v in clean.items()))
        if "enabled" in clean and site in self.connectors:
            if clean["enabled"]:
                self.connectors[site].start()
            else:
                self.connectors[site].stop()
        return self.settings[site]

    def start(self) -> None:
        for site, connector in self.connectors.items():
            if self.settings[site]["enabled"]:
                connector.start()

    def stop(self) -> None:
        for connector in self.connectors.values():
            connector.stop()

    # ------------------------------------------------------------- the record
    def event(self, site: str, level: str, message: str, game_id=None) -> None:
        log = {"error": LOGGER.error, "warning": LOGGER.warning}.get(level, LOGGER.info)
        log("%s%s: %s", site, f" game {game_id}" if game_id is not None else "", message)
        with self.lock:
            self.events.append({"at": time.time(), "site": site, "level": level, "game": game_id, "message": message})

    def game(self, site: str, game_id, url: str = "") -> GameState:
        with self.lock:
            state = self.games.get((site, game_id))
            if state is None:
                state = self.games[(site, game_id)] = GameState(site, game_id, url)
            return state

    def update_game(self, site: str, game_id, **fields) -> GameState:
        with self.lock:
            state = self.game(site, game_id, fields.pop("url", ""))
            for key, value in fields.items():
                setattr(state, key, value)
            state.updated = time.time()
            return state

    def moved(self, site: str, game_id, actor: str, description: str) -> None:
        self.update_game(site, game_id, last_move={"at": time.time(), "actor": actor, "description": description})
        self.event(site, "move", f"{actor}: {description}", game_id)

    def stuck(self, site: str, game_id, last_id: int) -> bool:
        """Whether the bots gave up at this position (until the game moves or the admin says retry)."""
        state = self.games.get((site, game_id))
        return state is not None and state.stuck is not None and state.stuck_at == last_id

    def give_up(self, site: str, game_id, last_id: int, reason: str) -> None:
        state = self.games.get((site, game_id))
        if state is None or state.stuck_at != last_id or state.stuck != reason:
            self.event(site, "error", f"the bots can't move at action {last_id}: {reason}", game_id)
        self.update_game(site, game_id, stuck=reason, stuck_at=last_id)

    def unstick(self, site: str, game_id) -> None:
        with self.lock:
            state = self.games.get((site, game_id))
            if state is not None and state.stuck is not None:
                state.stuck, state.stuck_at = None, None

    def retry(self, site: str, game_id) -> None:
        self.unstick(site, game_id)
        self.live.pop((site, game_id), None)
        self.event(site, "info", "retrying", game_id)

    def finished(self, site: str, game_ids) -> None:
        """Games that left a site's list (finished or deleted): shown a while longer, then dropped."""
        with self.lock:
            for game_id in game_ids:
                state = self.games.get((site, game_id))
                if state is not None and state.status != "finished":
                    state.status, state.bot_turn, state.waiting_on = "finished", False, ""
                self.live.pop((site, game_id), None)
            done = sorted(
                (s for s in self.games.values() if s.site == site and s.status == "finished"),
                key=lambda s: s.updated,
            )
            for state in done[:-FINISHED_KEPT]:
                del self.games[(site, state.game_id)]

    # --------------------------------------------------------------- the move
    def search_move(self, site: str, game: dict, view: dict, bots, readouts: int) -> Optional[dict]:
        """The model's move for the bot that must act in ``game`` (the site's game data; ``view``: the
        site engine's view of who acts, as ``autoplay.SiteView`` reads it; ``bots``: the bots' player
        ids): the advisor's job result, or None while another search runs. Raises :class:`CannotMove`.
        A game of a title in ``titled`` gets that policy's move instead (:meth:`titled_move`)."""
        if str(game.get("title") or "") in self.titled:
            return self.titled_move(site, game, view, bots)
        problem = support_problem(game)
        if problem:
            raise CannotMove(problem)
        key = (site, game["id"])
        try:
            live = self._follow(key, game)
            seats = [i + 1 for i, player in enumerate(game["players"]) if player["id"] in bots]
            job = self.advisor.start_move(live, SiteView.parse(view), seats, readouts)
        except SearchBusy:
            return None
        except (AutoplayError, UnsupportedGame) as e:
            self.live.pop(key, None)
            raise CannotMove(str(e)) from None
        while job.status == "running":
            time.sleep(JOB_POLL_SECONDS)
        if job.status != "done":
            raise CannotMove(f"the model's move failed: {job.error or job.status}")
        return job.result

    def titled_move(self, site: str, game: dict, view: dict, bots) -> dict:
        """The move of the ``titled`` policy for ``game``'s title (:meth:`titled.TitledPolicy.move`), played at
        once; the local site only. Raises :class:`CannotMove`."""
        from rl18xx.agent.advisor.titled import TitledGame

        title = str(game.get("title"))
        if site != LOCAL:
            raise CannotMove(f"The {title} bots play on the local site only.")
        key = (site, game["id"])
        try:
            live = self._follow(key, game, TitledGame)
            seats = [i + 1 for i, player in enumerate(game["players"]) if player["id"] in bots]
            return self.titled[title].move(live, SiteView.parse(view), seats)
        except (AutoplayError, UnsupportedGame) as e:
            self.live.pop(key, None)
            raise CannotMove(str(e)) from None

    def _follow(self, key, game: dict, kind=LiveGame):
        live = self.live.get(key)
        if isinstance(live, kind):
            try:
                live.update(game)
                return live
            except Exception:
                pass
        live = self.live[key] = kind(game)
        return live

    # ---------------------------------------------------------- the admin page
    def snapshot(self) -> dict:
        search = self.advisor.running_search()
        with self.lock:
            games = [asdict(state) for state in self.games.values()]
            events = list(self.events)[::-1]
            settings = copy.deepcopy(self.settings)
        sites = {}
        for site, values in settings.items():
            connector = self.connectors.get(site)
            sites[site] = {"settings": values, **(connector.status() if connector else {"state": "off"})}
        order = {"active": 0, "new": 1, "finished": 2}
        games.sort(key=lambda g: (order.get(g["status"], 3), g["site"], str(g["game_id"])))
        return {
            "now": time.time(),
            "started": self.started,
            "checkpoints": self.advisor.models.labels,
            "search": search.to_dict() if search else None,
            "sites": sites,
            "games": games,
            "events": events,
        }
