"""The bots on 18xx.games (:mod:`bot_hub`'s ``18xx.games`` connector), without a browser.

The bot accounts are ``<name>.token`` files in ``accounts_dir``, each holding that account's
``auth_token`` cookie value (never logged). Every ``poll_seconds`` the connector lists each account's
games (and joins the game ids appended to ``join_file`` or given on the admin page); a game waiting on
a bot is read, and the bot's move made:

* who must act and the move's auto actions come from the LOCAL site's copy of the site's Ruby engine
  (:meth:`site_bots.SiteClient.engine`) -- what the extension's page bridge got from the site's own
  engine in a headless Chromium, which this replaces (``extension/bot/run.js``);
* the move is the advisor's (:meth:`BotHub.search_move`), posted as the account of the bot that acts,
  as the site's page posts one (numbered, with its auto actions).

Every request waits its turn in one queue, ``min_interval`` seconds apart: the site allows 2 requests
a second per IP, and this machine shares its IP with the people playing (it has been blocklisted).
When the site stops answering, every request pauses, 2 minutes then doubling up to 30. A move that got
no answer may have landed, and the server numbers moves itself (it would take a resend as a second
move), so it is never resent: the game is read afresh first.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional

from rl18xx.agent.advisor.bot_hub import LOCAL, PUBLIC, CannotMove, Connector
from rl18xx.agent.advisor.site_bots import IMPLICIT_PASS_STEPS, SiteError, client_for, waiting_on

TITLE = "1830"
BACKOFF_START = 120.0
BACKOFF_MAX = 1800.0
LOST_POST_WAIT = 5.0  # after a move got no answer: time for the server to finish it
OUTAGE_STATUSES = (429, 502, 503, 504)


class NoAnswer(RuntimeError):
    """The site didn't answer (or said it is overloaded): every request pauses."""


class Refused(RuntimeError):
    """The site answered with an error."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Pacer:
    """One queue for every request to the site: at least ``interval()`` seconds apart, paused while
    the site isn't answering, counted for the admin page."""

    def __init__(self, interval, clock=time.time, sleep=time.sleep):
        self.interval = interval  # a callable: the setting may change while running
        self.clock, self.sleep = clock, sleep
        self.lock = threading.Lock()
        self.next = 0.0
        self.paused_until = 0.0
        self.backoff = BACKOFF_START
        self.sent: deque = deque()  # request times, the last hour

    def turn(self) -> None:
        with self.lock:
            now = self.clock()
            if now < self.paused_until:
                raise NoAnswer(f"paused until {time.strftime('%H:%M:%S', time.localtime(self.paused_until))}")
            wait = self.next - now
            if wait > 0:
                self.sleep(wait)
                now = self.clock()
            self.next = now + self.interval()
            self.sent.append(now)
            while self.sent and self.sent[0] < now - 3600:
                self.sent.popleft()

    def answered(self) -> None:
        self.backoff = BACKOFF_START

    def failed(self) -> float:
        """The site didn't answer: pause everything; the pause's length."""
        with self.lock:
            pause = self.backoff
            self.paused_until = self.clock() + pause
            self.backoff = min(BACKOFF_MAX, self.backoff * 2)
            return pause

    def counts(self) -> dict:
        now = self.clock()
        return {
            "last_minute": sum(1 for t in self.sent if t >= now - 60),
            "last_hour": len(self.sent),
            "paused_until": self.paused_until if self.paused_until > now else None,
        }


class PublicClient:
    """18xx.games' own API, as its pages use it: reads logged out, an account's requests with its cookie."""

    def __init__(self, url: str, pacer: Pacer, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.pacer = pacer
        self.timeout = timeout

    def request(self, path: str, token: Optional[str] = None, body: Optional[dict] = None) -> dict:
        self.pacer.turn()
        headers = {"Accept": "application/json"}
        if token:
            headers["Cookie"] = f"auth_token={token}"
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                message = json.loads(raw).get("error")
            except ValueError:
                message = None
            if e.code in OUTAGE_STATUSES or (e.code >= 500 and not message):
                raise NoAnswer(f"HTTP {e.code}") from None
            raise Refused(e.code, str(message or e.reason)) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise NoAnswer(str(getattr(e, "reason", None) or e)) from None
        self.pacer.answered()
        if isinstance(data, dict) and data.get("error"):
            raise Refused(200, str(data["error"]))
        return data

    def games(self, token: str) -> list:
        """The account's game list (its own new and active games among them)."""
        return self.request(f"/api/game?title={TITLE}", token=token).get("games") or []

    def game(self, game_id) -> dict:
        return self.request(f"/api/game/{game_id}")

    def join(self, game_id, token: str) -> dict:
        return self.request(f"/api/game/{game_id}/join", token=token, body={})

    def post(self, game_id, token: str, move: dict) -> dict:
        return self.request(f"/api/game/{game_id}/action", token=token, body=move)


def read_accounts(directory) -> dict:
    """``{account name: auth_token}`` from ``<name>.token`` files."""
    accounts = {}
    for path in sorted(Path(directory).expanduser().glob("*.token")):
        token = path.read_text().strip()
        if token and not any(c.isspace() for c in token):
            accounts[path.stem] = token
    return accounts


def take_join_requests(path) -> list:
    """Game ids appended to the join file since the last look (the file is emptied)."""
    path = Path(path).expanduser()
    try:
        text = path.read_text()
    except OSError:
        return []
    if not text.strip():
        return []
    path.write_text("")
    return [str(int(t)) for t in text.split() if t.isdigit()]


class PublicSite(Connector):
    site = PUBLIC

    def __init__(self, hub, client: Optional[PublicClient] = None, engine=None, clock=time.time):
        super().__init__(hub)
        self.pacer = client.pacer if client else Pacer(lambda: float(self.settings["min_interval"]))
        self._client = client
        self._engine = engine
        self.clock = clock
        self.accounts: dict = {}
        self.followed: dict = {}  # game id -> the latest listing of it (plus "bots": player ids)
        self.joins: deque = deque()  # (game id, number of bots) from the admin page
        self.next_look = 0.0

    @property
    def client(self) -> PublicClient:
        if self._client is None or self._client.url != self.settings["url"].rstrip("/"):
            self._client = PublicClient(self.settings["url"], self.pacer)
        return self._client

    @property
    def engine(self):
        """The local site's engine (its bot API)."""
        if self._engine is None:
            self._engine = client_for(self.hub.settings[LOCAL])
        return self._engine

    def join(self, game_id, bots: Optional[int] = None) -> None:
        """The admin page's "join": ``bots`` of the accounts (all of them by default) join on the next look."""
        self.joins.append((str(int(game_id)), bots))
        self.next_look = 0.0

    def status(self) -> dict:
        out = super().status()
        counts = self.pacer.counts()
        out.update(
            requests=counts, accounts=sorted(self.accounts) or sorted(read_accounts(self.settings["accounts_dir"]))
        )
        if counts["paused_until"] and out["state"] == "running":
            out["state"] = "paused"
        return out

    def tick(self) -> float:
        try:
            if self.clock() >= self.next_look:
                self.look()
                self.next_look = self.clock() + float(self.settings["poll_seconds"])
            played = False
            for game_id, listing in list(self.followed.items()):
                if self.stopping:
                    break
                if listing.get("status") == "active" and set(listing.get("acting") or []) & set(listing["bots"]):
                    played = self.play(game_id) or played
        except NoAnswer as e:
            pause = self.pacer.failed() if not str(e).startswith("paused until") else 0
            if pause:
                self.hub.event(self.site, "warning", f"the site isn't answering ({e}): pausing {int(pause)} s")
            self.state, self.detail = "running", f"paused: {e}"
            return 5.0
        self.detail = ""
        return 0.0 if played else max(1.0, self.next_look - self.clock())

    # ------------------------------------------------------------ the listing
    def look(self) -> None:
        self.accounts = read_accounts(self.settings["accounts_dir"])
        if not self.accounts:
            raise RuntimeError(f"no bot accounts: put <name>.token files in {self.settings['accounts_dir']}")
        requests = [(game_id, None) for game_id in take_join_requests(self.settings["join_file"])]
        while self.joins:
            requests.append(self.joins.popleft())
        for game_id, count in requests:
            self.join_game(game_id, count)
        listed = {}
        for name, token in self.accounts.items():
            for game in self.client.games(token):
                if str(game.get("title")) != TITLE or game.get("status") not in ("new", "active"):
                    continue
                if not any(p.get("name") == name for p in game.get("players") or []):
                    continue
                listed[game["id"]] = game
        self.last_look = self.clock()
        for game_id, game in listed.items():
            bots = [p["id"] for p in game.get("players") or [] if p.get("name") in self.accounts]
            self.followed[game_id] = {**game, "bots": bots}
            self._show(game, bots)
        gone = [game_id for game_id in self.followed if game_id not in listed]
        for game_id in gone:
            del self.followed[game_id]
        self.hub.finished(self.site, gone)

    def join_game(self, game_id, count: Optional[int]) -> None:
        names = list(self.accounts)[: count if count else None]
        game = self.client.game(game_id)
        if str(game.get("title")) != TITLE:
            self.hub.event(self.site, "warning", f"is {game.get('title')}: the bots play only {TITLE}", game_id)
            return
        for name in names:
            if any(p.get("name") == name for p in game.get("players") or []):
                continue
            if game.get("status") != "new":
                self.hub.event(self.site, "warning", f"has started: {name} can't join it any more", game_id)
                return
            try:
                game = self.client.join(game_id, self.accounts[name])
            except Refused as e:
                self.hub.event(self.site, "warning", f"{name} could not join: {e}", game_id)
                continue
            self.hub.event(self.site, "info", f"{name} joined", game_id)

    def _show(self, game: dict, bots) -> None:
        self.hub.update_game(
            self.site,
            game["id"],
            url=f"{self.client.url}/game/{game['id']}",
            title=str(game.get("title", "")),
            status=game.get("status", ""),
            round=f"{game.get('round') or ''} {game.get('turn') or ''}".strip(),
            players=[{"name": p["name"], "bot": p["id"] in bots} for p in game.get("players") or []],
            waiting_on=waiting_on(game),
            bot_turn=game.get("status") == "active" and bool(set(game.get("acting") or []) & set(bots)),
        )

    # --------------------------------------------------------------- the move
    def play(self, game_id) -> bool:
        """The model's move in ``game_id`` if a bot must act there; whether one was posted."""
        game = self.client.game(game_id)
        bots = [p["id"] for p in game.get("players") or [] if p.get("name") in self.accounts]
        listing = self.followed.setdefault(game_id, {})
        listing.update({k: game.get(k) for k in ("status", "acting", "round", "turn")}, bots=bots)
        self._show(game, bots)
        if game.get("status") != "active":
            return False
        try:
            view = self.engine.engine(game)["view"]
        except SiteError as e:
            raise RuntimeError(f"the local site's engine didn't answer: {e}") from None
        last_id = int(view.get("last_id") or 0)
        if self.hub.stuck(self.site, game_id, last_id):
            return False
        self.hub.unstick(self.site, game_id)
        if view.get("error"):
            self.hub.give_up(self.site, game_id, last_id, f"the local engine can't load it: {view['error']}")
            return False
        entity = view.get("entity") or {}
        if view.get("finished") or entity.get("player") not in bots:
            listing["acting"] = view.get("acting") or []
            return False
        try:
            result = self.hub.search_move(self.site, game, view, set(bots), self.settings["readouts"])
        except CannotMove as e:
            self.hub.give_up(self.site, game_id, last_id, str(e))
            return False
        if result is None:
            return False  # another search is running: on the next look
        token = self._token(game, entity["player"])
        try:
            response = self._post(game, view, token, result["action"])
        except Refused as e:
            pass_move = {"type": "pass", "entity": entity["id"], "entity_type": entity["type"]}
            implicit = (
                result["action"].get("type") != "pass"
                and "pass" in (view.get("actions") or [])
                and view.get("step") in IMPLICIT_PASS_STEPS
            )
            if not implicit:
                self.hub.give_up(self.site, game_id, last_id, f'the site refused "{result["description"]}": {e}')
                return False
            try:
                response = self._post(game, view, token, pass_move)
            except Refused:
                self.hub.give_up(self.site, game_id, last_id, f'the site refused "{result["description"]}": {e}')
                return False
            self.hub.moved(self.site, game_id, result["actor"], f"Pass ({view.get('step')}): the site refused {e}")
        except NoAnswer:
            # It may have landed: the game is read afresh (after the pause) before anything else.
            self.hub.event(
                self.site, "warning", f'"{result["description"]}" got no answer: reading the game again', game_id
            )
            self.wait(LOST_POST_WAIT)
            raise
        else:
            self.hub.moved(self.site, game_id, result["actor"], result["description"])
        listing.update({k: response.get(k) for k in ("status", "acting") if k in response})
        return True

    def _token(self, game: dict, player_id) -> str:
        for player in game.get("players") or []:
            if player.get("id") == player_id and player.get("name") in self.accounts:
                return self.accounts[player["name"]]
        raise CannotMove(f"no bot account plays player {player_id}")

    def _post(self, game: dict, view: dict, token: str, action: dict) -> dict:
        """``action`` checked by the local engine and posted as the site's page posts a move: numbered,
        with the auto actions it sets off (the server requires them to match its own)."""
        try:
            checked = self.engine.engine(game, action)
        except SiteError as e:
            raise RuntimeError(f"the local site's engine didn't answer: {e}") from None
        if checked.get("error"):
            raise Refused(0, f"the engine refuses it: {checked['error']}")
        move = dict(action)
        move.pop("created_at", None)
        move["id"] = int(view.get("last_id") or 0) + 1
        stored = checked.get("action") or {}
        if stored.get("auto_actions"):
            move["auto_actions"] = stored["auto_actions"]
        return self.client.post(game["id"], token, move)
