"""The advisor's local backend: a JSON API for the browser extension (``extension/``).

The extension's content script reads the game the user has open on
18xx.games (the site's own ``GET /api/game/<id>``) and its background worker
posts it here. This process never contacts 18xx.games or any other host, never
sees credentials or cookies, and keeps games in memory only (no game JSON in
files or logs).

- ``POST /api/advise``: body = the game JSON as ``GET /api/game/<id>`` returns
  it. Answers ``supported`` (and ``reason`` when not: another title, an
  optional rule the engines don't implement, ...) and, for a supported game,
  :meth:`Advisor.advise`'s advice: ``players`` in seat (engine) order, the
  ``acting`` entity and the player behind it, ``win`` estimates, the top
  ``moves`` (``description``, ``probability``, ``price`` with ``options``, and
  ``hex`` / ``tile`` / ``rotation`` / ``map`` for map moves), the model's view
  of the ``recent`` moves, ``warnings``, and ``position`` (the position's id).
  Games are cached by id and followed incrementally (``LiveGame``).
- ``POST /api/think``: ``{"game_id", "position", "readouts"}`` starts an MCTS
  search from the game's current position (one at a time; 409 while one runs or
  when ``position`` is no longer current).
- ``GET`` / ``POST /api/think/<job>``: its ``status``, ``progress`` and ``result``.
- ``GET`` / ``POST /api/health``: the loaded checkpoints.
- ``GET /debug`` (``--debug-page`` only): paste or upload a game JSON and see
  the extension's panel for it.

It serves on 127.0.0.1 unless ``--host`` says otherwise (``--host 0.0.0.0`` for a
browser on another machine of the LAN). Only requests addressed to an IP address,
localhost or this machine's host name (no DNS rebinding) from a browser extension
(``Origin: chrome-extension://...`` or ``moz-extension://...``, which web pages
can't forge) reach the API -- plus, with ``--debug-page``, the backend's own page.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_from_directory

from rl18xx.agent.advisor.advisor import DEFAULT_READOUTS, Advisor, SearchBusy
from rl18xx.agent.advisor.live_game import LiveGame, UnsupportedGame, support_problem

LOGGER = logging.getLogger(__name__)

EXTENSION_ORIGINS = ("chrome-extension://", "moz-extension://")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")
LOCAL_HOSTNAMES = ("127.0.0.1", "localhost", "::1")
MAX_GAMES = 8
MAX_BODY_BYTES = 32 * 1024 * 1024
EXTENSION_SRC = Path(__file__).resolve().parents[3] / "extension" / "src"
DEBUG_PAGE = Path(__file__).resolve().parent / "templates" / "debug.html"


class _Followed:
    def __init__(self):
        self.lock = threading.Lock()
        self.live: Optional[LiveGame] = None


class GameCache:
    """The followed games by id, the least recently used dropped beyond ``limit``."""

    def __init__(self, limit: int = MAX_GAMES):
        self._limit = limit
        self._games: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def entry(self, game_id, create: bool = True) -> Optional[_Followed]:
        key = str(game_id)
        with self._lock:
            followed = self._games.get(key)
            if followed is None:
                if not create:
                    return None
                followed = self._games[key] = _Followed()
            self._games.move_to_end(key)
            while len(self._games) > self._limit:
                self._games.popitem(last=False)
            return followed


def _players(game: dict) -> list:
    return [
        {"seat": i + 1, "id": p.get("id"), "name": p.get("name")}
        for i, p in enumerate(game.get("players") or [])
        if isinstance(p, dict)
    ]


def _failure(what: str, game_id, error: Exception):
    """Log an internal error without its message (it may quote the game) and answer 500."""
    frames = "".join(traceback.format_tb(error.__traceback__))
    LOGGER.error("%s failed (game %s): %s\n%s", what, game_id, type(error).__name__, frames)
    return jsonify({"error": f"{what} failed: {type(error).__name__}: {error}"}), 500


def host_allowed(hostname: Optional[str]) -> bool:
    """Whether a request's Host names this machine: an IP address, localhost or the
    machine's own name. Any other name could be a DNS-rebinding page's domain."""
    if not hostname:
        return False
    if hostname in LOCAL_HOSTNAMES or hostname.lower() in {socket.gethostname().lower(), socket.getfqdn().lower()}:
        return True
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


def create_app(advisor: Advisor, *, port: int, debug_page: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    games = GameCache()
    own_origins = {f"http://{host}:{port}" for host in LOCAL_HOSTS} if debug_page else set()

    def origin_allowed(origin: str) -> bool:
        return origin.startswith(EXTENSION_ORIGINS) or origin in own_origins

    @app.before_request
    def guard():
        if not host_allowed(urlsplit(f"//{request.host or ''}").hostname):
            return jsonify({"error": "This server only answers to an IP address or this machine's name"}), 403
        if request.path.startswith("/api/") and not origin_allowed(request.headers.get("Origin", "")):
            return jsonify({"error": "Only the advisor's browser extension may use this API"}), 403
        return None

    @app.after_request
    def cors(response):
        origin = request.headers.get("Origin", "")
        if request.path.startswith("/api/") and origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Max-Age"] = "600"
            if request.headers.get("Access-Control-Request-Private-Network"):
                response.headers["Access-Control-Allow-Private-Network"] = "true"
            response.headers["Vary"] = "Origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/health", methods=["GET", "POST"])
    def health():
        return jsonify({"ok": True, "checkpoints": advisor.models.labels})

    @app.route("/api/advise", methods=["POST"])
    def advise():
        game = request.get_json(silent=True)
        if (
            not isinstance(game, dict)
            or game.get("id") is None
            or not isinstance(game.get("players"), list)
            or not isinstance(game.get("actions"), list)
        ):
            return jsonify({"error": "Expected a game as 18xx.games' GET /api/game/<id> returns it"}), 400
        game_id = game["id"]
        base = {"game_id": game_id, "players": _players(game), "title": game.get("title")}
        problem = support_problem(game)
        if problem:
            return jsonify({**base, "supported": False, "reason": problem})
        if game.get("status") == "new":
            return jsonify({**base, "supported": True, "started": False, "reason": "The game hasn't started yet."})
        if game.get("status") == "archived":
            reason = "The game is archived: the site no longer has its moves."
            return jsonify({**base, "supported": False, "reason": reason})
        followed = games.entry(game_id)
        with followed.lock:
            try:
                if followed.live is None:
                    followed.live = LiveGame(game)
                else:
                    followed.live.update(game)
                advice = advisor.advise(followed.live, game.get("acting"))
            except UnsupportedGame as e:
                followed.live = None
                return jsonify({**base, "supported": False, "reason": str(e)})
            except Exception as e:  # a bug: start over next time
                followed.live = None
                return _failure("advise", game_id, e)
        return jsonify({**advice, "started": True, "title": game.get("title")})

    @app.route("/api/think", methods=["POST"])
    def think():
        body = request.get_json(silent=True) or {}
        game_id = body.get("game_id")
        followed = games.entry(game_id, create=False) if game_id is not None else None
        if followed is None or followed.live is None:
            return jsonify({"error": "Ask for advice on this game first"}), 404
        try:
            readouts = int(body.get("readouts") or DEFAULT_READOUTS)
        except (TypeError, ValueError):
            return jsonify({"error": "readouts must be a number"}), 400
        with followed.lock:
            live = followed.live
            position = body.get("position")
            if position and position != live.position_key:
                return jsonify({"error": "The game has moved on since that advice; refresh first"}), 409
            try:
                job = advisor.start_search(live, readouts)
            except SearchBusy as busy:
                return jsonify({"error": "A search is already running", "job": busy.job.to_dict()}), 409
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return _failure("think", game_id, e)
        return jsonify(job.to_dict())

    @app.route("/api/think/<job_id>", methods=["GET", "POST"])
    def think_status(job_id):
        job = advisor.job(job_id)
        if job is None:
            return jsonify({"error": f"No search {job_id}"}), 404
        return jsonify(job.to_dict())

    if debug_page:

        @app.route("/debug")
        def debug():
            return DEBUG_PAGE.read_text(), 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/debug/src/<path:name>")
        def debug_src(name):
            return send_from_directory(EXTENSION_SRC, name)

    return app


def run(args) -> None:
    """``main.py advisor``: load the checkpoints and serve on ``--host`` (127.0.0.1 by default)."""
    import torch

    from rl18xx.agent.advisor.advisor import AdvisorModels

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    models = AdvisorModels.load(args.policy, args.auction_policy, args.value, device)
    app = create_app(Advisor(models), port=args.port, debug_page=args.debug_page)
    host = getattr(args, "host", "127.0.0.1")
    LOGGER.info("Advisor backend on http://%s:%d (device %s)", host, args.port, device)
    if host not in ("127.0.0.1", "localhost", "::1"):
        LOGGER.info("Reachable from other machines: set the extension's backend URL to http://<this machine>:%d", args.port)
    if args.debug_page:
        LOGGER.info("Debug page: http://127.0.0.1:%d/debug", args.port)
    app.run(host=host, port=args.port, threaded=True, use_reloader=False)
