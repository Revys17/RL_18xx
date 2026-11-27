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
- ``POST /api/move``: ``{"game", "seats", "site", "readouts"}`` -- a hotseat (or live) game,
  the seats (1-based) the model plays, and the page bridge's view of who the
  site's Ruby engine waits for (``autoplay.SiteView``): starts the model's move
  (``Advisor.start_move``; 409 with ``busy`` while another search runs, with
  ``autoplay`` when it has no move to make). ``GET`` / ``POST /api/move/<job>``:
  its status; the result's ``action`` is the move in the site's format.
- ``GET`` / ``POST /api/health``: the loaded checkpoints.
- ``GET /debug`` (``--debug-page`` only): paste or upload a game JSON and see
  the extension's panel for it.

It serves on 127.0.0.1 unless ``--host`` says otherwise (``--host 0.0.0.0`` for a
browser on another machine of the LAN). Only requests addressed to an IP address,
localhost or this machine's host name (no DNS rebinding) from a browser extension
(``Origin: chrome-extension://...`` or ``moz-extension://...``, which web pages
can't forge) reach the API -- plus, with ``--debug-page``, the backend's own page, and
with ``--web-site-port P`` the pages of a site served on port P of this machine (the local
copy of 18xx.games, whose game page has an engine panel), for ``/api/health`` and
``/api/advise`` only.
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
from rl18xx.agent.advisor.autoplay import AutoplayError, SiteView
from rl18xx.agent.advisor.live_game import LiveGame, UnsupportedGame, support_problem

LOGGER = logging.getLogger(__name__)

EXTENSION_ORIGINS = ("chrome-extension://", "moz-extension://")
# What a web page (--web-site-port) may ask: advice, nothing that searches or plays.
WEB_PAGE_PATHS = ("/api/health", "/api/advise")
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


def web_page_origin(origin: str, ports) -> bool:
    """Whether ``origin`` is a page of a site on this machine served on one of ``ports``
    (``--web-site-port``): http(s), that port, and a host :func:`host_allowed` accepts."""
    if not ports:
        return False
    try:
        parts = urlsplit(origin)
        port = parts.port
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and port in ports and host_allowed(parts.hostname)


def create_app(advisor: Advisor, *, port: int, debug_page: bool = False, web_site_ports=()) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    games = GameCache()
    own_origins = {f"http://{host}:{port}" for host in LOCAL_HOSTS} if debug_page else set()
    web_ports = {int(p) for p in web_site_ports or ()}

    def origin_allowed(origin: str, path: str) -> bool:
        if origin.startswith(EXTENSION_ORIGINS) or origin in own_origins:
            return True
        return path in WEB_PAGE_PATHS and web_page_origin(origin, web_ports)

    @app.before_request
    def guard():
        if not host_allowed(urlsplit(f"//{request.host or ''}").hostname):
            return jsonify({"error": "This server only answers to an IP address or this machine's name"}), 403
        if request.path.startswith("/api/") and not origin_allowed(request.headers.get("Origin", ""), request.path):
            return jsonify({"error": "Only the advisor's browser extension may use this API"}), 403
        return None

    @app.after_request
    def cors(response):
        origin = request.headers.get("Origin", "")
        if request.path.startswith("/api/") and origin_allowed(origin, request.path):
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

    @app.route("/api/move", methods=["POST"])
    def move():
        body = request.get_json(silent=True) or {}
        game = body.get("game")
        if not isinstance(game, dict) or not isinstance(game.get("actions"), list) or game.get("id") is None:
            return jsonify({"error": "Expected {game, seats, site, readouts}"}), 400
        try:
            seats = [int(s) for s in body.get("seats") or []]
            readouts = int(body.get("readouts") or DEFAULT_READOUTS)
            site = SiteView.parse(body.get("site"))
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Bad request: {e}"}), 400
        if not seats:
            return jsonify({"error": "No seats for the model to play"}), 400
        game_id = game["id"]
        problem = support_problem(game)
        if problem:
            return jsonify({"error": problem}), 400
        followed = games.entry(game_id)
        with followed.lock:
            try:
                if followed.live is None:
                    followed.live = LiveGame(game)
                else:
                    followed.live.update(game)
                job = advisor.start_move(followed.live, site, seats, readouts)
            except SearchBusy as busy:
                return jsonify({"error": "A search is already running", "busy": True, "job": busy.job.to_dict()}), 409
            except AutoplayError as e:
                return jsonify({"error": str(e), "autoplay": True}), 409
            except UnsupportedGame as e:
                followed.live = None
                return jsonify({"error": str(e)}), 400
            except Exception as e:  # a bug: start over next time
                followed.live = None
                return _failure("move", game_id, e)
        return jsonify(job.to_dict())

    @app.route("/api/move/<job_id>", methods=["GET", "POST"])
    def move_status(job_id):
        job = advisor.job(job_id)
        if job is None:
            return jsonify({"error": f"No move {job_id}"}), 404
        return jsonify(job.to_dict())

    if debug_page:

        @app.route("/debug")
        def debug():
            return DEBUG_PAGE.read_text(), 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/debug/src/<path:name>")
        def debug_src(name):
            return send_from_directory(EXTENSION_SRC, name)

    return app


def load_models(args, device: str):
    """The ``main.py advisor`` options' checkpoints (``AdvisorModels.load``)."""
    from rl18xx.agent.advisor.advisor import AdvisorModels

    return AdvisorModels.load(
        args.policy,
        args.auction_policy,
        args.value,
        device,
        getattr(args, "search_value", None),
        play_policy=getattr(args, "play_policy", None),
        play_value=getattr(args, "play_value", None),
        play_auction_policy=getattr(args, "play_auction_policy", None),
        play_auction_value=getattr(args, "play_auction_value", None),
        play_value_head=getattr(args, "play_value_head", None) or "win",
        play_mode=getattr(args, "play_mode", None) or "search",
        play_temperature=float(getattr(args, "play_temperature", 0.0) or 0.0),
        play_prior_temperature=getattr(args, "play_prior_temperature", None),
    )


def parse_title_policy(spec: str) -> tuple:
    """``(title, checkpoint, temperature or None)`` of ``--title-policy <title>=[generic:]<checkpoint>[@<T>]``."""
    from rl18xx.agent.alphazero.game_network import spec_path

    title, sep, rest = str(spec).partition("=")
    if not sep or not title or not rest:
        raise ValueError(f"--title-policy {spec!r}: <title>=<checkpoint>[@<temperature>]")
    path, at, temperature = spec_path(rest).rpartition("@")
    if not at:
        path, temperature = spec_path(rest), ""
    return title, path, float(temperature) if temperature else None


def parse_title_search(spec: str, policy: str):
    """``(title, generic_search.SearchSpec)`` of ``--title-search <title>=[generic:]<critic>[#<head>][@<readouts>]
    [/<c_puct_init>][:<option>,...]`` searching ``policy`` (the title's ``--title-policy`` checkpoint): the 1830 bots'
    settings (``generic_search.BOT_SEARCH``) and ``BOT_READOUTS`` readouts unless given, the critic's ``score`` head
    unless ``#win`` / ``#blend=<w>``; ``self`` for the policy network's own value heads."""
    from rl18xx.agent.alphazero.generic_search import BOT_READOUTS, BOT_SEARCH, parse_search_spec

    title, sep, rest = str(spec).partition("=")
    if not sep or not title or not rest:
        raise ValueError(f"--title-search {spec!r}: <title>=<critic>[#<head>][@<readouts>][/<c_puct>][:<options>]")
    search = parse_search_spec(rest, default_readouts=BOT_READOUTS, default_head="score", policy=policy)
    if search.policy_only:
        raise ValueError(f"--title-search {spec!r}: sample=<t> is --title-policy's play (no search)")
    search.overrides = {**BOT_SEARCH, **search.overrides}
    return title, search


def load_title_policies(args, device: str) -> dict:
    """``--title-policy`` (and ``--title-search``) -> {title: ``titled.TitledPolicy``}: the bots' policies for titles
    other than 1830, each searched with its title's ``--title-search`` critic when one is given."""
    from rl18xx.agent.advisor.titled import TitledPolicy

    specs = {}
    for spec in getattr(args, "title_policy", None) or []:
        title, path, temperature = parse_title_policy(spec)
        if title in specs:
            raise ValueError(f"--title-policy: {title} given twice")
        specs[title] = (path, temperature)
    searches = {}
    for spec in getattr(args, "title_search", None) or []:
        title = str(spec).partition("=")[0]
        if title not in specs:
            raise ValueError(f"--title-search {spec!r}: no --title-policy for {title}")
        if title in searches:
            raise ValueError(f"--title-search: {title} given twice")
        searches[title] = parse_title_search(spec, specs[title][0])[1]
    return {
        title: TitledPolicy(title, path, device=device, temperature=temperature, search=searches.get(title))
        for title, (path, temperature) in specs.items()
    }


def run(args) -> None:
    """``main.py advisor``: load the checkpoints and serve on ``--host`` (127.0.0.1 by default)."""
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    models = load_models(args, device)
    labels = models.labels
    title_policies = load_title_policies(args, device)
    for title, policy in title_policies.items():
        labels[f"play_{title}"] = policy.label  # shown by /api/health and the admin page
        LOGGER.info(
            "The bots play %s seats with %s (%s)", title, policy.label, "searched" if policy.search else "policy only"
        )
    LOGGER.info(
        "The model plays seats with %s + value %s (%s head; private auction: %s + %s; mode %s; prior T %s)",
        labels["play_policy"],
        labels["play_value"],
        labels["play_value_head"],
        labels["play_auction_policy"],
        labels["play_auction_value"],
        labels["play_mode"],
        labels["play_prior_temperature"],
    )
    advisor = Advisor(models)
    web_site_ports = getattr(args, "web_site_port", None) or []
    app = create_app(advisor, port=args.port, debug_page=args.debug_page, web_site_ports=web_site_ports)
    if getattr(args, "bots", False):
        from rl18xx.agent.advisor import admin
        from rl18xx.agent.advisor.bot_hub import BotHub

        hub = BotHub(advisor, args.bot_settings, titled=title_policies)
        admin.register(app, hub, admin.read_password(args.admin_password_file))
        hub.start()
        LOGGER.info(
            "Bots: settings %s; admin page /admin (user admin, password in %s)",
            hub.settings_path,
            args.admin_password_file,
        )
    host = getattr(args, "host", "127.0.0.1")
    LOGGER.info("Advisor backend on http://%s:%d (device %s)", host, args.port, device)
    if host not in ("127.0.0.1", "localhost", "::1"):
        LOGGER.info(
            "Reachable from other machines: set the extension's backend URL to http://<this machine>:%d", args.port
        )
    if args.debug_page:
        LOGGER.info("Debug page: http://127.0.0.1:%d/debug", args.port)
    if web_site_ports:
        LOGGER.info("Pages of this machine's sites on port(s) %s may ask for advice", web_site_ports)
    app.run(host=host, port=args.port, threaded=True, use_reloader=False)
