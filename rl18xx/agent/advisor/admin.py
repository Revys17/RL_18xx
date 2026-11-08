"""The bots' admin page (``/admin`` on ``main.py advisor --bots``): switch each site's bots on and off,
change their search budget and pacing, have the 18xx.games accounts join a game, retry a game the
bots gave up on, and see the games, the running search and the event log (:mod:`bot_hub`).

HTTP Basic auth, user ``admin`` and the password in ``--admin-password-file`` (made on first start,
mode 600). Changes are POSTs carrying ``X-Admin: 1``, a header no other site's page can send here
without a CORS preflight this server never answers.
"""

from __future__ import annotations

import hmac
import secrets
from pathlib import Path

from flask import Response, jsonify, request

from rl18xx.agent.advisor.bot_hub import PUBLIC

PAGE = Path(__file__).resolve().parent / "templates" / "admin.html"
USER = "admin"


def read_password(path) -> str:
    """The admin password, made (random, mode 600) if the file doesn't exist yet."""
    path = Path(path).expanduser()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_text(secrets.token_urlsafe(18) + "\n")
    password = path.read_text().strip()
    if not password:
        raise ValueError(f"{path} is empty")
    return password


def register(app, hub, password: str) -> None:
    def authorized() -> bool:
        auth = request.authorization
        return (
            auth is not None
            and hmac.compare_digest((auth.username or "").encode(), USER.encode())
            and hmac.compare_digest((auth.password or "").encode(), password.encode())
        )

    @app.before_request
    def admin_guard():
        if not request.path.startswith("/admin"):
            return None
        if not authorized():
            return Response("The bots' admin page", 401, {"WWW-Authenticate": 'Basic realm="rl18xx bots"'})
        if request.method == "POST" and request.headers.get("X-Admin") != "1":
            return jsonify({"error": "Missing X-Admin header"}), 403
        return None

    @app.route("/admin")
    def admin_page():
        return Response(PAGE.read_text(), mimetype="text/html")

    @app.route("/admin/api/state")
    def admin_state():
        return jsonify(hub.snapshot())

    @app.route("/admin/api/site/<path:site>", methods=["POST"])
    def admin_site(site):
        try:
            settings = hub.change(site, request.get_json(silent=True) or {})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(settings)

    @app.route("/admin/api/retry", methods=["POST"])
    def admin_retry():
        body = request.get_json(silent=True) or {}
        hub.retry(body.get("site"), body.get("game_id"))
        return jsonify({"ok": True})

    @app.route("/admin/api/join", methods=["POST"])
    def admin_join():
        body = request.get_json(silent=True) or {}
        try:
            game_id = int(body.get("game_id"))
            bots = int(body["bots"]) if body.get("bots") else None
        except (TypeError, ValueError):
            return jsonify({"error": "game_id must be a number"}), 400
        connector = hub.connectors.get(PUBLIC)
        if connector is None or not connector.running:
            return jsonify({"error": "Switch the 18xx.games bots on first"}), 409
        connector.join(game_id, bots)
        hub.event(PUBLIC, "info", f"join requested ({bots or 'every'} bot{'s' if bots != 1 else ''})", game_id)
        return jsonify({"ok": True})
