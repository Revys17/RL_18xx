"""Build the 1830 advisor extension for Chrome and Firefox.

One source tree (``src/``), two Manifest V3 manifests: Chrome runs the
background worker as a service worker, Firefox as background scripts. Writes
``dist/chrome/`` and ``dist/firefox/``, each a directory to load unpacked
(see README.md).

    python extension/build.py                                   # https://18xx.games only
    python extension/build.py --extra-origin http://localhost:9292   # + a self-hosted server

Permissions: storage, the 18xx.games origin (and any ``--extra-origin``), and
the advisor backend on this machine (http://127.0.0.1 and http://localhost, any
port), and -- optional, asked for when the options page saves one -- a backend on
another machine (any http host). Match patterns can't name a port, so an extra
origin covers its host on every port.
"""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
FILES = ("common.js", "panel.js", "content.js", "background.js", "options.html", "options.js")
VERSION = "0.1.0"
SITE = "https://18xx.games/*"
BACKEND = ("http://127.0.0.1/*", "http://localhost/*")
REMOTE_BACKEND = "http://*/*"  # a backend on another machine (--host 0.0.0.0), granted from the options page
GECKO_ID = "rl18xx-advisor@rl18xx.local"


def origin_pattern(origin: str) -> str:
    """``scheme://host/*`` of an origin such as ``http://localhost:9292``."""
    parsed = urlparse(origin if "://" in origin else f"http://{origin}")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SystemExit(f"Not an http(s) origin: {origin!r}")
    return f"{parsed.scheme}://{parsed.hostname}/*"


def manifest(browser: str, extra_origins=()) -> dict:
    sites = [SITE] + [origin_pattern(o) for o in extra_origins]
    sites = list(dict.fromkeys(sites))
    data = {
        "manifest_version": 3,
        "name": "RL18xx 1830 advisor",
        "version": VERSION,
        "description": "Shows a local 1830 model's win estimates and suggested moves on 18xx.games game pages.",
        "permissions": ["storage"],
        "host_permissions": list(dict.fromkeys(sites + list(BACKEND))),
        "optional_host_permissions": [REMOTE_BACKEND],
        "content_scripts": [
            {"matches": sites, "js": ["common.js", "panel.js", "content.js"], "run_at": "document_idle"}
        ],
        "options_ui": {"page": "options.html", "open_in_tab": False},
    }
    if browser == "chrome":
        data["background"] = {"service_worker": "background.js"}
        data["minimum_chrome_version"] = "116"
    elif browser == "firefox":
        data["background"] = {"scripts": ["common.js", "background.js"]}
        data["browser_specific_settings"] = {
            "gecko": {
                "id": GECKO_ID,
                "strict_min_version": "128.0",
                # Nothing leaves the machine: games go from the page to the local backend.
                "data_collection_permissions": {"required": ["none"]},
            }
        }
    else:
        raise ValueError(browser)
    return data


def build(out: Path, extra_origins=()) -> list:
    built = []
    for browser in ("chrome", "firefox"):
        target = out / browser
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(SRC / name, target / name)
        (target / "manifest.json").write_text(json.dumps(manifest(browser, extra_origins), indent=2) + "\n")
        built.append(target)
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--extra-origin", action="append", default=[], help="Another 18xx server to run on, e.g. http://localhost:9292"
    )
    parser.add_argument("--out", default=str(HERE / "dist"), help="Output directory (default: extension/dist)")
    args = parser.parse_args()
    for target in build(Path(args.out), args.extra_origin):
        print(f"Built {target}")


if __name__ == "__main__":
    main()
