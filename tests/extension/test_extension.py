"""The browser extension (extension/): its build and, when node is installed, its JS unit tests."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

EXTENSION = Path(__file__).resolve().parents[2] / "extension"


def _build_module():
    spec = importlib.util.spec_from_file_location("extension_build", EXTENSION / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_build_writes_a_chrome_and_a_firefox_extension(tmp_path):
    build = _build_module()
    targets = build.build(tmp_path, extra_origins=["http://localhost:9292"])
    assert [t.name for t in targets] == ["chrome", "firefox"]
    for target in targets:
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["manifest_version"] == 3
        assert manifest["permissions"] == ["storage"]
        assert set(manifest["host_permissions"]) == {
            "https://18xx.games/*",
            "http://localhost/*",
            "http://127.0.0.1/*",
        }
        script = manifest["content_scripts"][0]
        assert script["matches"] == ["https://18xx.games/*", "http://localhost/*"]
        for name in script["js"] + [manifest["options_ui"]["page"]]:
            assert (target / name).is_file()
    chrome = json.loads((tmp_path / "chrome" / "manifest.json").read_text())
    firefox = json.loads((tmp_path / "firefox" / "manifest.json").read_text())
    assert chrome["background"] == {"service_worker": "background.js"}
    assert firefox["background"] == {"scripts": ["common.js", "background.js"]}
    assert firefox["browser_specific_settings"]["gecko"]["id"]


def test_the_extension_loads_no_remote_code():
    for path in (EXTENSION / "src").iterdir():
        text = path.read_text()
        assert "eval(" not in text and "new Function" not in text, path
        assert "<script src=\"http" not in text and "import(" not in text, path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_extension_js_unit_tests():
    tests = sorted(str(p) for p in (EXTENSION / "test").glob("*.test.js"))
    result = subprocess.run(["node", "--test", *tests], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
