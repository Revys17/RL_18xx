"""What deploy.sh records about a deployment (run with the deployment's own python, -I).

    deploy_record.py packages                       name==version of every installed distribution, sorted
    deploy_record.py record <dest> <commit> <ref> <repo>   DEPLOYMENT.json on stdout
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def packages() -> list[str]:
    return sorted({f"{d.metadata['Name'].lower()}=={d.version}" for d in importlib.metadata.distributions()})


def _run(*cmd, cwd=None) -> str:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        return (out.stdout or out.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({e})"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nvidia() -> dict:
    """The loaded kernel module's version and the installed userspace packages' (a mismatch is the
    'Driver/library version mismatch' that needs a reboot)."""
    module = Path("/proc/driver/nvidia/version")
    packages = _run("dpkg-query", "-W", "-f", "${Package} ${Version}\\n", "libnvidia-compute-*", "nvidia-driver-*")
    return {
        "kernel_module": module.read_text().splitlines()[0] if module.exists() else None,
        "packages": [line for line in packages.splitlines() if line and " " in line and not line.endswith(" ")],
    }


def record(dest: str, commit: str, ref: str, repo: str) -> dict:
    import torch  # the venv's (no CUDA initialisation: versions only)

    dest_path = Path(dest)
    engine = sorted(dest_path.glob(".venv/lib/python*/site-packages/engine_rs/engine_rs*.so"))
    return {
        "commit": commit,
        "ref": ref,
        "tags": _run("git", "-C", dest, "tag", "--points-at", commit).split(),
        "commit_date": _run("git", "-C", dest, "show", "-s", "--format=%cI", commit),
        "commit_subject": _run("git", "-C", dest, "show", "-s", "--format=%s", commit),
        "source_repo": repo,
        "deployed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "kernel": platform.release(),
        "nvidia": nvidia(),
        "python": {"version": platform.python_version(), "executable": os.path.realpath(sys.executable)},
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
        "uv": _run("uv", "--version"),
        "rustc": _run("rustc", "--version"),
        "cargo": _run("cargo", "--version"),
        "engine_rs": {str(p.relative_to(dest_path)): _sha256(p) for p in engine},
        "uv_lock_sha256": _sha256(dest_path / "uv.lock"),
        "cargo_lock_sha256": _sha256(dest_path / "engine-rs" / "Cargo.lock"),
        "build_env": {k: os.environ[k] for k in ("UV_LINK_MODE", "MATURIN_PEP517_ARGS") if k in os.environ},
    }


def main() -> None:
    if sys.argv[1:2] == ["packages"]:
        print("\n".join(packages()))
    elif sys.argv[1:2] == ["record"] and len(sys.argv) == 6:
        print(json.dumps(record(*sys.argv[2:6]), indent=2))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
