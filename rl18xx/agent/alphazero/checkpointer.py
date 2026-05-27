from rl18xx.agent.alphazero.model import AlphaZeroGNNModel, AlphaZeroModel
from rl18xx.agent.alphazero.config import ModelGNNConfig, ModelTransformerConfig
from pathlib import Path
from typing import Optional
import json
import logging
import os
import tempfile
import torch

LOGGER = logging.getLogger(__name__)

# Bundled `.pth` checkpoints include this discriminator field so the loader can
# pick the right (Config, Model) classes without sniffing other keys.
ARCHITECTURE_KEY = "architecture"

# Key under which the model's `state_dict()` is stored inside the bundled .pth.
# Presence of this key is also how the loader distinguishes new-format
# checkpoints (bundled dict) from legacy ones (raw state_dict).
STATE_DICT_KEY = "state_dict"

# Key under which the model's serialized config dict (the output of
# `config.to_json()`) is stored inside the bundled .pth.
CONFIG_KEY = "config"

# Per-arch pointer file recording which (session, checkpoint_num) is currently
# considered "best" (i.e. the model self-play workers and gating evaluations
# should load). Written atomically so concurrent readers can't see a partial file.
CURRENT_BEST_FILENAME = "current_best.json"


def _build_transformer_model(config_data: dict, checkpoint_path: Optional[str]) -> AlphaZeroModel:
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    config = ModelTransformerConfig.from_json(config_data)
    config.model_checkpoint_file = checkpoint_path
    return AlphaZeroTransformerModel(config)


def _build_gnn_model(config_data: dict, checkpoint_path: Optional[str]) -> AlphaZeroModel:
    config = ModelGNNConfig.from_json(config_data)
    config.model_checkpoint_file = checkpoint_path
    return AlphaZeroGNNModel(config)


# Maps architecture_name() -> factory. New architectures register here.
_ARCHITECTURE_REGISTRY = {
    "AlphaZeroTransformer": _build_transformer_model,
    "AlphaZeroGNN": _build_gnn_model,
}


def _infer_legacy_architecture(config_data: dict) -> str:
    """Best-effort guess for checkpoints saved before ARCHITECTURE_KEY was written.

    Transformer configs uniquely have d_entity / hex_transformer_layers; everything
    else is treated as GNN.
    """
    if "d_entity" in config_data or "hex_transformer_layers" in config_data:
        return "AlphaZeroTransformer"
    return "AlphaZeroGNN"


def _resolve_architecture(checkpoint_dict: dict, config_data: dict) -> str:
    """Pick the architecture name for a loaded checkpoint.

    Preference order: explicit `architecture` field on the .pth dict, explicit
    `architecture` field embedded in the saved config, then legacy inference.
    """
    arch = checkpoint_dict.get(ARCHITECTURE_KEY) if isinstance(checkpoint_dict, dict) else None
    if arch is None:
        arch = config_data.get(ARCHITECTURE_KEY)
    if arch is None:
        arch = _infer_legacy_architecture(config_data)
        LOGGER.warning(
            f"Checkpoint has no '{ARCHITECTURE_KEY}' field; inferred {arch!r} "
            f"from config shape. Re-save the model to write an explicit field."
        )
    return arch


def _load_checkpoint_dict(checkpoint_path: Path) -> dict:
    """Load a `.pth` file and normalize it to the bundled-dict layout.

    Returns a dict with keys `{state_dict, config, architecture}`.

    Handles two on-disk formats:
      * New: the .pth itself is a dict with those keys (config bundled inside).
      * Legacy: the .pth is a raw `state_dict`; the config lives in a sibling
        `config.json` file. The sidecar is parsed here so the rest of the
        loader is format-agnostic.
    """
    data = torch.load(str(checkpoint_path), map_location="cpu")

    if isinstance(data, dict) and STATE_DICT_KEY in data:
        # New bundled format. Make sure the architecture field is populated
        # (falling back to inference) so callers get a uniform shape.
        config_data = data.get(CONFIG_KEY, {}) or {}
        arch = _resolve_architecture(data, config_data)
        return {
            STATE_DICT_KEY: data[STATE_DICT_KEY],
            CONFIG_KEY: config_data,
            ARCHITECTURE_KEY: arch,
        }

    # Legacy format: raw state_dict. Pull config from sidecar.
    sidecar = checkpoint_path.parent / "config.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Legacy checkpoint {checkpoint_path} has no sidecar config.json at {sidecar}"
        )
    with open(sidecar, "r") as f:
        config_data = json.load(f)
    arch = _resolve_architecture({}, config_data)
    return {
        STATE_DICT_KEY: data,
        CONFIG_KEY: config_data,
        ARCHITECTURE_KEY: arch,
    }


def _pad_policy_head_state_dict(
    state_dict: dict, model: AlphaZeroModel
) -> dict:
    """Zero-pad legacy policy-head tensors when the on-disk action space is smaller.

    The action space was widened from 26535 → 26537 to add two new D-train
    slots at the very end. Old checkpoints have policy-head ``other_head``
    output tensors sized for the smaller space; this helper pads them with
    zeros along the output dim so loading into the new model succeeds without
    biasing the new slot logits. Buffers registered on the head (e.g.
    ``other_indices``) are dropped from the state_dict so the freshly-built
    model keeps its own correctly-sized copy.
    """
    head = getattr(model, "policy_head", None)
    if head is None or not hasattr(head, "policy_size"):
        return state_dict

    target_num_other = getattr(head, "num_other", None)
    if target_num_other is None:
        return state_dict

    out = dict(state_dict)
    # Determine the model's per-tensor reference shapes; only pad a state_dict
    # tensor when its shape differs from the model's by exactly the action-
    # space growth on dim 0. This way we never touch inner hidden-dim layers
    # whose first dim happens to be smaller than ``num_other`` (e.g. a
    # ``Sequential``'s 256-unit input Linear).
    model_state = model.state_dict()
    keys_to_pad = [k for k in out if k.startswith("policy_head.other_head.")]
    for key in keys_to_pad:
        tensor = out[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() == 0:
            continue
        ref = model_state.get(key)
        if ref is None or ref.shape == tensor.shape:
            continue
        # Only pad along dim 0, and only when every other dim matches the
        # reference exactly. Otherwise we don't know how to align the tensor.
        if tensor.dim() != ref.dim():
            continue
        if any(tensor.shape[d] != ref.shape[d] for d in range(1, tensor.dim())):
            continue
        if tensor.shape[0] >= ref.shape[0]:
            continue
        pad_amount = ref.shape[0] - tensor.shape[0]
        zeros_shape = list(tensor.shape)
        zeros_shape[0] = pad_amount
        zeros = torch.zeros(zeros_shape, dtype=tensor.dtype, device=tensor.device)
        out[key] = torch.cat([tensor, zeros], dim=0)
        LOGGER.info(
            f"Zero-padded legacy policy-head tensor {key}: "
            f"{tensor.shape} → {out[key].shape}"
        )

    # Drop any policy-head tensor whose shape still doesn't match the model
    # (e.g. the ``other_indices`` buffer, which grows from old num_other → new
    # num_other). Those values are reconstructed at model init, so dropping
    # lets ``strict=False`` keep the fresh value.
    for key in list(out):
        if not key.startswith("policy_head."):
            continue
        ref = model_state.get(key)
        if ref is None:
            continue
        if out[key].shape != ref.shape:
            LOGGER.info(
                f"Dropping legacy policy-head tensor {key} (shape "
                f"{tuple(out[key].shape)} → {tuple(ref.shape)}); "
                f"using freshly-initialized value."
            )
            del out[key]

    return out


def _instantiate_model(checkpoint_dict: dict, checkpoint_path: str) -> AlphaZeroModel:
    """Construct a model from a normalized checkpoint dict and load its weights.

    The factory is invoked with `model_checkpoint_file=None` so the model's own
    `load_weights` (which expects a raw state_dict on disk) is skipped — the
    state_dict has already been extracted from the bundled .pth and is loaded
    explicitly here.
    """
    arch = checkpoint_dict[ARCHITECTURE_KEY]
    config_data = dict(checkpoint_dict[CONFIG_KEY] or {})

    # Force the policy_size to the current value so the model is built with the
    # new action space; legacy checkpoints have 26535 in their saved config but
    # the in-memory action space is now 26537.
    from rl18xx.agent.alphazero.action_mapper import ActionMapper
    current_policy_size = ActionMapper().action_encoding_size
    if config_data.get("policy_size", current_policy_size) != current_policy_size:
        LOGGER.info(
            f"Overriding saved policy_size={config_data.get('policy_size')} "
            f"with current action_encoding_size={current_policy_size}"
        )
    config_data["policy_size"] = current_policy_size

    try:
        factory = _ARCHITECTURE_REGISTRY[arch]
    except KeyError:
        raise ValueError(
            f"Unknown model architecture {arch!r}. Known: {sorted(_ARCHITECTURE_REGISTRY)}"
        ) from None

    model = factory(config_data, None)
    # Record where the weights came from for downstream tooling that inspects
    # the in-memory model (e.g. logging which checkpoint a worker loaded).
    model.config.model_checkpoint_file = checkpoint_path
    state_dict = _pad_policy_head_state_dict(checkpoint_dict[STATE_DICT_KEY], model)
    # ``strict=False`` lets dropped legacy buffers (e.g. ``other_indices``) be
    # served by the freshly-initialized model copy.
    model.load_state_dict(state_dict, strict=False)
    return model


def _load_model_from_config(config_data: dict, checkpoint_path: str) -> AlphaZeroModel:
    """Backwards-compatible shim used by `arena.py` to construct a model from a
    saved config dict plus a path to a `.pth` file.

    Loads the checkpoint via the unified path so callers transparently get
    both bundled and legacy formats.
    """
    normalized = _load_checkpoint_dict(Path(checkpoint_path))
    # If the caller explicitly passed a config (e.g. read from a sidecar
    # config.json), prefer it over whatever was bundled — this preserves the
    # legacy behavior where the sidecar is authoritative.
    if config_data:
        normalized = {
            STATE_DICT_KEY: normalized[STATE_DICT_KEY],
            CONFIG_KEY: config_data,
            ARCHITECTURE_KEY: _resolve_architecture({}, config_data) or normalized[ARCHITECTURE_KEY],
        }
    return _instantiate_model(normalized, checkpoint_path)


def session_name_for(model: AlphaZeroModel) -> str:
    """Filesystem session directory name for a model: `<timestamp>_<seed>`.

    Note this is NOT the same as `model.get_name()`, which prefixes the
    architecture. Callers that need to write the `current_best.json` pointer
    or otherwise reference the on-disk session directory should use this.
    """
    seed = getattr(model.config, "seed", None)
    if seed is None:
        seed = "unknown"
    return f"{model.config.timestamp}_{seed}"


def _get_session_dir(model: AlphaZeroModel, checkpoint_dir: str) -> Path:
    """Get the session directory for a model: <checkpoint_dir>/<architecture>/<timestamp>_<seed>/"""
    arch = model.architecture_name()
    return Path(checkpoint_dir) / arch / session_name_for(model)


def _find_latest_session(checkpoint_dir: str, arch: Optional[str] = None) -> Path:
    """Find the latest session directory.

    If `arch` is provided, restrict the search to `<checkpoint_dir>/<arch>/`.
    Otherwise scan all architectures.
    """
    p = Path(checkpoint_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if arch is not None:
        arch_dirs = [p / arch]
    else:
        arch_dirs = [d for d in p.iterdir() if d.is_dir()]

    latest_session = None
    for arch_dir in arch_dirs:
        if not arch_dir.is_dir():
            continue
        for session_dir in arch_dir.iterdir():
            if not session_dir.is_dir():
                continue
            if latest_session is None or session_dir.name > latest_session.name:
                latest_session = session_dir

    if latest_session is None:
        raise FileNotFoundError(f"No session directories found in {checkpoint_dir}")
    return latest_session


def _find_latest_checkpoint(session_dir: Path) -> Path:
    """Find the highest-numbered .pth checkpoint in a session directory."""
    checkpoints = sorted(
        [p for p in session_dir.glob("*.pth") if p.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {session_dir}")
    return checkpoints[-1]


def _next_checkpoint_num(session_dir: Path) -> int:
    """Get the next checkpoint number for a session."""
    existing = [p for p in session_dir.glob("*.pth") if p.stem.isdigit()]
    if not existing:
        return 1
    return max(int(p.stem) for p in existing) + 1


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: write to a temp file in the same dir, then os.replace.

    Self-play workers poll `current_best.json` concurrently with the main loop
    updating it; a non-atomic write would let a worker observe a half-written
    or empty file. `os.replace` is atomic on POSIX, so readers always see
    either the old contents or the new contents, never a mix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Place the tempfile alongside the target so the eventual rename is on the
    # same filesystem (cross-fs renames are not atomic).
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup if anything went wrong before os.replace took over.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_save_torch(path: Path, payload: dict) -> None:
    """Write a torch payload atomically: torch.save to a sibling tmp, then os.replace.

    Mirrors `_atomic_write_json`. Self-play workers may be loading the
    `current_best` checkpoint concurrently with training writing the next one;
    a non-atomic `torch.save` would let a reader open a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    # We don't need the file descriptor opened by mkstemp — torch.save will
    # open the path itself. Close it immediately to avoid leaks on Windows.
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def set_current_best(
    checkpoint_dir: str,
    arch: str,
    session: str,
    checkpoint_num: int,
) -> None:
    """Write `current_best.json` under `<checkpoint_dir>/<arch>/`.

    Records which (session, checkpoint_num) is the current best for this
    architecture. Writes are atomic so concurrent readers always see a
    well-formed file.
    """
    pointer_path = Path(checkpoint_dir) / arch / CURRENT_BEST_FILENAME
    payload = {
        "arch": arch,
        "session": session,
        "checkpoint_num": int(checkpoint_num),
    }
    _atomic_write_json(pointer_path, payload)
    LOGGER.info(
        f"Updated current_best pointer for arch={arch}: session={session}, checkpoint={checkpoint_num}"
    )


def get_current_best(checkpoint_dir: str, arch: Optional[str] = None) -> Optional[dict]:
    """Read `current_best.json` for a given arch (or auto-detect if None).

    Returns a dict with keys `{arch, session, checkpoint_num}`, or `None` if no
    pointer exists (e.g. on a fresh repo, or before the first promotion).

    If `arch` is None, scans all arch subdirectories and returns the pointer
    from the lexicographically-latest session across architectures. (Multi-arch
    coexistence is rare, but this matches the "no arch specified" semantics of
    `get_latest_model`.)
    """
    base = Path(checkpoint_dir)
    if not base.exists() or not base.is_dir():
        return None

    if arch is not None:
        pointer_path = base / arch / CURRENT_BEST_FILENAME
        if not pointer_path.exists():
            return None
        try:
            with open(pointer_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            LOGGER.warning(f"Could not read current_best pointer at {pointer_path}: {e}")
            return None

    # No arch specified — search all arch dirs and pick the pointer with the
    # most recent session string.
    candidates: list[dict] = []
    for arch_dir in base.iterdir():
        if not arch_dir.is_dir():
            continue
        pointer_path = arch_dir / CURRENT_BEST_FILENAME
        if not pointer_path.exists():
            continue
        try:
            with open(pointer_path, "r") as f:
                data = json.load(f)
                candidates.append(data)
        except (json.JSONDecodeError, IOError) as e:
            LOGGER.warning(f"Could not read current_best pointer at {pointer_path}: {e}")
            continue

    if not candidates:
        return None
    return max(candidates, key=lambda d: d.get("session", ""))


def _load_model_from_session_checkpoint(session_dir: Path, checkpoint_path: Path) -> AlphaZeroModel:
    """Shared helper: load a model from a specific session+checkpoint.

    Reads the bundled .pth (or falls back to legacy state_dict + sidecar
    config.json) and instantiates the right model class from the registry.
    """
    checkpoint_dict = _load_checkpoint_dict(checkpoint_path)
    model = _instantiate_model(checkpoint_dict, str(checkpoint_path))
    checkpoint_num = int(checkpoint_path.stem)
    LOGGER.info(
        f"Loaded model from: {session_dir.parent.name}/{session_dir.name}/checkpoint {checkpoint_num}"
    )
    return model


def get_latest_model(
    model_checkpoint_dir: str,
    arch: Optional[str] = None,
) -> AlphaZeroModel:
    """Load the current-best checkpoint, falling back to the latest scan.

    Resolution order:
      1. If `<checkpoint_dir>/<arch>/current_best.json` exists (or, when arch is
         None, any arch has a pointer), load that specific checkpoint.
      2. Otherwise, fall back to the existing behavior: pick the
         lexicographically-latest session and its highest-numbered `.pth`.

    The pointer-based path lets gating decide which checkpoint is the "best"
    without requiring on-disk artifacts to be written in chronological order
    (a rejected candidate is still saved, but the pointer doesn't move).
    """
    pointer = get_current_best(model_checkpoint_dir, arch=arch)
    if pointer is not None:
        try:
            pointer_arch = pointer["arch"]
            session = pointer["session"]
            checkpoint_num = int(pointer["checkpoint_num"])
            session_dir = Path(model_checkpoint_dir) / pointer_arch / session
            checkpoint_path = session_dir / f"{checkpoint_num}.pth"
            if checkpoint_path.exists():
                return _load_model_from_session_checkpoint(session_dir, checkpoint_path)
            LOGGER.warning(
                f"current_best pointer references missing checkpoint "
                f"({checkpoint_path}); falling back to latest-session scan."
            )
        except (KeyError, ValueError, TypeError) as e:
            LOGGER.warning(
                f"current_best pointer is malformed ({pointer!r}): {e}. "
                f"Falling back to latest-session scan."
            )

    session_dir = _find_latest_session(model_checkpoint_dir, arch=arch)
    checkpoint_path = _find_latest_checkpoint(session_dir)
    return _load_model_from_session_checkpoint(session_dir, checkpoint_path)


def save_model(model: AlphaZeroModel, model_checkpoint_dir: str) -> int:
    """Save model as the next numbered checkpoint in its session directory.

    The checkpoint is a single self-describing `.pth` file containing the
    state_dict, the serialized config, and the architecture name. Writing
    bundled-and-atomically eliminates the previous race window where self-play
    workers could read a sidecar `config.json` that had been truncated and not
    yet refilled by the trainer.

    Returns the checkpoint number that was saved.
    """
    session_dir = _get_session_dir(model, model_checkpoint_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_num = _next_checkpoint_num(session_dir)
    checkpoint_path = session_dir / f"{checkpoint_num}.pth"

    config_payload = model.config.to_json()
    # Drop the in-memory torch.device — it isn't JSON-friendly and is
    # re-derived on load anyway.
    config_payload.pop("device", None)
    config_payload[ARCHITECTURE_KEY] = model.architecture_name()

    payload = {
        STATE_DICT_KEY: model.state_dict(),
        CONFIG_KEY: config_payload,
        ARCHITECTURE_KEY: model.architecture_name(),
    }

    try:
        _atomic_save_torch(checkpoint_path, payload)
        LOGGER.info(f"Successfully saved weights to {checkpoint_path}")
    except Exception as e:
        LOGGER.error(f"Error saving checkpoint to {checkpoint_path}: {e}")
        raise

    # Also write a sidecar config.json the first time a session is saved, for
    # human readability. This is a one-shot snapshot — we never rewrite it on
    # subsequent saves, so there is no race to worry about.
    sidecar_path = session_dir / "config.json"
    if not sidecar_path.exists():
        _atomic_write_json(sidecar_path, config_payload)

    return checkpoint_num


def save_optimizer_state(
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_checkpoint_dir: str,
    model: AlphaZeroModel,
) -> None:
    """Save optimizer and scheduler state to the model's session directory."""
    session_dir = _get_session_dir(model, model_checkpoint_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "optimizer.pth"
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        path,
    )
    LOGGER.info(f"Saved optimizer state to {path}")


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_checkpoint_dir: str,
    model: AlphaZeroModel,
) -> bool:
    """Load optimizer and scheduler state from the model's session directory.

    Returns True if state was loaded, False if no saved state exists.
    """
    session_dir = _get_session_dir(model, model_checkpoint_dir)
    path = session_dir / "optimizer.pth"
    if not path.exists():
        LOGGER.info("No saved optimizer state found. Starting fresh.")
        return False

    state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    LOGGER.info(f"Loaded optimizer state from {path}")
    return True
