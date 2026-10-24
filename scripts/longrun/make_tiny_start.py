"""A tiny policy + critic for dryrun.sh: a CPU search-learning run that updates every few seconds.

    <deployment>/.venv/bin/python <deployment>/scripts/longrun/make_tiny_start.py <dir>

Writes <dir>/learner/0.pth, <dir>/critic/0.pth (random transformers, d=32, one layer each) and
<dir>/config.json (learner_temperature 0.5, which search-learning reads from a --policy run).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from rl18xx.agent.alphazero.config import ModelTransformerConfig  # noqa: E402
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel  # noqa: E402
from rl18xx.agent.alphazero.policy_gradient import _save  # noqa: E402


def tiny(seed: int) -> AlphaZeroTransformerModel:
    torch.manual_seed(seed)
    config = ModelTransformerConfig(
        device=torch.device("cpu"), d_entity=32, econ_transformer_layers=1, econ_transformer_heads=2,
        econ_transformer_ff_dim=64, d_map=32, hex_transformer_layers=1, hex_transformer_heads=2,
        hex_transformer_ff_dim=64, cross_attn_heads=2, d_trunk=64, num_res_blocks=1, value_head_layers=1,
    )
    return AlphaZeroTransformerModel(config)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    _save(tiny(0), out / "learner" / "0.pth")
    _save(tiny(1), out / "critic" / "0.pth")
    (out / "config.json").write_text(json.dumps({"learner_temperature": 0.5}))
    print(out)


if __name__ == "__main__":
    main()
