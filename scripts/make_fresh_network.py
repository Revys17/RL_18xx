"""A randomly initialized network, for runs that start from nothing (no human data).

    uv run python scripts/make_fresh_network.py --out <dir> [--seed 0] [--value-entity-encoder]

writes ``<dir>/learner/0.pth`` -- an ``AlphaZeroTransformer`` with the default
``ModelTransformerConfig`` (the architecture of the supervised checkpoints: ~8.0M
parameters with the plain shared trunk, ~8.3M with ``--value-entity-encoder``), its
weights as the model's own initialization draws them (``torch.manual_seed(seed)``),
saved as policy-gradient / search-learning checkpoints are (``policy_gradient._save``)
-- and ``<dir>/config.json`` with ``learner_temperature`` 1, where
``search_learning.run_temperature`` reads a policy's temperature. Start a one-network
search-learning run from it: ``main.py search-learning --shared --policy
<dir>/learner/0.pth ...`` (the inference server loads the learner alone and the
search reads its own value heads). Refuses to overwrite an existing checkpoint."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def fresh_network(seed: int = 0, value_entity_encoder: bool = False):
    import torch

    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    torch.manual_seed(seed)
    config = ModelTransformerConfig(seed=seed, value_entity_encoder=value_entity_encoder, device=torch.device("cpu"))
    return AlphaZeroTransformerModel(config)


def write_fresh(out, seed: int = 0, value_entity_encoder: bool = False) -> Path:
    from rl18xx.agent.alphazero.policy_gradient import _save

    out = Path(out)
    path = out / "learner" / "0.pth"
    if path.exists():
        raise SystemExit(f"{path} exists")
    model = fresh_network(seed, value_entity_encoder)
    _save(model, path)
    params = sum(p.numel() for p in model.parameters())
    (out / "config.json").write_text(
        json.dumps(
            {
                "learner_temperature": 1.0,
                "source": "random initialization (scripts/make_fresh_network.py)",
                "seed": seed,
                "value_entity_encoder": value_entity_encoder,
                "parameters": params,
            },
            indent=1,
        )
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Directory for learner/0.pth and config.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-entity-encoder", action="store_true", help="The value side's own entity encoder")
    args = parser.parse_args()
    path = write_fresh(args.out, args.seed, args.value_entity_encoder)
    print(f"{path}: {json.loads((path.parent.parent / 'config.json').read_text())['parameters']} parameters")


if __name__ == "__main__":
    main()
