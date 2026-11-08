"""A randomly initialized network, for runs that start from nothing (no human data).

    uv run python scripts/make_fresh_network.py --out <dir> [--seed 0] [--value-entity-encoder]

writes ``<dir>/learner/0.pth`` -- an ``AlphaZeroTransformer`` with the default
``ModelTransformerConfig`` (the architecture of the supervised checkpoints: ~8.0M
parameters with the plain shared trunk, ~8.3M with ``--value-entity-encoder``), its
weights as the model's own initialization draws them (``torch.manual_seed(seed)``)
with the policy's output layers zeroed (``zero_policy_outputs``), saved as
policy-gradient / search-learning checkpoints are (``policy_gradient._save``)
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
    model = AlphaZeroTransformerModel(config)
    zero_policy_outputs(model)
    return model


def zero_policy_outputs(model) -> None:
    """Zero the policy's output layers, as the model's own initialization already does for
    its value, score and price heads. The trunk isn't normalized, so a Kaiming output layer
    starts with logits of std ~4: a confident random policy (2026-11-08 audit: Pass at
    1 - 1e-8 or more at track and token decisions, so a search put 3-6% of its visits on
    tile lays). Zeroed, every sub-head starts uniform: over the other moves, over hexes,
    tiles given a hex, rotations, token hexes and slots. The three blocks (lay track, place
    a token, everything else) are each log-softmaxed on their own, so between blocks a
    move's prior is one over its block's size: Pass ~60% against 20 legal lays (1/841
    against 20 x 1/25,668), a token 30x Pass (1/28). Search and root noise spread visits from
    there. The weights below the output layers get gradient from the first update on."""
    import torch

    head = model.policy_head
    with torch.no_grad():
        outputs = (head.hex_scorer, head.tile_head, head.rotation_head, head.place_token_slot_head)
        for layer in [module[-1] for module in outputs] + [head.other_head]:
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()


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
                "policy_outputs": "zeroed",
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
