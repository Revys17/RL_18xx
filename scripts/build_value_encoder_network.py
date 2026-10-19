"""One network from a policy and a critic: the policy's whole network plus a value-only
entity encoder (``ModelTransformerConfig.value_entity_encoder``) copied from the critic.

The result's policy path (entity encoder, map encoder, fusion, trunk, policy and price
heads) is the policy checkpoint's, bit for bit; its value heads read
``[value encoder's player tokens, shared trunk]``: the value encoder starts as the
critic's entity encoder (or, for a critic that is itself a shared value-encoder network,
its value encoder) and the value heads as the critic's (whose trunk half now sees
the policy's trunk -- fit them, e.g. ``distill_value_on_children.py --train-only value``).
``<out>/../config.json`` carries the policy run's ``learner_temperature``.

    uv run python scripts/build_value_encoder_network.py \\
        --policy model_checkpoints_pg/az1u280_shared_20261018/learner/280.pth \\
        --critic model_checkpoints_search/az1_20261017/critic/280.pth \\
        --out model_checkpoints_pg/shared_diag_venc_20261018/learner/280.pth
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def build(policy_path: str, critic_path: str):
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
    from rl18xx.agent.alphazero.policy_gradient import _load

    policy, critic = _load(policy_path), _load(critic_path)
    config = ModelTransformerConfig.from_json({**policy.config.to_json(), "value_entity_encoder": True})
    model = AlphaZeroTransformerModel(config).to(next(policy.parameters()).device)
    missing, unexpected = model.load_state_dict(policy.state_dict(), strict=False)
    if unexpected or any(not k.startswith("value_econ_transformer.") for k in missing):
        raise RuntimeError(f"unexpected state mismatch: missing {missing}, unexpected {unexpected}")
    # A critic that is itself a shared value-encoder network (e.g. az7b) gives its value
    # encoder; a plain critic its (whole-network) entity encoder.
    source = getattr(critic, "value_econ_transformer", None)
    source = critic.econ_transformer if source is None else source
    model.value_econ_transformer.load_state_dict(source.state_dict())
    model.win_loss_head.load_state_dict(critic.win_loss_head.state_dict())
    model.score_head.load_state_dict(critic.score_head.state_dict())
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--critic", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from rl18xx.agent.alphazero import value_refit as vr

    model = build(args.policy, args.critic)
    vr.write_model_checkpoint(
        model, args.policy, Path(args.out), {"value_encoder": {"policy": args.policy, "critic": args.critic}}
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
