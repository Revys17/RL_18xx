"""The systems the probes load, by name: ``gen6`` (the live bots), ``u1528`` (A4 u1528 with its own critic) and
``ca<N>`` (A4 u1528's policy with the critic of ``u1528_critic_all_20261125`` at update N: retrained on every seat's
decisions, 2026-11-25). ``ONLY=a,b`` picks the systems of the probes that compare several (default gen6,u1528);
``EXTRA_SPECS="name=spec;..."`` adds other eval_head_to_head-style player specs."""
import os

LR1 = "/home/revys/longrun/work/lr1/model_checkpoints_pg/gen6_20261110"
A4 = "/home/revys/workspace/RL_18xx/model_checkpoints_generic_pg/robust_A4_20261118"
CA = "/home/revys/workspace/RL_18xx/model_checkpoints_generic_pg/u1528_critic_all_20261125"
SPECS = {
    "gen6": f"{LR1}/learner/2400.pth+{LR1}/critic/2400.pth#score",
    "u1528": f"generic:{A4}/learner/1528.pth+generic:{A4}/critic/1528.pth#score",
}
for _item in filter(None, os.environ.get("EXTRA_SPECS", "").split(";")):
    _name, _spec = _item.split("=", 1)
    SPECS[_name] = _spec


def spec(name: str) -> str:
    if name.startswith("ca") and name[2:].isdigit():
        return f"generic:{A4}/learner/1528.pth+generic:{CA}/critic/{name[2:]}.pth#score"
    return SPECS[name]


def compared() -> dict:
    """The systems a comparing probe loads: ``ONLY`` or gen6 and u1528."""
    names = os.environ.get("ONLY", "gen6,u1528").split(",")
    return {name: spec(name) for name in names}
