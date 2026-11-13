"""unpin_edit.py <population json> <state json>: drops every pin and the 8 round-2 president exploiters (restart3)."""
import json, os, sys

RETIRE = [f"model_checkpoints_pg/gx_{c}_20261010/learner/200.pth" for c in ("prr", "nyc", "cpr", "bo", "co", "erie", "nynh", "bm")]
pop_path, state_path = sys.argv[1], sys.argv[2]
pop = json.load(open(pop_path))
kept = []
for m in pop["members"]:
    path = m["path"] if isinstance(m, dict) else m
    if path in RETIRE:
        continue
    if isinstance(m, dict):
        m.pop("min_share", None); m.pop("min_share_players", None)
    kept.append(m)
assert len(pop["members"]) - len(kept) == len(RETIRE), "a president exploiter is missing from the population"
pop["members"] = kept
state = json.load(open(state_path))
for n, lg in state["league"]["by_count"].items():
    for path in RETIRE:
        assert lg["stats"].pop(path, None) is not None, (n, path)
    lg["min_shares"] = {}
for path, data in ((pop_path, pop), (state_path, state)):
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=2)
    os.replace(path + ".tmp", path)
print(f"population {len(kept)} members, no pins; retired {len(RETIRE)} from the file and every count's stats")
