"""The title-generic policy / value model (``model_generic``, RL bridge milestone B2) on real positions:
random walks of 1830 (2-6 players; legal indices and descriptors from the 1830 layout) and 1867
(3-6 players; legal indices and descriptors from the 1867 layout ``1867-v1`` of milestone A2, prices on
each move's step grid).

Checked: shapes over mixed titles and player counts in one batch; probabilities sum to 1 over each
position's legal actions (and the map factors inside each group); invariance to the order of the legal
rows; padding is inert (a position alone vs in a mixed batch, plain vs de-duplicated maps); forward /
backward of every loss, under bf16 autocast too; the value-only path; the price head's distribution over
every legal price; checkpoint round trip and refusals; a tiny overfit run (a fixed action and a fixed
price cell). ``-m benchmark`` times the forward pass against the 1830 model on the same positions.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero.generic_tokens import encode_batch, raw_game, schema, spec_info  # noqa: E402
from rl18xx.agent.alphazero.titles import new_game  # noqa: E402
from scripts.g1867_fuzz import pick as pick_1867  # noqa: E402

V = mg.vocab()
FAM = {name: i for i, name in enumerate(V["families"])}

SMALL = dict(
    d_model=32,
    num_heads=4,
    hex_layers=1,
    entity_layers=1,
    fusion_layers=1,
    d_trunk=64,
    trunk_blocks=1,
    d_head=32,
    d_edge=8,
    d_cond=16,
    d_id=8,
)


# ---------------------------------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------------------------------


def walk_1830(num_players: int, seed: int, steps: int = 700, every: int = 4) -> list:
    rng = random.Random(seed)
    game = raw_game(new_game("1830", num_players, auction_unlock=True))
    positions = []
    for step in range(steps):
        if game.finished:
            break
        legal = [int(i) for i in game.factored_legal_indices()]
        if not legal:
            break
        if step % every == 0:
            positions.append(game.pickle_clone())
        index = rng.choice(legal)
        price_range = game.price_range_for_index(index)
        game.apply_action_index(index, rng.randint(*price_range) if price_range else None)
    return positions


def walk_1867(num_players: int, seed: int, steps: int = 700, every: int = 4) -> list:
    rng = random.Random(seed * 1_000_003 + num_players)
    game = new_game("1867", num_players, seed=seed)
    positions = []
    for step in range(steps):
        if game.finished:
            break
        moves = json.loads(game.g1867_legal_moves_json(None))
        if not moves:
            break
        if step % every == 0:
            positions.append(game.pickle_clone())
        game.process_action(pick_1867(moves, rng))
    return positions


def legal_of(game) -> mg.LegalActions:
    return mg.legal_actions(game)


def _features(la: mg.LegalActions) -> set:
    fam = la.rows[:, mg.C_FAMILY]
    out = set()
    if (fam == FAM["LayTile"]).any():
        out.add("lay")
    if np.isin(fam, [FAM["PlaceToken"], FAM["RemoveToken"]]).any():
        out.add("token")
    if (la.rows[:, mg.C_PRICE_KIND] >= 0).any():
        out.add("price")
        out |= {f"price:{V['price_kinds'][k]}" for k in np.unique(la.rows[:, mg.C_PRICE_KIND]) if k >= 0}
    if len(la) > 1:
        out.add("choice")
    return out


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    """Small models on a shared, loaded machine: torch's default intra-op pool (one thread per core)
    spends its time synchronizing; a few threads run these tests several times faster."""
    before = torch.get_num_threads()
    torch.set_num_threads(min(before, 4))
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def walks() -> dict:
    out = {}
    for n in (2, 3, 4, 5, 6):
        games = walk_1830(n, seed=n)
        out[("1830", n)] = [(g, legal_of(g)) for g in games]
    for n in (3, 4, 5, 6):
        games = walk_1867(n, seed=n)
        out[("1867", n)] = [(g, legal_of(g)) for g in games]
    return out


def diverse(items: list, k: int) -> list:
    """Up to ``k`` positions covering tile lays, tokens, each price kind and plain choices first."""
    chosen, covered = [], set()
    for want in ("lay", "token", "price:bid", "price:train_transfer", "price:company_purchase", "choice"):
        for item in items:
            f = _features(item[1])
            if want in f and want not in covered and item not in chosen:
                chosen.append(item)
                covered |= f
                break
    for item in items:
        if len(chosen) >= k:
            break
        if item not in chosen and len(item[1]):
            chosen.append(item)
    return chosen[:k]


@pytest.fixture(scope="module")
def mixed(walks) -> list:
    """A batch over both titles and every player count, tile lays, tokens and prices included."""
    items = []
    for key in sorted(walks):
        items += diverse(walks[key], 5)
    return items


def small_model(seed: int = 0, **overrides) -> mg.GenericPolicyValueModel:
    torch.manual_seed(seed)
    model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**{**SMALL, **overrides}))
    # non-zero value / price outputs (their last layers start at zero)
    for head in (model.win_head[-1], model.score_head[-1], model.price_mlp[-1]):
        torch.nn.init.normal_(head.weight, std=0.3)
    return model.eval()


def inputs_for(items: list, dedupe_maps: bool = False, model=None) -> tuple[dict, dict]:
    games = [g for g, _ in items]
    legal = [la for _, la in items]
    arrays = mg.build_inputs(games, legal, dedupe_maps=dedupe_maps, config=None if model is None else model.config)
    return arrays, mg.to_device(arrays)


def per_position(arrays: dict, values: torch.Tensor) -> list:
    off = arrays["legal_offsets"]
    return [values[off[i] : off[i + 1]] for i in range(len(off) - 1)]


def price_per_position(arrays: dict, values: torch.Tensor) -> list:
    pos = arrays["legal_pos"][arrays["price_rows"]]
    return [values[torch.from_numpy(np.flatnonzero(pos == i))] for i in range(len(arrays["legal_offsets"]) - 1)]


# ---------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------


def test_default_size():
    model = mg.GenericPolicyValueModel()
    n = model.num_parameters()
    print(f"\ndefault GenericPolicyValueModel: {n:,} parameters")
    assert 7_000_000 < n < 10_000_000


def test_static_hex_bias_is_the_per_map_lookup():
    """The distance / direction bias computed once per title and gathered equals the per-map embedding
    lookup bit for bit, and so do the embeddings' gradients."""
    model = small_model()
    for emb in (model.dist_bias, model.dir_bias):
        torch.nn.init.normal_(emb.weight)
    titles = torch.tensor([1, 2, 1, 1, 2, 2, 1])
    h = int(mg.title_tables()["h_max"])
    bias = model._static_hex_bias(titles, h)
    dist = model.t_hex_dist[:, :h, :h][titles]
    direction = model.t_hex_dir[:, :h, :h][titles]
    ref = (model.dist_bias(dist) + model.dir_bias(direction)).permute(0, 3, 1, 2)
    assert bias.shape == ref.shape and torch.equal(bias, ref)
    weight = torch.randn_like(ref)
    grads = []
    for out in (bias, ref):
        model.zero_grad()
        (out * weight).sum().backward(retain_graph=True)
        grads.append([model.dist_bias.weight.grad.clone(), model.dir_bias.weight.grad.clone()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5 * float(b.abs().max()))  # fp32 summation order


def test_tile_exits_match_laid_tiles(walks):
    """The rotation convention of ``title_tables()["tile_exits"]``: a laid tile's track edges (the
    encoder writes them rotated) are the catalog tile's at rotation 0 rotated by the hex's rotation."""
    names = {n: k for k, n in enumerate(schema()["hex_features"])}
    edge_track = mg._hex_feature_columns()["edge_track"]
    tables = mg.title_tables()
    checked = 0
    for (title, n), items in walks.items():
        game = items[-1][0]
        arrays = encode_batch([game])
        hf, ids = arrays["hex_feat"][0], arrays["hex_ids"][0]
        t = arrays["meta"][0, 0]
        for h in range(spec_info(title)["num_hexes"]):
            tile = ids[h, 1] - 1
            if hf[h, names["tile_laid"]] == 0 or not 0 <= tile < tables["num_tiles"][t]:
                continue
            rot = int(np.argmax(hf[h, names["rot_0"] : names["rot_0"] + 6]))
            expect = (hf[h] @ edge_track > 0).astype(np.float32)
            assert np.array_equal(tables["tile_exits"][t, tile, rot], expect), (title, h, tile, rot)
            checked += 1
    assert checked > 50
    for title in ("1830", "1867"):
        nbr = np.asarray(spec_info(title)["hex_neighbors"])
        for h, k in zip(*np.nonzero(nbr >= 0)):
            assert nbr[nbr[h, k], (k + 3) % 6] == h


def test_mixed_batch_shapes_and_normalization(mixed):
    model = small_model()
    arrays, inputs = inputs_for(mixed, model=model)
    b = len(mixed)
    titles = {g.title for g, _ in mixed}
    counts = {(g.title, len(g.players)) for g, _ in mixed}
    assert titles == {"1830", "1867"} and len(counts) == 9
    assert len(arrays["map_rows"]) and len(arrays["price_rows"])
    with torch.no_grad():
        out = model(inputs)
    n = len(arrays["legal_rows"])
    assert out["logits"].shape == out["logp"].shape == (n,)
    assert out["win_logits"].shape == out["score"].shape == (b, 6)
    assert out["price_logp"].shape == (len(arrays["price_rows"]), mg.NUM_CELLS)
    for key in ("logits", "logp", "price_logp", "win_logits", "score", "trunk"):
        assert torch.isfinite(out[key]).all(), key
    sums = torch.zeros(b, dtype=torch.float64).index_add(0, inputs["legal_pos"], out["logp"].double().exp())
    assert torch.allclose(sums, torch.ones(b, dtype=torch.float64), atol=1e-5)
    counts = torch.from_numpy(np.diff(arrays["legal_offsets"]))
    entropy = mg.policy_entropy(out, inputs)
    assert (entropy > -1e-6).all() and (entropy <= torch.log(counts.float()) + 1e-5).all()
    last = counts - 1
    assert torch.equal(mg.chosen_log_prob(out, inputs, last), out["logp"][inputs["legal_offsets"][1:] - 1])
    # price cells: a distribution over the range's non-empty cells, ~0 on empty ones
    p = out["price_logp"].exp()
    mask = inputs["price_cell_mask"]
    assert torch.allclose(p.sum(-1), torch.ones(len(p)), atol=1e-5)
    assert (p[~mask] < 1e-6).all()
    # value: win shares over present seats only, score 0 on absent seats
    win = torch.softmax(out["win_logits"], -1)
    players = torch.tensor([len(g.players) for g, _ in mixed])
    present = torch.arange(6)[None] < players[:, None]
    assert torch.equal(out["present"], present)
    assert torch.allclose(win.sum(-1), torch.ones(b), atol=1e-5)
    assert (win[~present] == 0).all() and (out["score"][~present] == 0).all()


def test_map_factors_are_distributions(mixed):
    """Inside each map group: P(hex) sums to 1 over its legal hexes, P(tile | city | hex) over each
    hex's legal values, P(rot | ...) over each (hex, tile)'s legal rotations; a map row's logit is
    its group score plus the three log factors."""
    model = small_model(1)
    arrays, inputs = inputs_for(mixed)
    with torch.no_grad():
        out = model(inputs)
    comp = out["policy_components"]
    k1_group, k2_k1, map_k2 = inputs["k1_group"], inputs["k2_k1"], inputs["map_k2"]
    g = int(k1_group.max()) + 1
    assert torch.allclose(torch.zeros(g).index_add(0, k1_group, comp["log_p_hex"].exp()), torch.ones(g), atol=1e-5)
    k1 = len(inputs["k1_rep"])
    assert torch.allclose(torch.zeros(k1).index_add(0, k2_k1, comp["log_p_second"].exp()), torch.ones(k1), atol=1e-5)
    k2 = len(inputs["k2_rep"])
    assert torch.allclose(torch.zeros(k2).index_add(0, map_k2, comp["log_p_rot"].exp()), torch.ones(k2), atol=1e-5)
    # group mass: the rows of a group share the group score; their factors sum to 1
    mr = inputs["map_rows"]
    lay_rows = (inputs["legal_rows"][mr, mg.C_FAMILY] == FAM["LayTile"]).sum()
    assert lay_rows > 0
    within = comp["log_p_hex"][inputs["map_k1"]] + comp["log_p_second"][map_k2] + comp["log_p_rot"]
    group_of_row = k1_group[inputs["map_k1"]]
    assert torch.allclose(torch.zeros(g).index_add(0, group_of_row, within.exp()), torch.ones(g), atol=1e-5)


def test_legal_order_invariance(mixed):
    model = small_model(2)
    rng = np.random.default_rng(0)
    shuffled, perms = [], []
    for game, la in mixed:
        perm = rng.permutation(len(la))
        perms.append(perm)
        shuffled.append((game, mg.LegalActions(rows=la.rows[perm], price_range=la.price_range[perm])))
    a_arrays, a_inputs = inputs_for(mixed)
    b_arrays, b_inputs = inputs_for(shuffled)
    with torch.no_grad():
        a, b = model(a_inputs), model(b_inputs)
    a_logp, b_logp = per_position(a_arrays, a["logp"]), per_position(b_arrays, b["logp"])
    a_price, b_price = price_per_position(a_arrays, a["price_logp"]), price_per_position(b_arrays, b["price_logp"])
    for i, perm in enumerate(perms):
        assert torch.allclose(a_logp[i][torch.from_numpy(perm)], b_logp[i], atol=1e-5)
        priced = np.flatnonzero(mixed[i][1].rows[:, mg.C_PRICE_KIND] >= 0)
        order = np.argsort(np.argsort(perm)[priced])  # priced rows in shuffled order
        assert torch.allclose(a_price[i][torch.from_numpy(order)], b_price[i], atol=1e-5)
    assert torch.allclose(a["win_logits"], b["win_logits"]) and torch.allclose(a["score"], b["score"])


@pytest.mark.parametrize("dedupe", [False, True])
def test_padding_is_inert(mixed, dedupe):
    model = small_model(3)
    arrays, inputs = inputs_for(mixed + mixed[:3], dedupe_maps=dedupe)  # repeated maps for the dedupe
    if dedupe:
        assert len(arrays["hex_feat"]) < len(mixed) + 3
    with torch.no_grad():
        batched = model(inputs)
    logp = per_position(arrays, batched["logp"])
    price = price_per_position(arrays, batched["price_logp"])
    for i, item in enumerate(mixed[::3]):
        i *= 3
        alone_arrays, alone_inputs = inputs_for([item])
        with torch.no_grad():
            alone = model(alone_inputs)
        assert torch.allclose(alone["logp"], logp[i], atol=2e-4), i
        assert torch.allclose(alone["price_logp"], price[i], atol=2e-4), i
        assert torch.allclose(alone["win_logits"][0], batched["win_logits"][i], atol=2e-4)
        assert torch.allclose(alone["score"][0], batched["score"][i], atol=2e-4)


def _targets(arrays: dict, inputs: dict, seed: int = 0):
    rng = np.random.default_rng(seed)
    counts = np.diff(arrays["legal_offsets"])
    index = torch.from_numpy(np.array([rng.integers(c) if c else -1 for c in counts], dtype=np.int64))
    probs = np.zeros(len(arrays["legal_rows"]), dtype=np.float32)
    for i, c in enumerate(counts):
        if c:
            w = rng.random(c).astype(np.float32)
            probs[arrays["legal_offsets"][i] : arrays["legal_offsets"][i + 1]] = w / w.sum()
    prices = []
    for (lo, hi, step), grid in zip(arrays["price_range"], arrays["price_grid"]):
        unit = int(step) if grid else 1
        prices.append(int(lo + unit * rng.integers((hi - lo) // unit + 1)))
    pt = mg.to_device(mg.price_targets(arrays, np.array(prices, dtype=np.int64)))
    b = len(counts)
    players = torch.from_numpy(arrays["meta"][:, 4])
    present = torch.arange(6)[None] < players[:, None]
    win = present.float() / present.float().sum(-1, keepdim=True)
    score = torch.rand(b, 6) * present.float()
    return index, torch.from_numpy(probs), pt, win, score


def _all_losses(model, arrays, inputs, seed=0):
    out = model(inputs)
    index, probs, pt, win, score = _targets(arrays, inputs, seed)
    v = mg.value_losses(out, win, score)
    losses = {
        "policy_index": mg.policy_loss(out, inputs, target_index=index),
        "policy_probs": mg.policy_loss(out, inputs, target_probs=probs),
        "price": mg.price_loss(out, pt["price_cell"], pt["price_log_count"], pt["price_valid"]),
        "win": v["ce"],
        "score": v["mse"],
    }
    return out, losses


@pytest.mark.parametrize("autocast", [False, True])
def test_forward_backward(mixed, autocast):
    model = small_model(4).train()
    arrays, inputs = inputs_for(mixed)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out, losses = _all_losses(model, arrays, inputs)
    for name, loss in losses.items():
        assert torch.isfinite(loss) and loss.item() > 0, name
    assert losses["price"].item() > 0
    sum(losses.values()).backward()
    sums = torch.zeros(len(mixed), dtype=torch.float64).index_add(0, inputs["legal_pos"], out["logp"].double().exp())
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)  # heads stay fp32 under autocast
    for key in ("entities", "hexes", "trunk", "win_logits", "score", "price_logp"):
        assert torch.isfinite(out[key]).all(), key  # padded rows included
    no_grad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            no_grad.append(name)
            continue
        assert torch.isfinite(p.grad).all(), name
    # only the input projections of entity types no 1830 / 1867 position has (1822's concessions and
    # minor proxies) get no gradient
    present = set(np.unique(arrays["ent_type"]).tolist())
    absent = [t for t in range(1, len(V["entity_types"])) if t not in present]
    assert {V["entity_types"][t] for t in absent} == {"concession", "minor_proxy"}
    assert sorted(no_grad) == sorted(f"ent_in.{t - 1}.{w}" for t in absent for w in ("weight", "bias")), no_grad


def test_value_only_matches_the_full_forward(mixed):
    model = small_model(5)
    arrays, inputs = inputs_for(mixed)
    with torch.no_grad():
        full = model(inputs)
        value = model(inputs, value_only=True)
        plain = model(mg.to_device(mg.build_inputs([g for g, _ in mixed])))  # no legal rows at all
    for out in (value, plain):
        assert out["logp"] is None and out["price_logp"] is None
        assert torch.allclose(out["win_logits"], full["win_logits"], atol=1e-5)
        assert torch.allclose(out["score"], full["score"], atol=1e-5)


def test_edge_term_matches_the_reference(mixed):
    """The rotation factor's edge term, projected per hex before the lay rows gather it and computed on
    lay rows only, equals B2's original (projection after the gather, every map row): values, the policy
    and the gradients."""
    model = small_model(6)
    for p in (model.edge_tok.bias, model.edge_raw.bias):  # an off-map neighbor reads the bias: make it count
        torch.nn.init.normal_(p, std=0.5)
    arrays, inputs = inputs_for(mixed)
    assert (arrays["legal_rows"][:, mg.C_FAMILY] == FAM["LayTile"]).sum() > 20
    outs, grads = [], []
    for impl in (mg.GenericPolicyValueModel._edge_term, mg.GenericPolicyValueModel._edge_term_reference):
        model.zero_grad()
        model._edge_term = impl.__get__(model)
        out = model(inputs)
        out["logp"].sum().backward()
        outs.append(out["logp"].detach())
        grads.append([p.grad.clone() for p in (model.edge_tok.weight, model.edge_tok.bias, model.edge_dir)])
    del model._edge_term
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    for a, b in zip(*grads):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)
    # the term itself, every map row (0 for non-lays)
    with torch.no_grad():
        out = model(inputs)
        hexes = out["hexes"]
        _, _, _, edge_raw = model._map_encoder(inputs, inputs["meta"][:, 0], inputs["ent_type"].shape[1])
        mi = torch.arange(len(mixed))
        new = model._edge_term(inputs, hexes, edge_raw, inputs["map_rows"], mi)
        ref = model._edge_term_reference(inputs, hexes, edge_raw, inputs["map_rows"], mi)
    assert new.shape == ref.shape and torch.allclose(new, ref, atol=1e-5)


def test_checkpointed_layers_match(mixed):
    """Gradient checkpointing of the attention blocks (``model.checkpoint_layers``): the same outputs and
    gradients as storing the activations."""
    model = small_model(8).train()
    arrays, inputs = inputs_for(mixed)
    results = []
    for checkpoint in (False, True):
        model.checkpoint_layers = checkpoint
        model.zero_grad()
        torch.manual_seed(0)  # _targets draws the score targets from torch's generator
        out, losses = _all_losses(model, arrays, inputs)
        sum(losses.values()).backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        results.append((out["logp"].detach(), out["win_logits"].detach(), grads))
    (lp0, w0, g0), (lp1, w1, g1) = results
    assert torch.equal(lp0, lp1) and torch.equal(w0, w1)
    assert g0.keys() == g1.keys()
    # the recomputed blocks' gradients may accumulate in another order: fp32 rounding only
    for name in g0:
        assert torch.allclose(g0[name], g1[name], rtol=1e-4, atol=1e-6), name


def walk_1822_family(title: str, num_players: int, seed: int, steps: int = 240, every: int = 12) -> list:
    from scripts.g1822_fuzz import pick as pick_1822

    rng = random.Random(seed * 1_000_003 + num_players)
    ids = {1000 + i + 1: f"Player {i + 1}" for i in range(num_players)}
    game = engine_rs.BaseGame.new_titled(title, ids, None, seed)
    positions = []
    for step in range(steps):
        if game.finished:
            break
        moves = json.loads(game.g1822_legal_moves_json(None))
        if not moves:
            break
        if step % every == 0:
            positions.append(game.pickle_clone())
        game.process_action(dict(pick_1822(moves, rng)))
    return positions


def stub_lays(game, n: int, seed: int) -> mg.LegalActions:
    """A well-formed stub legal list for a title without an RL layout yet: ``n`` tile lays (random hexes,
    tiles and rotations of the title's tables) and a pass."""
    rng = random.Random(seed)
    info = spec_info(game.title)
    rows = [
        [FAM["LayTile"], mg.EK_CORPORATION, 0, -1, -1, -1, rng.randrange(info["num_hexes"]), -1,
         rng.randrange(info["num_tiles"]), rng.randrange(6), -1, -1, -1, 0]
        for _ in range(n)
    ]  # fmt: skip
    rows = np.unique(np.asarray(rows, dtype=np.int32), axis=0).tolist() + [[FAM["Pass"]] + [-1] * 12 + [0]]
    rows = np.asarray(rows, dtype=np.int32)
    return mg.LegalActions(rows=rows, price_range=np.full((len(rows), 3), -1, dtype=np.int64))


def test_g1822_family_positions(mixed):
    """1822, 1822PNW, 1822MX and 1822CA positions (encoder v2; stub tile-lay lists until the family's
    layouts exist) in one batch with 1830 / 1867 ones: finite outputs and gradients, probabilities sum to
    1, every entity type's projection trained, and a family position alone gives the same values as in
    the padded mixed batch."""
    family = []
    for k, title in enumerate(("1822", "1822PNW", "1822MX", "1822CA")):
        for n in (3, 5):
            games = walk_1822_family(title, n, seed=k + n)
            family += [(g, stub_lays(g, 40, seed=i)) for i, g in enumerate(games[-3:])]
    items = mixed[:3] + mixed[-3:] + family
    model = small_model(7).train()
    arrays, inputs = inputs_for(items, model=model)
    assert set(np.unique(arrays["meta"][:, 0])) == {1, 2, 3, 4, 5, 6}
    out, losses = _all_losses(model, arrays, inputs)
    for name, loss in losses.items():
        assert torch.isfinite(loss), name
    sum(losses.values()).backward()
    sums = torch.zeros(len(items), dtype=torch.float64).index_add(0, inputs["legal_pos"], out["logp"].double().exp())
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    for key in ("entities", "hexes", "trunk", "win_logits", "score"):
        assert torch.isfinite(out[key]).all(), key
    for name, p in model.named_parameters():
        assert p.grad is not None, name  # concessions and minor proxies included now
        assert torch.isfinite(p.grad).all(), name
    model.eval()
    with torch.no_grad():
        batch = model(inputs, value_only=True)
        for i in (6, len(items) - 1):  # an 1822 and an 1822CA position alone
            alone = model(mg.to_device(mg.build_inputs([items[i][0]])), value_only=True)
            assert torch.allclose(alone["win_logits"][0], batch["win_logits"][i], atol=1e-4)


def test_price_head_covers_every_legal_price(mixed):
    """``P(price) = P(cell) / |cell|`` sums to 1 over every integer of each priced row's legal range,
    and the price loss is ``-log P(price)``."""
    model = small_model(6)
    arrays, inputs = inputs_for(mixed)
    with torch.no_grad():
        out = model(inputs)
    logp = out["price_logp"].double()
    kinds = set()
    for j, i in enumerate(arrays["price_rows"]):
        lo, hi, step = (int(x) for x in arrays["price_range"][j])
        kind = int(arrays["legal_rows"][i, mg.C_PRICE_KIND])
        kinds.add(V["price_kinds"][kind])
        grid = bool(arrays["price_grid"][j])
        cells = mg.price_cells_for(kind, lo, hi, step, grid=grid)
        legal_prices = range(lo, hi + 1, step if grid else 1)  # 1867: the move's step grid only
        total = sum(math_exp(logp[j, cells.cell_of(p)].item()) / cells.counts[cells.cell_of(p)] for p in legal_prices)
        assert abs(total - 1.0) < 1e-6, (lo, hi)
        if hi > lo:
            price = legal_prices[len(legal_prices) // 2]
            pt = mg.price_targets(
                {
                    **arrays,
                    "price_rows": arrays["price_rows"][j : j + 1],
                    "price_range": arrays["price_range"][j : j + 1],
                    "price_grid": arrays["price_grid"][j : j + 1],
                },
                np.array([price]),
            )
            single = {"price_logp": out["price_logp"][j : j + 1], "trunk": out["trunk"]}
            loss = mg.price_loss(
                single, *(torch.from_numpy(pt[k]) for k in ("price_cell", "price_log_count", "price_valid"))
            )
            c = cells.cell_of(price)
            assert abs(loss.item() - (-(logp[j, c].item() - np.log(cells.counts[c])))) < 1e-4
    assert {"bid", "train_transfer", "company_purchase"} <= kinds


def math_exp(x: float) -> float:
    return float(np.exp(x))


def test_build_inputs_checks_vocab(mixed):
    model = small_model(7)
    arrays, _ = inputs_for(mixed[:2])
    bad = {**arrays, "legal_rows": arrays["legal_rows"].copy()}
    bad["legal_rows"][0, mg.C_FAMILY] = model.config.num_families
    with pytest.raises(ValueError, match="family"):
        mg.check_inputs(model.config, bad)
    bad = {**arrays, "legal_rows": arrays["legal_rows"].copy()}
    bad["legal_rows"][0, mg.C_PRICE_KIND] = len(mg.vocab()["price_kinds"])
    with pytest.raises(ValueError, match="price kind"):
        mg.check_inputs(model.config, bad)


def test_amount_prices_have_cells(mixed):
    """``amount`` (the 1822 family's partial debt payoff, ``docs/rl_descriptors.md`` §9 price census)
    prices through "BuyTrain" cells: atoms at the lowest payment, the highest (everything the player
    can pay) and one below it, plus 20 bins, in step units on a generator layout's grid."""
    v = mg.vocab()
    amount = v["price_kinds"].index("amount")
    assert v["price_cell_types"][amount] == "BuyTrain"
    cells = mg.price_cells_for(amount, 1, 160, 1, grid=True)
    assert int(cells.counts.sum()) == 160
    for p in (1, 159, 160):
        assert cells.counts[cells.cell_of(p)] == 1, p
    la = mg.LegalActions(
        rows=np.array([[0] * 12 + [amount, 0]], dtype=np.int32), price_range=np.array([[1, 160, 1]]), price_grid=True
    )
    arrays = mg.build_inputs([mixed[0][0]], [la])
    assert arrays["price_rows"].tolist() == [0]
    assert (arrays["price_cell_mask"][0] == cells.nonempty).all()


def test_checkpoint_round_trip_and_refusals(mixed, tmp_path):
    model = small_model(8)
    _, inputs = inputs_for(mixed[:6])
    path = tmp_path / "generic.pth"
    mg.save_checkpoint(model, path, titles=["1830", "1867"], extra={"note": "test"})
    loaded, meta = mg.load_checkpoint(path)
    assert meta["encoder_version"] == schema()["version"]
    assert meta["layouts"]["1830"]["fingerprint"] == "e721aa6520116e2a"
    assert meta["layouts"]["1867"]["fingerprint"] == "4987cd19b7e912e2"  # 1867-v1
    with torch.no_grad():
        a, b = model(inputs), loaded.eval()(inputs)
    for key in ("logp", "price_logp", "win_logits", "score"):
        assert torch.equal(a[key], b[key]), key

    def tampered(edit):
        payload = torch.load(path, weights_only=True)
        edit(payload)
        bad = tmp_path / "bad.pth"
        torch.save(payload, bad)
        return bad

    cases = {
        "encoder version": lambda p: p["generic_meta"].__setitem__(
            "encoder_version", p["generic_meta"]["encoder_version"] + 1
        ),
        "layout": lambda p: p["generic_meta"]["layouts"]["1830"].__setitem__("fingerprint", "0" * 16),
        "vocabulary": lambda p: p["generic_meta"]["vocab"].__setitem__("families", ["BuyShares", "Pass"]),
        "architecture": lambda p: p.__setitem__("architecture", "AlphaZeroTransformer"),
    }
    for name, edit in cases.items():
        with pytest.raises(mg.CheckpointMismatch):
            mg.load_checkpoint(tampered(edit))
    # an appended vocabulary entry is fine (the saved list is a prefix of the current one)
    ok = tampered(
        lambda p: p["generic_meta"]["vocab"].__setitem__("families", p["generic_meta"]["vocab"]["families"][:-1])
    )
    mg.load_checkpoint(ok)
    # a checkpoint written while ``amount`` had no cells loads (it never priced that kind); a kind
    # whose cells changed does not
    amount = mg.vocab()["price_kinds"].index("amount")

    def cells_were(value):
        def edit(p):
            p["generic_meta"]["vocab"]["price_cell_types"][amount] = value

        return edit

    mg.load_checkpoint(tampered(cells_were(None)))
    with pytest.raises(mg.CheckpointMismatch):
        mg.load_checkpoint(tampered(cells_were("Bid")))


def test_overfits_a_fixed_action_and_price(walks):
    """A small model learns to pick a fixed legal action and a fixed price cell on a handful of 1830
    positions in a few hundred CPU steps."""
    items = []
    for n in (3, 4):
        pool = [it for it in walks[("1830", n)] if len(it[1]) > 1]
        priced = [
            it
            for it in pool
            if "price" in _features(it[1]) and (it[1].price_range[:, 1] > it[1].price_range[:, 0] + 4).any()
        ]
        lays = [it for it in pool if "lay" in _features(it[1])]
        items += priced[:2] + lays[:2]
    assert len(items) >= 6
    model = small_model(9).train()
    arrays, inputs = inputs_for(items)
    counts = np.diff(arrays["legal_offsets"])
    target = torch.from_numpy((np.arange(len(counts)) * 7 + 3) % counts)  # a fixed legal action each
    prices = np.array([lo + 2 * (hi - lo) // 3 for lo, hi, _ in arrays["price_range"]], dtype=np.int64)
    pt = mg.to_device(mg.price_targets(arrays, prices))
    assert pt["price_valid"].sum() >= 2
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(300):
        out = model(inputs)
        loss = mg.policy_loss(out, inputs, target_index=target) + mg.price_loss(
            out, pt["price_cell"], pt["price_log_count"], pt["price_valid"]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    p_target = out["logp"][inputs["legal_offsets"][:-1] + target].exp()
    assert (p_target > 0.9).all(), p_target
    valid = pt["price_valid"]
    cell_p = out["price_logp"].exp()[valid].gather(1, pt["price_cell"][valid][:, None]).squeeze(1)
    assert (cell_p > 0.9).all(), cell_p


@pytest.mark.benchmark
def test_throughput_vs_1830_model(walks):
    """Positions/s of a forward pass (CPU, fp32, no grad, batch 64): the default generic model vs the
    default 1830 v2 model on the same 1830 positions -- 64 spread over the 3-5 player walks, and 64 with a
    legal tile lay (repeated if the walks have fewer) -- plus encoding + input building. On a loaded
    shared machine only the single-thread numbers are stable."""
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    pool = [it for n in (3, 4, 5) for it in walks[("1830", n)] if len(it[1])]
    spread = [pool[i] for i in np.linspace(0, len(pool) - 1, 64).round().astype(int)]
    lay_pool = [it for n in (2, 3, 4, 5, 6) for it in walks[("1830", n)] if "lay" in _features(it[1])]
    lays = [lay_pool[i % len(lay_pool)] for i in range(64)]
    torch.manual_seed(0)
    generic = mg.GenericPolicyValueModel().eval()
    legacy = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu"))).eval()
    print(f"\ngeneric {generic.num_parameters():,} params, 1830 v2 {sum(p.numel() for p in legacy.parameters()):,}")

    def generic_inputs(games):
        return mg.to_device(mg.build_inputs(games, [mg.legal_actions(g) for g in games]))

    def legacy_inputs(games):
        by_count: dict = {}
        for g in games:
            by_count.setdefault(len(g.players), []).append(g)
        parts, legal = [], []
        for n, gs in by_count.items():
            batch = engine_rs.encode_games(gs)
            extras = np.array([(int(rt), 0, int(rot), n) for rt, rot in zip(batch["round_types"], batch["rotations"])])
            parts.append((batch["game_states"], batch["node_features"], extras, None))
            legal += [np.asarray(g.factored_legal_indices(), dtype=np.int64) for g in gs]
        return legacy.packed_inputs(parts), legacy.lay_tile_rows(legal)

    def best(fn, reps=10):
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        return min(times)

    for label, items in (("spread", spread), ("tile-lay", lays)):
        games = [g for g, _ in items]
        g_in = generic_inputs(games)
        l_in, l_rows = legacy_inputs(games)
        rows = sum(len(la) for _, la in items)
        n_lay = sum("lay" in _features(la) for _, la in items)
        print(f"{label}: 64 positions, {n_lay} with tile lays, {rows} legal rows")
        for threads in (1, 8):
            torch.set_num_threads(threads)
            with torch.no_grad():
                tg = best(lambda: generic(g_in))
                tl = best(lambda: legacy.forward_packed(l_in, lay_tile_rows=l_rows))
                tgv = best(lambda: generic(g_in, value_only=True))
                tlv = best(lambda: legacy.forward_packed(l_in, value_only=True))
            print(
                f"  {threads} thread(s): forward generic {64 / tg:,.0f}/s (value-only {64 / tgv:,.0f}/s), "
                f"1830 v2 {64 / tl:,.0f}/s (value-only {64 / tlv:,.0f}/s)"
            )
        te = best(lambda: generic_inputs(games))
        tle = best(lambda: legacy_inputs(games))
        print(f"  inputs (encode + legal + build): generic {64 / te:,.0f}/s, 1830 {64 / tle:,.0f}/s")
