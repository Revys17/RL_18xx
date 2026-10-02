"""Can MCTS find the human's price? Current Gaussian price head vs. a price menu.

Fits two price heads on held-out-by-game human decisions from
``scripts/price_census.py`` output, using the same small MLP body and feature
vector (a stand-in for the trunk; the real network sees the whole state):

  * the shipped ``ContinuousPriceHead`` parametrization and loss
    (``Normal(150 + 100*raw, exp(raw + 3))`` in absolute dollars, truncated-Normal
    NLL exactly as ``train._compute_price_nll_loss``), and
  * a range-relative categorical "price menu": the min-bid ladder
    (min_bid + $5*j) for bids; {min, max, max-1} atoms + 20 interior bins for
    cross-corp trains.

It then measures what search actually depends on: after k price expansions
under one slot, does the candidate set contain the price the human played?
(exact for bids and for the min / max(-1) train atoms; within 5% of the legal
range for interior train prices).

    uv run python scripts/price_head_compare.py --census price_census.csv
"""

import argparse
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import truncnorm


# The Gaussian head's MCTS sampler as shipped before the price pmf replaced it
# (rejection sampling with a uniform fallback), kept here so the comparison
# stays reproducible. The "correct truncnorm sampler" rows below correspond to
# the interim fix on the training branch (d846456).
_PRICE_GRID = {"Bid": 5, "BuyTrain": 1, "BuyCompany": 1}


def _snap_price(price, action_type, price_min, price_max):
    step = _PRICE_GRID.get(action_type, 1)
    snapped = int(round(price / step) * step)
    if snapped < price_min:
        rem = price_min % step
        snapped = price_min if rem == 0 else price_min + (step - rem)
    if snapped > price_max:
        snapped = price_max - (price_max % step) if step > 1 else price_max
    return int(max(min(snapped, price_max), price_min))


def sample_price_for_pw(price_mean, price_log_std, action_type, price_range, rng):
    p_min, p_max = price_range
    if p_min == p_max:
        return int(p_min)
    sigma = float(np.exp(np.clip(price_log_std, -1.0, 8.5)))
    for _ in range(8):
        sample = rng.normal(float(price_mean), sigma)
        if p_min - sigma <= sample <= p_max + sigma:
            return _snap_price(sample, action_type, int(p_min), int(p_max))
    step = _PRICE_GRID.get(action_type, 1)
    return int(p_min) + int(rng.integers(0, max(1, (int(p_max) - int(p_min)) // step + 1))) * step


torch.set_num_threads(4)
N_BINS = 20  # interior bins for cross-corp trains
N_LADDER = 20  # min_bid + $5*j, j < N_LADDER (covers every human bid in the corpus)
TRAIN_TYPES = ["2", "3", "4", "5", "6", "D"]
COMPANIES = ["SV", "CS", "DH", "MH", "CA", "BO"]
KS = (1, 2, 4, 8, 14)


def train_features(t):
    f = [t.lo, t.hi, t.face, t.buyer_cash, t.seller_cash, t.next_depot.fillna(0)]
    f = [x / 1000 for x in f] + [t.buyer_ntrains, t.seller_ntrains, t.np / 6]
    f += [(t.what.astype(str) == x).astype(float) for x in TRAIN_TYPES]
    return np.stack(f, 1).astype(np.float32)


def bid_features(b):
    f = [b.lo / 1000, b.hi / 1000, b.face / 1000, b.np / 6] + [(b.what == x).astype(float) for x in COMPANIES]
    return np.stack(f, 1).astype(np.float32)


def train_cell(p, lo, hi):
    if p == lo:
        return 0
    if p == hi:
        return 1
    if p == hi - 1:
        return 2
    return 3 + min(int((p - lo) / max(hi - lo, 1) * N_BINS), N_BINS - 1)


def train_cell_price(c, lo, hi):
    if c < 3:
        return (lo, hi, max(lo, hi - 1))[c]
    return int(round(lo + (c - 3 + 0.5) / N_BINS * (hi - lo)))


def bid_cell(p, lo, hi):
    j, r = divmod(p - lo, 5)
    return int(j) if r == 0 and j < N_LADDER else N_LADDER  # last cell = off-ladder


def bid_cell_price(c, lo, hi):
    return min(lo + 5 * c, hi) if c < N_LADDER else None


def mlp(d_in, d_out):
    return nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, d_out))


def gauss_nll(raw, price, lo, hi):
    """``train._compute_price_nll_loss`` on the shipped head parametrization."""
    mu = 150.0 + 100.0 * raw[:, 0]
    log_sigma = (raw[:, 1] + 3.0).clamp(-1.0, 8.5)
    sigma = log_sigma.exp()
    z = (price - mu) / sigma
    cdf = lambda x: 0.5 * (1 + torch.erf((x - mu) / sigma / math.sqrt(2)))
    lp = -0.5 * z * z - log_sigma - 0.5 * math.log(2 * math.pi) - torch.log((cdf(hi) - cdf(lo)).clamp(min=1e-8))
    return -lp.mean(), mu, log_sigma


def fit(model, loss_fn, X, *targets, epochs=60):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    X, T = torch.tensor(X), [torch.tensor(np.asarray(t)) for t in targets]
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            idx = perm[i : i + 512]
            loss = loss_fn(model(X[idx]), *[t[idx] for t in T])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def compare(kind, rows, feats, cell_of, cell_price, n_cells, atype, hit, anchors, rng, test_games):
    tr, te = rows[~rows.game.isin(test_games)], rows[rows.game.isin(test_games)]
    Xtr, Xte = feats(tr), feats(te)
    lo, hi, p = (te[c].values.astype(int) for c in ("lo", "hi", "price"))
    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))

    torch.manual_seed(0)
    gauss_targets = (np.float32(tr[c]) for c in ("price", "lo", "hi"))
    g = fit(mlp(Xtr.shape[1], 2), lambda o, *t: gauss_nll(o, *t)[0], Xtr, *gauss_targets)
    with torch.no_grad():
        _, mu, ls = gauss_nll(g(f32(Xte)), f32(p), f32(lo), f32(hi))
    mu, ls = mu.numpy(), ls.numpy()

    cells = np.array([cell_of(*x) for x in zip(tr.price, tr.lo, tr.hi)])
    c = fit(mlp(Xtr.shape[1], n_cells), lambda o, y: nn.functional.cross_entropy(o, y), Xtr, cells)
    with torch.no_grad():
        probs = torch.softmax(c(f32(Xte)), -1).numpy()

    def truncnorm_sample(i):
        s = float(np.exp(np.clip(ls[i], -1, 8.5)))
        a, b = (lo[i] - 0.5 - mu[i]) / s, (hi[i] + 0.5 - mu[i]) / s
        return _snap_price(truncnorm.rvs(a, b, loc=mu[i], scale=s, random_state=rng), atype, lo[i], hi[i])

    def sampled(draw, seed=()):
        return lambda i, k: set(seed(i)) | {draw(i) for _ in range(k)} if seed else {draw(i) for _ in range(k)}

    shipped = lambda m, s: (lambda i: sample_price_for_pw(m[i], s[i], atype, (lo[i], hi[i]), rng))
    mechanisms = {
        "Gaussian head UNTRAINED (init)": sampled(shipped(np.full(len(te), 150.0), np.full(len(te), 3.0))),
        "Gaussian head trained, shipped sampler": sampled(shipped(mu, ls)),
        "Gaussian head trained, correct truncnorm sampler": sampled(truncnorm_sample),
        "Gaussian head trained, shipped sampler + anchors": sampled(shipped(mu, ls), lambda i: anchors(lo[i], hi[i])),
        "price menu, PW samples from menu": lambda i, k: {
            cell_price(x, lo[i], hi[i]) for x in rng.choice(n_cells, size=k, p=probs[i] / probs[i].sum())
        },
        "price menu, top-k by prior": lambda i, k: {cell_price(x, lo[i], hi[i]) for x in np.argsort(-probs[i])[:k]},
    }
    print(f"\n=== {kind}: train n={len(tr)}, held-out n={len(te)} ({te.game.nunique()} games)")
    print(f"trained Gaussian: mu below legal range {np.mean(mu < lo):.3f}, above {np.mean(mu > hi):.3f}; "
          f"median sigma ${np.median(np.exp(ls)):.0f}; median range ${np.median(hi - lo):.0f}")
    # The same categorical model read as an exact pmf over EVERY legal integer
    # price: a cell's mass is spread uniformly over the integers it contains
    # (cells that contain none are dropped and the rest renormalized). Any
    # human price is then a valid target verbatim, and sampling can produce any
    # legal price -- the cells are a resolution limit of the density, not an
    # action list.
    members, pmf_probs = [], []
    for i in range(len(te)):
        ps = np.arange(lo[i], hi[i] + 1)
        cell_ids = np.array([cell_of(q, lo[i], hi[i]) for q in ps])
        members.append([ps[cell_ids == cc] for cc in range(n_cells)])
        pr = probs[i] * np.array([len(m) > 0 for m in members[-1]])
        pmf_probs.append(pr / pr.sum())

    def pmf_sample(i, k):
        cells = rng.choice(n_cells, size=k, p=pmf_probs[i])
        return {int(rng.choice(members[i][cc])) for cc in cells}

    def pmf_gumbel_topk(i, k):
        # sampling WITHOUT replacement over cells (Gumbel-top-k), then a uniform price inside each
        g = np.log(pmf_probs[i] + 1e-30) + rng.gumbel(size=n_cells)
        return {int(rng.choice(members[i][cc])) for cc in np.argsort(-g)[:k] if pmf_probs[i][cc] > 0}

    mechanisms["exact-pmf head, i.i.d. samples"] = pmf_sample
    mechanisms["exact-pmf head, Gumbel-top-k samples"] = pmf_gumbel_topk

    print(f"{'coverage of the human price after k expansions':52s}" + "".join(f"  k={k:<4d}" for k in KS))
    for name, propose in mechanisms.items():
        cov = [np.mean([hit(p[i], lo[i], hi[i], propose(i, k)) for i in range(len(te))]) for k in KS]
        print(f"{name:52s}" + "".join(f"  {x:.3f} " for x in cov))

    # Exact log-likelihood of the human's integer price (bits; lower is better).
    s = np.exp(np.clip(ls, -1, 8.5))
    gauss_bits = -(_log_mass(p - 0.5, p + 0.5, mu, s) - _log_mass(lo - 0.5, hi + 0.5, mu, s)) / np.log(2)
    pmf_bits = -np.array(
        [np.log2(pmf_probs[i][cell_of(p[i], lo[i], hi[i])] / len(members[i][cell_of(p[i], lo[i], hi[i])]))
         for i in range(len(te))]
    )
    print(f"exact log-likelihood of the human's price, bits/decision (median | mean):  "
          f"uniform over legal range {np.median(np.log2(hi - lo + 1)):.2f} | {np.mean(np.log2(hi - lo + 1)):.2f};  "
          f"trained Gaussian {np.median(gauss_bits):.2f} | {np.mean(gauss_bits):.2f};  "
          f"exact-pmf head {np.median(pmf_bits):.2f} | {np.mean(pmf_bits):.2f}")


def _log_mass(a, b, mu, s):
    """log(Phi((b-mu)/s) - Phi((a-mu)/s)), stable in both tails."""
    from scipy.special import log_ndtr

    za, zb = (a - mu) / s, (b - mu) / s
    upper = za > 0  # both in the upper tail: use the survival function
    hi_l = np.where(upper, log_ndtr(-za), log_ndtr(zb))
    lo_l = np.where(upper, log_ndtr(-zb), log_ndtr(za))
    return hi_l + np.log1p(-np.exp(np.minimum(lo_l - hi_l, -1e-12)))


def legacy_grid_fidelity(d):
    """Share of human price decisions the pre-May fixed-grid action space could
    represent without rewriting the action (and, for bids, the encoded state)."""
    b = d[d.type == "bid"]
    t = d[(d.type == "buy_train") & (d.seller != "The Depot")]
    c = d[d.type == "buy_company"]
    grid = {1, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900}
    on_t = t.price.isin(grid) | (t.price >= t.hi - 1)
    on_b = (b.price == b.lo) & (b.lo % 5 == 0)
    on_c = (c.price == c.lo) | (c.price == c.hi)
    print("\nlegacy fixed-grid action space: share of human decisions representable WITHOUT rewriting")
    print(f"  Bid {on_b.mean():.3f} (rest rewrote the encoded bid ladder);  cross-corp BuyTrain {on_t.mean():.3f};  "
          f"BuyCompany {on_c.mean():.3f}   (continuous / exact-pmf heads: 1.000 for all three)")


def train_hit(p, lo, hi, cands):
    if p == lo:
        return lo in cands
    if p >= hi - 1:
        return any(q >= hi - 1 for q in cands)
    tol = max(1, 0.05 * (hi - lo))
    return any(abs(q - p) <= tol for q in cands if q is not None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="price_census.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    d = pd.read_csv(args.census).dropna(subset=["lo", "hi"])
    d = d[d.hi > d.lo]
    games = d.game.unique()
    rng.shuffle(games)
    test_games = set(games[: len(games) // 5])

    legacy_grid_fidelity(d)
    compare("cross-corp BuyTrain", d[(d.type == "buy_train") & (d.seller != "The Depot")], train_features,
            train_cell, train_cell_price, 3 + N_BINS, "BuyTrain", train_hit, lambda lo, hi: (lo, hi), rng, test_games)
    compare("Bid", d[d.type == "bid"], bid_features, bid_cell, bid_cell_price, N_LADDER + 1, "Bid",
            lambda p, lo, hi, cands: p in cands, lambda lo, hi: (lo,), rng, test_games)
