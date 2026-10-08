"""
Group 3, steps 1-3  (paper Eq. 17-23): user node vectors -> influence -> attention group vectors.

Works on the dicts returned by group3_data_access (WeiboData.topic_view / FakeNewsNetData.topic_view).
Put this file next to group3_data_access.py.  numpy + gensim only.

    from group3_steps1_3 import Cfg, run_weibo_topic, run_static_topic
    views = wb.topic_view(tid)
    res = run_weibo_topic(views, Cfg())
    res[k]["groups"][gid]["vec"]    # Fea(g, t)   (dim,)
    res[k]["groups"][gid]["attn"]   # {uid: a_i}  (sums to 1)  -> use for interpretability (Eq. 43)

Choices the paper does not fix (all documented here so the report can cite them):
  * Walk graph = reply edges (Eq. 19 weights) + attribute-kNN edges.  Half the users in a Weibo slice have
    no reply edge (they only replied to the source post), so reply edges alone leave them without a walk.
    The kNN part mirrors Group 2's Eq. 14 (k=10, gamma=0.4); each node's attribute mass is gamma, its reply mass <= 1.
  * gamma_ij in Eq. 20 is cosine of z-scored behavioural features mapped to [0, 1] via (cos+1)/2
    (a raw cosine would zero out half of the reply edges).
  * Eq. 1: all replies are comment type (I=1), l_ij=1 (no friendship graph), time normalised to [-1, 0]
    over the topic's elapsed time (the paper gives no time unit).
  * Node2vec p, q, walk length, dim are not given in the paper: defaults in Cfg.
  * Warm start: slice t continues training from slice t-1's skip-gram model, so vectors of the same user
    stay in one coordinate space across slices (the paper does not address cross-slice alignment).
  * Influence (Eq. 2-5) proxy for Weibo: followers/followees and likes/retweets do not exist, so
    W_i = replies received before this slice (comment weight omega_c), dynamic part = replies received in this slice,
    fans/follow ratio dropped (=1).  Min-max over users of the slice (Eq. 2).
  * Attention vector q (Eq. 22) is frozen random here; it is meant to be trained jointly with Group 4's GRU.
"""
from __future__ import annotations

import bisect
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gensim.models import Word2Vec

logging.getLogger("gensim.models.word2vec").setLevel(logging.ERROR)   # 'alpha higher than previous cycles' is expected on warm start


@dataclass
class Cfg:
    dim: int = 64
    num_walks: int = 8
    walk_len: int = 15
    window: int = 5
    walk_p: float = 1.0          # Eq. 18 return parameter
    walk_q: float = 1.0          # Eq. 18 in-out parameter
    epochs_first: int = 5
    epochs_update: int = 3
    alpha_update: float = 0.01   # restart learning rate for warm-started slices (first slice uses gensim's 0.025)
    attr_knn: int = 10           # Group 2: attr_knn
    attr_mass: float = 0.4       # Group 2: gamma (Table VI)
    beta: float = 0.4            # Eq. 3 (Table VI)
    omega: Tuple[float, float, float] = (0.2, 0.5, 0.3)   # omega_l, omega_r, omega_c (Table VI)
    seed: int = 42               # fully reproducible only with PYTHONHASHSEED fixed and workers=1


# ----------------------------------------------------------------------------- features
def standardize(feat: Dict, log_cols: Sequence[int] = ()):
    """log1p on heavy-tailed columns, then z-score. Returns (users, X)."""
    users = sorted(feat, key=str)
    X = np.stack([np.asarray(feat[u], dtype=float) for u in users])
    for c in log_cols:
        X[:, c] = np.log1p(np.maximum(X[:, c], 0.0))
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return users, (X - X.mean(0)) / sd


def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def knn_cosine(Xn, k, block=512):
    n = len(Xn)
    k = min(k, n - 1)
    if k <= 0:
        return np.zeros((n, 0), dtype=int), np.zeros((n, 0))
    idx = np.zeros((n, k), dtype=int)
    sim = np.zeros((n, k))
    for s in range(0, n, block):
        S = Xn[s:s + block] @ Xn.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -2.0       # drop self
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        order = np.argsort(-ps, 1)
        idx[s:s + block] = np.take_along_axis(part, order, 1)
        sim[s:s + block] = np.take_along_axis(ps, order, 1)
    return idx, sim


# ----------------------------------------------------------------------------- Eq. 1 / 19: edge strength
def _parse(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()


def weibo_pair_strength(s: dict) -> Dict[Tuple, float]:
    """UserInteract(u, v, t) per unordered reply pair, from the cumulative interactions of slice s."""
    inter = s["interactions"]
    if not inter:
        return {}
    mid2uid = {i["mid"]: i["uid"] for i in inter}
    times = [_parse(i["date"]) for i in inter]
    t0, t1 = min(times), max(times)
    span = max(t1 - t0, 1.0)
    per_pair = defaultdict(list)
    for i, ti in zip(inter, times):
        p = i["parent"]
        if p and p in mid2uid and mid2uid[p] != i["uid"]:
            key = tuple(sorted((i["uid"], mid2uid[p]), key=str))
            per_pair[key].append((ti - t1) / span)                    # in [-1, 0]
    # sum_m I * exp(M * (t_m - t)),  I = 1 (comment), l_ij = 1, M = messages exchanged by the pair
    return {k: float(sum(math.exp(len(v) * d) for d in v)) for k, v in per_pair.items()}


# ----------------------------------------------------------------------------- walk graph
def build_adjacency(users: List, X: np.ndarray, pair_strength: Dict[Tuple, float], cfg: Cfg):
    n = len(users)
    pos = {u: i for i, u in enumerate(users)}
    Xn = _unit(X)

    nbr_ui = defaultdict(dict)
    for (u, v), st in pair_strength.items():
        if u in pos and v in pos and u != v:
            i, j = pos[u], pos[v]
            nbr_ui[i][j] = nbr_ui[i].get(j, 0.0) + st
            nbr_ui[j][i] = nbr_ui[j].get(i, 0.0) + st

    w = [defaultdict(float) for _ in range(n)]
    for i, d in nbr_ui.items():                                       # Eq. 19, row-normalised per user
        tot = sum(d.values())
        for j, st in d.items():
            gamma = 0.5 * (float(Xn[i] @ Xn[j]) + 1.0)                # Eq. 20, mapped to [0, 1]
            x = gamma * st / tot
            w[i][j] += x
            w[j][i] += x                                              # undirected walk graph

    idx, sim = knn_cosine(Xn, cfg.attr_knn)
    k = idx.shape[1]
    for i in range(n):                                                # attribute edges (Group 2, Eq. 14)
        for j, sv in zip(idx[i], sim[i]):
            x = cfg.attr_mass * max(float(sv), 0.0) / k
            w[i][int(j)] += x
            w[int(j)][i] += x
    covered = len(nbr_ui) / max(n, 1)
    return w, covered


def biased_walks(w, cfg: Cfg, rng: random.Random):
    """Returns (walks, number of nodes whose edge weights were all zero -> uniform fallback)."""
    n = len(w)
    nbrs = [list(d.keys()) for d in w]
    cum, n_degenerate = [], 0
    for d, nb in zip(w, nbrs):
        if not nb:
            cum.append([])
            continue
        c = np.cumsum([max(d[j], 0.0) for j in nb])
        if c[-1] <= 1e-12:                       # every edge weight is 0 (e.g. all-zero feature vectors, no reply edge)
            n_degenerate += 1                    # -> uniform over neighbours instead of NaN
            c = np.arange(1, len(nb) + 1, dtype=float)
        cum.append((c / c[-1]).tolist())
    general = not (cfg.walk_p == 1.0 and cfg.walk_q == 1.0)
    sets = [set(nb) for nb in nbrs] if general else None

    walks, order = [], list(range(n))
    for _ in range(cfg.num_walks):
        rng.shuffle(order)
        for s in order:
            walk = [s]
            while len(walk) < cfg.walk_len:
                cur = walk[-1]
                nb = nbrs[cur]
                if not nb:
                    break
                if not general or len(walk) == 1:
                    j = min(bisect.bisect(cum[cur], rng.random()), len(nb) - 1)
                    walk.append(nb[j])
                else:                                                  # Eq. 18 bias
                    prev = walk[-2]
                    ws = [(w[cur][x] + 1e-12) * (1.0 / cfg.walk_p if x == prev
                                       else 1.0 if x in sets[prev] else 1.0 / cfg.walk_q) for x in nb]
                    walk.append(rng.choices(nb, weights=ws)[0])
            walks.append(walk)
    return walks, n_degenerate


class SliceEmbedder:
    """Skip-gram over walks (Eq. 21); each slice continues from the previous slice's model."""

    def __init__(self, cfg: Cfg):
        self.cfg, self.model = cfg, None

    def fit(self, sentences: List[List[str]]):
        c = self.cfg
        if self.model is None:
            self.model = Word2Vec(vector_size=c.dim, window=c.window, min_count=1, sg=1,
                                  workers=1, seed=c.seed, negative=5)
            self.model.build_vocab(sentences)
            self.model.train(sentences, total_examples=len(sentences), epochs=c.epochs_first)
        else:
            self.model.build_vocab(sentences, update=True)
            self.model.train(sentences, total_examples=len(sentences), epochs=c.epochs_update,
                             start_alpha=c.alpha_update, end_alpha=0.0001)

    def vectors(self, users) -> Dict:
        return {u: self.model.wv[str(u)] for u in users}


# ----------------------------------------------------------------------------- Eq. 2-5 influence
def influence_weibo(users, cum: Counter, prev: Counter, cfg: Cfg) -> Dict:
    hist = {u: prev.get(u, 0) for u in users}
    new = {u: cum.get(u, 0) - prev.get(u, 0) for u in users}
    S = {u: cfg.omega[2] * hist[u] for u in users}                    # replies are comment type only
    smax, nmax = max(S.values()) + 1e-6, max(new.values()) + 1e-6
    raw = {u: S[u] / smax + cfg.beta * new[u] / nmax for u in users}
    lo, hi = min(raw.values()), max(raw.values())
    return {u: (raw[u] - lo) / (hi - lo + 1e-9) for u in users}       # Eq. 2


def influence_upfd(users, view: dict, cfg: Cfg = Cfg()) -> Dict:
    """Eq. 2-5 analogue for UPFD. The 10 profile columns are PRE-SCALED (e.g. followers max 0.02), so raw counts
    are not recoverable; this is a proxy.  Column order (UPFD): 0 verified, 1 geo, 2 followers, 3 friends,
    4 statuses, 5 favourites, 6 listed, 7 account age, 8 name words, 9 description words.
      ratio   = log1p(followers / (friends + 1e-3))   fans/follow term of Eq. 3
      W_i     = listed / max(listed)                  reputation, Eq. 4 shape
      dynamic = number of users who retweeted via u   (children in the propagation tree)"""
    f, root = view["features"], view["root"]
    fol = np.array([f[u][2] for u in users], float)
    fri = np.array([f[u][3] for u in users], float)
    lis = np.array([f[u][6] for u in users], float)
    ratio = np.log1p(fol / (fri + 1e-3))
    ratio /= ratio.max() + 1e-9
    W = lis / (lis.max() + 1e-6)
    kids = Counter(a for a, b in view["edges"] if a != root)
    dyn = np.array([kids.get(u, 0) for u in users], float)
    dyn /= dyn.max() + 1e-6
    raw = W * ratio + cfg.beta * dyn
    lo, hi = raw.min(), raw.max()
    return {u: float((r - lo) / (hi - lo + 1e-9)) for u, r in zip(users, raw)}


# ----------------------------------------------------------------------------- Eq. 22-23 attention
def default_q(cfg: Cfg, seed: Optional[int] = None) -> np.ndarray:
    r = np.random.RandomState(cfg.seed if seed is None else seed)
    return r.normal(0.0, 1.0, cfg.dim) / math.sqrt(cfg.dim)


def group_vectors(groups: Dict, emb: Dict, infl: Dict, q: np.ndarray) -> Dict:
    out = {}
    for gid, members in groups.items():
        mem = [u for u in members if u in emb]
        if not mem:
            continue
        M = np.stack([emb[u] for u in mem])
        score = M @ q + np.array([infl.get(u, 0.0) for u in mem])      # q^T Fea + Influence
        score -= score.max()
        a = np.exp(score)
        a /= a.sum()
        out[gid] = {"vec": a @ M, "attn": dict(zip(mem, a.tolist())), "size": len(mem)}
    return out


# ----------------------------------------------------------------------------- drivers
def run_weibo_topic(views: List[dict], cfg: Cfg = Cfg(), q: Optional[np.ndarray] = None) -> List[dict]:
    """views = WeiboData.topic_view(topic_id). Returns one dict per slice."""
    q = default_q(cfg) if q is None else q
    rng, emb = random.Random(cfg.seed), SliceEmbedder(cfg)
    prev, res = Counter(), []
    for s in views:
        users_g = set(s["user_group"])
        if not users_g or not s["user_features"]:
            res.append({"slice_index": s["slice_index"], "groups": {}, "n_users": 0})
            continue
        users, X = standardize(s["user_features"], log_cols=(0,))     # col 0 = n_interactions (heavy tailed)
        w, covered = build_adjacency(users, X, weibo_pair_strength(s), cfg)
        walks, ndeg = biased_walks(w, cfg, rng)
        emb.fit([[str(users[i]) for i in wk] for wk in walks])
        vec = emb.vectors(users)
        cum = Counter(v for _, v in s["edges"])
        infl = influence_weibo(users, cum, prev, cfg)
        prev = cum
        res.append({"slice_index": s["slice_index"], "n_users": len(users), "reply_edge_cover": covered,
                    "degenerate_frac": ndeg / len(users), "influence": infl, "groups": group_vectors(s["groups"], vec, infl, q)})
    return res


def run_static_topic(view: dict, cfg: Cfg = Cfg(), q: Optional[np.ndarray] = None,
                     influence_fn: Optional[Callable[[List, dict], Dict]] = None) -> List[dict]:
    """view = FakeNewsNetData.topic_view(topic_id): one static tree, news root excluded.
    influence_fn(users, view) -> {user: [0,1]}; None = zero influence (attention uses q only) until the
    UPFD profile columns are checked."""
    q = default_q(cfg) if q is None else q
    root = view["root"]
    users_all = [n for n in view["nodes"] if n in view["user_group"]]
    if not users_all:
        return [{"slice_index": 1, "groups": {}, "n_users": 0}]
    feat = {n: view["features"][n] for n in users_all}
    users, X = standardize(feat)
    pairs = {tuple(sorted((u, v), key=str)): 1.0 for u, v in view["edges"] if u != root and v != root}
    w, covered = build_adjacency(users, X, pairs, cfg)
    walks, ndeg = biased_walks(w, cfg, random.Random(cfg.seed))
    emb = SliceEmbedder(cfg)
    emb.fit([[str(users[i]) for i in wk] for wk in walks])
    vec = emb.vectors(users)
    infl = influence_fn(users, view) if influence_fn else {u: 0.0 for u in users}
    return [{"slice_index": 1, "n_users": len(users), "reply_edge_cover": covered,
             "degenerate_frac": ndeg / len(users), "influence": infl,
             "groups": group_vectors(view["groups"], vec, infl, q)}]
