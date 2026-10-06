"""
TA-Louvain: Topology + Attribute Louvain hidden-community mining, with
Jaccard-based group tracking across time slices.

Reference: Wang et al., "A Guided Topic Detection Model Based on Topic Evolution
and Group Stance", IEEE TCSS, vol. 13(3), 2026  (Sec. IV-A, Eq. 11-16).

Paper equations implemented
---------------------------
Eq.11  S_N(i,j)  = N_ij / (N_i + N_j)                      shared-neighbour similarity
Eq.12  S_NN(i,k) = S_N(i,j) * S_N(j,k)                     2nd-order similarity via bridge j
Eq.13  S(i,j)    = l1*S_N + (1-l1)*S_NN + l2*S_att         composite similarity
Eq.14  W_ij      = A_ij + gamma * Sim_att(i,j)             attribute-enhanced adjacency
Eq.15  Q'        = 1/m * sum_ij [W_ij - d_i d_j / 2m] * S(i,j) * delta(c_i,c_j)
Eq.16  Jaccard(g_i^{t-1}, g_j^t) > 0.5  -> same group across slices

Implementation notes (where the paper is under-specified)
---------------------------------------------------------
* Eq.15 is realised by folding S into the weights: w'_ij = W_ij * S_ij, then standard
  Louvain modularity on w' (own implementation, numpy only).  Subtracting the null term d_i d_j/2m only on the
  sparse set of pairs where S is defined was tested and collapses everything into one
  group, so it is not used.
* Eq.14 applied to *all* pairs makes the graph dense (O(N^2)).  By default we only
  add attribute edges between each node and its `attr_knn` most similar nodes.
  Set `attr_knn=None` for the literal dense version (small graphs only).
* S, W are evaluated only on candidate pairs = structural edges U attribute-kNN
  pairs (S is treated as 0 elsewhere).
* Eq.12 has several bridges j for one pair (i,k); we take the max (default) or
  the clipped sum (`snn_mode="sum"`).
* Paper's sensitivity study uses l2 = 1 - l1, Table VI uses 0.7/0.3 -> consistent
  with defaults l1=0.7, l2=0.3, gamma=0.4.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Node = Any   # user ids: str or int (any hashable, sortable value)


# --------------------------------------------------------------------------- #
# Configuration / results
# --------------------------------------------------------------------------- #
@dataclass
class TALouvainConfig:
    lambda1: float = 0.7          # Eq.13 topological mixing (Table VI)
    lambda2: float = 0.3          # Eq.13 attribute weight   (Table VI)
    gamma: float = 0.4            # Eq.14 attribute weight in W (Table VI)
    snn_mode: str = "max"         # "max" | "sum" over bridges in Eq.12
    attr_knn: Optional[int] = 10  # attribute neighbours per node; None = dense
    attr_min_sim: float = 0.0     # ignore attribute pairs below this cosine
    jaccard_threshold: float = 0.5  # Eq.16 continuity threshold (strictly greater)
    resolution: float = 1.0       # modularity resolution (1.0 = standard)
    seed: int = 42


@dataclass
class SliceResult:
    labels: Dict[Node, int]                    # node -> community id (this slice)
    groups: Dict[int, Set[Node]]               # community id -> members
    q_prime: float                             # modularity of final partition on the S-weighted graph
    persistent_ids: Dict[int, int] = field(default_factory=dict)  # local id -> tracked id


# --------------------------------------------------------------------------- #
# Step 1: structure, similarities, W, M
# --------------------------------------------------------------------------- #
def _norm_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _attr_knn_pairs(Xn: np.ndarray, k: int, min_sim: float, chunk: int = 2048):
    """Top-k cosine neighbours per row -> dict {(i,j) i<j : sim}."""
    n = Xn.shape[0]
    pairs: Dict[Tuple[int, int], float] = {}
    k = min(k, n - 1)
    if k <= 0:
        return pairs
    for s in range(0, n, chunk):
        blk = Xn[s:s + chunk] @ Xn.T
        for r in range(blk.shape[0]):
            blk[r, s + r] = -1.0  # no self
        idx = np.argpartition(-blk, k - 1, axis=1)[:, :k]
        for r in range(blk.shape[0]):
            i = s + r
            for j in idx[r]:
                sim = float(blk[r, j])
                if sim >= min_sim and sim > 0:
                    a, b = (i, int(j)) if i < j else (int(j), i)
                    pairs[(a, b)] = max(sim, 0.0)
    return pairs


def build_pair_model(
    nodes: Sequence[Node],
    edges: Iterable[Tuple],
    attrs: Optional[Dict[Node, np.ndarray]],
    cfg: TALouvainConfig,
):
    """Return (index, pairs, W, S, deg, m) where pairs is a list of (i,j), i<j."""
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    # structural adjacency A (undirected, weights summed, no self loops)
    A: Dict[Tuple[int, int], float] = defaultdict(float)
    for e in edges:
        u, v = e[0], e[1]
        w = float(e[2]) if len(e) > 2 else 1.0
        if u == v or u not in index or v not in index:
            continue
        i, j = index[u], index[v]
        A[(i, j) if i < j else (j, i)] += w
    nbr: List[Set[int]] = [set() for _ in range(n)]
    for (i, j) in A:
        nbr[i].add(j)
        nbr[j].add(i)

    # attribute similarity
    Xn = None
    if attrs is not None and cfg.lambda2 + cfg.gamma > 0:
        d = len(next(iter(attrs.values())))
        X = np.zeros((n, d), dtype=np.float32)
        for u, vec in attrs.items():
            if u in index:
                X[index[u]] = vec
        Xn = _norm_rows(X)

    # candidate pairs = structural edges U attribute-kNN pairs
    cand = set(A.keys())
    knn_sims: Dict[Tuple[int, int], float] = {}
    if Xn is not None:
        if cfg.attr_knn is None:  # literal Eq.14: all pairs (O(N^2) memory!)
            iu = np.triu_indices(n, 1)
            sims = np.clip(Xn @ Xn.T, 0, 1)[iu]
            keep = sims > cfg.attr_min_sim
            knn_sims = {(int(a), int(b)): float(s) for a, b, s in zip(iu[0][keep], iu[1][keep], sims[keep])}
        elif cfg.attr_knn > 0:
            knn_sims = _attr_knn_pairs(Xn, cfg.attr_knn, cfg.attr_min_sim)
        cand |= set(knn_sims.keys())

    def att_sim(i, j):
        if Xn is None:
            return 0.0
        s = knn_sims.get((i, j))
        return s if s is not None else float(max(Xn[i] @ Xn[j], 0.0))

    # Eq.11 with cache (topology only, from A)
    sn_cache: Dict[Tuple[int, int], float] = {}

    def S_N(i, j):
        key = (i, j) if i < j else (j, i)
        v = sn_cache.get(key)
        if v is None:
            denom = len(nbr[i]) + len(nbr[j])
            v = len(nbr[i] & nbr[j]) / denom if denom else 0.0
            sn_cache[key] = v
        return v

    # Eq.12: via common neighbours j of i and k
    def S_NN(i, k):
        bridges = nbr[i] & nbr[k]
        if not bridges:
            return 0.0
        vals = [S_N(i, j) * S_N(j, k) for j in bridges]
        return max(vals) if cfg.snn_mode == "max" else min(1.0, sum(vals))

    pairs = sorted(cand)
    W = np.zeros(len(pairs))
    S = np.zeros(len(pairs))
    for t, (i, j) in enumerate(pairs):
        sa = att_sim(i, j)
        W[t] = A.get((i, j), 0.0) + cfg.gamma * sa                                   # Eq.14
        S[t] = cfg.lambda1 * S_N(i, j) + (1 - cfg.lambda1) * S_NN(i, j) + cfg.lambda2 * sa  # Eq.13

    deg = np.zeros(n)
    for t, (i, j) in enumerate(pairs):
        deg[i] += W[t]
        deg[j] += W[t]
    m = W.sum()
    return index, pairs, W, S, deg, m


# --------------------------------------------------------------------------- #
# Step 2: modularity optimisation on similarity-weighted graph  (Eq.15)
# --------------------------------------------------------------------------- #
def _louvain(n: int, wedges: Dict[Tuple[int, int], float], resolution: float, seed: int):
    """Weighted Louvain (Blondel et al. 2008) on an undirected graph with n nodes.
    wedges: {(i, j): w} with i < j.  Returns (labels list, modularity)."""
    rng = random.Random(seed)
    adj: List[Dict[int, float]] = [defaultdict(float) for _ in range(n)]
    for (i, j), w in wedges.items():
        adj[i][j] += w
        adj[j][i] += w
    self_w = [0.0] * n
    m2 = 2.0 * sum(wedges.values())              # total degree = 2m
    if m2 <= 0:
        return list(range(n)), 0.0

    node_to_super = list(range(n))
    cur_adj, cur_self = adj, self_w
    while True:
        nn = len(cur_adj)
        k = [sum(cur_adj[i].values()) + 2.0 * cur_self[i] for i in range(nn)]
        comm = list(range(nn))
        tot = k[:]
        moved_any = False
        while True:
            order = list(range(nn))
            rng.shuffle(order)
            moved = False
            for i in order:
                ci = comm[i]
                w_to: Dict[int, float] = defaultdict(float)
                for j, w in cur_adj[i].items():
                    w_to[comm[j]] += w
                tot[ci] -= k[i]                  # remove i from its community
                best_c = ci
                best_gain = w_to.get(ci, 0.0) - resolution * tot[ci] * k[i] / m2
                for c, w in w_to.items():
                    g = w - resolution * tot[c] * k[i] / m2
                    if g > best_gain + 1e-12:
                        best_c, best_gain = c, g
                tot[best_c] += k[i]
                if best_c != ci:
                    comm[i] = best_c
                    moved = True
            if not moved:
                break
            moved_any = True
        if not moved_any:
            break
        # aggregate communities into super-nodes
        uniq = {c: x for x, c in enumerate(sorted(set(comm)))}
        lab = [uniq[c] for c in comm]
        new_adj: List[Dict[int, float]] = [defaultdict(float) for _ in range(len(uniq))]
        new_self = [0.0] * len(uniq)
        for i in range(nn):
            ci = lab[i]
            new_self[ci] += cur_self[i]
            for j, w in cur_adj[i].items():
                cj = lab[j]
                if ci == cj:
                    new_self[ci] += w / 2.0      # each internal edge seen twice
                else:
                    new_adj[ci][cj] += w
        node_to_super = [lab[x] for x in node_to_super]
        cur_adj, cur_self = new_adj, new_self
        if len(uniq) == nn:
            break

    # modularity of the final partition on the original graph
    inside: Dict[int, float] = defaultdict(float)
    degsum: Dict[int, float] = defaultdict(float)
    for i in range(n):
        degsum[node_to_super[i]] += sum(adj[i].values())
        for j, w in adj[i].items():
            if node_to_super[j] == node_to_super[i]:
                inside[node_to_super[i]] += w
    q = sum(inside[c] / m2 - resolution * (degsum[c] / m2) ** 2 for c in degsum)
    return node_to_super, q


def _optimise(nodes, index, pairs, W, S, cfg: TALouvainConfig):
    """Eq.15 realised by folding S into the weights, w'_ij = W_ij * S_ij, then standard
    modularity optimisation on w' (degree-based null model on w').

    Eq.15 does not define the null-model term on pairs where S is undefined.  Subtracting
    d_i d_j / 2m only on the sparse candidate pairs was tested and is far too weak: all
    nodes collapse into one giant group.  Folding avoids that."""
    wedges = {}
    for t, (i, j) in enumerate(pairs):
        w = float(W[t] * S[t])
        if w > 0:
            wedges[(i, j)] = w
    return _louvain(len(nodes), wedges, cfg.resolution, cfg.seed)


# --------------------------------------------------------------------------- #
# Public API: one slice
# --------------------------------------------------------------------------- #
def ta_louvain(
    nodes: Sequence[Node],
    edges: Iterable[Tuple],
    attrs: Optional[Dict[Node, np.ndarray]] = None,
    cfg: Optional[TALouvainConfig] = None,
) -> SliceResult:
    """Detect hidden groups in one time slice.

    nodes : all user ids active in the slice
    edges : (u, v) or (u, v, weight) interactions (treated as undirected)
    attrs : node -> attribute vector (numeric, already scaled; see standardize_attributes)
    """
    cfg = cfg or TALouvainConfig()
    nodes = list(nodes)
    if not nodes:
        return SliceResult({}, {}, 0.0)
    index, pairs, W, S, deg, m = build_pair_model(nodes, edges, attrs, cfg)
    lab, q = _optimise(nodes, index, pairs, W, S, cfg)
    remap = {c: k for k, c in enumerate(sorted(set(lab)))}
    lab = [remap[c] for c in lab]
    labels = {u: lab[index[u]] for u in nodes}
    groups: Dict[int, Set[Node]] = defaultdict(set)
    for u, c in labels.items():
        groups[c].add(u)
    return SliceResult(labels, dict(groups), q)


# --------------------------------------------------------------------------- #
# Step 3: group tracking across slices (Eq.16)
# --------------------------------------------------------------------------- #
def jaccard(a: Set, b: Set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


class GroupTracker:
    """Assigns persistent ids: a group at t inherits the id of the group at t-1
    with which its Jaccard overlap is > threshold (greedy, one-to-one)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.prev: Dict[int, Set[Node]] = {}
        self._next = 0
        self.history: List[Dict[int, Set[Node]]] = []   # persistent id -> members, per slice

    def update(self, groups: Dict[int, Set[Node]]) -> Dict[int, int]:
        cands = []
        for lc, mem in groups.items():
            for pid, pm in self.prev.items():
                j = jaccard(pm, mem)
                if j > self.threshold:
                    cands.append((j, lc, pid))
        cands.sort(reverse=True)
        mapping: Dict[int, int] = {}
        used = set()
        for j, lc, pid in cands:
            if lc not in mapping and pid not in used:
                mapping[lc] = pid
                used.add(pid)
        for lc in groups:
            if lc not in mapping:
                mapping[lc] = self._next
                self._next += 1
        self.prev = {mapping[lc]: set(mem) for lc, mem in groups.items()}
        self.history.append({k: set(v) for k, v in self.prev.items()})
        return mapping


def run_over_slices(slices: Sequence[dict], cfg: Optional[TALouvainConfig] = None) -> List[SliceResult]:
    """slices: list of dicts {"nodes": [...], "edges": [...], "attrs": {node: vec}}"""
    cfg = cfg or TALouvainConfig()
    tracker = GroupTracker(cfg.jaccard_threshold)
    out = []
    for s in slices:
        r = ta_louvain(s["nodes"], s["edges"], s.get("attrs"), cfg)
        r.persistent_ids = tracker.update(r.groups)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Helper: attribute preprocessing
# --------------------------------------------------------------------------- #
def standardize_attributes(
    raw: Dict[Node, Sequence[float]], log_cols: Sequence[int] = ()
) -> Dict[Node, np.ndarray]:
    """log1p heavy-tailed count columns, then z-score every column. Cosine similarity
    is meaningless on raw counts (follower counts dominate), so do this first.
    Note: z-scored features can be negative; ta_louvain clips cosine to [0, 1]."""
    keys = list(raw)
    X = np.array([raw[k] for k in keys], dtype=np.float64)
    for c in log_cols:
        X[:, c] = np.log1p(np.maximum(X[:, c], 0))
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    return {k: X[i] for i, k in enumerate(keys)}


# --------------------------------------------------------------------------- #
# Self-test on a synthetic network where topology alone is too sparse
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_per, k = 60, 4
    nodes = list(range(n_per * k))
    truth = {u: u // n_per for u in nodes}
    centers = rng.normal(0, 3, (k, 8))
    attrs = {u: centers[truth[u]] + rng.normal(0, 1, 8) for u in nodes}
    edges = []
    for u in nodes:
        for v in range(u + 1, len(nodes)):
            p = 0.06 if truth[u] == truth[v] else 0.01  # very sparse
            if rng.random() < p:
                edges.append((u, v))

    def nmi(a, b):
        try:
            from sklearn.metrics import normalized_mutual_info_score as f
        except ImportError:        # sklearn is only needed for this self-test
            return float("nan")
        return f([a[u] for u in nodes], [b[u] for u in nodes])

    cfg_topo = TALouvainConfig(lambda1=1.0, lambda2=0.0, gamma=0.0)
    r_topo = ta_louvain(nodes, edges, attrs, cfg_topo)
    r_ta = ta_louvain(nodes, edges, attrs, TALouvainConfig())
    print(f"topology-only : {len(r_topo.groups):3d} groups  NMI={nmi(truth, r_topo.labels):.3f}  Q'={r_topo.q_prime:.3f}")
    print(f"TA-Louvain    : {len(r_ta.groups):3d} groups  NMI={nmi(truth, r_ta.labels):.3f}  Q'={r_ta.q_prime:.3f}")

    # tracking test: 2 slices with 90% member overlap
    s2_nodes = nodes[: int(len(nodes) * 0.9)]
    res = run_over_slices(
        [
            {"nodes": nodes, "edges": edges, "attrs": attrs},
            {"nodes": s2_nodes, "edges": [e for e in edges if e[0] in set(s2_nodes) and e[1] in set(s2_nodes)], "attrs": attrs},
        ]
    )
    print("persistent ids slice1:", sorted(res[0].persistent_ids.values()))
    print("persistent ids slice2:", sorted(res[1].persistent_ids.values()))
