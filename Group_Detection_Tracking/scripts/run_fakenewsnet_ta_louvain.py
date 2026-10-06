"""
Run TA-Louvain on FakeNewsNet / UPFD (Politifact, Gossipcop).

UPFD has no timestamps, so there are no time slices: TA-Louvain runs ONCE per topic on
the full propagation graph.  The output uses the same schema as the Weibo output with a
single slice, so downstream code can treat both datasets alike.

Graph per topic
  nodes : topic["node_indices"] (global UPFD node ids)
  edges : rows of <dataset>_edges_cleaned.txt ("u,v"), assigned to the topic that owns u
  root  : the lowest node index of a topic is the NEWS node (all first-level tweeters
          connect to it).  It is a hub, not a user, and would glue everything into one
          group, so it is removed before grouping (use --keep-root to disable).
  attrs : optional --features file (UPFD profile features, row i = global node i),
          .npz (scipy sparse) or .npy.  If omitted, <dataset>_new_profile_feature.npz in
          data/processed/fakenewsnet/ is used when it exists; otherwise only topology is used.

Run from the project root:
python Group_Detection_Tracking/scripts/run_fakenewsnet_ta_louvain.py --dataset politifact
python Group_Detection_Tracking/scripts/run_fakenewsnet_ta_louvain.py --dataset gossipcop
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ta_louvain import TALouvainConfig, standardize_attributes, ta_louvain


def load_features(path):
    """Load UPFD node features; row i must describe global node i."""
    if path is None:
        return None
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        if {"indices", "indptr", "data"} <= set(z.files):      # scipy sparse matrix
            try:
                import scipy.sparse as sp
            except ImportError:
                raise SystemExit("This feature file is a scipy sparse matrix. Install scipy first:  pip install scipy")
            return sp.load_npz(path).tocsr()
        return z[z.files[0]]                                     # plain numpy archive
    return np.load(path)


def process_topic(job):
    meta, nodes, edges, feats, cfg_kwargs = job
    cfg = TALouvainConfig(**cfg_kwargs)
    attrs = None
    if feats is not None:
        attrs = standardize_attributes({n: feats[i] for i, n in enumerate(nodes)})
    r = ta_louvain(nodes, edges, attrs, cfg)
    groups = {str(g): sorted(int(u) for u in m) for g, m in r.groups.items()}
    return {
        **meta,
        "static": True,
        "slices": [{
            "slice_index": 1,
            "n_users": len(r.labels),
            "n_groups": len(groups),
            "modularity": round(r.q_prime, 4),
            "groups": groups,
        }],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["politifact", "gossipcop"])
    ap.add_argument("--topics", default=None)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--features", default=None, help="UPFD node feature file (npz/npy), optional")
    ap.add_argument("--output", default=None)
    ap.add_argument("--keep-root", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--lambda1", type=float, default=0.7)
    ap.add_argument("--lambda2", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.4)
    ap.add_argument("--attr-knn", type=int, default=10)
    a = ap.parse_args()
    base = os.path.join(
        "Preprocessing",
        "data",
        "processed",
        "fakenewsnet"
    )

    topics_path = a.topics or os.path.join(
        base, f"{a.dataset}_topics_with_categories.json"
    )

    edges_path = a.edges or os.path.join(
        base, f"{a.dataset}_edges_cleaned.txt"
    )

    out_path = a.output or os.path.join(
        "Group_Detection_Tracking",
        "data_output",
        f"{a.dataset}_topics_groups.json"
    )
    t0 = time.time()
    if a.features is None:
        feature_path = os.path.join(
            "Group_Detection_Tracking",
            "data_output",
            f"{a.dataset}_new_profile_feature.npz"
        )
        if os.path.exists(feature_path):
            a.features = feature_path
            print(f"using features: {a.features}")

    with open(topics_path, encoding="utf-8") as f:
        topics = json.load(f)
    if a.limit:
        topics = topics[: a.limit]
    edges = np.loadtxt(edges_path, delimiter=",", dtype=np.int64, ndmin=2)
    feats = load_features(a.features)
    if feats is not None:
        print(f"feature matrix shape: {feats.shape}")

    # global node id -> owning topic
    n_max = max(max(t["node_indices"]) for t in topics)
    if feats is not None and feats.shape[0] <= n_max:
        raise SystemExit(f"feature file has {feats.shape[0]} rows but topics use node index up to {n_max}; "
                         "wrong file or wrong dataset?")
    owner = np.full(n_max + 1, -1, dtype=np.int64)
    for k, t in enumerate(topics):
        owner[np.asarray(t["node_indices"], dtype=np.int64)] = k
    keep = (edges[:, 0] <= n_max) & (edges[:, 1] <= n_max)
    edges = edges[keep]
    ou, ov = owner[edges[:, 0]], owner[edges[:, 1]]
    cross = int(np.sum((ou != ov) & (ou >= 0) & (ov >= 0)))
    edges, ou = edges[(ou == ov) & (ou >= 0)], ou[(ou == ov) & (ou >= 0)]
    order = np.argsort(ou, kind="stable")
    edges, ou = edges[order], ou[order]
    starts = np.searchsorted(ou, np.arange(len(topics) + 1))
    print(f"topics={len(topics)}  edges kept={len(edges)}  cross-topic edges dropped={cross}")

    cfg_kwargs = dict(lambda1=a.lambda1, lambda2=a.lambda2, gamma=a.gamma, attr_knn=a.attr_knn)
    jobs, root_is_max_deg, root_only_source = [], 0, 0
    for k, t in enumerate(topics):
        idx = sorted(int(x) for x in t["node_indices"])
        e = edges[starts[k]:starts[k + 1]]
        root = idx[0]
        deg = np.bincount(e.ravel(), minlength=n_max + 1)[idx] if len(e) else np.zeros(len(idx))
        root_is_max_deg += int(deg.argmax() == 0) if len(e) else 0
        if len(e):                       # in a parent -> child edge list only the root is never a target
            targets = set(e[:, 1].tolist())
            root_only_source += int([n for n in idx if n not in targets] == [root])
        nodes = idx if a.keep_root else idx[1:]
        e_use = [(int(u), int(v)) for u, v in e if a.keep_root or (u != root and v != root)]
        f = None
        if feats is not None:
            f = feats[nodes]
            f = f.toarray() if hasattr(f, "toarray") else np.asarray(f)
        meta = {k2: t.get(k2) for k2 in ("topic_id", "guided_label", "original_label", "topic_category")}
        meta["root_removed"] = not a.keep_root
        jobs.append((meta, nodes, e_use, f, cfg_kwargs))
    print(f"root check 1: the lowest node index is the ONLY node that never appears as an edge target "
          f"in {root_only_source}/{len(topics)} topics (confirms it is the news root if edges are parent->child)")
    print(f"root check 2: the lowest node index has the highest degree in {root_is_max_deg}/{len(topics)} topics "
          "(informational: a popular first-level tweeter can out-rank the root)")

    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(process_topic, jobs, chunksize=16))
    else:
        res = [process_topic(j) for j in jobs]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    ng = [r["slices"][0]["n_groups"] for r in res]
    print(f"mean groups/topic={np.mean(ng):.1f}  time={time.time() - t0:.0f}s  -> {out_path}")


if __name__ == "__main__":
    main()
