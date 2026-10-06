"""
Sanity checks for TA-Louvain output (Weibo topics_groups.json or FakeNewsNet *_topics_groups.json).

Reports: users/groups per slice, group-size distribution, singleton share, largest-group
share, modularity, slice-to-slice persistence (static FakeNewsNet files have 1 slice, so
persistence is skipped) and guided vs non-guided summaries.

Optional ablation (Weibo only): --ablation N --sliced <time-sliced json> re-runs the LAST
slice of N topics with topology-only vs full TA-Louvain and compares group structure
(cf. paper Fig. 15-16).  There is no ground truth for Weibo groups, so Q/NMI vs truth
cannot be computed; fragmentation (singleton share) is the meaningful comparison.

  python scripts/check_groups.py --groups data/processed/weibo_ced/topics_groups.json
  python scripts/check_groups.py --groups ... --ablation 100 --sliced data/processed/weibo_ced/topics_time_sliced.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def summarise(topics):
    rows = defaultdict(list)
    for t in topics:
        for s in t["slices"]:
            sizes = [len(m) for m in s["groups"].values()]
            if not sizes:
                continue
            n = sum(sizes)
            rows["n_users"].append(n)
            rows["n_groups"].append(len(sizes))
            rows["largest_share"].append(max(sizes) / n)
            rows["singleton_share"].append(sum(1 for x in sizes if x == 1) / len(sizes))
            rows["modularity"].append(s["modularity"])
    print("\n== per-slice structure (mean / median) ==")
    for k, v in rows.items():
        print(f"  {k:16s} {np.mean(v):10.3f} / {np.median(v):10.3f}")

    pers_id, pers_j = [], []
    for t in topics:
        for a, b in zip(t["slices"][:-1], t["slices"][1:]):
            if not a["groups"] or not b["groups"]:
                continue
            common = set(a["groups"]) & set(b["groups"])
            pers_id.append(len(common) / len(b["groups"]))
            pers_j.extend(jaccard(a["groups"][g], b["groups"][g]) for g in common)
    if pers_id:
        print("\n== tracking across slices ==")
        print(f"  groups keeping their id from the previous slice: {np.mean(pers_id):.3f}")
        print(f"  mean Jaccard of matched groups:                   {np.mean(pers_j):.3f}")

    print("\n== guided vs non-guided (last slice) ==")
    for lab in ("guided", "nonguided", "non-guided"):
        sel = [t for t in topics if t.get("guided_label") == lab and t["slices"][-1]["groups"]]
        if not sel:
            continue
        last = [t["slices"][-1] for t in sel]
        print(f"  {lab:11s} topics={len(sel):5d}  groups={np.mean([s['n_groups'] for s in last]):7.1f}  "
              f"users={np.mean([s['n_users'] for s in last]):7.1f}  "
              f"largest-share={np.mean([max(len(m) for m in s['groups'].values()) / s['n_users'] for s in last]):.3f}")


def ablation(sliced_path, n):
    from run_weibo_ta_louvain import process_topic
    with open(sliced_path, encoding="utf-8") as f:
        topics = json.load(f)[:n]
    print(f"\n== ablation on {len(topics)} topics (last slice) ==")
    for name, cfg in [("topology-only", dict(lambda1=1.0, lambda2=0.0, gamma=0.0, attr_knn=0)),
                      ("TA-Louvain   ", dict(attr_knn=10))]:
        ng, sing, big = [], [], []
        for t in topics:
            r = process_topic((t, cfg, True))
            if not r:
                continue
            s = r["slices"][-1]
            sizes = [len(m) for m in s["groups"].values()]
            ng.append(len(sizes))
            sing.append(sum(1 for x in sizes if x == 1) / len(sizes))
            big.append(max(sizes) / s["n_users"])
        print(f"  {name} groups={np.mean(ng):7.1f}  singleton share={np.mean(sing):.3f}  largest-group share={np.mean(big):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--ablation", type=int, default=0)
    ap.add_argument(
        "--sliced",
        default="Preprocessing/data/processed/weibo_ced/topics_time_sliced.json"
    )
    a = ap.parse_args()
    with open(a.groups, encoding="utf-8") as f:
        summarise(json.load(f))
    if a.ablation:
        ablation(a.sliced, a.ablation)
