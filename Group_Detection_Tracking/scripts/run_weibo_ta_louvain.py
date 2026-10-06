"""
Run TA-Louvain on topics_time_sliced.json (Weibo CED, 10 equal-width slices / topic).

Per topic and per slice t:
  nodes  = users that have interacted up to slice t   (cumulative=True, default)
           or only in slice t                           (--per-slice)
  edges  = user -> user of the interaction it replies to (via `parent` -> `mid`).
           Interactions with empty `parent` reply to the source post; the source
           author is NOT in this file and a root hub would glue everything into a
           single group, so those interactions create no edge (the user is still a
           node and can be grouped through attribute similarity).
  attrs  = behavioural features (no profile data exists for commenters, see below)
Groups are tracked across slices with Jaccard > 0.5 (Eq.16).

Usage (run from the project root; both scripts live in scripts/):
  python scripts/run_weibo_ta_louvain.py --limit 50      # quick test
  python scripts/run_weibo_ta_louvain.py --workers 6     # full run
Defaults: input  data/processed/weibo_ced/topics_time_sliced.json
          output data/processed/weibo_ced/topics_groups.json
          
          
To RUN: python Group_Detection_Tracking/scripts/run_weibo_ta_louvain.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ta_louvain import TALouvainConfig, run_over_slices, standardize_attributes

EMOJI = re.compile(r"\[[^\[\]]{1,8}\]")      # Weibo emoticons like [蜡烛]
MENTION = re.compile(r"@[\w\u4e00-\u9fff-]+")
URL = re.compile(r"https?://|t\.cn/")
FEATURES = [
    "n_interactions", "mean_log_len", "empty_frac", "emoji_per_msg", "mention_per_msg",
    "url_frac", "reply_to_user_frac", "log_first_delay_min", "log_mean_delay_min",
    "uid_numeric", "uid_len",
]
LOG_COLS = [0]  # n_interactions is heavy tailed


def _parse(ts: str) -> float:
    from datetime import datetime
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()


def user_features(interactions, t0):
    """Behavioural attribute vector per user from the interactions we have for it.
    (Weibo CED gives profile data only for original posters, so 'age/sex/followers'
    from Eq.9 are unavailable for commenters; these are proxies.)"""
    by_user = defaultdict(list)
    for i in interactions:
        by_user[i["uid"]].append(i)
    raw = {}
    for u, items in by_user.items():
        texts = [x["text"] or "" for x in items]
        delays = [max(_parse(x["date"]) - t0, 0) / 60.0 for x in items]
        raw[u] = [
            len(items),
            float(np.mean([math.log1p(len(t)) for t in texts])),
            float(np.mean([len(t.strip()) == 0 for t in texts])),
            float(np.mean([len(EMOJI.findall(t)) for t in texts])),
            float(np.mean([len(MENTION.findall(t)) for t in texts])),
            float(np.mean([bool(URL.search(t)) for t in texts])),
            float(np.mean([bool(x["parent"]) for x in items])),
            math.log1p(min(delays)),
            math.log1p(float(np.mean(delays))),
            1.0 if u.isdigit() else 0.0,
            float(len(u)),
        ]
    return raw


def process_topic(args):
    topic, cfg_kwargs, cumulative = args
    cfg = TALouvainConfig(**cfg_kwargs)
    all_inter = [i for s in topic["time_slices"] for i in s["new_interactions_this_slice"]]
    if not all_inter:
        return None
    mid2uid = {i["mid"]: i["uid"] for i in all_inter}
    t0 = min(_parse(i["date"]) for i in all_inter)

    slices_in, seen = [], []
    for s in sorted(topic["time_slices"], key=lambda x: x["slice_index"]):
        new = s["new_interactions_this_slice"]
        seen = seen + new if cumulative else new
        if not seen:
            slices_in.append({"nodes": [], "edges": [], "attrs": None})
            continue
        nodes = sorted({i["uid"] for i in seen})
        edges = []
        for i in seen:
            p = i["parent"]
            if p and p in mid2uid and mid2uid[p] != i["uid"]:
                edges.append((i["uid"], mid2uid[p]))
        attrs = standardize_attributes(user_features(seen, t0), log_cols=LOG_COLS)
        slices_in.append({"nodes": nodes, "edges": edges, "attrs": attrs})

    results = run_over_slices(slices_in, cfg)
    out_slices = []
    for s, r in zip(sorted(topic["time_slices"], key=lambda x: x["slice_index"]), results):
        groups = {}
        for lid, members in r.groups.items():
            groups[str(r.persistent_ids[lid])] = sorted(members, key=str)
        out_slices.append({
            "slice_index": s["slice_index"],
            "n_users": len(r.labels),
            "n_groups": len(groups),
            "modularity": round(r.q_prime, 4),
            "groups": groups,                       # persistent group id -> member uids
        })
    return {
        "topic_id": topic["topic_id"],
        "guided_label": topic["guided_label"],
        "original_label": topic["original_label"],
        "slices": out_slices,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"Preprocessing/data/processed/weibo_ced/topics_time_sliced.json")
    ap.add_argument("--output", default=r"Group_Detection_Tracking/data_output/topics_groups.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--per-slice", action="store_true", help="graph from each slice's new interactions only")
    ap.add_argument("--lambda1", type=float, default=0.7)
    ap.add_argument("--lambda2", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.4)
    ap.add_argument("--attr-knn", type=int, default=10)
    a = ap.parse_args()

    t_start = time.time()
    with open(a.input, encoding="utf-8") as f:
        topics = json.load(f)
    if a.limit:
        topics = topics[: a.limit]
    cfg_kwargs = dict(lambda1=a.lambda1, lambda2=a.lambda2, gamma=a.gamma, attr_knn=a.attr_knn)
    jobs = [(t, cfg_kwargs, not a.per_slice) for t in topics]

    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(process_topic, jobs, chunksize=8))
    else:
        res = []
        for k, j in enumerate(jobs, 1):
            res.append(process_topic(j))
            if k % 100 == 0:
                print(f"{k}/{len(jobs)} topics, {time.time() - t_start:.0f}s")
    res = [r for r in res if r]

    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)

    ng = [s["n_groups"] for r in res for s in r["slices"]]
    nu = [s["n_users"] for r in res for s in r["slices"]]
    print(f"topics={len(res)}  mean groups/slice={np.mean(ng):.1f}  mean users/slice={np.mean(nu):.1f}  "
          f"time={time.time() - t_start:.0f}s  -> {a.output}")


if __name__ == "__main__":
    main()
