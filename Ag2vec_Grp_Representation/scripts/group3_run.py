"""
Batch runner for steps 1-3.  Chunked + resumable (re-run the same call after a Colab disconnect and finished
chunks are skipped), multiprocess, results saved compactly (float32) to a folder - point it at Drive.

    import sys; sys.path.insert(0, '/content/g3')
    import group3_run as G
    from group3_steps1_3 import Cfg

    G.run(wb, 'weibo', 'train', '/content/drive/MyDrive/guided-topic-detection_2/data/processed/group3_out', Cfg(), limit=20)
    G.run(fn, 'fnn',   'train', OUT, Cfg())        # fn = FakeNewsNetData('politifact'); use a different OUT per dataset

Each chunk file <out>/<kind>_<split>_<NNNN>.pkl holds {"cfg", "topics": {topic_id: [per-slice dict]}, "errors"}.
Per-slice dict: slice_index, n_users, reply_edge_cover, groups {gid: {vec, members, attn}},
influence_users, influence.   (Everything Group 3's stance step and Group 4's matrix needs.)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from dataclasses import asdict

import numpy as np

from group3_steps1_3 import Cfg, influence_upfd, run_static_topic, run_weibo_topic

DATA, KIND, CFG = None, None, Cfg()      # set by run() before the pool is forked
def _init_worker(data, kind, cfg):
    global DATA, KIND, CFG
    DATA, KIND, CFG = data, kind, cfg
    
    

def compact(res):
    out = []
    for s in res:
        groups = {}
        for gid, g in s["groups"].items():
            members = list(g["attn"].keys())
            groups[gid] = {"vec": g["vec"].astype(np.float32), "members": members,
                           "attn": np.array([g["attn"][u] for u in members], np.float32)}
        infl = s.get("influence", {})
        out.append({"slice_index": s["slice_index"], "n_users": s.get("n_users", 0),
                    "reply_edge_cover": s.get("reply_edge_cover"),
                    "degenerate_frac": s.get("degenerate_frac"), "groups": groups,
                    "influence_users": list(infl.keys()),
                    "influence": np.array(list(infl.values()), np.float32)})
    return out


def _work(tid):
    try:
        if KIND == "weibo":
            res = run_weibo_topic(DATA.topic_view(tid), CFG)
        else:
            res = run_static_topic(DATA.topic_view(tid), CFG,
                                   influence_fn=lambda users, view: influence_upfd(users, view, CFG))
        return tid, compact(res), None
    except Exception as e:                      # keep going; failures are listed in the chunk file
        return tid, None, repr(e)


def run(data, kind, split, out_dir, cfg: Cfg = Cfg(), chunk=50, workers=2, limit=None):
    """kind: 'weibo' | 'fnn'.  split: 'train' | 'val' | 'test' | 'all'."""
    global DATA, KIND, CFG
    DATA, KIND, CFG = data, kind, cfg
    os.makedirs(out_dir, exist_ok=True)
    ids = data.topic_ids(None if split == "all" else split)
    if limit:
        ids = ids[:limit]
    # fork = children inherit `data` copy-on-write (Linux / macOS / Colab). Windows has no fork -> run sequentially.
    use_pool = workers > 1 and "fork" in mp.get_all_start_methods()
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    if workers > 1 and not use_pool:
        print("fork not available on this OS: running with 1 process (use WSL/Linux for parallel runs)")
    t_all = time.time()
    for ci, s in enumerate(range(0, len(ids), chunk)):
        path = os.path.join(out_dir, f"{kind}_{split}_{ci:04d}.pkl")
        if os.path.exists(path):
            continue
        part, t0 = ids[s:s + chunk], time.time()
        if use_pool:
            with ctx.Pool(
                workers,
                initializer=_init_worker,
                initargs=(data, kind, cfg)
            ) as pool:
                results = pool.map(_work, part, chunksize=1)
        else:
            results = [_work(t) for t in part]
        topics = {tid: comp for tid, comp, err in results if err is None}
        errors = {tid: err for tid, comp, err in results if err is not None}
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"cfg": asdict(cfg), "topics": topics, "errors": errors}, f, protocol=4)
        os.replace(tmp, path)                   # a chunk file is either complete or absent
        print(f"chunk {ci}: {len(topics)} ok, {len(errors)} errors, {time.time() - t0:.0f}s "
              f"({(time.time() - t0) / len(part):.1f}s/topic) | total {time.time() - t_all:.0f}s", flush=True)
    print("done:", len(ids), "topics ->", out_dir)


def load_all(out_dir, kind, split):
    """Merge every finished chunk: {topic_id: [per-slice dict]}."""
    topics = {}
    for f in sorted(os.listdir(out_dir)):
        if f.startswith(f"{kind}_{split}_") and f.endswith(".pkl"):
            with open(os.path.join(out_dir, f), "rb") as fh:
                topics.update(pickle.load(fh)["topics"])
    return topics
