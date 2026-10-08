"""
Export Group 3's output in the format Group 2's hand-off proposed for Group 4 (Faisal):

    X     float32 (n_topics, T, K, d)   d = 64 group-vector dims + 3 stance dims (Mut_sup, Mut_opo, Mut_none)
    mask  bool    (n_topics, T, K)      True = real group, False = padding
    gid   int32   (n_topics, T, K)      persistent group id in that slot (-1 = padding)
    labels int8   (n_topics,)           guided = 1
    split  str    (n_topics,)           train / val / test
    topic_ids str (n_topics,)
    stance_included bool                False until steps 4-6 are done (stance dims are zeros then)

Usage (CPU only, no GPU needed):
    import group3_run as G, group3_export as E
    topics = G.load_all(OUT, 'weibo', 'all')
    E.export(topics, E.meta_from(wb), 'weibo_group_feature_matrices.npz', T=10, K=30, slot_mode='stable')

slot_mode (agree with Faisal):
  'stable' - a group keeps the same slot k in every slice it exists, so the GRU can follow it over time
             (a slot is freed when its group dissolves and reused by a new group; if all K slots are taken the
             smallest new group is dropped).
  'size'   - slot = rank by group size in each slice (the hand-off's proposal); slot k holds different groups
             in different slices, so a flattened GRU input cannot follow one group.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def meta_from(data) -> Dict[str, Tuple[int, str]]:
    """{topic_id: (label, split)} from WeiboData / FakeNewsNetData (guided = 1)."""
    out = {}
    for tid in data.topic_ids():
        rec = data.groups[tid]
        out[tid] = (1 if rec["guided_label"] == "guided" else 0, data.split_of(tid) or "none")
    return out


def slot_layout(slices: List[dict], K: int, mode: str) -> List[Dict[str, int]]:
    layouts, slot_of = [], {}
    for s in slices:
        gs = s["groups"]
        if mode == "size":
            ranked = sorted(gs, key=lambda g: (-len(gs[g]["members"]), int(g)))[:K]
            layouts.append({g: k for k, g in enumerate(ranked)})
            continue
        for g in [g for g in slot_of if g not in gs]:              # dissolved groups free their slot
            del slot_of[g]
        used = set(slot_of.values())
        for g in sorted((g for g in gs if g not in slot_of), key=lambda g: (-len(gs[g]["members"]), int(g))):
            free = [k for k in range(K) if k not in used]
            if not free:
                break                                              # no slot left: drop smallest new groups
            slot_of[g] = free[0]
            used.add(free[0])
        layouts.append(dict(slot_of))
    return layouts


def export(topics: Dict[str, List[dict]], meta: Dict[str, Tuple[int, str]], path: str,
           T: int = 10, K: int = 30, slot_mode: str = "stable", dim: int = 64,
           stance: Optional[Dict[Tuple[str, int, str], np.ndarray]] = None):
    """stance: {(topic_id, slice_position, group_id): array(3)} = Mut(g, t); None -> zeros + stance_included=False."""
    ids = sorted(t for t in topics if t in meta)
    N, D = len(ids), dim + 3
    X = np.zeros((N, T, K, D), np.float32)
    mask = np.zeros((N, T, K), bool)
    gid = np.full((N, T, K), -1, np.int32)
    dropped = 0
    for n, tid in enumerate(ids):
        slices = topics[tid]
        layouts = slot_layout(slices, K, slot_mode)
        for t, (s, lay) in enumerate(zip(slices[:T], layouts[:T])):
            dropped += len(s["groups"]) - len(lay)
            for g, k in lay.items():
                X[n, t, k, :dim] = s["groups"][g]["vec"]
                if stance is not None and (tid, t, g) in stance:
                    X[n, t, k, dim:] = stance[(tid, t, g)]
                mask[n, t, k] = True
                gid[n, t, k] = int(g)
    np.savez_compressed(path, X=X, mask=mask, gid=gid,
                        labels=np.array([meta[t][0] for t in ids], np.int8),
                        split=np.array([meta[t][1] for t in ids]),
                        topic_ids=np.array(ids),
                        stance_included=np.array(stance is not None))
    print(f"saved {path}: X{X.shape}, {int(mask.sum())} real group slots, {dropped} groups dropped (K={K}), "
          f"slot_mode={slot_mode}, stance_included={stance is not None}")
    return path
