from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Set
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUP2_SCRIPTS = ROOT / "Group_Detection_Tracking" / "scripts"

if str(GROUP2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GROUP2_SCRIPTS))

from run_weibo_ta_louvain import FEATURES, _parse, user_features

__all__ = ["FEATURES", "WeiboData", "FakeNewsNetData", "group_member_sequences"]

WEIBO_DIR = "Preprocessing/data/processed/weibo_ced"
FNN_DIR = "Preprocessing/data/processed/fakenewsnet"
GROUP_DIR = "Group_Detection_Tracking/data_output"

def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _id_set(path) -> Set[str]:
    obj = _load(path)
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                obj = v
                break
    if obj and isinstance(obj[0], dict):
        return {str(t["topic_id"]) for t in obj}
    return {str(t) for t in obj}

def _split_map(prefix: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ("train", "val", "test"):
        p = f"{prefix}split_{name}.json"
        if os.path.exists(p):
            for tid in _id_set(p):
                out[tid] = name
    return out

def group_member_sequences(group_record: dict) -> Dict[str, List[Set]]:
    ids = sorted({g for s in group_record["slices"] for g in s["groups"]}, key=int)
    return {g: [set(s["groups"].get(g, [])) for s in group_record["slices"]] for g in ids}

class WeiboData:
    def __init__(
        self,
        base: str = WEIBO_DIR,
        sliced: Optional[str] = None,
        groups: Optional[str] = None
    ):
        self.base = base

        self.sliced = {
            t["topic_id"]: t
            for t in _load(sliced or f"{base}/topics_time_sliced.json")
        }

        self.groups = {
            t["topic_id"]: t
            for t in _load(
                groups or "Group_Detection_Tracking/data_output/topics_groups.json"
            )
        }
        self._split = _split_map(f"{base}/topics_")

    def topic_ids(self, split: Optional[str] = None) -> List[str]:
        ids = [t for t in self.groups if t in self.sliced]
        return [t for t in ids if self._split.get(t) == split] if split else ids

    def split_of(self, topic_id: str) -> Optional[str]:
        return self._split.get(topic_id)

    def topic_view(self, topic_id: str) -> List[dict]:
        topic, grec = self.sliced[topic_id], self.groups[topic_id]
        slices = sorted(topic["time_slices"], key=lambda s: s["slice_index"])
        all_inter = [i for s in slices for i in s["new_interactions_this_slice"]]
        mid2uid = {i["mid"]: i["uid"] for i in all_inter}
        t0 = min(_parse(i["date"]) for i in all_inter) if all_inter else 0.0
        seen: List[dict] = []
        out = []
        for s, gs in zip(slices, grec["slices"]):
            new = s["new_interactions_this_slice"]
            seen = seen + new
            edges = [(i["uid"], mid2uid[i["parent"]]) for i in seen
                     if i["parent"] and i["parent"] in mid2uid and mid2uid[i["parent"]] != i["uid"]]
            user_group = {u: g for g, members in gs["groups"].items() for u in members}
            out.append({
                "slice_index": s["slice_index"],
                "boundary_time": s["boundary_time"],
                "groups": gs["groups"],
                "user_group": user_group,
                "modularity": gs["modularity"],
                "interactions": seen,
                "new_interactions": new,
                "edges": edges,
                "user_features": {u: np.asarray(v) for u, v in user_features(seen, t0).items()} if seen else {},
                "topic_heat": s["cumulative_interaction_count"],
            })
        return out

class FakeNewsNetData:
    def __init__(self, dataset: str, base: str = FNN_DIR):
        import scipy.sparse as sp
        self.dataset = dataset
        pre = f"{base}/{dataset}"

        self.topics = {
            t["topic_id"]: t
            for t in _load(f"{pre}_topics_with_categories.json")
        }

        group_path = os.path.join(GROUP_DIR, f"{dataset}_topics_groups.json")
        self.groups = {
            t["topic_id"]: t
            for t in _load(group_path)
        }

        self._edges = np.loadtxt(
            f"{pre}_edges_cleaned.txt",
            delimiter=",",
            dtype=np.int64,
            ndmin=2
        )

        self._feats = sp.load_npz(
            f"{pre}_new_profile_feature.npz"
        ).tocsr()

        n_max = int(max(max(t["node_indices"]) for t in self.topics.values()))
        self._owner = np.full(n_max + 1, -1, dtype=np.int64)
        self._names = list(self.topics)

        for k, name in enumerate(self._names):
            self._owner[
                np.asarray(
                    self.topics[name]["node_indices"],
                    dtype=np.int64
                )
            ] = k

        self._split = _split_map(f"{pre}_")

    def topic_ids(self, split: Optional[str] = None) -> List[str]:
        ids = list(self.groups)
        return [t for t in ids if self._split.get(t) == split] if split else ids

    def split_of(self, topic_id: str) -> Optional[str]:
        return self._split.get(topic_id)

    def topic_view(self, topic_id: str) -> dict:
        t = self.topics[topic_id]
        k = self._names.index(topic_id)
        mask = self._owner[self._edges[:, 0]] == k
        nodes = sorted(int(x) for x in t["node_indices"])
        root = nodes[0]
        e = self._edges[mask]
        gs = self.groups[topic_id]["slices"][0]
        feats = self._feats[nodes].toarray()
        return {
            "topic_id": topic_id,
            "label": t["guided_label"],
            "original_label": t["original_label"],
            "category": t.get("topic_category"),
            "root": root,
            "nodes": nodes,
            "edges": [(int(u), int(v)) for u, v in e],
            "features": {n: feats[i] for i, n in enumerate(nodes)},
            "groups": gs["groups"],
            "user_group": {u: g for g, m in gs["groups"].items() for u in m},
            "modularity": gs["modularity"],
        }