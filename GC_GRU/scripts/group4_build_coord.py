"""Build bot-human coordination features (Gap 1), aligned to the topic order of the Weibo feature npz.

  python group4_build_coord.py --npz W.npz --sliced topics_time_sliced.json --chunks CHUNKDIR --out weibo_coord.npz [--limit 50]

Needs: weibo_group_feature_matrices.npz (topic_ids, labels, mask), topics_time_sliced.json, weibo_all_*.pkl chunks.
Output npz: C (N,T,9) coordination features, V (N,T,2) volume control, names, vnames, topic_ids.
Slice t only uses interactions up to slice t (cumulative, same as Group 3), so no future leakage.
"""
import argparse
import glob
import json
import os
import pickle
import time

import numpy as np

from group4_coord_lib import COORD_NAMES, VOL_NAMES, prep, slice_features


def load_chunks(chunk_dir, kind="weibo"):
    topics = {}
    for f in sorted(glob.glob(os.path.join(chunk_dir, f"{kind}_all_*.pkl"))):
        with open(f, "rb") as fh:
            topics.update(pickle.load(fh).get("topics", {}))
    return topics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--sliced", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only first N topics (smoke test); rest stay zero")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=False)
    ids, y, T = z["topic_ids"].astype(str), z["labels"].astype(int), z["mask"].shape[1]
    print("loading sliced json ...", flush=True)
    with open(a.sliced, encoding="utf-8") as f:
        sliced = {str(t["topic_id"]): t for t in json.load(f)}
    chunks = {str(k): v for k, v in load_chunks(a.chunks).items()}
    print(f"npz topics={len(ids)} T={T} | sliced={len(sliced)} | chunks={len(chunks)} | positive-label share={y.mean():.3f}")

    N = len(ids) if not a.limit else min(a.limit, len(ids))
    C = np.zeros((len(ids), T, len(COORD_NAMES)), np.float32)
    V = np.zeros((len(ids), T, len(VOL_NAMES)), np.float32)
    miss_s = miss_c = 0
    t0 = time.time()
    for n in range(N):
        tid = ids[n]
        top = sliced.get(tid)
        if top is None:
            miss_s += 1
            continue
        ch = chunks.get(tid)
        if ch is None:
            miss_c += 1
        slices = sorted(top["time_slices"], key=lambda s: s["slice_index"])[:T]
        recs = []
        for k, s in enumerate(slices):
            recs += prep(s["new_interactions_this_slice"], k)
            groups = {g: v["members"] for g, v in ch[k]["groups"].items()} if ch and k < len(ch) else {}
            C[n, k], V[n, k] = slice_features(recs, groups)       # recs = everything seen up to slice k
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{N} topics, {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(a.out, C=C, V=V, names=np.array(COORD_NAMES), vnames=np.array(VOL_NAMES), topic_ids=ids)
    print(f"saved {a.out}: C{C.shape} V{V.shape} | missing in sliced={miss_s}, missing in chunks={miss_c} "
          f"| NaN={int(np.isnan(C).sum() + np.isnan(V).sum())} | {time.time() - t0:.0f}s")

    # sanity: univariate AUC of each feature at the last slice (a volume feature with a big AUC = leakage warning)
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y[:N])) > 1:
        print("\nunivariate test (last slice, all built topics) -> AUC vs label")
        for nm, arr in list(zip(COORD_NAMES, C[:N, -1].T)) + list(zip(VOL_NAMES, V[:N, -1].T)):
            if arr.std() > 0:
                print(f"  {nm:18s} auc={roc_auc_score(y[:N], arr):.3f}   mean={arr.mean():.3f}")
            else:
                print(f"  {nm:18s} constant (all {arr.mean():.3f})")


if __name__ == "__main__":
    main()
