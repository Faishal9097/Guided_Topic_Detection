"""Build per-topic, per-slice structural features that do NOT depend on the arbitrary embedding
coordinates (group sizes, attention entropy, influence, cosine-to-previous-slice, group turnover).
Output npz is aligned to the topic order of the feature npz.

  python group4_build_struct.py --npz W.npz --chunks CHUNKDIR --kind weibo --out weibo_struct.npz
(kind: 'weibo' or 'fnn')"""
import argparse

import numpy as np

from group4_common import load_chunks

NAMES = ["log_n_groups", "log_users", "log_mean_size", "max_size_share", "singleton_share",
         "mean_attn_entropy", "mean_max_attn", "mean_influence", "cos_prev_mean",
         "new_group_frac", "dissolved_frac"]


def slice_feats(sl, prev):
    g = sl.get("groups", {})
    if not g:
        return np.zeros(len(NAMES), np.float32), {}
    sizes = np.array([len(v["members"]) for v in g.values()], float)
    ent, mx = [], []
    for v in g.values():
        a = np.asarray(v["attn"], float)
        a = a / (a.sum() + 1e-12)
        ent.append(float(-(a * np.log(a + 1e-12)).sum() / np.log(len(a))) if len(a) > 1 else 0.0)
        mx.append(float(a.max()))
    infl = np.asarray(sl.get("influence", []), float)
    cos = []
    for gid, v in g.items():
        if gid in prev:
            a, b = np.asarray(v["vec"], float), np.asarray(prev[gid], float)
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    new_frac = float(np.mean([gid not in prev for gid in g])) if prev else 0.0
    dis_frac = (sum(1 for gid in prev if gid not in g) / max(len(prev), 1)) if prev else 0.0
    f = [np.log1p(len(g)), np.log1p(sizes.sum()), np.log1p(sizes.mean()), sizes.max() / sizes.sum(),
         float(np.mean(sizes == 1)), np.mean(ent), np.mean(mx), float(infl.mean()) if infl.size else 0.0,
         float(np.mean(cos)) if cos else 0.0, new_frac, dis_frac]
    return np.asarray(f, np.float32), {gid: v["vec"] for gid, v in g.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--kind", required=True, choices=["weibo", "politifact", "gossipcop"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=False)
    ids, T = z["topic_ids"].astype(str), z["mask"].shape[1]
    topics = load_chunks(a.chunks, a.kind)
    S = np.zeros((len(ids), T, len(NAMES)), np.float32)
    missing = 0
    for n, tid in enumerate(ids):
        sl = topics.get(tid)
        if sl is None and tid.isdigit():
                sl = topics.get(int(tid))
        if sl is None:
            missing += 1
            continue
        prev = {}
        for t, s in enumerate(sl[:T]):
            S[n, t], prev = slice_feats(s, prev)
    np.savez_compressed(a.out, S=S, names=np.array(NAMES), topic_ids=ids)
    print(f"saved {a.out}: S{S.shape}; topics missing from chunks: {missing}/{len(ids)}")


if __name__ == "__main__":
    main()