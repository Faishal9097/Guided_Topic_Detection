"""Interpretability for a model trained on X-derived blocks only (count/vec/mut).
  python group4_interpret.py --npz W.npz --model m.pt --chunks CHUNKDIR --kind weibo --n-topics 100
Group contribution = P_full - P(group removed from pooling), Eq. 42 (removal, not zeroing, because the
input is a pooled mean). Stance dominance SDS = Mut_sup - Mut_opo (Eq. 44). Attention stats come from chunks.
NOTE: Eq. 43's *mean* attention is always 1/|group| (weights sum to 1), so we also report max attention."""
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from group4_common import DEVICE, GCGRU, POOLED, build_features_arrays, load_chunks, load_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--kind", default="weibo")
    ap.add_argument("--n-topics", type=int, default=100)
    ap.add_argument("--tstar", type=int, default=None, help="0-based slice; default = slice with most groups")
    ap.add_argument("--across", action="store_true", help="remove the group in ALL slices (stable slot)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d, _ = load_data(a.npz)
    try:
        ck = torch.load(a.model, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(a.model, map_location="cpu")
    blocks, c = ck["blocks"], ck["cfg"]
    if not all(b in POOLED for b in blocks):
        raise SystemExit("interpretation needs a model trained on blocks from {count,vec,mut}")
    model = GCGRU(ck["in_dim"], c["hidden"], c["layers"], c["dropout"], c["fc"], ck["unit"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    chunks = load_chunks(a.chunks, a.kind) if a.chunks else None

    @torch.no_grad()
    def P(X1, m1):
        F = (build_features_arrays(X1, m1, blocks) - ck["mu"]) / ck["sd"]
        F = torch.tensor(F[:, :ck["T"]], dtype=torch.float32).to(DEVICE)
        return float(torch.softmax(model(F), 1)[0, 1])

    def info(tid, t, gid):
        if chunks is None or tid not in chunks or t >= len(chunks[tid]):
            return None
        g = chunks[tid][t]["groups"]
        return g.get(str(gid)) or g.get(int(gid))

    X, M, y = d["X"], d["mask"], d["labels"]
    rng = np.random.RandomState(a.seed)
    test_idx = np.where(d["split"] == "test")[0]
    sel = rng.choice(test_idx, min(a.n_topics, len(test_idx)), replace=False)
    rows, case = [], None
    D = X.shape[3]
    for n in sel:
        X1, m1 = X[n:n + 1], M[n:n + 1]
        p_full = P(X1, m1)
        cnt = m1[0].sum(1)
        ts = a.tstar if a.tstar is not None else int(np.where(cnt == cnt.max())[0][-1])
        slots = np.where(m1[0, ts])[0]
        if len(slots) < 2:
            continue
        trows = []
        for k in slots:
            m2 = m1.copy()
            if a.across:
                m2[0, :, k] = False
            else:
                m2[0, ts, k] = False
            r = {"topic": d["topic_ids"][n], "slot": int(k), "dP": p_full - P(X1, m2),
                 "sds": float(X1[0, ts, k, D - 3] - X1[0, ts, k, D - 2]), "size": np.nan,
                 "abar": np.nan, "maxattn": np.nan}
            g = info(d["topic_ids"][n], ts, d["gid"][n, ts, k])
            if g is not None:
                at = np.asarray(g["attn"], float)
                r.update(size=len(at), abar=float(at.mean()), maxattn=float(at.max()))
            trows.append(r)
        rows += trows
        if y[n] == 1 and (case is None or p_full > case[0]):
            case = (p_full, n, ts, trows)

    print(f"{len(rows)} (topic, group) pairs from {len(sel)} test topics")
    dP = np.array([r["dP"] for r in rows])
    for name, v in (("|SDS|", np.abs([r["sds"] for r in rows])), ("max attention", [r["maxattn"] for r in rows]),
                    ("group size", [r["size"] for r in rows])):
        v = np.asarray(v, float)
        ok = ~np.isnan(v)
        if ok.sum() > 3:
            print(f"Spearman(dP, {name}) = {spearmanr(dP[ok], v[ok]).correlation:+.3f}  (n={ok.sum()})")

    if case:
        p_full, n, ts, trows = case
        top = sorted(trows, key=lambda r: -r["dP"])[:3]
        print(f"\nCase topic {d['topic_ids'][n]}  t*={ts}  P_guided={p_full:.3f}")
        print("rank slot  size  abar(=1/size)  maxattn   SDS     dP")
        for i, r in enumerate(top, 1):
            print(f"{i:>4} {r['slot']:>4} {r['size']:>5} {r['abar']:>12.3f} {r['maxattn']:>9.3f} {r['sds']:>+7.3f} {r['dP']:>+7.3f}")
        print("\nStance perturbation (Mut_sup*gamma, Mut_opo/gamma) -> P_guided")
        for r in top:
            out = []
            for gm in (0.8, 1.0, 1.2):
                X2 = X[n:n + 1].copy()
                X2[0, ts, r["slot"], D - 3] *= gm
                X2[0, ts, r["slot"], D - 2] /= gm
                out.append(f"g={gm}: {P(X2, M[n:n + 1]):.3f}")
            print(f"slot {r['slot']:>2}  " + "   ".join(out))


if __name__ == "__main__":
    main()