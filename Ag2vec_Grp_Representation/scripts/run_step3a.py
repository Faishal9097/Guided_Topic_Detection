"""Step 3a: retrain the stance classifier on the v2 labels, build Mut ('full' vs 'clean'), run the leakage ablation."""
import collections
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Ag2vec_Grp_Representation" / "data_output"

STANCE_CSV = (
    ROOT
    / "Ag2vec_Grp_Representation"
    / "data_output"
    / "stance_sample_labeled.csv"
)

PROBS_PKL = OUT / "stance_probs_v2.pkl"
EXPORT_DIR = OUT


def main():
    os.chdir(ROOT)                                     # WeiboData uses project-relative paths
    import group3_run as G
    import group3_stance as S
    import group3_stance_features as F
    import group3_export as E
    from group3_data_access import WeiboData
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    wb = WeiboData()

    # ---- AG2Vec results (label-free) ----
    steps = {}
    for sp in ("all", "train", "val", "test"):
        steps.update(G.load_all(str(OUT), "weibo", sp))
    steps = {t: v for t, v in steps.items() if t in wb.sliced and t in wb.groups}
    if not steps:
        raise SystemExit(f"No weibo_*.pkl chunks found in {OUT}")
    missing = set(wb.topic_ids()) - set(steps)
    print(f"topics with AG2Vec results: {len(steps)} | topics without: {len(missing)}")

    # ---- stance classifier on the corrected labels ----
    import torch
    print("GPU available:", torch.cuda.is_available())
    embed = S.sbert_embedder()
    df = S.load_labeled(str(STANCE_CSV))
    S.check_overlap(df, wb)
    X = S.features(df.comment, df.source, embed, False)
    res = S.tune_and_report(X, df.label.values)
    clf = S.fit(X, df.label.values, res["C"])

    if PROBS_PKL.exists():
        with open(PROBS_PKL, "rb") as f:
            probs = pickle.load(f)
        print("loaded cached stance probs:", PROBS_PKL)
    else:
        probs = S.predict_topics(wb, list(steps), clf, embed, None, pair=False, save_to=str(PROBS_PKL))

    # ---- leakage ablation ----
    def auc_report(Xm, ids):
        y = np.array([1 if wb.groups[t]["original_label"] == "rumor" else 0 for t in ids])
        sp = np.array([wb.split_of(t) or "none" for t in ids])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(Xm[sp == "train"], y[sp == "train"])
        return {s: round(roc_auc_score(y[sp == s], m.predict_proba(Xm[sp == s])[:, 1]), 3) for s in ("val", "test")}

    def mut_auc(stance):
        agg = collections.defaultdict(list)
        for (tid, t, g), m in stance.items():
            agg[tid].append(m)
        ids = sorted(agg)
        return auc_report(np.array([np.mean(agg[t], 0) for t in ids]), ids)

    def size_auc():                                    # reference: group counts/sizes only, no stance
        ids, rows = sorted(steps), []
        for t in ids:
            ng = [len(s["groups"]) for s in steps[t]]
            ms = [np.mean([len(g["members"]) for g in s["groups"].values()]) if s["groups"] else 0 for s in steps[t]]
            mx = [max([len(g["members"]) for g in s["groups"].values()], default=0) for s in steps[t]]
            rows.append([np.mean(ng), np.mean(ms), np.mean(mx), sum(1 for n in ng if n)])
        return auc_report(np.array(rows, float), ids)

    print("size-only baseline AUC:", size_auc())
    for mode in ("full", "clean"):
        feats = {t: F.topic_features(t, wb, steps[t], probs[t], mode=mode) for t in steps}
        rho = F.fit_rho(feats, [t for t in feats if wb.split_of(t) == "train"])
        stance = {k: v["mut"] for k, v in F.mut_vectors(feats, rho).items()}
        print(mode, "| mean-pooled Mut-only AUC:", mut_auc(stance))
        E.export(steps, E.meta_from(wb), str(EXPORT_DIR / f"weibo_group_feature_matrices_{mode}.npz"),
                 T=10, K=30, slot_mode="stable", stance=stance)


if __name__ == "__main__":                              # required on Windows
    main()