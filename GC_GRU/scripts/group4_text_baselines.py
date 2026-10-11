"""Weibo text baseline (original post text) vs struct/mut, on the SAME CV folds as group4_baselines.py.
Also stacks the text probability with struct(+mut) to test whether stance adds anything beyond the text."""
import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from group4_common import build_features_arrays, cv_splits, fmt, load_data, metrics, summarize


def pool(F):
    return np.concatenate([F.mean(1), F[:, -1], F[:, -1] - F[:, 0]], 1)


def text_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3, max_features=200000, sublinear_tf=True),
        LogisticRegression(C=3.0, max_iter=2000))


def gb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--struct", required=True)
    ap.add_argument("--om", required=True, help="path to original_microblog_cleaned.json")
    a = ap.parse_args()

    d, struct = load_data(a.npz, a.struct)
    y, ids = d["labels"], d["topic_ids"]
    with open(a.om, encoding="utf-8") as f:
        om = json.load(f)
    texts = np.array([str((om.get(t) or {}).get("text", "")) for t in ids], dtype=object)
    found = int(sum(1 for t in texts if t.strip()))
    print(f"topics with text: {found}/{len(ids)}")
    if found < 0.95 * len(ids):
        raise SystemExit("Too many topics without text: check --om path and the topic-id keys.")

    F_s = pool(build_features_arrays(d["X"], d["mask"], ["struct"], struct))
    F_sm = pool(build_features_arrays(d["X"], d["mask"], ["struct", "mut"], struct))

    res = {k: [] for k in ("text", "struct", "struct+mut", "text+struct", "text+struct+mut")}
    for i, sp in enumerate(cv_splits(y)):
        tr, te = np.isin(sp, ["train", "val"]), sp == "test"
        tm = text_model().fit(texts[tr], y[tr])
        pt_te = tm.predict_proba(texts[te])[:, 1]
        oof = cross_val_predict(text_model(), texts[tr], y[tr], method="predict_proba",
                                cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
        res["text"].append({"test": metrics(y[te], pt_te)})
        for name, F in (("struct", F_s), ("struct+mut", F_sm)):
            m = gb().fit(F[tr], y[tr])
            res[name].append({"test": metrics(y[te], m.predict_proba(F[te])[:, 1])})
            m2 = gb().fit(np.c_[F[tr], oof], y[tr])
            res["text+" + name].append({"test": metrics(y[te], m2.predict_proba(np.c_[F[te], pt_te])[:, 1])})
        print(f"fold {i + 1}/15 done", flush=True)

    for k, runs in res.items():
        print(f"[{k:16s}] {fmt(summarize(runs, 'test'))}")


if __name__ == "__main__":
    main()