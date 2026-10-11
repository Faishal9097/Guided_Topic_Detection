import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from group4_common import build_features_arrays, cv_splits, fmt, load_data, metrics, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--blocks", default="size")
    a = ap.parse_args()
    d, struct = load_data(a.npz, a.struct)
    y = d["labels"]
    F = build_features_arrays(d["X"], d["mask"], a.blocks.split(","), struct)
    Fg = np.concatenate([F.mean(1), F[:, -1], F[:, -1] - F[:, 0]], 1) if F.shape[1] > 1 else F[:, 0]
    runs = []
    for sp in cv_splits(y):
        tr, te = np.isin(sp, ["train", "val"]), sp == "test"
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(Fg[tr], y[tr])
        runs.append({"test": metrics(y[te], m.predict_proba(Fg[te])[:, 1])})
    print(f"GB {a.blocks}:", fmt(summarize(runs, "test")))


if __name__ == "__main__":
    main()