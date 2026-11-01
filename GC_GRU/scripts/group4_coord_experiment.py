"""Gap 1 ablation: does bot-human coordination add anything on top of struct+mut?  Same folds, same seeds for every row.

  python group4_coord_experiment.py --npz W.npz --struct weibo_struct.npz --coord weibo_coord.npz \
         --out results_coord.json [--max-folds 2 --epochs 2]   (smoke test)

Rows (all use repeated stratified 5x3 CV = 15 folds from group4_common.cv_splits, fold i uses model seed i):
  base          struct + mut
  +coord        struct + mut + coord(9)
  +volume       struct + mut + volume(2)          <- control: does a plain volume feature explain any gain?
  +coord_shuf   struct + mut + coord rows shuffled across topics   <- control: same dims, no information
  GB variants   HistGradientBoosting on the flattened features (--gb)
The claim "coord helps" needs: +coord > base AND +coord > +volume AND +coord > +coord_shuf.
"""
import argparse
import json

import numpy as np
from scipy.stats import t as t_dist

from group4_common import DEFAULT_CFG, build_features_arrays, cv_splits, fmt, load_data, metrics, run_seeds, summarize


def load_coord(path, topic_ids):
    z = np.load(path, allow_pickle=False)
    if not np.array_equal(z["topic_ids"].astype(str), topic_ids):
        raise SystemExit("coord file topic_ids do not match the feature npz (rebuild it from the same npz).")
    return z["C"].astype(np.float32), z["V"].astype(np.float32), [str(x) for x in z["names"]]


def cv_gru(F, y, cfg, max_folds, unit="gru"):
    runs = []
    for i, sp in enumerate(cv_splits(y)):
        if max_folds and i >= max_folds:
            break
        r, _, _ = run_seeds(F, y, sp, cfg, [i], unit)
        runs.append(r[0])
        print(f"    fold {i + 1}: auc={r[0]['test']['auc']:.3f} pr_auc={r[0]['test']['pr_auc']:.3f}", flush=True)
    return runs


def cv_gb(F, y, max_folds):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, runs = F.reshape(len(y), -1), []
    for i, sp in enumerate(cv_splits(y)):
        if max_folds and i >= max_folds:
            break
        tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
        m = HistGradientBoostingClassifier(random_state=i).fit(X[tr], y[tr])
        runs.append({"test": metrics(y[te], m.predict_proba(X[te])[:, 1]), "best_epoch": -1})
    return runs


def compare(runs, base, metric):
    x = np.array([r["test"][metric] for r in runs]) - np.array([r["test"][metric] for r in base])
    k = len(x)
    if k < 3 or x.std(ddof=1) == 0:
        return dict(diff=float(x.mean()), folds_better=int((x > 0).sum()), k=k, p_naive=float("nan"), p_corrected=float("nan"))
    var = x.var(ddof=1)
    t_naive = x.mean() / np.sqrt(var / k)
    t_corr = x.mean() / np.sqrt((1 / k + 0.2 / 0.72) * var)         # Nadeau-Bengio corrected resampled t (test 20%, train 72%)
    p = lambda t: float(2 * t_dist.sf(abs(t), k - 1))
    return dict(diff=float(x.mean()), folds_better=int((x > 0).sum()), k=k, p_naive=p(t_naive), p_corrected=p(t_corr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--struct", required=True)
    ap.add_argument("--coord", required=True)
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--gb", action="store_true")
    ap.add_argument("--out", default="results_coord.json")
    a = ap.parse_args()

    d, struct = load_data(a.npz, a.struct)
    y = d["labels"]
    C, V, names = load_coord(a.coord, d["topic_ids"])
    cfg = dict(DEFAULT_CFG, epochs=a.epochs)
    print(f"topics={len(y)} positive-label share={y.mean():.3f} (check this is the TARGET you intend: rumor vs guided)")

    base = build_features_arrays(d["X"], d["mask"], ["struct", "mut"], struct)
    perm = np.random.RandomState(0).permutation(len(y))
    feats = {"base (struct+mut)": base,
             "+coord": np.concatenate([base, C], 2),
             "+volume (control)": np.concatenate([base, V], 2),
             "+coord_shuffled (control)": np.concatenate([base, C[perm]], 2)}

    results = {}
    for name, F in feats.items():
        print(f"\n[{name}] features {F.shape}", flush=True)
        results["GRU " + name] = cv_gru(F, y, cfg, a.max_folds)
        print(f"  GRU {name}: {fmt(summarize(results['GRU ' + name]))}", flush=True)
        if a.gb:
            results["GB " + name] = cv_gb(F, y, a.max_folds)
            print(f"  GB  {name}: {fmt(summarize(results['GB ' + name], keys=('acc', 'macro_f1', 'auc', 'pr_auc')))}")

    print("\n== paired comparison vs base (same folds). p_corrected = Nadeau-Bengio; still optimistic: CED has near-duplicate events ==")
    stats = {}
    for fam in ("GRU", "GB"):
        b = results.get(f"{fam} base (struct+mut)")
        if b is None:
            continue
        for name in list(feats)[1:]:
            for m in ("auc", "pr_auc", "macro_f1"):
                s = compare(results[f"{fam} {name}"], b, m)
                stats[f"{fam} {name} {m}"] = s
                print(f"{fam} {name:26s} {m:9s} diff={s['diff']:+.4f}  better in {s['folds_better']}/{s['k']} folds  "
                      f"p_naive={s['p_naive']:.3f} p_corrected={s['p_corrected']:.3f}")

    with open(a.out, "w") as f:
        json.dump({"cfg": cfg, "coord_names": names, "summary": {k: summarize(v, keys=("acc", "bal_acc", "macro_f1", "auc", "pr_auc"))
                                                                   for k, v in results.items()},
                   "stats": stats}, f, indent=1)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
