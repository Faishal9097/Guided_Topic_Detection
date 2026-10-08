"""Experiment suites.
  --suite ablation : feature-block ablation (+ trivial size/count baselines)
  --suite temporal : GRU vs LSTM vs Transformer vs MLP (no temporal order) on --blocks
  --suite early    : first T = 2,4,6,8,Tmax slices on --blocks
Significance vs the reference config: paired t-test over seeds + paired bootstrap on test predictions."""
import argparse
import json

import numpy as np
from scipy.stats import ttest_rel

from group4_common import (DEFAULT_CFG, build_features_arrays, fmt, load_data, run_seeds, summarize)

ABLATION = [("size", ["size"]), ("count", ["count"]), ("vec", ["vec"]), ("mut", ["mut"]),
            ("vec+mut", ["vec", "mut"]), ("struct", ["struct"]), ("struct+mut", ["struct", "mut"]),
            ("vec+struct+mut", ["vec", "struct", "mut"]), ("flat", ["flat"])]


def paired_bootstrap(y, pa, pb, n=2000, seed=0):
    ca, cb = ((pa >= .5) == y).astype(float), ((pb >= .5) == y).astype(float)
    diff = ca - cb
    rng = np.random.RandomState(seed)
    boots = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(n)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(diff.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], float(min(p, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--suite", required=True, choices=["ablation", "temporal", "early"])
    ap.add_argument("--blocks", default="vec,mut")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d, struct = load_data(a.npz, a.struct)
    y, split = d["labels"], d["split"]
    Tmax, stance = d["X"].shape[1], bool(d["stance_included"])
    cfg = dict(DEFAULT_CFG, epochs=a.epochs, class_weight=a.class_weight)
    seeds = list(range(a.seeds))
    blocks = a.blocks.split(",")

    configs = []                                   # (name, blocks, unit, T)
    if a.suite == "ablation":
        for name, b in ABLATION:
            if ("struct" in b or "size" in b) and struct is None:
                print(f"skip {name}: no --struct"); continue
            if "mut" in b and not stance:
                print(f"skip {name}: dataset has no stance features"); continue
            configs.append((name, b, "gru", None))
    elif a.suite == "temporal":
        configs = [(u, blocks, u, None) for u in ("gru", "lstm", "transformer", "mlp")]
    else:
        if Tmax < 2:
            raise SystemExit("early-detection suite needs T>1 (Weibo only)")
        for k in sorted(set([2, 4, 6, 8, Tmax])):
            if k <= Tmax:
                configs.append((f"T={k}", blocks, "gru", k))

    cache, results = {}, {}
    for name, b, unit, T in configs:
        key = tuple(b)
        if key not in cache:
            cache[key] = build_features_arrays(d["X"], d["mask"], b, struct)
        runs, _, _ = run_seeds(cache[key], y, split, cfg, seeds, unit, T)
        results[name] = runs
        print(f"[{name:15s}] {fmt(summarize(runs))}   (epochs*: {[r['best_epoch'] for r in runs]})", flush=True)

    ref = a.ref or (configs[-1][0] if a.suite == "early" else
                    ("vec+mut" if "vec+mut" in results else configs[0][0]))
    print(f"\n== significance vs reference '{ref}' (test set, {a.seeds} seeds) ==")
    yt = y[split == "test"]
    pr = np.mean([r["test_prob"] for r in results[ref]], 0)
    stats = {}
    for name, runs in results.items():
        if name == ref:
            continue
        row = {}
        for m in ("acc", "f1"):
            x = [r["test"][m] for r in runs]
            z = [r["test"][m] for r in results[ref]]
            row[f"t_p_{m}"] = float(ttest_rel(x, z).pvalue) if len(seeds) > 1 and np.std(np.subtract(x, z)) > 0 else float("nan")
        dm, ci, pb = paired_bootstrap(yt, np.mean([r["test_prob"] for r in runs], 0), pr)
        row.update(acc_diff=dm, acc_diff_ci95=ci, boot_p=pb)
        stats[name] = row
        print(f"{name:15s} acc_diff={dm:+.3f} CI95=[{ci[0]:+.3f},{ci[1]:+.3f}] boot_p={pb:.3f} "
              f"t_p(acc)={row['t_p_acc']:.3f} t_p(f1)={row['t_p_f1']:.3f}")
    out = a.out or f"results_{a.suite}.json"
    with open(out, "w") as f:
        json.dump({"suite": a.suite, "ref": ref, "cfg": cfg, "stats": stats,
                   "results": {n: {"summary": summarize(r), "runs": [{k: v for k, v in x.items() if k != "test_prob"} for x in r]}
                               for n, r in results.items()}}, f, indent=1)
    print("saved ->", out)


if __name__ == "__main__":
    main()