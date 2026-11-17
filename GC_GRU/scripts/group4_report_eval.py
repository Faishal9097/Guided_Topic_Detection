"""Detailed evaluation with predictions: ROC/PR, threshold sweep, confusion matrices, probability
histograms, bootstrap AUC CI, and (optional) univariate-AUC leakage diagnostic.

  python group4_report_eval.py --npz W.npz --struct weibo_struct.npz --configs "size,mut,struct,struct+mut,vec+mut" --seeds 3 --univariate
Threshold is chosen on the VALIDATION split (max macro-F1) and then applied to test."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

from group4_common import DEFAULT_CFG, DEVICE, GCGRU, build_features_arrays, load_data, pool_blocks, predict, run_seeds

ROOT = Path(__file__).resolve().parent.parent
THR = np.linspace(0.02, 0.98, 97)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def cls_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"acc": float((pred == y).mean()), "bal_acc": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "f1_guided": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "f1_nonguided": float(f1_score(y, pred, pos_label=0, zero_division=0)),
            "rec_nonguided": float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "rec_guided": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))}


def best_thr(y, p):
    sc = [f1_score(y, (p >= t).astype(int), average="macro", zero_division=0) for t in THR]
    return float(THR[int(np.argmax(sc))])


def boot_auc(y, p, n=1000, seed=0):
    rng, vals = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ms(v, nd=3):
    return f"{np.mean(v):.{nd}f} ± {np.std(v):.{nd}f}"


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / name)


def univariate(d, struct, out, tag):
    X, mask, y = d["X"], d["mask"], d["labels"]
    P = pool_blocks(X, mask)
    names, cols = ["log_n_groups"], [P["count"][:, -1, 0]]
    names.append("group_vec_norm")
    cols.append(P["vec"][:, -1, -1])
    if bool(d["stance_included"]):
        for stat in ("mean", "max", "std"):
            for st in ("sup", "opo", "none"):
                names.append(f"mut_{st}_{stat}")
        cols += [P["mut"][:, -1, i] for i in range(9)]
    if struct is not None:
        S, snames = struct
        for i, n in enumerate(snames):
            if n not in names:
                names.append(n)
                cols.append(S[:, -1, i])
    rows = []
    for n, c in zip(names, cols):
        c = np.asarray(c, float)
        auc = roc_auc_score(y, c) if np.std(c) > 0 else 0.5
        rows.append((n, auc))
    rows.sort(key=lambda r: -max(r[1], 1 - r[1]))
    with open(out / f"univariate_auc_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "auc_raw", "auc_oriented"])
        w.writerows([[n, f"{a:.4f}", f"{max(a, 1 - a):.4f}"] for n, a in rows])
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(rows) + 1.5))
    y_ = np.arange(len(rows))
    ax.barh(y_, [max(a, 1 - a) for _, a in rows],
            color=["#4C78A8" if a >= 0.5 else "#F58518" for _, a in rows])
    ax.axvline(0.5, ls="--", c="red")
    ax.set_yticks(y_)
    ax.set_yticklabels([n for n, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Univariate ROC-AUC for 'guided' (last slice, all topics); orange = higher value => less guided")
    ax.set_title(f"Which single features already predict the label? ({tag})")
    savefig(fig, out, f"fig_univariate_auc_{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--configs", default="size,struct,struct+mut,vec+mut")
    ap.add_argument("--unit", default="gru")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--univariate", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else ROOT / "results" / "report"
    out.mkdir(parents=True, exist_ok=True)
    tag = (a.tag or Path(a.npz).stem.split("_")[0]) + ("_cw" if a.class_weight else "")
    d, struct = load_data(a.npz, a.struct)
    y, split = d["labels"], d["split"]
    va, te = np.where(split == "val")[0], np.where(split == "test")[0]
    cfg = dict(DEFAULT_CFG, epochs=a.epochs, class_weight=a.class_weight)
    print(f"{tag}: test={len(te)} (nonguided {int((y[te] == 0).sum())}), val={len(va)}, "
          f"guided share={y.mean():.3f}, class_weight={a.class_weight}")
    if a.univariate:
        univariate(d, struct, out, tag)

    R = {}
    for name in [c.strip() for c in a.configs.split(",")]:
        blocks = name.split("+")
        if "mut" in blocks and not bool(d["stance_included"]):
            print(f"skip {name}: no stance features")
            continue
        if ("struct" in blocks or "size" in blocks) and struct is None:
            print(f"skip {name}: needs --struct")
            continue
        F = build_features_arrays(d["X"], d["mask"], blocks, struct)
        runs, mu, sd = run_seeds(F, y, split, cfg, list(range(a.seeds)), a.unit, None, keep_state=True)
        Fn = torch.tensor(((F - mu) / sd).astype(np.float32))
        pvs, pts, seeds_out = [], [], []
        for r in runs:
            model = GCGRU(F.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"], cfg["fc"], a.unit).to(DEVICE)
            model.load_state_dict(r["state"])
            pv, pt = predict(model, Fn[torch.as_tensor(va)]), predict(model, Fn[torch.as_tensor(te)])
            thr = best_thr(y[va], pv)
            seeds_out.append({"auc": float(roc_auc_score(y[te], pt)), "thr": thr,
                              "m05": cls_metrics(y[te], pt, 0.5), "mthr": cls_metrics(y[te], pt, thr)})
            pvs.append(pv)
            pts.append(pt)
        pv_m, pt_m = np.mean(pvs, 0), np.mean(pts, 0)
        thr_e = best_thr(y[va], pv_m)
        lo, hi = boot_auc(y[te], pt_m, a.n_boot)
        R[name] = {"pv": pv_m, "pt": pt_m, "thr": thr_e, "seeds": seeds_out, "auc_ci": (lo, hi),
                   "auc_ens": float(roc_auc_score(y[te], pt_m)),
                   "ap_g": float(average_precision_score(y[te], pt_m)),
                   "ap_n": float(average_precision_score(1 - y[te], 1 - pt_m))}
        print(f"[{name:15s}] AUC seeds {ms([s['auc'] for s in seeds_out])} | ensemble {R[name]['auc_ens']:.3f} "
              f"CI[{lo:.3f},{hi:.3f}] | macro-F1@0.5 {ms([s['m05']['macro_f1'] for s in seeds_out])} "
              f"| macro-F1@valthr {ms([s['mthr']['macro_f1'] for s in seeds_out])}", flush=True)
    if not R:
        raise SystemExit("no configuration was run")
    yt = y[te]

    # ---- tables
    keys = ["acc", "bal_acc", "macro_f1", "f1_nonguided", "rec_nonguided"]
    header = (["Config", "AUC (mean±std)", "AUC ensemble [95% CI]", "AP guided", "AP nonguided", "val thr"]
              + [f"{k}@0.5" for k in keys] + [f"{k}@valthr" for k in keys])
    body = []
    for n, r in R.items():
        s = r["seeds"]
        body.append([n, ms([x["auc"] for x in s]), f"{r['auc_ens']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]",
                     f"{r['ap_g']:.3f}", f"{r['ap_n']:.3f}", ms([x["thr"] for x in s], 2)]
                    + [ms([x["m05"][k] for x in s]) for k in keys] + [ms([x["mthr"][k] for x in s]) for k in keys])
    with open(out / f"eval_summary_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + body)
    md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    md += "".join("| " + " | ".join(map(str, r)) + " |\n" for r in body)
    (out / f"eval_summary_{tag}.md").write_text(md, encoding="utf-8")
    print("saved", out / f"eval_summary_{tag}.md")

    # ---- ROC + PR
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for n, r in R.items():
        fpr, tpr, _ = roc_curve(yt, r["pt"])
        axes[0].plot(fpr, tpr, label=f"{n} (AUC {r['auc_ens']:.3f})")
        p_, r_, _ = precision_recall_curve(yt, r["pt"])
        axes[1].plot(r_, p_, label=f"{n} (AP {r['ap_g']:.3f})")
        p_, r_, _ = precision_recall_curve(1 - yt, 1 - r["pt"])
        axes[2].plot(r_, p_, label=f"{n} (AP {r['ap_n']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test, seed-ensemble)")
    axes[1].axhline(yt.mean(), c="k", ls="--", lw=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR curve: guided class")
    axes[2].axhline(1 - yt.mean(), c="k", ls="--", lw=1)
    axes[2].set(xlabel="Recall", ylabel="Precision", title="PR curve: nonguided (minority) class")
    for ax in axes:
        ax.legend(fontsize=8)
    savefig(fig, out, f"fig_roc_pr_{tag}.png")

    # ---- threshold sweeps
    n_cfg = len(R)
    fig, axes = plt.subplots(1, n_cfg, figsize=(4.6 * n_cfg, 4), sharey=True, squeeze=False)
    for ax, (n, r) in zip(axes[0], R.items()):
        sweep = [cls_metrics(yt, r["pt"], t) for t in THR]
        for k, lab in [("acc", "accuracy"), ("bal_acc", "balanced acc"), ("macro_f1", "macro-F1"),
                       ("f1_nonguided", "F1 nonguided")]:
            ax.plot(THR, [s[k] for s in sweep], label=lab)
        ax.axvline(0.5, c="gray", ls=":")
        ax.axvline(r["thr"], c="red", ls="--")
        ax.set(title=f"{n}\n(red = val-chosen thr {r['thr']:.2f})", xlabel="Decision threshold")
    axes[0][0].legend(fontsize=8)
    savefig(fig, out, f"fig_threshold_sweep_{tag}.png")

    # ---- confusion matrices
    fig, axes = plt.subplots(n_cfg, 2, figsize=(7, 3.1 * n_cfg), squeeze=False)
    for row, (n, r) in zip(axes, R.items()):
        for ax, (thr, lab) in zip(row, [(0.5, "thr 0.5"), (r["thr"], f"val thr {r['thr']:.2f}")]):
            cm = confusion_matrix(yt, (r["pt"] >= thr).astype(int), labels=[0, 1])
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred nonguided", "pred guided"], fontsize=8)
            ax.set_yticklabels(["true nonguided", "true guided"], fontsize=8)
            ax.set_title(f"{n} @ {lab}", fontsize=9)
            ax.grid(False)
    savefig(fig, out, f"fig_confusion_{tag}.png")

    # ---- probability histograms
    fig, axes = plt.subplots(1, n_cfg, figsize=(4.6 * n_cfg, 3.8), sharey=True, squeeze=False)
    for ax, (n, r) in zip(axes[0], R.items()):
        ax.hist(r["pt"][yt == 0], bins=20, range=(0, 1), alpha=0.6, density=True, label="true nonguided")
        ax.hist(r["pt"][yt == 1], bins=20, range=(0, 1), alpha=0.6, density=True, label="true guided")
        ax.set(title=n, xlabel="Predicted P(guided)")
    axes[0][0].legend(fontsize=8)
    savefig(fig, out, f"fig_prob_hist_{tag}.png")

    np.savez_compressed(out / f"predictions_{tag}.npz", y_test=yt, y_val=y[va], topic_ids_test=d["topic_ids"][te],
                        **{f"{n}__test": r["pt"] for n, r in R.items()},
                        **{f"{n}__val": r["pv"] for n, r in R.items()})
    print("saved predictions ->", out / f"predictions_{tag}.npz")


if __name__ == "__main__":
    main()