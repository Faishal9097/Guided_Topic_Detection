"""Report figures + tables from the Group 4 result JSONs (no retraining).
  python group4_report_plots.py --data "<Ag2vec data_output folder>"
--data is optional: it adds dataset facts (sizes, guided share, test minority count) and trivial-baseline lines."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["acc", "prec", "rec", "f1", "macro_f1", "auc"]
SUITES = {"weibo": "results_ablation.json", "politifact": "results_politifact.json",
          "gossipcop": "results_gossipcop.json"}
NPZ = {ds: f"{ds}_group_feature_matrices.npz" for ds in SUITES}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.25})


def jload(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def pm(t, nd=3):
    return f"{t[0]:.{nd}f} ± {t[1]:.{nd}f}"


def as_suite(js):
    if "results" in js:
        return js

    if "blocks" in js:
        name = "+".join(js["blocks"]) + f" ({js['unit']})"
        return {
            "suite": "single",
            "ref": None,
            "stats": {},
            "results": {
                name: {
                    "summary": js["test_summary"],
                    "runs": js["runs"]
                }
            }
        }

    # JSON containing configuration -> metric summaries
    return {
        "suite": "summary",
        "ref": None,
        "stats": {},
        "results": {
            name: {"summary": metrics, "runs": []}
            for name, metrics in js.items()
        }
    }

def rows_of(js):
    out = []

    for name, v in js["results"].items():
        runs = v["runs"]
        s = v["summary"]

        ng = [
            r["test"]["f1_nonguided"]
            for r in runs
        ]

        out.append({
            "name": name,
            **{k: tuple(s[k]) for k in METRICS},
            "f1_nonguided": (
                float(np.mean(ng)) if ng else float("nan"),
                float(np.std(ng)) if ng else float("nan")
            ),
            "ng_runs": ng,
            "test_auc_runs": [r["test"]["auc"] for r in runs],
            "test_f1m_runs": [r["test"]["macro_f1"] for r in runs],
            "epochs": (
                float(np.mean([r["best_epoch"] for r in runs]))
                if runs else float("nan")
            ),
            "secs": (
                float(np.mean([r["seconds"] for r in runs]))
                if runs else float("nan")
            ),
            "n": len(runs)
        })

    return out


def trivial(p):
    """Metrics of always predicting the majority class (guided = 1, share p)."""
    if p is None:
        return None
    if p >= 0.5:
        f1g = 2 * p / (1 + p)
        return {"acc": p, "f1": f1g, "macro_f1": f1g / 2}
    q = 1 - p
    return {"acc": q, "f1": 0.0, "macro_f1": (2 * q / (1 + q)) / 2}


def write_table(stem, header, body, out):
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + body)
    md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    md += "".join("| " + " | ".join(map(str, r)) + " |\n" for r in body)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    esc = lambda s: str(s).replace("_", "\\_").replace("±", "$\\pm$").replace("%", "\\%")
    tex = "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}\n\\hline\n"
    tex += " & ".join(esc(h) for h in header) + " \\\\\n\\hline\n"
    tex += "".join(" & ".join(esc(c) for c in r) + " \\\\\n" for r in body) + "\\hline\n\\end{tabular}\n"
    (out / f"{stem}.tex").write_text(tex, encoding="utf-8")
    return md


def metric_table(rows):
    header = ["Config", "Accuracy", "Precision", "Recall", "F1 (guided)", "Macro-F1",
              "F1 (nonguided)", "ROC-AUC", "Best epoch", "Sec/seed"]
    body = [[r["name"], pm(r["acc"]), pm(r["prec"]), pm(r["rec"]), pm(r["f1"]), pm(r["macro_f1"]),
             pm(r["f1_nonguided"]), pm(r["auc"]), f"{r['epochs']:.1f}", f"{r['secs']:.0f}"] for r in rows]
    return header, body


def stats_table(js):
    header = ["Config", "Δ accuracy vs ref", "95% CI", "bootstrap p", "paired t p (acc)", "paired t p (F1)"]
    body = [[n, f"{s['acc_diff']:+.3f}", f"[{s['acc_diff_ci95'][0]:+.3f}, {s['acc_diff_ci95'][1]:+.3f}]",
             f"{s['boot_p']:.3f}", f"{s['t_p_acc']:.3f}", f"{s['t_p_f1']:.3f}"] for n, s in js["stats"].items()]
    return header, body


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / name)


# ----------------------------------------------------------------------------- plots
def plot_suite_bars(rows, title, out, fname, share=None):
    base = trivial(share)
    panels = [("auc", "ROC-AUC", 0.5), ("macro_f1", "Macro-F1", base and base["macro_f1"]),
              ("acc", "Accuracy", base and base["acc"]), ("f1_nonguided", "F1 (nonguided)", None)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 0.5 * len(rows) + 2.4), sharey=True)
    y = np.arange(len(rows))
    for ax, (k, lab, b) in zip(axes, panels):
        m, s = [r[k][0] for r in rows], [r[k][1] for r in rows]
        ax.barh(y, m, xerr=s, capsize=2, color="#4C78A8")
        if b is not None:
            ax.axvline(b, ls="--", c="red", lw=1.2)
        for yi, mi in zip(y, m):
            ax.text(mi + 0.01, yi, f"{mi:.3f}", va="center", fontsize=8)
        ax.set_title(lab)
        ax.set_xlim(0, 1.12)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["name"] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle(title + ("  (red dashed = chance / always-majority baseline)" if share is not None else
                          "  (red dashed in AUC = chance)"))
    savefig(fig, out, fname)


def plot_strip(rows, title, out, fname):
    rows = [r for r in rows if len(r["ng_runs"]) > 0]

    if not rows:
        print(f"Skipping {fname}: no per-seed data available")
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.8))
    rng = np.random.RandomState(0)
    for i, r in enumerate(rows):
        v = np.array(r["ng_runs"])
        ax.scatter(v, i + rng.uniform(-0.12, 0.12, len(v)), s=30, color="#E45756", zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.28, i + 0.28], c="k", lw=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Test F1 of the nonguided (minority) class, one dot per seed; bar = mean")
    ax.set_title(title)
    savefig(fig, out, fname)


def plot_temporal(rows, out, fname):
    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(rows)), 0.25
    for j, (k, lab) in enumerate([("auc", "ROC-AUC"), ("macro_f1", "Macro-F1"), ("acc", "Accuracy")]):
        ax.bar(x + (j - 1) * w, [r[k][0] for r in rows], w, yerr=[r[k][1] for r in rows], capsize=2, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows])
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="lower center")
    ax.set_title("Temporal unit comparison (Weibo)")
    savefig(fig, out, fname)


def plot_early(rows, share, out, fname):
    rows = sorted(rows, key=lambda r: int(r["name"].split("=")[1]))
    T = [int(r["name"].split("=")[1]) for r in rows]
    base = trivial(share)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (k, lab, b) in zip(axes, [("auc", "ROC-AUC", 0.5), ("macro_f1", "Macro-F1", base and base["macro_f1"]),
                                      ("f1_nonguided", "F1 (nonguided)", None)]):
        ax.errorbar(T, [r[k][0] for r in rows], yerr=[r[k][1] for r in rows], marker="o", capsize=3)
        if b is not None:
            ax.axhline(b, ls="--", c="red", lw=1)
        ax.set_xlabel("Time slices available (T)")
        ax.set_title(lab)
        ax.set_xticks(T)
    fig.suptitle("Early detection (Weibo); red dashed = chance / always-majority baseline")
    savefig(fig, out, fname)


def plot_significance(js, out, fname, title):
    st = js.get("stats") or {}
    if not st:
        return
    names = list(st)
    d = np.array([st[n]["acc_diff"] for n in names])
    lo = np.array([st[n]["acc_diff_ci95"][0] for n in names])
    hi = np.array([st[n]["acc_diff_ci95"][1] for n in names])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.8))
    y = np.arange(len(names))
    ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="o", capsize=3)
    ax.axvline(0, c="k", lw=1)
    for yi, n in zip(y, names):
        ax.text(hi[yi] + 0.003, yi, f"p={st[n]['boot_p']:.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"Accuracy difference vs reference '{js.get('ref')}' (paired bootstrap, 95% CI)")
    ax.set_title(title)
    savefig(fig, out, fname)


def plot_cross(per_ds, out):
    ds_names, size_auc, best_auc, best_nm = [], [], [], []
    for ds, rows in per_ds.items():
        sz = [r for r in rows if r["name"] == "size"]
        b = max(rows, key=lambda r: r["auc"][0])
        ds_names.append(ds)
        size_auc.append(sz[0]["auc"][0] if sz else np.nan)
        best_auc.append(b["auc"][0])
        best_nm.append(b["name"])
    x, w = np.arange(len(ds_names)), 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, size_auc, w, label="size baseline")
    ax.bar(x + w / 2, best_auc, w, label="best configuration")
    for xi, v, nm in zip(x, best_auc, best_nm):
        ax.text(xi + w / 2, v + 0.01, f"{v:.3f}\n{nm}", ha="center", fontsize=8)
    ax.axhline(0.5, ls="--", c="red", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylim(0.4, 1.08)
    ax.set_ylabel("Test ROC-AUC")
    ax.legend(loc="lower right")
    ax.set_title("Cross-dataset comparison")
    savefig(fig, out, "fig_cross_dataset.png")


def plot_main_runs(named_rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, lab) in zip(axes, [("test_auc_runs", "Test ROC-AUC"), ("test_f1m_runs", "Test macro-F1")]):
        for i, (n, r) in enumerate(named_rows):
            v = np.array(r[key])
            ax.scatter([i] * len(v), v, s=35, zorder=3)
            ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, c="k", lw=2)
        ax.set_xticks(range(len(named_rows)))
        ax.set_xticklabels([n for n, _ in named_rows], rotation=15)
        ax.set_ylabel(lab)
    fig.suptitle("Per-seed test results of the main models (bar = mean)")
    savefig(fig, out, "fig_main_runs_per_seed.png")


# ----------------------------------------------------------------------------- facts
def facts_of(data_dir, ds):
    p = Path(data_dir) / NPZ[ds]
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    y, sp = z["labels"], z["split"].astype(str)
    te = sp == "test"
    return {"topics": len(y), "guided_share": float(y.mean()), "train": int((sp == "train").sum()),
            "val": int((sp == "val").sum()), "test": int(te.sum()),
            "test_nonguided": int((y[te] == 0).sum()), "test_guided": int((y[te] == 1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "report"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Group 4 results summary\n"]
    facts = {}
    if a.data:
        for ds in SUITES:
            f = facts_of(a.data, ds)
            if f:
                facts[ds] = f
        if facts:
            header = ["Dataset", "Topics", "Guided share", "Train", "Val", "Test", "Test guided", "Test nonguided"]
            body = [[ds, f["topics"], f"{f['guided_share']:.3f}", f["train"], f["val"], f["test"],
                     f["test_guided"], f["test_nonguided"]] for ds, f in facts.items()]
            md += ["## Dataset facts\n", write_table("table_dataset_facts", header, body, out), "\n"]
    share = lambda ds: facts[ds]["guided_share"] if ds in facts else None

    # main runs
    named = []
    for fn in ("r_vecmut.json", "r_structmut.json"):
        js = jload(res / fn)
        if js:
            for r in rows_of(as_suite(js)):
                named.append((r["name"], r))
    if named:
        h, b = metric_table([r for _, r in named])
        md += ["## Main models (test, mean ± std over seeds)\n", write_table("table_main_models", h, b, out), "\n"]
        plot_main_runs(named, out)

    per_ds = {}
    for ds, fn in SUITES.items():
        js = jload(res / fn)
        if not js:
            continue
        rows = rows_of(as_suite(js))
        per_ds[ds] = rows
        h, b = metric_table(rows)
        md += [f"## Feature ablation: {ds}\n", write_table(f"table_ablation_{ds}", h, b, out), "\n"]
        plot_suite_bars(rows, f"Feature ablation: {ds}", out, f"fig_ablation_{ds}.png", share(ds))
        plot_strip(rows, f"Minority-class F1 per seed: {ds}", out, f"fig_nonguided_f1_{ds}.png")
        if js.get("stats"):
            h, b = stats_table(js)
            md += [f"### Significance vs reference '{js['ref']}' ({ds})\n", write_table(f"table_stats_{ds}", h, b, out), "\n"]
            plot_significance(js, out, f"fig_significance_{ds}.png", f"Ablation significance: {ds}")
    if len(per_ds) > 1:
        plot_cross(per_ds, out)

    js = jload(res / "results_temporal.json")
    if js:
        rows = rows_of(js)
        h, b = metric_table(rows)
        md += ["## Temporal unit comparison (Weibo)\n", write_table("table_temporal", h, b, out), "\n"]
        plot_temporal(rows, out, "fig_temporal.png")
        if js.get("stats"):
            h, b = stats_table(js)
            md += ["### Significance vs reference\n", write_table("table_stats_temporal", h, b, out), "\n"]
            plot_significance(js, out, "fig_significance_temporal.png", "Temporal unit significance")
    js = jload(res / "results_early.json")
    if js:
        rows = rows_of(js)
        h, b = metric_table(rows)
        md += ["## Early detection (Weibo)\n", write_table("table_early", h, b, out), "\n"]
        plot_early(rows, share("weibo"), out, "fig_early_detection.png")
        if js.get("stats"):
            h, b = stats_table(js)
            md += ["### Significance vs full horizon\n", write_table("table_stats_early", h, b, out), "\n"]
            plot_significance(js, out, "fig_significance_early.png", "Early detection significance")

    (out / "REPORT_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\nDone. Everything is in", out)


if __name__ == "__main__":
    main()