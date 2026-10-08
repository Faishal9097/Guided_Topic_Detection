"""Train/evaluate GC-GRU for one feature configuration over several seeds.

  python group4_train.py --npz W.npz --struct weibo_struct.npz --blocks vec,mut --seeds 5 --out r.json
blocks: count | vec | mut | flat | struct | size   (comma separated)
--T N uses only the first N time slices (early detection). --save-model saves the best-val seed."""
import argparse
import json

import numpy as np
import torch

from group4_common import (DEFAULT_CFG, build_features_arrays, fmt, load_data, run_seeds, summarize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--blocks", default="vec,mut")
    ap.add_argument("--unit", default="gru", choices=["gru", "lstm", "transformer", "mlp"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    ap.add_argument("--batch", type=int, default=DEFAULT_CFG["batch"])
    ap.add_argument("--hidden", type=int, default=DEFAULT_CFG["hidden"])
    ap.add_argument("--layers", type=int, default=DEFAULT_CFG["layers"])
    ap.add_argument("--dropout", type=float, default=DEFAULT_CFG["dropout"])
    ap.add_argument("--fc", type=int, default=DEFAULT_CFG["fc"])
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d, struct = load_data(a.npz, a.struct)
    blocks = a.blocks.split(",")
    if "mut" in blocks and not bool(d["stance_included"]):
        print("WARNING: this dataset has no stance features (Mut = 0); 'mut' carries no information.")
    cfg = dict(DEFAULT_CFG, epochs=a.epochs, lr=a.lr, batch=a.batch, hidden=a.hidden, layers=a.layers,
               dropout=a.dropout, fc=a.fc, class_weight=a.class_weight)
    y, split = d["labels"], d["split"]
    print({s: int((split == s).sum()) for s in ("train", "val", "test")},
          "guided share:", round(float(y.mean()), 3))
    F = build_features_arrays(d["X"], d["mask"], blocks, struct)
    print("features", F.shape, "blocks", blocks, "unit", a.unit)
    runs, mu, sd = run_seeds(F, y, split, cfg, list(range(a.seeds)), a.unit, a.T,
                             keep_state=bool(a.save_model))
    for r in runs:
        print(f"seed {r['seed']}: best_epoch={r['best_epoch']} val_auc={r['val']['auc']:.3f} "
              f"test_acc={r['test']['acc']:.3f} test_f1={r['test']['f1']:.3f} ({r['seconds']}s)")
    print("VAL :", fmt(summarize(runs, "val")))
    print("TEST:", fmt(summarize(runs, "test")))
    if a.save_model:
        b = max(runs, key=lambda r: r["val"]["auc"])
        torch.save({"state": b["state"], "mu": mu, "sd": sd, "blocks": blocks, "unit": a.unit, "cfg": cfg,
                    "in_dim": F.shape[2], "T": a.T or F.shape[1], "seed": b["seed"]}, a.save_model)
        print("saved model ->", a.save_model)
    if a.out:
        slim = [{k: v for k, v in r.items() if k not in ("state", "test_prob")} for r in runs]
        with open(a.out, "w") as f:
            json.dump({"blocks": blocks, "unit": a.unit, "T": a.T, "cfg": cfg, "runs": slim,
                       "test_summary": summarize(runs, "test")}, f, indent=1)


if __name__ == "__main__":
    main()