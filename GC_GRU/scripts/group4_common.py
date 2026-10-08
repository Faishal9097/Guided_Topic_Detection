"""Group 4 shared utilities: data loading, feature blocks, GC-GRU model, training, metrics."""
import glob
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POOLED = ("count", "vec", "mut")
SIZE_COLS = ["log_n_groups", "log_users", "log_mean_size", "max_size_share", "singleton_share"]
# Paper's GC-GRU_B (Table V) + lr 0.004, batch 32, 32 epochs
DEFAULT_CFG = dict(hidden=128, layers=2, dropout=0.5, fc=256, lr=0.004, batch=32,
                   epochs=32, class_weight=False, clip=5.0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
def load_data(npz_path, struct_path=None):
    z = np.load(npz_path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["labels"] = d["labels"].astype(np.int64)
    d["split"] = d["split"].astype(str)
    d["topic_ids"] = d["topic_ids"].astype(str)
    d["stance_included"] = np.array(bool(np.any(d["X"][..., -3:] != 0)))   # trust the data, not the flag
    struct = None
    if struct_path:
        s = np.load(struct_path, allow_pickle=False)
        if not np.array_equal(s["topic_ids"].astype(str), d["topic_ids"]):
            raise SystemExit("struct file topic_ids do not match the feature npz (rebuild it).")
        struct = (s["S"].astype(np.float32), [str(x) for x in s["names"]])
    return d, struct


def load_chunks(chunk_dir, kind):
    """Merge Group 3 chunk files <kind>_all_NNNN.pkl: {topic_id: [per-slice dict]}."""
    topics = {}
    for f in sorted(glob.glob(os.path.join(chunk_dir, f"{kind}_all_*.pkl"))):
        with open(f, "rb") as fh:
            topics.update(pickle.load(fh).get("topics", {}))
    return topics


def pool_blocks(X, mask, chunk=256):
    """Masked pooling over the K group slots. Returns dict of (N,T,*) arrays.
       count: log1p(#groups)            vec: mean group vector (64) + mean norm
       mut:   mean/max/std of Mut (3 each = 9)"""
    N, T, K, D = X.shape
    out = {k: [] for k in POOLED}
    for s in range(0, N, chunk):
        x = X[s:s + chunk].astype(np.float32)
        m = mask[s:s + chunk]
        mf = m[..., None].astype(np.float32)
        cnt = m.sum(2).astype(np.float32)
        den = np.maximum(cnt, 1.0)[..., None]
        vec, mut = x[..., :D - 3], x[..., D - 3:]
        vmean = (vec * mf).sum(2) / den
        vnorm = (np.linalg.norm(vec, axis=-1) * m).sum(2) / den[..., 0]
        mmean = (mut * mf).sum(2) / den
        mmax = np.where(m[..., None], mut, -np.inf).max(2)
        mmax = np.where(np.isfinite(mmax), mmax, 0.0)
        mstd = np.sqrt((((mut - mmean[:, :, None, :]) ** 2) * mf).sum(2) / den)
        out["count"].append(np.log1p(cnt)[..., None])
        out["vec"].append(np.concatenate([vmean, vnorm[..., None]], -1))
        out["mut"].append(np.concatenate([mmean, mmax, mstd], -1))
    return {k: np.concatenate(v, 0).astype(np.float32) for k, v in out.items()}


def build_features_arrays(X, mask, blocks, struct=None):
    """blocks subset of: count, vec, mut, flat, struct, size. Returns (N,T,F) float32."""
    P = pool_blocks(X, mask) if any(b in POOLED for b in blocks) else None
    parts = []
    for b in blocks:
        if b in POOLED:
            parts.append(P[b])
        elif b == "flat":
            parts.append((X * mask[..., None]).reshape(X.shape[0], X.shape[1], -1))
        elif b in ("struct", "size"):
            if struct is None:
                raise SystemExit(f"block '{b}' needs --struct (run group4_build_struct.py first)")
            S, names = struct
            parts.append(S if b == "struct" else S[..., [names.index(n) for n in SIZE_COLS]])
        else:
            raise SystemExit(f"unknown block '{b}'")
    return np.concatenate(parts, axis=2).astype(np.float32)


def fit_norm(F, split):
    flat = F[split == "train"].reshape(-1, F.shape[2])        # TRAIN topics only
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


# ----------------------------------------------------------------------------- model
class GCGRU(nn.Module):
    """unit: gru (paper) | lstm | transformer | mlp (no temporal order: mean over time)."""

    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.5, fc=256, unit="gru", max_len=64):
        super().__init__()
        self.unit = unit
        if unit in ("gru", "lstm"):
            cls = nn.GRU if unit == "gru" else nn.LSTM
            self.rnn = cls(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        elif unit == "transformer":
            self.proj = nn.Linear(in_dim, hidden)
            self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
            layer = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        elif unit == "mlp":
            self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        else:
            raise ValueError(unit)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, fc), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(fc, 2))

    def forward(self, x):
        if self.unit in ("gru", "lstm"):
            h = self.rnn(x)[0][:, -1]
        elif self.unit == "transformer":
            h = self.enc(self.proj(x) + self.pos[:, :x.shape[1]]).mean(1)
        else:
            h = self.proj(x).mean(1)
        return self.head(h)                                   # softmax applied in predict()


# ----------------------------------------------------------------------------- train / eval
def metrics(y, p):
    y = np.asarray(y)
    pred = (np.asarray(p) >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {"acc": float(accuracy_score(y, pred)), "prec": float(pr[0]), "rec": float(rc[0]),
            "f1": float(f1[0]), "f1_nonguided": float(f1[1]), "macro_f1": float(f1.mean()), "auc": auc}


@torch.no_grad()
def predict(model, Xt, bs=512):
    model.eval()
    out = [torch.softmax(model(Xt[i:i + bs].to(DEVICE)), 1)[:, 1].cpu() for i in range(0, len(Xt), bs)]
    return torch.cat(out).numpy()


def train_eval(F, y, split, cfg, seed, unit="gru", keep_state=False):
    """F already normalised. Best epoch is chosen on validation AUC; test is reported at that epoch."""
    set_seed(seed)
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    Xt, yt = torch.tensor(F), torch.tensor(y)
    model = GCGRU(F.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"], cfg["fc"], unit).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    w = None
    if cfg["class_weight"]:
        cnt = np.bincount(y[tr], minlength=2).astype(np.float32)
        w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best, best_state, best_ep, t0 = -1.0, None, -1, time.time()
    Xva = Xt[torch.as_tensor(va)]
    for ep in range(cfg["epochs"]):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), cfg["batch"]):
            idx = torch.as_tensor(perm[i:i + cfg["batch"]])
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx].to(DEVICE)), yt[idx].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
        mv = metrics(y[va], predict(model, Xva))
        score = mv["auc"] if not np.isnan(mv["auc"]) else mv["acc"]
        if score > best:
            best, best_ep = score, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pv = predict(model, Xva)
    pt = predict(model, Xt[torch.as_tensor(te)])
    res = {"seed": seed, "best_epoch": best_ep, "val": metrics(y[va], pv), "test": metrics(y[te], pt),
           "test_prob": pt.tolist(), "seconds": round(time.time() - t0, 1)}
    if keep_state:
        res["state"] = best_state
    return res


def run_seeds(F_raw, y, split, cfg, seeds, unit="gru", T=None, keep_state=False):
    F = F_raw if T is None else F_raw[:, :T]
    mu, sd = fit_norm(F, split)
    Fn = ((F - mu) / sd).astype(np.float32)
    return [train_eval(Fn, y, split, cfg, s, unit, keep_state) for s in seeds], mu, sd


def summarize(runs, part="test", keys=("acc", "prec", "rec", "f1", "macro_f1", "auc")):
    return {k: (float(np.mean([r[part][k] for r in runs])), float(np.std([r[part][k] for r in runs])))
            for k in keys}


def fmt(summary):
    return "  ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in summary.items())