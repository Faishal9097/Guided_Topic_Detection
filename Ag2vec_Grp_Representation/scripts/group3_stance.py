"""
Stance classifier for Weibo comments (paper Eq. 24-25 inputs): support / oppose / observe.

Plan: multilingual sentence embeddings (GPU helps for the encoding step) + logistic regression trained on the
300 hand-labelled comments (stance_sample_labeled.csv).  The classifier's predicted probabilities are used directly as
the soft label p^j (paper Eq. 24) and max(p^j) as the confidence alpha_j (Eq. 25), instead of label smoothing.

Colab (GPU runtime):
    !pip -q install sentence-transformers
    import sys; sys.path.insert(0, '/content/g3')
    import group3_stance as S
    embed = S.sbert_embedder()                                   # paraphrase-multilingual-MiniLM-L12-v2
    df = S.load_labeled('/content/stance_sample_labeled.csv')
    for pair in (False, True):                                   # comment only vs comment x source features
        X = S.features(df.comment, df.source, embed, pair)
        print('pair' if pair else 'comment only', S.tune_and_report(X, df.label.values))
    clf = S.fit(X_best, df.label.values, C_best)
    probs = S.predict_topics(wb, wb.topic_ids(), clf, embed, om, pair=...)   # {topic_id: {mid: p(3)}}

Class order everywhere: [support, oppose, observe]  (= Mut_sup, Mut_opo, Mut_none).
Empty-text interactions (pure reposts, ~46% of Weibo interactions) get EMPTY_P: mostly observe, low confidence.
That value is an assumption - state it in the report.
"""
from __future__ import annotations

import pickle
from typing import Callable, Dict, List, Optional

import numpy as np

LABELS = ["support", "oppose", "observe"]
EMPTY_P = np.array([0.1, 0.1, 0.8], np.float32)


# ----------------------------------------------------------------------------- data / features
def load_labeled(path):
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    df["comment"] = df["comment"].astype(str).str.strip()
    return df


def sbert_embedder(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: Optional[str] = None,
                   batch: int = 256) -> Callable:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=device)

    def embed(texts):
        return m.encode([str(t) for t in texts], batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=False)
    return embed


def features(comments, sources, embed: Callable, pair: bool = False) -> np.ndarray:
    ec = np.asarray(embed(list(comments)))
    if not pair:
        return ec
    es = np.asarray(embed(list(sources)))
    return np.hstack([ec, ec * es])                    # comment + agreement-with-source features


# ----------------------------------------------------------------------------- train / evaluate
def _lr(C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


def cv_scores(X, y, C=1.0, k=5, seed=0):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    pred = cross_val_predict(_lr(C), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed))
    return {"acc": round(float(accuracy_score(y, pred)), 3),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 3)}, pred


def tune_and_report(X, y, Cs=(0.1, 0.3, 1.0, 3.0, 10.0)):
    """5-fold CV over C.  The best-C score is slightly optimistic (C chosen on the same folds) - say so in the report."""
    from sklearn.metrics import classification_report
    best = None
    for C in Cs:
        sc, pred = cv_scores(X, y, C)
        print(f"C={C:<5} {sc}")
        if best is None or sc["macro_f1"] > best[1]["macro_f1"]:
            best = (C, sc, pred)
    print(classification_report(y, best[2], digits=2))
    return {"C": best[0], **best[1]}


def fit(X, y, C=1.0):
    return _lr(C).fit(X, y)


def _ordered(clf, P):
    idx = [list(clf.classes_).index(l) for l in LABELS]
    return P[:, idx].astype(np.float32)


# ----------------------------------------------------------------------------- inference over interactions
def predict_topics(wb, tids: List[str], clf, embed: Callable, om: Optional[dict] = None, pair: bool = False,
                   block: int = 20000, save_to: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """p^j for every interaction of the given topics.  om = original_microblog_cleaned.json (needed if pair=True)."""
    out = {tid: {} for tid in tids}
    rows = []                                           # (tid, mid, text)
    for tid in tids:
        for s in wb.sliced[tid]["time_slices"]:
            for i in s["new_interactions_this_slice"]:
                txt = (i["text"] or "").strip()
                if txt:
                    rows.append((tid, i["mid"], txt))
                else:
                    out[tid][i["mid"]] = EMPTY_P
    src_cache = {}
    for b in range(0, len(rows), block):
        print(f"Encoding interactions {b:,}/{len(rows):,}", flush=True)
        part = rows[b:b + block]
        ec = np.asarray(embed([r[2] for r in part]))
        if pair:
            for tid in {r[0] for r in part} - set(src_cache):
                src_cache[tid] = np.asarray(embed([om[tid]["text"]]))[0]
            ec = np.hstack([ec, ec * np.stack([src_cache[r[0]] for r in part])])
        P = _ordered(clf, clf.predict_proba(ec))
        for (tid, mid, _), p in zip(part, P):
            out[tid][mid] = p
    if save_to:
        with open(save_to, "wb") as f:
            pickle.dump(out, f, protocol=4)
    return out
