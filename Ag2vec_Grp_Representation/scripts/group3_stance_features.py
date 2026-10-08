"""
Group 3, steps 4-6  (paper Eq. 24-39): soft stance -> group stance proportions -> internal/external factors
-> linear stance influence (regression) -> mutual influence Mut(g, t) = [Mut_sup, Mut_opo, Mut_none].

Inputs
  wb     : WeiboData (needs .sliced and .groups; no topic_view call, so it is cheap)
  steps  : {topic_id: [per-slice dict]} from group3_run.load_all  (needs influence_users / influence per slice)
  probs  : {topic_id: {mid: p(3)}} from group3_stance.predict_topics   (order: support, oppose, observe)

    import group3_stance_features as F
    feats = {tid: F.topic_features(tid, wb, steps[tid], probs[tid]) for tid in steps}
    train = [t for t in feats if wb.split_of(t) == 'train']
    rho = F.fit_rho(feats, train)                    # fitted on TRAIN topics only
    out = F.mut_vectors(feats, rho)                  # {(topic_id, slice_pos, gid): {"mut","ptilde","S"}}
    stance = {k: v["mut"] for k, v in out.items()}   # -> group3_export.export(..., stance=stance)

Proxies (the Weibo data has no followees / retweet-vs-original split) - cite these in the report:
  * P~_k (Eq. 26): confidence-weighted mean of the classifier's probabilities over the interactions of the group's
    members in the slice (alpha_j = max p^j).  If the group has no new interaction in the slice, all its interactions
    so far are used; if none at all, uniform.  The classifier's probabilities replace Eq. 24's label smoothing.
  * TopicAware (Eq. 6): own interaction count in the topic so far / count of the most active user (followees unavailable).
  * Act(g) (Eq. 8): share of the group's interactions so far that are pure reposts (empty text) = N_r / (N_r + N_ori).
  * TopicHeat (Eq. 7): log1p(cumulative interaction count)  (log keeps the heavy tail from dominating the regression).
  * Influence: the step-2 proxy (replies received).
  * Regression target (the paper does not say): P~_k of the SAME persistent group in the NEXT slice; one shared
    (rho0, rho1, rho2) for the three stances (as in Eq. 29-31), so every (group, slice, k) is one row.
  * Group level (Eq. 33): weights sum to 1 and E_out does not depend on the user, so aggregating users first gives the
    same S_k(g, t) as aggregating after the user-level regression.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

OMEGA = (0.5, 0.5)                       # Table VI (game-theory parameters omega_1, omega_2)
K3 = 3                                   # support, oppose, observe


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def topic_features(tid: str, wb, steps_t: List[dict], probs_t: Dict[str, np.ndarray]) -> List[dict]:
    """Per slice: {gid: {"ptilde": (3,), "ein": (3,), "eout": (3,), "has_data": bool, "size": int}}"""
    slices = sorted(wb.sliced[tid]["time_slices"], key=lambda s: s["slice_index"])
    gslices = wb.groups[tid]["slices"]
    cum: List[dict] = []
    out = []
    for pos, (s, gs) in enumerate(zip(slices, gslices)):
        new = s["new_interactions_this_slice"]
        cum = cum + new
        groups = gs["groups"]
        if not groups or pos >= len(steps_t):
            out.append({})
            continue
        st = steps_t[pos]
        infl = dict(zip(st.get("influence_users", []), st.get("influence", [])))
        n_user = Counter(i["uid"] for i in cum)
        n_max = max(n_user.values()) if n_user else 1
        heat = math.log1p(s["cumulative_interaction_count"])
        slice_out = {}
        for gid, members in groups.items():
            mem = set(members)
            g_cum = [i for i in cum if i["uid"] in mem]
            g_new = [i for i in new if i["uid"] in mem]
            use = g_new if g_new else g_cum
            num, den = np.zeros(K3), 0.0
            for i in use:
                p = probs_t.get(i["mid"])
                if p is None:
                    continue
                a = float(np.max(p))                       # Eq. 25
                num += a * np.asarray(p, float)
                den += a
            has = den > 0
            ptilde = num / den if has else np.full(K3, 1.0 / K3)      # Eq. 26
            act = (sum(1 for i in g_cum if not (i["text"] or "").strip()) / len(g_cum)) if g_cum else 0.0
            iv = np.array([infl.get(u, 0.0) for u in members], float)
            w = iv / iv.sum() if iv.sum() > 1e-9 else np.full(len(members), 1.0 / len(members))   # Eq. 32
            ta = np.array([n_user.get(u, 0) / n_max for u in members], float)
            ta_infl = float((w * ta * iv).sum())
            slice_out[gid] = {"ptilde": ptilde, "ein": ptilde * ta_infl,            # Eq. 27 (aggregated by w)
                              "eout": ptilde * act * heat,                          # Eq. 28
                              "has_data": bool(has), "size": len(members)}
        out.append(slice_out)
    return out


def fit_rho(feats: Dict[str, List[dict]], train_ids, require_data: bool = True):
    """Least squares for (rho0, rho1, rho2): P~_k(g, t+1) ~ rho0 + rho1*Ein_k(g,t) + rho2*Eout_k(g,t)."""
    A, y = [], []
    for tid in train_ids:
        sl = feats.get(tid, [])
        for t in range(len(sl) - 1):
            for gid, a in sl[t].items():
                b = sl[t + 1].get(gid)
                if b is None or (require_data and not (a["has_data"] and b["has_data"])):
                    continue
                for k in range(K3):
                    A.append([1.0, a["ein"][k], a["eout"][k]])
                    y.append(b["ptilde"][k])
    if not A:
        raise ValueError("no (group, slice) pairs to fit - check train ids / steps / probs")
    A, y = np.asarray(A), np.asarray(y)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = float(((y - A @ coef) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    print(f"rho fitted on {len(y)} rows: rho0={coef[0]:.4f} rho1={coef[1]:.4f} rho2={coef[2]:.4f}  R2={1 - ss_res / ss_tot:.3f}")
    return {"rho0": float(coef[0]), "rho1": float(coef[1]), "rho2": float(coef[2]), "n_rows": int(len(y)),
            "r2": float(1 - ss_res / ss_tot)}


def mut_vectors(feats: Dict[str, List[dict]], rho: dict, omega: Tuple[float, float] = OMEGA):
    """Eq. 29-39 -> {(topic_id, slice_pos, gid): {"mut": (3,), "ptilde": (3,), "S": (3,)}}"""
    o1, o2 = omega
    res = {}
    for tid, sl in feats.items():
        for t, groups in enumerate(sl):
            for gid, a in groups.items():
                S = rho["rho0"] + rho["rho1"] * a["ein"] + rho["rho2"] * a["eout"]      # Eq. 29-31, 33-35
                ssup, sopo, snone = float(S[0]), float(S[1]), float(S[2])
                mut = np.array([_sigmoid(ssup + snone - sopo),                           # Eq. 36
                                _sigmoid(sopo + snone - ssup),                           # Eq. 37
                                _sigmoid(snone + o1 * ssup + o2 * sopo)], np.float32)    # Eq. 38
                res[(tid, t, gid)] = {"mut": mut, "ptilde": a["ptilde"].astype(np.float32), "S": S.astype(np.float32)}
    return res
