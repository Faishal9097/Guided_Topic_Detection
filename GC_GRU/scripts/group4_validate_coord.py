"""Synthetic-injection validation of the bot-like score (no ground truth exists, so we plant known bots).

  python group4_validate_coord.py --sliced topics_time_sliced.json --n-topics 200

For each real topic we add synthetic bot accounts and check how many get flagged (recall), and how many REAL users
are flagged before/after (false-positive rate).  Scenarios:
  rapid     : bots post 5 messages each with unique texts, 5-20 s apart
  copy_K    : K bots each post ONE identical text within +-30 s of each other (a botnet burst); K = 2,3,5,10
  both      : identical text AND rapid
  stealth   : unique texts, hours apart (a careful bot) -> expected NOT detected; reported honestly
"""
import argparse
import json
import random

import numpy as np

from group4_coord_lib import analyse, prep


def mk(uid, mid, ts, text):
    from group4_coord_lib import norm_text
    return {"uid": uid, "mid": mid, "parent": "", "ts": ts, "nt": norm_text(text), "sl": 0}


def rnd_text(rng, n=10):
    return "".join(chr(rng.randrange(0x4E00, 0x4E00 + 2000)) for _ in range(n))


def inject(recs, scenario, K, rng):
    t_lo, t_hi = min(r["ts"] for r in recs), max(r["ts"] for r in recs)
    t_mid = rng.uniform(t_lo, t_hi)
    out, bots = [], []
    shared = "这个消息大家快转发一下" + rnd_text(rng, 4)
    for b in range(K):
        u = f"BOT{b}"
        bots.append(u)
        if scenario == "rapid":
            t = t_mid + rng.uniform(0, 300)
            for m in range(5):
                out.append(mk(u, f"{u}_{m}", t, rnd_text(rng))); t += rng.uniform(5, 20)
        elif scenario == "copy":
            out.append(mk(u, f"{u}_0", t_mid + rng.uniform(-30, 30), shared))
        elif scenario == "both":
            t = t_mid + rng.uniform(-30, 30)
            for m in range(5):
                out.append(mk(u, f"{u}_{m}", t, shared)); t += rng.uniform(5, 20)
        elif scenario == "stealth":
            t = rng.uniform(t_lo, t_hi)
            for m in range(5):
                out.append(mk(u, f"{u}_{m}", t, rnd_text(rng))); t += rng.uniform(3600, 7200)
    return recs + out, set(bots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sliced", required=True)
    ap.add_argument("--n-topics", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    with open(a.sliced, encoding="utf-8") as f:
        topics = json.load(f)
    pool = []
    for t in topics:
        inter = [i for s in t["time_slices"] for i in s["new_interactions_this_slice"]]
        if len(inter) >= 50:
            pool.append(prep(inter))
        if len(pool) >= a.n_topics:
            break
    print(f"{len(pool)} real topics (>=50 interactions)")

    base_fp = np.mean([np.mean([d["bot"] for d in analyse(r)[0].values()]) for r in pool])
    print(f"real users flagged bot-like with NO injection: {base_fp:.3%}  <- false-positive baseline\n")

    from sklearn.metrics import roc_auc_score
    rows = [("rapid", 10), ("copy", 2), ("copy", 3), ("copy", 5), ("copy", 10), ("both", 10), ("stealth", 10)]
    print(f"{'scenario':10s}{'K':>4s}{'bot recall':>12s}{'real FP after':>15s}{'AUC(score)':>12s}")
    for sc, K in rows:
        rec, fp, aucs = [], [], []
        for r in pool:
            r2, bots = inject(r, sc, K, rng)
            users = analyse(r2)[0]
            rec.append(np.mean([users[b]["bot"] for b in bots]))
            real = [d for u, d in users.items() if u not in bots]
            fp.append(np.mean([d["bot"] for d in real]))
            y = [int(u in bots) for u in users]
            aucs.append(roc_auc_score(y, [d["score"] for d in users.values()]))
        print(f"{sc:10s}{K:4d}{np.mean(rec):12.3f}{np.mean(fp):15.3%}{np.mean(aucs):12.3f}")
    print("\nExpected: rapid/both ~1.0, copy needs K>=3, stealth ~0 (limitation: slow unique-text bots are invisible).")


if __name__ == "__main__":
    main()
