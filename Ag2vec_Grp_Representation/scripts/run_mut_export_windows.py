"""
Final outputs:
D:\#Project\Innovative 7th\Guided_Topic_Detection\Ag2vec_Grp_Representation\data_output\weibo_group_feature_matrices.npz
D:\#Project\Innovative 7th\Guided_Topic_Detection\Ag2vec_Grp_Representation\data_output\politifact_group_feature_matrices.npz
D:\#Project\Innovative 7th\Guided_Topic_Detection\Ag2vec_Grp_Representation\data_output\gossipcop_group_feature_matrices.npz
"""


from __future__ import annotations

import os
import sys
import pickle
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "Ag2vec_Grp_Representation", "scripts")
OUT = os.path.join(ROOT, "Ag2vec_Grp_Representation", "data_output")

if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import group3_run as G
import group3_stance_features as F
import group3_export as E
from group3_data_access import WeiboData, FakeNewsNetData


def main():

    t0 = time.time()

    print("=" * 70)
    print("GROUP 3 — MUTUAL INFLUENCE + FINAL EXPORT")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load Weibo
    # ------------------------------------------------------------
    print("\n[1/4] Loading Weibo...")
    wb = WeiboData()

    # ------------------------------------------------------------
    # 2. Load stance probabilities
    # ------------------------------------------------------------
    print("[2/4] Loading stance probabilities...")

    stance_path = os.path.join(
        OUT, "stance_probs_weibo.pkl"
    )

    if not os.path.exists(stance_path):
        raise FileNotFoundError(
            f"Missing stance file:\n{stance_path}"
        )

    with open(stance_path, "rb") as f:
        probs = pickle.load(f)

    print(f"Stance topics loaded: {len(probs)}")

    # ------------------------------------------------------------
    # 3. Load Steps 1-3
    # ------------------------------------------------------------
    print("\n[3/4] Loading Steps 1-3 chunks...")

    steps = G.load_all(
        OUT,
        "weibo",
        "all"
    )

    print(f"Weibo Step 1-3 topics loaded: {len(steps)}")

    if len(steps) != len(wb.topic_ids()):
        raise RuntimeError(
            f"Weibo topic mismatch: "
            f"steps={len(steps)}, data={len(wb.topic_ids())}"
        )

    # ------------------------------------------------------------
    # 4. Steps 4-6
    # ------------------------------------------------------------
    print("\n[4/4] Computing stance features...")

    feats = {}

    total = len(steps)

    for i, tid in enumerate(steps, 1):

        if tid not in probs:
            raise RuntimeError(
                f"Missing stance probabilities for topic: {tid}"
            )

        feats[tid] = F.topic_features(
            tid,
            wb,
            steps[tid],
            probs[tid]
        )

        if i % 100 == 0 or i == total:
            print(
                f"\rProcessed {i}/{total} topics",
                end="",
                flush=True
            )

    print()

    # ------------------------------------------------------------
    # Fit regression ONLY on Weibo train topics
    # ------------------------------------------------------------
    train_ids = [
        tid for tid in feats
        if wb.split_of(tid) == "train"
    ]

    print(f"\nTraining topics for rho: {len(train_ids)}")

    rho = F.fit_rho(
        feats,
        train_ids
    )

    print("\nFitted rho:")
    print(rho)

    # ------------------------------------------------------------
    # Mutual influence vectors
    # ------------------------------------------------------------
    print("\nComputing mutual influence vectors...")

    mut_out = F.mut_vectors(
        feats,
        rho
    )

    stance = {
        key: value["mut"]
        for key, value in mut_out.items()
    }

    print(
        f"Mutual-influence vectors: {len(stance)}"
    )

    # Save Weibo Mut information
    mut_path = os.path.join(
        OUT,
        "weibo_stance_mut.pkl"
    )

    with open(mut_path, "wb") as f:
        pickle.dump(
            {
                "rho": rho,
                "mut": mut_out
            },
            f,
            protocol=4
        )

    print(f"Saved: {mut_path}")

    # ------------------------------------------------------------
    # WEIBO FINAL EXPORT
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPORTING WEIBO")
    print("=" * 70)

    weibo_topics = G.load_all(
        OUT,
        "weibo",
        "all"
    )

    weibo_path = os.path.join(
        OUT,
        "weibo_group_feature_matrices.npz"
    )

    E.export(
        weibo_topics,
        E.meta_from(wb),
        weibo_path,
        T=10,
        K=30,
        slot_mode="stable",
        dim=64,
        stance=stance
    )

    # ------------------------------------------------------------
    # POLITIFACT FINAL EXPORT
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPORTING POLITIFACT")
    print("=" * 70)

    pf = FakeNewsNetData("politifact")

    pf_topics = G.load_all(
        OUT,
        "politifact",
        "all"
    )

    pf_path = os.path.join(
        OUT,
        "politifact_group_feature_matrices.npz"
    )

    # FakeNewsNet has no comment-level stance probabilities.
    # Keep the 3 stance dimensions as zero.
    E.export(
        pf_topics,
        E.meta_from(pf),
        pf_path,
        T=1,
        K=30,
        slot_mode="stable",
        dim=64,
        stance={}
    )

    # ------------------------------------------------------------
    # GOSSIPCOP FINAL EXPORT
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPORTING GOSSIPCOP")
    print("=" * 70)

    gc = FakeNewsNetData("gossipcop")

    gc_topics = G.load_all(
        OUT,
        "gossipcop",
        "all"
    )

    gc_path = os.path.join(
        OUT,
        "gossipcop_group_feature_matrices.npz"
    )

    E.export(
        gc_topics,
        E.meta_from(gc),
        gc_path,
        T=1,
        K=30,
        slot_mode="stable",
        dim=64,
        stance={}
    )

    # ------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------
    elapsed = (time.time() - t0) / 60

    print("\n" + "=" * 70)
    print("MUT_EXPORT COMPLETE")
    print("=" * 70)

    print(f"Total time: {elapsed:.2f} minutes")

    print("\nFinal outputs:")
    print(weibo_path)
    print(pf_path)
    print(gc_path)


if __name__ == "__main__":
    main()