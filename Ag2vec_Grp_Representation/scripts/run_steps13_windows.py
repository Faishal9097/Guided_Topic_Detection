"""
Windows runner for Group 3 — Steps 1-3.

Pipeline:
    Group 2 outputs
        ↓
    Step 1-3:
        - User node vectors / Node2vec
        - User influence
        - Attention-weighted group vectors
        ↓
    Chunked .pkl outputs
        ↓
    Later: mut_export

This replaces the coder's run_group3_local.py steps13 command.


Expected Output:
Ag2vec_Grp_Representation/
└── data_output/
    ├── stance_clf.pkl
    ├── stance_probs_weibo.pkl
    │
    ├── weibo_all_0000.pkl
    ├── weibo_all_0001.pkl
    ├── weibo_all_0002.pkl
    ├── ...
    │
    ├── politifact_all_0000.pkl
    ├── politifact_all_0001.pkl
    ├── ...
    │
    ├── gossipcop_all_0000.pkl
    ├── gossipcop_all_0001.pkl
    └── ...
"""
"""
Windows runner for Group 3 — Steps 1-3.

Runs:
1. Weibo
2. Politifact
3. Gossipcop

The actual chunk processing remains inside group3_run.py.
This script only controls the execution order and displays progress.
"""

import argparse
import os
import sys
import time


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..")
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ============================================================
# IMPORTS
# ============================================================

from group3_data_access import WeiboData, FakeNewsNetData
import group3_run as G
from group3_steps1_3 import Cfg


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "Ag2vec_Grp_Representation",
    "data_output"
)


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(name, data, cfg, workers, chunk, limit=None):

    topic_ids = data.topic_ids()
    total_topics = len(topic_ids)

    if limit is not None:
        total_topics = min(limit, total_topics)

    total_chunks = (total_topics + chunk - 1) // chunk

    print()
    print("=" * 70)
    print(f"DATASET: {name.upper()}")
    print("=" * 70)
    print(f"Total topics : {len(topic_ids)}")
    print(f"Topics to run: {total_topics}")
    print(f"Chunks       : {total_chunks}")
    print(f"Chunk size   : {chunk}")
    print(f"Workers      : {workers}")
    print()

    start = time.time()

    G.run(
        data=data,
        kind=name,
        split="all",
        out_dir=OUT_DIR,
        cfg=cfg,
        chunk=chunk,
        workers=workers,
        limit=limit,
    )

    elapsed = time.time() - start

    print()
    print("-" * 70)
    print(f"{name.upper()} FINISHED")
    print(f"Time: {elapsed / 60:.2f} minutes")
    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run Group 3 Steps 1-3"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=50
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "weibo",
            "politifact",
            "gossipcop",
            "all"
        ],
        default="all"
    )

    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = Cfg()

    print()
    print("=" * 70)
    print("GROUP 3 — STEPS 1-3")
    print("AG2vec / Node2vec / Influence / Attention")
    print("=" * 70)

    print(f"Output : {OUT_DIR}")
    print(f"Workers: {args.workers}")
    print(f"Chunk  : {args.chunk}")
    print(f"Limit  : {args.limit}")
    print()

    total_start = time.time()

    # ========================================================
    # 1. WEIBO
    # ========================================================

    if args.dataset in ("weibo", "all"):

        print("[1/3] Loading Weibo...")

        wb = WeiboData()

        run_dataset(
            "weibo",
            wb,
            cfg,
            args.workers,
            args.chunk,
            args.limit
        )

    # ========================================================
    # 2. POLITIFACT
    # ========================================================

    if args.dataset in ("politifact", "all"):

        print("[2/3] Loading Politifact...")

        politifact = FakeNewsNetData("politifact")

        run_dataset(
            "politifact",
            politifact,
            cfg,
            args.workers,
            args.chunk,
            args.limit
        )

    # ========================================================
    # 3. GOSSIPCOP
    # ========================================================

    if args.dataset in ("gossipcop", "all"):

        print("[3/3] Loading Gossipcop...")

        gossipcop = FakeNewsNetData("gossipcop")

        run_dataset(
            "gossipcop",
            gossipcop,
            cfg,
            args.workers,
            args.chunk,
            args.limit
        )

    # ========================================================
    # FINAL
    # ========================================================

    elapsed = time.time() - total_start

    print()
    print("=" * 70)
    print("STEPS 1-3 COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed / 60:.2f} minutes")
    print()
    print("Output:")
    print(OUT_DIR)
    print()
    print("Next: verify the generated .pkl chunks before mut_export.")


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()