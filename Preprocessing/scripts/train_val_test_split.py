"""
Step 8: Train/Val/Test Split
Splits topics into 80/10/10 train/val/test sets, stratified by
guided/nonguided label to preserve class balance. Applied to Weibo,
Politifact, and Gossipcop.
"""

import json
import random
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
random.seed(42)  # reproducibility

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def _tid(t):
    return str(t.get("topic_id", t.get("post_id", t.get("graph_id"))))


def stratified_split(topics: list, label_key: str = "original_label", keep_in_train=frozenset()):
    by_label = {}
    for t in topics:
        lab = t.get(label_key)
        if lab is None:
            raise KeyError(f"topic {_tid(t)} has no '{label_key}'; carry it through topic construction first")
        by_label.setdefault(lab, []).append(t)
    train, val, test = [], [], []
    for label, items in by_label.items():
        n = len(items)
        n_val = int(n * VAL_RATIO)
        n_test = n - int(n * TRAIN_RATIO) - n_val
        forced = [t for t in items if _tid(t) in keep_in_train]
        free = [t for t in items if _tid(t) not in keep_in_train]
        random.shuffle(free)
        val.extend(free[:n_val])
        test.extend(free[n_val:n_val + n_test])
        train.extend(free[n_val + n_test:] + forced)
    return train, val, test



def process_file(input_path: Path, output_prefix: Path, label_key: str = "original_label", keep_in_train=frozenset()):
    print(f"\nProcessing: {input_path.name}")
    with open(input_path, "r", encoding="utf-8") as f:
        topics = json.load(f)

    train, val, test = stratified_split(topics, label_key, keep_in_train)

    print(f"  Total topics: {len(topics)}")
    print(f"  Train: {len(train)} ({len(train)/len(topics)*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/len(topics)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(topics)*100:.1f}%)")

    from collections import Counter
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        print(f"    {split_name}: {dict(Counter(t.get(label_key) for t in split_data))}")
        print(f"    {split_name} guided_label (secondary): {dict(Counter(t.get('guided_label') for t in split_data))}")
    # Save each split
    with open(f"{output_prefix}_train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(f"{output_prefix}_val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    with open(f"{output_prefix}_test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {output_prefix}_{{train,val,test}}.json")


def main():
    # --- Weibo ---
    import pandas as pd
    keep = set(pd.read_csv("scripts/stance_sample_labeled.csv", encoding="utf-8-sig")["topic_id"].astype(str))
    process_file(
        PROCESSED_DIR / "weibo_ced" / "topics_time_sliced.json",
        PROCESSED_DIR / "weibo_ced" / "topics_split", keep_in_train=keep
    )

    # --- FakeNewsNet: Politifact ---
    process_file(
        PROCESSED_DIR / "fakenewsnet" / "politifact_topics_constructed.json",
        PROCESSED_DIR / "fakenewsnet" / "politifact_split",
    )

    # --- FakeNewsNet: Gossipcop ---
    process_file(
        PROCESSED_DIR / "fakenewsnet" / "gossipcop_topics_constructed.json",
        PROCESSED_DIR / "fakenewsnet" / "gossipcop_split",
    )


if __name__ == "__main__":
    main()