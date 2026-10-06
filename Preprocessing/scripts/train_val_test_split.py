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


def stratified_split(topics: list, label_key: str = "guided_label"):
    """
    Splits topics into train/val/test, stratified by label_key,
    so each split preserves the overall guided/nonguided ratio.
    """
    by_label = {}
    for topic in topics:
        label = topic.get(label_key, "unknown")
        by_label.setdefault(label, []).append(topic)

    train, val, test = [], [], []

    for label, items in by_label.items():
        shuffled = items[:]
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # remainder goes to test, so all items are accounted for
        n_test = n - n_train - n_val

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    return train, val, test


def process_file(input_path: Path, output_prefix: Path, label_key: str = "guided_label"):
    print(f"\nProcessing: {input_path.name}")
    with open(input_path, "r", encoding="utf-8") as f:
        topics = json.load(f)

    train, val, test = stratified_split(topics, label_key)

    print(f"  Total topics: {len(topics)}")
    print(f"  Train: {len(train)} ({len(train)/len(topics)*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/len(topics)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(topics)*100:.1f}%)")

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        guided_count = sum(1 for t in split_data if t.get(label_key) == "guided")
        nonguided_count = len(split_data) - guided_count
        print(f"    {split_name}: guided={guided_count}, nonguided={nonguided_count}")

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
    process_file(
        PROCESSED_DIR / "weibo_ced" / "topics_time_sliced.json",
        PROCESSED_DIR / "weibo_ced" / "topics_split",
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