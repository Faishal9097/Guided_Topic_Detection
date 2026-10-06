"""
Step 3 (follow-up): Check how many of the 87 excluded original posts
belong to rumor vs. non-rumor threads.
"""

from pathlib import Path

BASE_DIR = Path("data/raw/weibo_ced/CED_Dataset")
RUMOR_REPOST_DIR = BASE_DIR / "rumor-repost"
NON_RUMOR_REPOST_DIR = BASE_DIR / "non-rumor-repost"

EXCLUDED_IDS_FILE = Path("data/processed/weibo_ced/excluded_original_ids.txt")


def main():
    with open(EXCLUDED_IDS_FILE, "r") as f:
        excluded_ids = set(line.strip() for line in f if line.strip())

    print(f"Total excluded IDs: {len(excluded_ids)}")

    rumor_ids = set(f.stem for f in RUMOR_REPOST_DIR.glob("*.json"))
    non_rumor_ids = set(f.stem for f in NON_RUMOR_REPOST_DIR.glob("*.json"))

    excluded_rumor = excluded_ids & rumor_ids
    excluded_non_rumor = excluded_ids & non_rumor_ids
    excluded_neither = excluded_ids - rumor_ids - non_rumor_ids

    print(f"\nExcluded posts that are RUMOR threads: {len(excluded_rumor)}")
    print(f"Excluded posts that are NON-RUMOR threads: {len(excluded_non_rumor)}")
    print(f"Excluded posts matching NEITHER folder (orphaned): {len(excluded_neither)}")

    if excluded_neither:
        print(f"\nOrphaned IDs (sample): {list(excluded_neither)[:5]}")

    print(f"\nNew rumor thread count after cleaning: {len(rumor_ids) - len(excluded_rumor)}")
    print(f"New non-rumor thread count after cleaning: {len(non_rumor_ids) - len(excluded_non_rumor)}")


if __name__ == "__main__":
    main()