"""
Step 3: Data Cleaning — Weibo (CED) Dataset
Cleans original-microblog, rumor-repost, and non-rumor-repost data.
Outputs cleaned data + a cleaning report to data/processed/.
"""

import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("data/raw/weibo_ced/CED_Dataset")
ORIGINAL_DIR = BASE_DIR / "original-microblog"
RUMOR_REPOST_DIR = BASE_DIR / "rumor-repost"
NON_RUMOR_REPOST_DIR = BASE_DIR / "non-rumor-repost"

OUTPUT_DIR = Path("data/processed/weibo_ced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json_files(folder: Path):
    records = {}
    for f in folder.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                records[f.stem] = json.load(file)
        except Exception as e:
            print(f"❌ Failed to load {f.name}: {e}")
    return records


def clean_original_posts(posts: dict):
    """
    Clean original-microblog posts.
    Returns: (cleaned_posts, report)
    """
    report = {
        "total_input": len(posts),
        "malformed_user_field": 0,
        "empty_text": 0,
        "missing_time": 0,
        "excluded_ids": [],
        "final_count": 0,
    }

    cleaned = {}

    for post_id, post in posts.items():
        issues = []

        # Check user field type
        user = post.get("user", {})
        if not isinstance(user, dict):
            report["malformed_user_field"] += 1
            issues.append("malformed_user")

        # Check text
        text = post.get("text", "")
        if not text or not isinstance(text, str) or text.strip() == "":
            report["empty_text"] += 1
            issues.append("empty_text")

        # Check time
        time_val = post.get("time")
        if time_val is None or not isinstance(time_val, (int, float)):
            report["missing_time"] += 1
            issues.append("missing_time")

        # Decision: exclude if user field is malformed (can't compute
        # Influence() at all without it) OR text is empty (core content
        # missing). Missing time is logged but NOT excluded (can be
        # imputed/handled downstream if needed).
        if "malformed_user" in issues or "empty_text" in issues:
            report["excluded_ids"].append(post_id)
            continue

        cleaned[post_id] = post

    report["final_count"] = len(cleaned)
    return cleaned, report


def clean_repost_folder(repost_data: dict):
    """
    Clean rumor-repost / non-rumor-repost data.
    Returns: (cleaned_data, report)
    """
    report = {
        "total_threads_input": len(repost_data),
        "total_records_input": 0,
        "records_with_missing_uid": 0,
        "records_with_missing_mid": 0,
        "total_records_output": 0,
        "empty_threads_excluded": 0,
    }

    cleaned = {}

    for post_id, replies in repost_data.items():
        if not isinstance(replies, list):
            continue

        report["total_records_input"] += len(replies)
        valid_replies = []

        for reply in replies:
            uid = reply.get("uid")
            mid = reply.get("mid")

            if uid is None or uid == "":
                report["records_with_missing_uid"] += 1
                continue
            if mid is None or mid == "":
                report["records_with_missing_mid"] += 1
                continue

            valid_replies.append(reply)

        if len(valid_replies) == 0:
            report["empty_threads_excluded"] += 1
            continue

        cleaned[post_id] = valid_replies
        report["total_records_output"] += len(valid_replies)

    return cleaned, report


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_report(title: str, report: dict):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for k, v in report.items():
        if k == "excluded_ids":
            print(f"  {k}: {len(v)} IDs (see cleaning log for details)")
        else:
            print(f"  {k}: {v}")


def main():
    # --- Original posts ---
    print("Loading and cleaning original-microblog posts...")
    original_posts = load_json_files(ORIGINAL_DIR)
    cleaned_original, original_report = clean_original_posts(original_posts)
    print_report("ORIGINAL POSTS — Cleaning Report", original_report)

    save_json(cleaned_original, OUTPUT_DIR / "original_microblog_cleaned.json")

    # Save excluded IDs log separately
    with open(OUTPUT_DIR / "excluded_original_ids.txt", "w") as f:
        f.write("\n".join(original_report["excluded_ids"]))

    # --- Rumor repost ---
    print("\nLoading and cleaning rumor-repost data...")
    rumor_repost = load_json_files(RUMOR_REPOST_DIR)
    cleaned_rumor, rumor_report = clean_repost_folder(rumor_repost)
    print_report("RUMOR-REPOST — Cleaning Report", rumor_report)

    save_json(cleaned_rumor, OUTPUT_DIR / "rumor_repost_cleaned.json")

    # --- Non-rumor repost ---
    print("\nLoading and cleaning non-rumor-repost data...")
    non_rumor_repost = load_json_files(NON_RUMOR_REPOST_DIR)
    cleaned_non_rumor, non_rumor_report = clean_repost_folder(non_rumor_repost)
    print_report("NON-RUMOR-REPOST — Cleaning Report", non_rumor_report)

    save_json(cleaned_non_rumor, OUTPUT_DIR / "non_rumor_repost_cleaned.json")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Original posts: {original_report['total_input']} -> {original_report['final_count']} "
          f"({original_report['total_input'] - original_report['final_count']} excluded)")
    print(f"Rumor threads: {rumor_report['total_threads_input']} -> "
          f"{len(cleaned_rumor)} ({rumor_report['empty_threads_excluded']} excluded)")
    print(f"Non-rumor threads: {non_rumor_report['total_threads_input']} -> "
          f"{len(cleaned_non_rumor)} ({non_rumor_report['empty_threads_excluded']} excluded)")
    print(f"\nCleaned data saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()