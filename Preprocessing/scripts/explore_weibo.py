"""
Step 2: Weibo (CED) Dataset Exploration
Walks through original-microblog, rumor-repost, non-rumor-repost folders,
parses all JSON files, and prints a summary inventory.
"""

import json
from pathlib import Path
from collections import defaultdict

# ---- Config ----
BASE_DIR = Path("data/raw/weibo_ced/CED_Dataset")
ORIGINAL_DIR = BASE_DIR / "original-microblog"
RUMOR_REPOST_DIR = BASE_DIR / "rumor-repost"
NON_RUMOR_REPOST_DIR = BASE_DIR / "non-rumor-repost"


def load_json_files(folder: Path):
    """Load all .json files in a folder, skip hidden/system files."""
    records = {}
    if not folder.exists():
        print(f"⚠️  Folder not found: {folder}")
        return records

    json_files = [f for f in folder.glob("*.json")]
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                records[f.stem] = json.load(file)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse {f.name}: {e}")
        except Exception as e:
            print(f"❌ Error reading {f.name}: {e}")
    return records


def analyze_original_posts(posts: dict):
    print("\n" + "=" * 60)
    print("ORIGINAL MICROBLOG POSTS")
    print("=" * 60)
    print(f"Total posts loaded: {len(posts)}")

    has_full_user_profile = 0
    genders = defaultdict(int)
    missing_fields = defaultdict(int)
    malformed_user_field = 0

    expected_fields = ["text", "user", "time", "likes", "reposts", "comments"]
    expected_user_fields = ["verified", "gender", "messages", "followers", "friends"]

    for post_id, post in posts.items():
        for field in expected_fields:
            if field not in post:
                missing_fields[field] += 1

        user = post.get("user", {})

        # Defensive check: 'user' should be a dict, but some records may
        # have it as a string, null, or missing entirely — a data quality
        # issue worth documenting rather than crashing on.
        if not isinstance(user, dict):
            malformed_user_field += 1
            continue

        if all(f in user for f in expected_user_fields):
            has_full_user_profile += 1
        genders[user.get("gender", "unknown")] += 1

    print(f"Posts with complete user profile: {has_full_user_profile} / {len(posts)}")
    print(f"Posts with malformed 'user' field (not a dict): {malformed_user_field}")
    print(f"Gender distribution: {dict(genders)}")
    if missing_fields:
        print(f"Missing fields found: {dict(missing_fields)}")
    else:
        print("No missing top-level fields detected.")


def analyze_repost_folder(folder_name: str, repost_data: dict):
    print("\n" + "=" * 60)
    print(f"{folder_name.upper()}")
    print("=" * 60)
    print(f"Total original-post threads: {len(repost_data)}")

    total_replies = 0
    unique_uids = set()
    empty_text_count = 0
    has_nested_replies = 0

    for post_id, replies in repost_data.items():
        if not isinstance(replies, list):
            continue
        total_replies += len(replies)
        for reply in replies:
            unique_uids.add(reply.get("uid"))
            if reply.get("text", "") == "":
                empty_text_count += 1
            if reply.get("parent", "") != "":
                has_nested_replies += 1

    print(f"Total reply/repost records: {total_replies}")
    print(f"Unique reposting/commenting users: {len(unique_uids)}")
    print(f"Empty-text (pure repost) records: {empty_text_count}")
    print(f"Records with non-empty 'parent' (nested reply): {has_nested_replies}")

    return unique_uids


def main():
    print("Loading original-microblog posts...")
    original_posts = load_json_files(ORIGINAL_DIR)
    analyze_original_posts(original_posts)

    original_uids = set(original_posts.keys())

    print("\nLoading rumor-repost data...")
    rumor_repost = load_json_files(RUMOR_REPOST_DIR)
    rumor_uids = analyze_repost_folder("rumor-repost", rumor_repost)

    print("\nLoading non-rumor-repost data...")
    non_rumor_repost = load_json_files(NON_RUMOR_REPOST_DIR)
    non_rumor_uids = analyze_repost_folder("non-rumor-repost", non_rumor_repost)

    # ---- Cross-check: profile coverage gap ----
    print("\n" + "=" * 60)
    print("PROFILE DATA COVERAGE CHECK")
    print("=" * 60)
    all_reposting_uids = rumor_uids | non_rumor_uids
    print(f"Total unique users who reposted/commented: {len(all_reposting_uids)}")
    print(f"Total users with full profile (original posters): {len(original_uids)}")
    print("Note: original post filenames appear to be post IDs, not user IDs —")
    print("cross-referencing user-level coverage requires checking 'user' field")
    print("inside each original post separately (not just filename).")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original posts: {len(original_posts)}")
    print(f"Rumor-repost threads: {len(rumor_repost)}")
    print(f"Non-rumor-repost threads: {len(non_rumor_repost)}")
    print(f"Expected total (per paper Table III): 1538 nonguided + 1849 guided = 3387")
    print(f"Actual total original posts found: {len(original_posts)}")


if __name__ == "__main__":
    main()