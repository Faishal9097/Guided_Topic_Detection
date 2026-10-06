"""
Step 5: Topic Construction — Weibo (CED)
Assembles unified per-topic records combining original post, its
guided/nonguided label (Step 4), and its full interaction list
(reposts/comments), ready for time-slicing in Step 6.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "weibo_ced"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "weibo_ced"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Loading all processed Weibo data...")
    original_posts = load_json(PROCESSED_DIR / "original_microblog_cleaned.json")
    rumor_repost = load_json(PROCESSED_DIR / "rumor_repost_cleaned.json")
    non_rumor_repost = load_json(PROCESSED_DIR / "non_rumor_repost_cleaned.json")
    guided_labels = load_json(PROCESSED_DIR / "guided_labels_computed.json")

    # Build lookup: post_id -> guided label info
    guided_lookup = {item["post_id"]: item for item in guided_labels}

    topics = []
    skipped_no_repost_data = 0

    for post_id, post in original_posts.items():
        # Find matching repost/comment thread (only one of the two will match)
        interactions = rumor_repost.get(post_id) or non_rumor_repost.get(post_id)

        if interactions is None:
            skipped_no_repost_data += 1
            continue

        guided_info = guided_lookup.get(post_id, {})

        topic_record = {
            "topic_id": post_id,
            "original_label": guided_info.get("original_label", "unknown"),
            "guided_label": guided_info.get("guided_label", "unknown"),
            "combined_guided_score": guided_info.get("combined_score"),
            "original_post": {
                "text": post.get("text", ""),
                "time": post.get("time"),
                "user_profile": post.get("user", {}),
                "likes": post.get("likes", 0),
                "reposts": post.get("reposts", 0),
                "comments": post.get("comments", 0),
            },
            "interactions": [
                {
                    "uid": r.get("uid"),
                    "text": r.get("text", ""),
                    "date": r.get("date"),
                    "parent": r.get("parent", ""),
                    "mid": r.get("mid"),
                    "has_profile_data": False,  # see note: no shared ID system
                }
                for r in interactions
            ],
            "num_interactions": len(interactions),
        }

        topics.append(topic_record)

    print(f"\nTotal topics constructed: {len(topics)}")
    print(f"Skipped (no matching repost data): {skipped_no_repost_data}")

    if topics:
        interaction_counts = [t["num_interactions"] for t in topics]
        print(f"Avg interactions per topic: {sum(interaction_counts)/len(interaction_counts):.1f}")
        print(f"Min interactions: {min(interaction_counts)}")
        print(f"Max interactions: {max(interaction_counts)}")

    guided_count = sum(1 for t in topics if t["guided_label"] == "guided")
    print(f"\nGuided topics: {guided_count}")
    print(f"Nonguided topics: {len(topics) - guided_count}")

    output_path = OUTPUT_DIR / "topics_constructed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()