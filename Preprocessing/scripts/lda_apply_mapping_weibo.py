"""
Step 7 (Option C), Stage 2: Apply manual cluster-to-category mapping
based on inspection of LDA topic word lists from Stage 1.
"""

import json
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "weibo_ced"

# Mapping derived from manual inspection of Stage 1's discovered topics.
# Topics 3 and 5 were borderline calls — documented as such.
CLUSTER_TO_CATEGORY = {
    0: "social_events",
    1: "social_events",
    2: "health",
    3: "other",
    4: "politics",
    5: "social_events",
    6: "social_events",       # borderline: contains a health-adjacent word (生病/sick)
    7: "other",
    8: "social_events",
    9: "social_events",
    10: "social_events",
    11: "other",              # low confidence, too mixed
    12: "entertainment",      # borderline: also contains a political term (钓鱼岛)
    13: "entertainment",
    14: "social_events",
    15: "other",
    16: "other",
    17: "social_events",
    18: "other",
    19: "health",             # high confidence: explicit "健康" keyword
    20: "entertainment",
    21: "other",
    22: "social_events",
    23: "health",             # borderline: could be politics instead
}


def main():
    print("Loading topics with LDA cluster assignments...")
    with open(PROCESSED_DIR / "topics_with_lda_clusters.json", "r", encoding="utf-8") as f:
        topics = json.load(f)

    print(f"Applying category mapping to {len(topics)} topics...")

    for topic in topics:
        cluster = topic.get("lda_topic_cluster")
        category = CLUSTER_TO_CATEGORY.get(cluster, "other")
        topic["topic_category"] = category

    category_counts = Counter(t["topic_category"] for t in topics)

    print("\n" + "=" * 60)
    print("FINAL CATEGORY DISTRIBUTION (LDA-based)")
    print("=" * 60)
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count} ({count/len(topics)*100:.1f}%)")

    output_path = PROCESSED_DIR / "topics_with_categories_lda.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()