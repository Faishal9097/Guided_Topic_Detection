"""
Step 7: Topic Category Tagging
FakeNewsNet: direct source-based mapping (Politifact->Politics,
Gossipcop->Entertainment), per paper's stated methodology.
Weibo: keyword-based approximation (paper used LDA + semantic tags,
which requires more NLP infrastructure than available here).
"""

import json
import jieba
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CATEGORY_KEYWORDS = {
    "politics": ["政府", "官员", "政策", "选举", "党", "腐败", "官方", "拆迁",
                 "维权", "人大", "法律", "法院", "警察", "公安"],
    "entertainment": ["明星", "电影", "演员", "娱乐", "综艺", "歌手", "演唱会",
                       "网红", "微博", "粉丝", "偶像", "选秀"],
    "health": ["医院", "医生", "疾病", "病毒", "健康", "治疗", "药", "癌症",
               "手术", "患者", "疫情", "感染"],
    "social_events": ["事故", "灾难", "地震", "火灾", "交通", "案件", "犯罪",
                       "失踪", "死亡", "救援", "袭击", "爆炸"],
}


def tag_weibo_topic(text: str) -> str:
    if not text:
        return "other"
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in text)
    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "other"
    return best_category


def tag_weibo_dataset():
    print("\n" + "=" * 60)
    print("WEIBO — Topic Category Tagging (keyword-based approximation)")
    print("=" * 60)

    input_path = PROCESSED_DIR / "weibo_ced" / "topics_constructed.json"
    with open(input_path, "r", encoding="utf-8") as f:
        topics = json.load(f)

    print(f"Tagging {len(topics)} topics...")
    category_counts = Counter()

    for i, topic in enumerate(topics):
        text = topic.get("original_post", {}).get("text", "")
        category = tag_weibo_topic(text)
        topic["topic_category"] = category
        category_counts[category] += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(topics)}...")

    print("\nCategory distribution:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count} ({count/len(topics)*100:.1f}%)")

    output_path = PROCESSED_DIR / "weibo_ced" / "topics_with_categories.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")


def tag_fakenewsnet_dataset(name: str, category: str):
    print("\n" + "=" * 60)
    print(f"{name.upper()} — Topic Category Tagging (direct source mapping)")
    print("=" * 60)

    input_path = PROCESSED_DIR / "fakenewsnet" / f"{name}_topics_constructed.json"
    with open(input_path, "r") as f:
        topics = json.load(f)

    for topic in topics:
        topic["topic_category"] = category

    print(f"Tagged {len(topics)} topics as '{category}' (100%, direct mapping)")

    output_path = PROCESSED_DIR / "fakenewsnet" / f"{name}_topics_with_categories.json"
    with open(output_path, "w") as f:
        json.dump(topics, f, indent=2)
    print(f"Saved to: {output_path}")


def main():
    tag_weibo_dataset()
    tag_fakenewsnet_dataset("politifact", "politics")
    tag_fakenewsnet_dataset("gossipcop", "entertainment")


if __name__ == "__main__":
    main()