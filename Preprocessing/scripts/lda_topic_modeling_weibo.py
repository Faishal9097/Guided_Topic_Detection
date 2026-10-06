"""
Step 7 (Option C): LDA Topic Modeling — Weibo
Stage 1: Segments Chinese text via jieba, builds a document-term
matrix, trains LDA, and prints each discovered topic's top words for
manual inspection/labeling in Stage 2.
"""

import json
import jieba
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "weibo_ced"
NUM_TOPICS = 24  # increased from 12, to isolate finer/rarer clusters
TOP_WORDS_PER_TOPIC = 15

# Basic Chinese stopwords (common function words with little topical meaning)
STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "那", "还", "为", "与", "对", "但",
    "被", "从", "把", "让", "给", "他", "她", "它", "们", "这个", "什么",
    "可以", "因为", "所以", "如果", "转发", "微博", "回复", "评论",
}


def segment_text(text: str) -> str:
    words = jieba.lcut(text)
    filtered = [w for w in words if w.strip() and w not in STOPWORDS and len(w) > 1]
    return " ".join(filtered)


def main():
    print("Loading Weibo topics...")
    with open(PROCESSED_DIR / "topics_constructed.json", "r", encoding="utf-8") as f:
        topics = json.load(f)

    print(f"Segmenting text for {len(topics)} topics...")
    documents = []
    for i, topic in enumerate(topics):
        text = topic.get("original_post", {}).get("text", "")
        segmented = segment_text(text)
        documents.append(segmented if segmented else "empty")
        if (i + 1) % 1000 == 0:
            print(f"  Segmented {i + 1}/{len(topics)}...")

    print("\nBuilding document-term matrix...")
    vectorizer = CountVectorizer(max_df=0.75, min_df=3, max_features=3000)
    doc_term_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Training LDA with {NUM_TOPICS} topics...")

    lda = LatentDirichletAllocation(
        n_components=NUM_TOPICS,
        random_state=42,
        max_iter=30,
        learning_method="online",
    )
    lda.fit(doc_term_matrix)

    print("\n" + "=" * 60)
    print("DISCOVERED TOPICS (top words per cluster)")
    print("=" * 60)
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-TOP_WORDS_PER_TOPIC:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"\nTopic {topic_idx}: {', '.join(top_words)}")

    doc_topic_matrix = lda.transform(doc_term_matrix)
    dominant_topics = doc_topic_matrix.argmax(axis=1)

    for topic, dom_topic in zip(topics, dominant_topics):
        topic["lda_topic_cluster"] = int(dom_topic)

    output_path = PROCESSED_DIR / "topics_with_lda_clusters.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    print(f"\n\nSaved topics with LDA cluster assignments to: {output_path}")
    print("\nNEXT STEP: review the topic word lists above and map each")
    print("Topic N to one of: politics / entertainment / health / social_events / other")

if __name__ == "__main__":
    main()