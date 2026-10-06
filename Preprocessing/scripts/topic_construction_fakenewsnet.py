"""
Step 5 (light consolidation): Topic Construction — FakeNewsNet (UPFD)
UPFD's graph structure is inherently topic-organized already (one
graph_id = one topic/news article). This step consolidates edges,
node features, and Step 4's guided labels into unified per-topic
records, matching Weibo's structure for downstream consistency.

NOTE: No timestamp data exists in UPFD, so these topic records will
NOT support time-slicing (Step 6) — documented explicitly.
"""

import numpy as np
import scipy.sparse as sp
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

BASE_DIR = ROOT_DIR / "data" / "raw" / "fakenewsnet"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "fakenewsnet"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "fakenewsnet"

DATASETS = ["politifact", "gossipcop"]


def load_edge_list(folder: Path):
    a_file = folder / "A.txt"
    if not a_file.exists():
        a_file = folder / "A"
    edges = []
    with open(a_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                edges.append((int(parts[0].strip()), int(parts[1].strip())))
    return edges


def process_dataset(name: str):
    print("\n" + "=" * 60)
    print(f"CONSOLIDATING: {name.upper()}")
    print("=" * 60)

    raw_folder = BASE_DIR / name
    edges = load_edge_list(raw_folder)
    labels = np.load(raw_folder / "graph_labels.npy")
    node_graph_id = np.load(raw_folder / "node_graph_id.npy")

    guided_labels_path = PROCESSED_DIR / f"{name}_guided_labels_computed.json"
    with open(guided_labels_path, "r") as f:
        guided_data = json.load(f)
    guided_lookup = {item["graph_id"]: item for item in guided_data}

    edges_by_graph = {}
    for src, dst in edges:
        gid = int(node_graph_id[src])
        edges_by_graph.setdefault(gid, []).append((src, dst))

    topics = []
    for graph_id in range(len(labels)):
        node_indices = np.where(node_graph_id == graph_id)[0].tolist()
        guided_info = guided_lookup.get(graph_id, {})

        topic_record = {
            "topic_id": f"{name}_{graph_id}",
            "original_label": "fake" if labels[graph_id] == 1 else "real",
            "guided_label": guided_info.get("guided_label", "unknown"),
            "combined_guided_score": guided_info.get("combined_score"),
            "num_nodes": len(node_indices),
            "num_edges": len(edges_by_graph.get(graph_id, [])),
            "node_indices": node_indices,  # references into feature matrices
            "has_timestamp_data": False,  # confirmed limitation, see docs
        }
        topics.append(topic_record)

    print(f"Topics consolidated: {len(topics)}")
    guided_count = sum(1 for t in topics if t["guided_label"] == "guided")
    print(f"Guided: {guided_count}, Nonguided: {len(topics) - guided_count}")

    avg_nodes = sum(t["num_nodes"] for t in topics) / len(topics)
    avg_edges = sum(t["num_edges"] for t in topics) / len(topics)
    print(f"Avg nodes per topic: {avg_nodes:.1f}")
    print(f"Avg edges per topic: {avg_edges:.1f}")

    output_path = OUTPUT_DIR / f"{name}_topics_constructed.json"
    with open(output_path, "w") as f:
        json.dump(topics, f, indent=2)
    print(f"Saved to: {output_path}")


def main():
    for name in DATASETS:
        process_dataset(name)


if __name__ == "__main__":
    main()