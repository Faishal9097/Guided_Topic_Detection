"""
Step 3: Data Cleaning — FakeNewsNet (UPFD format)
Validates structural consistency across edge list, node-graph mapping,
feature matrices, and labels for Politifact and Gossipcop.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path

BASE_DIR = Path("data/raw/fakenewsnet")
OUTPUT_DIR = Path("data/processed/fakenewsnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
                try:
                    src, dst = int(parts[0].strip()), int(parts[1].strip())
                    edges.append((src, dst))
                except ValueError:
                    continue
    return edges


def clean_dataset(name: str):
    print("\n" + "=" * 60)
    print(f"CLEANING: {name.upper()}")
    print("=" * 60)

    folder = BASE_DIR / name
    report = {}

    edges = load_edge_list(folder)
    labels = np.load(folder / "graph_labels.npy")
    node_graph_id = np.load(folder / "node_graph_id.npy")

    num_nodes = len(node_graph_id)
    num_graphs = len(labels)

    report["num_nodes"] = num_nodes
    report["num_graphs"] = num_graphs
    report["num_edges_raw"] = len(edges)

    invalid_edges = [
        (s, d) for s, d in edges
        if s < 0 or s >= num_nodes or d < 0 or d >= num_nodes
    ]
    report["invalid_edges"] = len(invalid_edges)
    valid_edges = [(s, d) for s, d in edges if (s, d) not in invalid_edges]
    report["valid_edges"] = len(valid_edges)

    unique_graph_ids = np.unique(node_graph_id)
    report["unique_graph_ids_in_mapping"] = len(unique_graph_ids)
    report["graph_id_range"] = f"{unique_graph_ids.min()} to {unique_graph_ids.max()}"
    report["labels_count"] = num_graphs
    report["graph_id_label_mismatch"] = len(unique_graph_ids) != num_graphs

    feature_files = [
        "new_profile_feature.npz",
        "new_content_feature.npz",
        "new_spacy_feature.npz",
        "new_bert_feature.npz",
    ]
    feature_shapes = {}
    for feat_file in feature_files:
        path = folder / feat_file
        if path.exists():
            mat = sp.load_npz(path)
            feature_shapes[feat_file] = mat.shape[0]
    report["feature_row_counts"] = feature_shapes
    report["feature_rows_match_nodes"] = all(
        v == num_nodes for v in feature_shapes.values()
    )

    graph_node_counts = np.bincount(node_graph_id.astype(int), minlength=num_graphs)
    empty_graphs = np.where(graph_node_counts == 0)[0]
    report["empty_graphs"] = len(empty_graphs)

    unique_labels = np.unique(labels)
    report["unique_label_values"] = unique_labels.tolist()
    report["labels_valid_binary"] = set(unique_labels.tolist()).issubset({0.0, 1.0})

    for k, v in report.items():
        print(f"  {k}: {v}")

    edge_output_path = OUTPUT_DIR / f"{name}_edges_cleaned.txt"
    with open(edge_output_path, "w") as f:
        for s, d in valid_edges:
            f.write(f"{s},{d}\n")
    print(f"\n  Cleaned edge list saved to: {edge_output_path}")

    return report


def main():
    all_reports = {}
    for name in DATASETS:
        all_reports[name] = clean_dataset(name)

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    for name, report in all_reports.items():
        status = "CLEAN" if (
            report["invalid_edges"] == 0
            and not report["graph_id_label_mismatch"]
            and report["feature_rows_match_nodes"]
            and report["empty_graphs"] == 0
            and report["labels_valid_binary"]
        ) else "ISSUES FOUND"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()