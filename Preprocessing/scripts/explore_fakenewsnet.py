"""
Step 2: FakeNewsNet (UPFD format via safe-graph/GNN-FakeNews) Exploration
Inspects the structure and content of Politifact and Gossipcop datasets.
"""

import numpy as np
from pathlib import Path

BASE_DIR = Path("data/raw/fakenewsnet")
DATASETS = ["politifact", "gossipcop"]


def inspect_dataset(name: str):
    print("\n" + "=" * 60)
    print(f"DATASET: {name.upper()}")
    print("=" * 60)

    folder = BASE_DIR / name
    if not folder.exists():
        print(f"⚠️  Folder not found: {folder}")
        return

    # --- Edge list (A) ---
        # --- Edge list (A) ---
    a_file = folder / "A.txt"
    if not a_file.exists():
        a_file = folder / "A"  # fallback, no extension
    if a_file.exists():
        with open(a_file, "r") as f:
            lines = f.readlines()
        print(f"Edge list (A): {len(lines)} edges")
        print(f"  Sample lines: {lines[:3]}")
    else:
        print("⚠️  'A' edge list file not found (checked both 'A.txt' and 'A')")
    # --- Graph labels ---
    labels_path = folder / "graph_labels.npy"
    if labels_path.exists():
        labels = np.load(labels_path)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nGraph labels shape: {labels.shape}")
        print(f"Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # --- Node -> graph mapping ---
    node_graph_path = folder / "node_graph_id.npy"
    if node_graph_path.exists():
        node_graph_id = np.load(node_graph_path)
        print(f"\nnode_graph_id shape: {node_graph_id.shape}")
        print(f"Total nodes (users) across all graphs: {len(node_graph_id)}")
        print(f"Total unique graphs referenced: {len(np.unique(node_graph_id))}")

    # --- Feature files (sparse) ---
    feature_files = [
        "new_profile_feature.npz",
        "new_content_feature.npz",
        "new_spacy_feature.npz",
        "new_bert_feature.npz",
    ]

    for feat_file in feature_files:
        feat_path = folder / feat_file
        if not feat_path.exists():
            print(f"\n⚠️  {feat_file} not found")
            continue

        try:
            import scipy.sparse as sp
            matrix = sp.load_npz(feat_path)
            print(f"\n{feat_file}:")
            print(f"  Shape: {matrix.shape}  (rows=nodes, cols=feature_dims)")
            print(f"  Non-zero entries: {matrix.nnz}")
            print(f"  Sample row (first node, first 10 values): "
                  f"{matrix[0, :10].toarray().flatten()}")
        except Exception as e:
            print(f"\n⚠️  Could not load {feat_file} as sparse matrix: {e}")

    # --- Train/val/test split sizes ---
    for split in ["train_idx.npy", "val_idx.npy", "test_idx.npy"]:
        split_path = folder / split
        if split_path.exists():
            idx = np.load(split_path)
            print(f"\n{split}: {len(idx)} graphs")


def main():
    for dataset_name in DATASETS:
        inspect_dataset(dataset_name)


if __name__ == "__main__":
    main()