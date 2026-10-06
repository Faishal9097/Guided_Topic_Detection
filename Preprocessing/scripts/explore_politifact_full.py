"""
Step 2 (continued): Explore the richer Politifact/BuzzFeed FakeNewsNet
release (Shu et al. 2018 version) - CSVs + .mat user feature files.
"""

import pandas as pd
from pathlib import Path
from scipy.io import loadmat

BASE_DIR = Path("data/raw/fakenewsnet/full_labels")


def inspect_csv(filename: str):
    path = BASE_DIR / filename
    if not path.exists():
        print(f"⚠️  Not found: {filename}")
        return
    try:
        df = pd.read_csv(path)
        print(f"\n{filename}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"⚠️  Could not read {filename} as CSV: {e}")


def inspect_text_file(filename: str, max_lines: int = 5):
    path = BASE_DIR / filename
    if not path.exists():
        print(f"⚠️  Not found: {filename}")
        return
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        print(f"\n{filename}")
        print(f"  Total lines: {len(all_lines)}")
        print(f"  First {max_lines} lines:")
        for line in all_lines[:max_lines]:
            print(f"    {line.strip()}")
    except Exception as e:
        print(f"⚠️  Could not read {filename}: {e}")


def inspect_mat_file(filename: str):
    path = BASE_DIR / filename
    if not path.exists():
        print(f"⚠️  Not found: {filename}")
        return
    try:
        mat = loadmat(path)
        print(f"\n{filename}")
        keys = [k for k in mat.keys() if not k.startswith("__")]
        print(f"  Keys: {keys}")
        for k in keys:
            print(f"  {k}: shape={mat[k].shape}, dtype={mat[k].dtype}")
    except Exception as e:
        print(f"⚠️  Could not read {filename} as .mat: {e}")


def main():
    print("=" * 60)
    print("POLITIFACT — Content CSVs")
    print("=" * 60)
    inspect_csv("PolitiFact_fake_news_content.csv")
    inspect_csv("PolitiFact_real_news_content.csv")

    print("\n" + "=" * 60)
    print("POLITIFACT — Network/User files (text format)")
    print("=" * 60)
    inspect_text_file("PolitifactNews")
    inspect_text_file("PolitifactNewsUser")
    inspect_text_file("PolitifactUser")
    inspect_text_file("PolitifactUserUser")

    print("\n" + "=" * 60)
    print("POLITIFACT — User Features (.mat)")
    print("=" * 60)
    inspect_mat_file("PolitifactUserFeature.mat")


if __name__ == "__main__":
    main()