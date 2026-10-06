"""
Before/after comparison for the label-construction rewrite.

Compares the OLD calibrated-heuristic guided_labels_computed.json output
(produced by the previous version of label_construction_weibo.py /
label_construction_fakenewsnet.py) against the NEW rule-based rubric
output, and reports:
  - how many topics flipped label (guided <-> nonguided)
  - for topics where the new file has per-dimension flags, *why* they
    flipped (which dimension flags fired)
  - the overall guided-ratio shift

Usage:
  python3 compare_labels_before_after.py \
      --old-weibo path/to/old_guided_labels_computed.json \
      --new-weibo data/processed/weibo_ced/guided_labels_computed.json \
      --old-politifact path/to/old_politifact_guided_labels_computed.json \
      --new-politifact data/processed/fakenewsnet/politifact_guided_labels_computed.json \
      --old-gossipcop path/to/old_gossipcop_guided_labels_computed.json \
      --new-gossipcop data/processed/fakenewsnet/gossipcop_guided_labels_computed.json

You must supply the OLD files yourself (a copy of the calibrated
version's output, saved before re-running the new script) — this repo
does not fabricate a "before" run. If you don't have the old JSON
saved, re-run the previous script version once to produce it, or skip
whichever --old-* argument you don't have; that dataset is reported as
"no prior run available for comparison" instead of guessed at.
"""

import argparse
import json
from pathlib import Path
from collections import Counter


def load_json(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [!] File not found, skipping: {path}")
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(old_records, new_records, id_field, label, paper_pct=None):
    print("\n" + "=" * 60)
    print(f"COMPARISON: {label}")
    print("=" * 60)

    if old_records is None or new_records is None:
        print("  Skipped — missing old or new file for this dataset.")
        return

    old_lookup = {r[id_field]: r for r in old_records}
    new_lookup = {r[id_field]: r for r in new_records}

    common_ids = set(old_lookup) & set(new_lookup)
    only_old = set(old_lookup) - set(new_lookup)
    only_new = set(new_lookup) - set(old_lookup)

    print(f"  Old run: {len(old_lookup)} topics | New run: {len(new_lookup)} topics")
    print(f"  In common: {len(common_ids)} | Only in old: {len(only_old)} | Only in new: {len(only_new)}")

    old_guided = sum(1 for r in old_lookup.values() if r.get("guided_label") == "guided")
    new_guided = sum(1 for r in new_lookup.values() if r.get("guided_label") == "guided")
    old_pct = old_guided / len(old_lookup) * 100 if old_lookup else 0.0
    new_pct = new_guided / len(new_lookup) * 100 if new_lookup else 0.0

    print(f"\n  Guided ratio — old (calibrated): {old_guided}/{len(old_lookup)} ({old_pct:.1f}%)")
    print(f"  Guided ratio — new (rule-based): {new_guided}/{len(new_lookup)} ({new_pct:.1f}%)")
    if paper_pct is not None:
        print(f"  Paper's published ratio (for reference): {paper_pct}%")

    flipped_to_guided = []
    flipped_to_nonguided = []
    unchanged = 0

    for _id in common_ids:
        old_lbl = old_lookup[_id].get("guided_label")
        new_lbl = new_lookup[_id].get("guided_label")
        if old_lbl == new_lbl:
            unchanged += 1
            continue
        entry = {
            "id": _id,
            "old_label": old_lbl,
            "new_label": new_lbl,
        }
        # If the new record carries per-dimension flags, explain the flip.
        for dim in ("structure", "narrative", "dynamics"):
            if dim in new_lookup[_id]:
                entry[f"{dim}_flag"] = new_lookup[_id][dim].get("flag")
                entry[f"{dim}_evaluable"] = new_lookup[_id][dim].get("evaluable")
        if new_lbl == "guided":
            flipped_to_guided.append(entry)
        else:
            flipped_to_nonguided.append(entry)

    total_flipped = len(flipped_to_guided) + len(flipped_to_nonguided)
    flip_rate = total_flipped / len(common_ids) * 100 if common_ids else 0.0

    print(f"\n  Flipped label: {total_flipped} / {len(common_ids)} ({flip_rate:.1f}%)")
    print(f"    nonguided -> guided: {len(flipped_to_guided)}")
    print(f"    guided -> nonguided: {len(flipped_to_nonguided)}")
    print(f"  Unchanged: {unchanged}")

    if flipped_to_guided or flipped_to_nonguided:
        print("\n  Sample of flipped topics (up to 5 each direction), with the new")
        print("  rubric's dimension flags as the explanation for the flip:")
        for entry in flipped_to_guided[:5]:
            print(f"    {entry}")
        for entry in flipped_to_nonguided[:5]:
            print(f"    {entry}")

    return {
        "old_guided_pct": old_pct,
        "new_guided_pct": new_pct,
        "flipped_to_guided": flipped_to_guided,
        "flipped_to_nonguided": flipped_to_nonguided,
        "unchanged": unchanged,
        "flip_rate_pct": flip_rate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-weibo")
    parser.add_argument("--new-weibo", default="data/processed/weibo_ced/guided_labels_computed.json")
    parser.add_argument("--old-politifact")
    parser.add_argument("--new-politifact", default="data/processed/fakenewsnet/politifact_guided_labels_computed.json")
    parser.add_argument("--old-gossipcop")
    parser.add_argument("--new-gossipcop", default="data/processed/fakenewsnet/gossipcop_guided_labels_computed.json")
    parser.add_argument("--output", default="label_flip_report.json")
    args = parser.parse_args()

    report = {}

    report["weibo"] = compare(
        load_json(args.old_weibo), load_json(args.new_weibo),
        id_field="post_id", label="Weibo (CED)", paper_pct=54.6,
    )
    report["politifact"] = compare(
        load_json(args.old_politifact), load_json(args.new_politifact),
        id_field="graph_id", label="FakeNewsNet — Politifact",
    )
    report["gossipcop"] = compare(
        load_json(args.old_gossipcop), load_json(args.new_gossipcop),
        id_field="graph_id", label="FakeNewsNet — Gossipcop",
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull flip report saved to: {args.output}")


if __name__ == "__main__":
    main()
