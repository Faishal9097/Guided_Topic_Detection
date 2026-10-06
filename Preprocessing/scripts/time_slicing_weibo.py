"""
Step 6: Time Slicing — Weibo (CED)
Splits each topic's timeline into N cumulative time slices
G = (G1, G2, ..., Gn), per the paper's formalization (Section III-B).
Each slice Gt contains all interactions from topic start up to that
time boundary (cumulative growth, not disjoint buckets).
"""

import json
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "weibo_ced"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "weibo_ced"

NUM_SLICES = 10  # matches paper's tunable "time step" range (1-15), default mid-point


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def build_time_slices(topic: dict, num_slices: int = NUM_SLICES) -> list:
    post_time = topic["original_post"].get("time")
    if post_time is None:
        return []

    post_datetime = datetime.fromtimestamp(post_time)

    interactions_with_dates = []
    for interaction in topic["interactions"]:
        dt = parse_date(interaction.get("date"))
        if dt is not None:
            interactions_with_dates.append((dt, interaction))

    if not interactions_with_dates:
        return []

    interactions_with_dates.sort(key=lambda x: x[0])
    last_interaction_time = interactions_with_dates[-1][0]

    total_duration = (last_interaction_time - post_datetime).total_seconds()
    if total_duration <= 0:
        return []

    slice_duration = total_duration / num_slices

    slices = []
    cumulative_count = 0
    cumulative_users = set()
    already_included_idx = 0  # pointer into sorted interactions_with_dates

    for i in range(1, num_slices + 1):
        slice_boundary = post_datetime.timestamp() + (slice_duration * i)
        slice_boundary_dt = datetime.fromtimestamp(slice_boundary)

        # Only collect NEW interactions since the last slice boundary (delta)
        delta_interactions = []
        while (already_included_idx < len(interactions_with_dates)
               and interactions_with_dates[already_included_idx][0] <= slice_boundary_dt):
            delta_interactions.append(interactions_with_dates[already_included_idx][1])
            already_included_idx += 1

        cumulative_count += len(delta_interactions)
        for ia in delta_interactions:
            if ia.get("uid"):
                cumulative_users.add(ia.get("uid"))

        slices.append({
            "slice_index": i,
            "boundary_time": slice_boundary_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "cumulative_interaction_count": cumulative_count,
            "cumulative_unique_users": len(cumulative_users),
            "new_interactions_this_slice": delta_interactions,  # DELTA, not cumulative
        })

    return slices

def main():
    print("Loading constructed topics...")
    with open(PROCESSED_DIR / "topics_constructed.json", "r", encoding="utf-8") as f:
        topics = json.load(f)

    print(f"Building {NUM_SLICES} time slices per topic for {len(topics)} topics...")

    topics_with_slices = []
    skipped_no_valid_dates = 0

    for i, topic in enumerate(topics):
        slices = build_time_slices(topic, NUM_SLICES)
        if not slices:
            skipped_no_valid_dates += 1
            continue

        topic_record = {
            "topic_id": topic["topic_id"],
            "guided_label": topic["guided_label"],
            "original_label": topic["original_label"],
            "num_slices": len(slices),
            "time_slices": slices,
        }
        topics_with_slices.append(topic_record)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(topics)}...")

    print(f"\nTopics with valid time slices: {len(topics_with_slices)}")
    print(f"Skipped (no valid interaction dates): {skipped_no_valid_dates}")

    if topics_with_slices:
        final_slice_counts = [t["time_slices"][-1]["cumulative_interaction_count"] for t in topics_with_slices]
        avg_final = sum(final_slice_counts) / len(final_slice_counts)
        print(f"Avg interactions captured by final slice: {avg_final:.1f}")

    output_path = OUTPUT_DIR / "topics_time_sliced.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(topics_with_slices, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()