# Step 6: Time Slicing — Weibo (CED)

**Date:** 18-09-2026
**Author:** Adrija Pal

## Approach

Per the paper's formalization (Section III-B: `G = (G1, G2, ..., Gn)`),
each topic's timeline was split into **10 cumulative time slices**.
Each slice `Gt` contains all interactions from the topic's original
post time up to that slice's time boundary (cumulative growth, not
disjoint time buckets) — consistent with how the paper describes
tracking group structure evolution across time steps.

## Design decision: fixed slice COUNT, not fixed time interval

The paper tunes a "time step" hyperparameter from 1 to 15 (Section
V-A-5, Fig. 9), applied uniformly across the whole dataset for GRU
input. This only makes sense if every topic produces the **same
number of slices** regardless of its actual lifespan — a fixed time
interval (e.g., "1 slice per hour") or fixed interaction count per
slice would instead produce a *variable* number of slices per topic
(since topics range from 3 to 964 interactions in our data), which
would be incompatible with a single global "time step" setting.
Default set to 10 slices per topic (mid-point of the paper's tested
1–15 range); parameterized in the script for later tuning.

## Results

| Metric | Value |
|---|---|
| Topics with valid time slices | 3,300 / 3,300 (100%) |
| Skipped (no valid interaction dates) | 0 |
| Slices per topic | 10 (fixed) |
| Avg interactions captured by final slice | 378.4 (matches Step 5 total avg — confirms correct cumulative logic) |

## Output structure

Each topic record contains 10 ordered slices, each with cumulative
interaction count, cumulative unique user count, and the full
cumulative interaction list up to that time boundary:
```json
{
  "topic_id": "...",
  "guided_label": "guided" | "nonguided",
  "num_slices": 10,
  "time_slices": [
    {
      "slice_index": 1,
      "boundary_time": "...",
      "cumulative_interaction_count": int,
      "cumulative_unique_users": int,
      "interactions": [...]
    },
    ...
  ]
}
```

## Applicability note
This step applies to **Weibo only**. FakeNewsNet (UPFD) has no
timestamp data (confirmed in Step 5 documentation) and therefore
cannot be time-sliced — this is a structural limitation of the
substitute dataset, not an omission.

## Output file
`data/processed/weibo_ced/topics_time_sliced.json`

## Script used
See `scripts/time_slicing_weibo.py`.