# Step 5: Topic Construction — FakeNewsNet (UPFD)

**Date:** 18-09-2026
**Author:** Adrija Pal

## Approach

UPFD's graph structure is inherently topic-organized already — each
`graph_id` corresponds to one news article/topic, with nodes as users
and edges as propagation links. This step is a **light consolidation**,
not a full construction: it combines the edge list, node counts, and
Step 4's computed guided labels into unified per-topic records matching
Weibo's structure, for consistency across the pipeline.

## ⚠️ Confirmed limitation: no timestamp data

UPFD provides no `date`/`time` fields anywhere in its files (edge list,
node features, or elsewhere). This was first noted in Step 4 (where it
ruled out the propagation-dynamics guided-signal dimension) and is
**re-confirmed here**: FakeNewsNet topic records cannot support
time-slicing (Step 6 of this pipeline). Step 6 will therefore apply to
**Weibo only** — this is a structural limitation of the substitute
dataset (see Step 2 documentation for why UPFD was used instead of the
paper's original live-crawled data), not a gap introduced during
processing.

## Results

| Metric | Politifact | Gossipcop |
|---|---|---|
| Topics consolidated | 314 | 5,464 |
| Guided | 158 | 2,733 |
| Nonguided | 156 | 2,731 |
| Avg nodes per topic | 130.7 | 57.5 |
| Avg edges per topic | 129.7 | 56.5 |

## Output structure

Each topic record contains:
```json
{
  "topic_id": "politifact_0",
  "original_label": "real" | "fake",
  "guided_label": "guided" | "nonguided",
  "combined_guided_score": float,
  "num_nodes": int,
  "num_edges": int,
  "node_indices": [...],
  "has_timestamp_data": false
}
```

## Output files
data/processed/fakenewsnet/
├── politifact_topics_constructed.json
└── gossipcop_topics_constructed.json


## Script used
See `scripts/topic_construction_fakenewsnet.py`.