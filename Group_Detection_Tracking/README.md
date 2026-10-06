This module implements the **TA-Louvain-based hidden group detection and temporal group tracking** component of the paper:

> **A Guided Topic Detection Model Based on Topic Evolution and Group Stance**

The implementation applies Topology + Attribute Louvain (TA-Louvain) to identify hidden user groups associated with constructed topics and, for Weibo CED, tracks those groups across temporal slices using Jaccard similarity.

---

## 1. Module Purpose

The `Group_Detection_Tracking` module is responsible for:

1. Constructing topic-specific interaction graphs from the preprocessed datasets.
2. Detecting hidden groups using TA-Louvain.
3. Incorporating behavioural/profile attributes when available.
4. Tracking groups across temporal slices for Weibo CED.
5. Producing JSON outputs for downstream components.

---

## 2. Supported Datasets

### Weibo CED

- Temporal topic processing
- 10 equal-width time slices per topic
- Cumulative interaction graphs by default
- Optional per-slice processing
- Behavioural user attributes
- TA-Louvain group detection
- Jaccard-based group tracking

### FakeNewsNet / UPFD

Supported datasets:

- Politifact
- Gossipcop

UPFD/FakeNewsNet does not provide the temporal information required for the Weibo temporal pipeline. Therefore, TA-Louvain is applied once per topic using the complete propagation graph.

The output uses a single static slice so downstream components can treat both datasets consistently.

---

## 3. Directory Structure

```text
Group_Detection_Tracking/
│
├── README.md
│
├── data_output/
│   ├── politifact_new_profile_feature.npz
│   ├── gossipcop_new_profile_feature.npz
│   ├── topics_groups.json
│   ├── politifact_topics_groups.json
│   └── gossipcop_topics_groups.json
│
└── scripts/
    ├── ta_louvain.py
    ├── run_weibo_ta_louvain.py
    └── run_fakenewsnet_ta_louvain.py
```

### `scripts/ta_louvain.py`

Core TA-Louvain implementation.

Responsibilities:

- Attribute similarity
- Topological similarity
- Second-order similarity
- Attribute-enhanced adjacency
- Similarity-weighted graph construction
- Louvain optimisation
- Modularity calculation
- Jaccard-based group tracking
- Attribute standardisation

### `scripts/run_weibo_ta_louvain.py`

Dataset-specific execution script for Weibo CED.

Responsibilities:

- Load temporal topic data
- Construct topic/slice graphs
- Generate behavioural attributes
- Run TA-Louvain over slices
- Track groups across slices
- Save results

### `scripts/run_fakenewsnet_ta_louvain.py`

Dataset-specific execution script for FakeNewsNet / UPFD.

Responsibilities:

- Load topic information
- Load propagation edges
- Load optional UPFD profile features
- Remove the news root node by default
- Run TA-Louvain once per topic
- Save static group results

---

## 4. TA-Louvain

The core implementation combines:

- Network topology
- Attribute similarity
- Higher-order structural similarity
- Louvain community detection

### Equation 11 — Shared-neighbour similarity

\[
S_N(i,j)=\frac{N_{ij}}{N_i+N_j}
\]

### Equation 12 — Second-order similarity

\[
S_{NN}(i,k)=S_N(i,j)S_N(j,k)
\]

When multiple bridge nodes exist, the implementation uses the maximum by default.

### Equation 13 — Composite similarity

\[
S(i,j)=\lambda_1S_N+(1-\lambda_1)S_{NN}+\lambda_2S_{att}
\]

### Equation 14 — Attribute-enhanced adjacency

\[
W_{ij}=A_{ij}+\gamma Sim_{att}(i,j)
\]

### Equation 15 — Enhanced modularity

The implementation folds the similarity term into the graph weights:

\[
w'_{ij}=W_{ij}S_{ij}
\]

and performs weighted Louvain optimisation on the resulting graph.

### Equation 16 — Group tracking

Groups between consecutive temporal slices are linked when:

\[
Jaccard(g_i^{t-1},g_j^t)>0.5
\]

The threshold is therefore strictly greater than `0.5`.

---

## 5. Default TA-Louvain Configuration

```text
lambda1   = 0.7
lambda2   = 0.3
gamma     = 0.4
attr_knn  = 10
threshold = 0.5
seed      = 42
```

| Parameter | Default | Meaning |
|---|---:|---|
| `lambda1` | `0.7` | Weight of first-order topology |
| `lambda2` | `0.3` | Attribute contribution |
| `gamma` | `0.4` | Attribute contribution to adjacency |
| `attr_knn` | `10` | Number of attribute neighbours |
| Jaccard threshold | `0.5` | Group continuity threshold |
| seed | `42` | Reproducibility |

---

# 6. Weibo CED

## Input

```text
Preprocessing/data/processed/weibo_ced/topics_time_sliced.json
```

This file contains the topic-level temporal representation generated during preprocessing.

Each topic contains temporal slices with:

```text
slice_index
new_interactions_this_slice
```

## Graph Construction

For every topic and temporal slice:

### Nodes

Users participating in the interactions.

By default, the implementation uses cumulative interactions:

```text
slice 1
slice 1 + slice 2
slice 1 + slice 2 + slice 3
...
```

The alternative is available through:

```text
--per-slice
```

which uses only the interactions introduced in the current slice.

### Edges

Edges are constructed using:

```text
interaction user -> parent interaction user
```

through the `parent` -> `mid` relationship.

If an interaction has an empty `parent`, it does not generate an edge.

The source post author is not present in the commenter interaction file. Creating a root hub would connect unrelated users into one group, so those root interactions do not generate edges. The user is still retained as a node.

---

# 7. Weibo Behavioural Attributes

Because commenter profile information is unavailable in the Weibo CED interaction data, behavioural features are used.

The implementation calculates:

```text
n_interactions
mean_log_len
empty_frac
emoji_per_msg
mention_per_msg
url_frac
reply_to_user_frac
log_first_delay_min
log_mean_delay_min
uid_numeric
uid_len
```

These attributes are standardised before calculating cosine similarity.

The number of interactions is log-transformed because it is heavy-tailed.

---

# 8. Weibo Execution

Run commands from the **project root**:

```text
Guided_Topic_Detection/
```

### Quick Test

```powershell
python Group_Detection_Tracking/scripts/run_weibo_ta_louvain.py --limit 50
```

This processes the first 50 topics and is recommended before the full run.

### Full Run

For the current hardware configuration:

```text
Intel i5-12450H
16 GB DDR4 RAM
```

use:

```powershell
python Group_Detection_Tracking/scripts/run_weibo_ta_louvain.py --workers 6
```

`--workers` controls CPU processes. The implementation uses CPU multiprocessing; GPU VRAM does not determine the worker count.

---

# 9. Weibo Output

Default output:

```text
Group_Detection_Tracking/data_output/topics_groups.json
```

The output contains, for every topic:

```text
topic_id
guided_label
original_label
slices
```

Each slice contains:

```text
slice_index
n_users
n_groups
modularity
groups
```

The `groups` field has the form:

```text
persistent_group_id -> list of user IDs
```

---

# 10. FakeNewsNet / UPFD

Supported datasets:

```text
politifact
gossipcop
```

The implementation does not perform temporal group tracking for FakeNewsNet because UPFD does not provide the required temporal interaction information.

TA-Louvain therefore runs once per topic on the complete propagation graph.

The output is represented using a single static slice.

---

# 11. FakeNewsNet Inputs

### Politifact

```text
Preprocessing/data/processed/fakenewsnet/politifact_topics_with_categories.json
Preprocessing/data/processed/fakenewsnet/politifact_edges_cleaned.txt
```

### Gossipcop

```text
Preprocessing/data/processed/fakenewsnet/gossipcop_topics_with_categories.json
Preprocessing/data/processed/fakenewsnet/gossipcop_edges_cleaned.txt
```

---

# 12. FakeNewsNet Profile Features

Optional UPFD profile features are stored in:

```text
Group_Detection_Tracking/data_output/
```

Expected files:

```text
politifact_new_profile_feature.npz
gossipcop_new_profile_feature.npz
```

If the corresponding `.npz` file exists, the execution script automatically loads it.

If it does not exist, TA-Louvain can run without profile attributes.

The feature matrix must use:

```text
row i = global UPFD node i
```

---

# 13. FakeNewsNet Root Handling

For each topic, the lowest node index is treated as the news/root node.

The root is removed by default because it can act as a hub connecting otherwise unrelated users into one large group.

Default:

```text
root removed
```

To retain the root:

```powershell
--keep-root
```

---

# 14. FakeNewsNet Execution

## Politifact

```powershell
python Group_Detection_Tracking/scripts/run_fakenewsnet_ta_louvain.py --dataset politifact
```

Output:

```text
Group_Detection_Tracking/data_output/politifact_topics_groups.json
```

## Gossipcop

```powershell
python Group_Detection_Tracking/scripts/run_fakenewsnet_ta_louvain.py --dataset gossipcop
```

Output:

```text
Group_Detection_Tracking/data_output/gossipcop_topics_groups.json
```

---

# 15. FakeNewsNet Quick Test

```powershell
python Group_Detection_Tracking/scripts/run_fakenewsnet_ta_louvain.py --dataset politifact --limit 50
```

or:

```powershell
python Group_Detection_Tracking/scripts/run_fakenewsnet_ta_louvain.py --dataset gossipcop --limit 50
```

---

# 16. Parallel Processing

Both execution scripts support:

```text
--workers
```

Example:

```powershell
python Group_Detection_Tracking/scripts/run_fakenewsnet_ta_louvain.py --dataset politifact --workers 6
```

and:

```powershell
python Group_Detection_Tracking/scripts/run_weibo_ta_louvain.py --workers 6
```

The implementation uses `ProcessPoolExecutor`.

For the current system:

```text
Intel i5-12450H
16 GB RAM
```

`6` workers is the recommended starting point.

---

# 17. Important Execution Rule

Run commands from the **project root**:

```text
Guided_Topic_Detection/
```

Correct:

```powershell
cd "D:\#Project\Innovative 7th\Guided_Topic_Detection"
```

Then:

```powershell
python Group_Detection_Tracking/scripts/run_weibo_ta_louvain.py --limit 50
```

Do not run the commands from:

```text
Group_Detection_Tracking/scripts/
```

when using the relative dataset paths.

---

# 18. Custom Input/Output Paths

### Weibo

```powershell
python Group_Detection_Tracking/scripts/run_weibo_ta_louvain.py `
    --input "Preprocessing/data/processed/weibo_ced/topics_time_sliced.json" `
    --output "Group_Detection_Tracking/data_output/topics_groups.json"
```

### FakeNewsNet

```powershell
python Group_Detection_Tracking/scripts/run_fakenewsnet_ta_louvain.py `
    --dataset politifact `
    --topics "Preprocessing/data/processed/fakenewsnet/politifact_topics_with_categories.json" `
    --edges "Preprocessing/data/processed/fakenewsnet/politifact_edges_cleaned.txt" `
    --output "Group_Detection_Tracking/data_output/politifact_topics_groups.json"
```

---

# 19. Output Summary

Main generated group-detection outputs:

```text
Group_Detection_Tracking/data_output/

├── topics_groups.json
├── politifact_topics_groups.json
└── gossipcop_topics_groups.json
```

The `.npz` files are input feature files:

```text
├── politifact_new_profile_feature.npz
└── gossipcop_new_profile_feature.npz
```

They are not generated by TA-Louvain.

---

# 20. Implementation Notes

### Attribute KNN

The implementation does not construct attribute edges between every possible pair by default.

Instead:

```text
attr_knn = 10
```

is used to limit attribute neighbours.

This avoids the `O(N²)` memory requirement of a fully dense attribute graph.

For small graphs, the dense formulation can be enabled by setting:

```text
attr_knn = None
```

### Similarity Candidate Pairs

Similarity is calculated over:

```text
structural edges ∪ attribute-KNN pairs
```

rather than every possible node pair.

### Second-Order Similarity

When multiple bridge nodes exist, the default implementation uses the maximum bridge similarity.

---

# 21. Reproducibility

The TA-Louvain configuration uses:

```text
seed = 42
```

---

# 22. Overall Project Pipeline

```text
Raw Dataset
    │
    ▼
Data Acquisition
    │
    ▼
Cleaning
    │
    ▼
Topic Construction
    │
    ▼
Temporal Processing
    │
    ▼
Labels
    │
    ▼
Train / Validation / Test
    │
    ▼
TA-Louvain
    │
    ├── Topology
    ├── Attributes
    ├── Similarity
    └── Louvain
    │
    ▼
Group Detection
    │
    ▼
Group Tracking
    │
    ▼
Downstream Topic Evolution / Group Stance Components
```

---

# 23. Reference

**Wang et al., "A Guided Topic Detection Model Based on Topic Evolution and Group Stance", IEEE Transactions on Computational Social Systems, Vol. 13, No. 3, 2026.**

The TA-Louvain implementation corresponds to the topology/attribute group detection and Jaccard-based group tracking component of the reference paper.

---

## Status

- [x] TA-Louvain core implementation
- [x] Attribute similarity
- [x] Topological similarity
- [x] Second-order similarity
- [x] Attribute-enhanced adjacency
- [x] Louvain optimisation
- [x] Jaccard group tracking
- [x] Weibo CED temporal processing
- [x] FakeNewsNet / UPFD static processing
- [x] Politifact support
- [x] Gossipcop support
- [x] Optional UPFD profile features
- [x] Parallel topic processing
'''