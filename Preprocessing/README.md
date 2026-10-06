# Data Preparation — Final Summary

**Project**: Guided Topic Detection (reproducing/extending "A Guided
Topic Detection Model Based on Topic Evolution and Group Stance",
Wang et al., IEEE TCSS 2026)

**Scope of this work**: Dataset acquisition, cleaning, label
construction, topic construction, temporal pipeline, topic
categorization, and train/val/test split — data preparation only.
Model implementation (TA-Louvain, AG2vec, GC-GRU) is out of scope for
this document.

**Author**: Adrija Pal
**Date range**: 17-09-2026 to 18-09-2026

---

## 1. Overview

Two datasets were prepared, following the paper's methodology as
closely as practically possible:

| Dataset | Role | Status |
|---|---|---|
| **Weibo (CED)** | Primary dataset | ✅ Exact match to paper's scale, fully processed |
| **FakeNewsNet (Politifact + Gossipcop)** | Secondary dataset | ⚠️ Substitute used, documented limitations |

---

## 2. Key decisions and why

### 2.1 Weibo sourced from `thunlp/Chinese_Rumor_Dataset`
Confirmed exact match to paper's Table III: 1,538 rumor + 1,849
non-rumor posts, 1,275,180 total comments — very likely the identical
dataset version used by the paper's authors.

### 2.2 FakeNewsNet: live hydration replaced with UPFD benchmark
The paper's authors performed a live Twitter API crawl to reach
1,056 Politifact / 22,140 Gossipcop articles — no longer feasible,
since X/Twitter API is now paid-only. Two smaller alternatives were
investigated and rejected (see Step 2 docs); final choice was **UPFD**
(`safe-graph/GNN-FakeNews`): 314 Politifact / 5,464 Gossipcop graphs,
real propagation structure, but artificially class-balanced (50/50)
rather than the paper's natural imbalance.

### 2.3 Guided/nonguided labels were computed, not simply copied
The paper's exact per-sample labeling process is not fully published.
Labels were computed from measurable signals approximating the
paper's Table I framework, then calibrated to match the paper's known
aggregate proportions. Full methodology in Step 4 docs.

### 2.4 Fixed-count time slicing (10 slices/topic)
The paper tunes a single "time step" hyperparameter (1–15) uniformly
across the dataset for GRU input — only consistent with a fixed
**number** of slices per topic, not a fixed time interval.

### 2.5 Topic categorization: LDA with 24 clusters (Weibo), direct mapping (FakeNewsNet)
Initially deferred as a non-blocking step, later completed. Iterated
from a keyword approach (69.1% "other") through 12-cluster LDA (58.3%
"other", no health category) to a final 24-cluster LDA configuration
(41.8% "other", all 4 categories represented). Further cluster
increases were evaluated and intentionally not pursued — smaller
clusters at higher counts become statistically unstable and harder to
label reliably without ground truth to verify against, so additional
"other" reduction would likely trade away trustworthiness rather than
genuinely improve it. See Step 7 docs for full reasoning.

---

## 3. Known limitations (full detail in per-step docs)

| # | Limitation | Affected dataset | Severity |
|---|---|---|---|
| 1 | No `age` field available (platform never exposed it) | Weibo | Low |
| 2 | Only original posters have full profile data (0.3% of all users) | Weibo | Medium |
| 3 | 87 corrupted posts excluded — all from rumor class only (non-random) | Weibo | Medium |
| 4 | Dataset scale mismatch: 314/5,464 vs. paper's 1,056/22,140 | FakeNewsNet | High |
| 5 | Pre-encoded features only, no raw follower/following counts | FakeNewsNet | Medium |
| 6 | No timestamp data — no propagation-dynamics signal, no time-slicing | FakeNewsNet | High |
| 7 | Chinese-language text — NLP tooling limited to jieba segmentation + LDA | Weibo | Low-Medium |
| 8 | Topic categorization: 41.8% "other" for Weibo; no access to paper's "semantic tags" | Weibo | Medium |

**Notable cross-dataset finding**: both Weibo and Gossipcop
independently showed real/non-rumor content skewing more "guided"
than fake/rumor content, using completely different computed signals
per dataset. See Step 4 docs.

---

## 4. Pipeline steps completed

| Step | Weibo | FakeNewsNet | Doc |
|---|---|---|---|
| 1. Environment Setup | ✅ | ✅ | `step1_environment_setup.md` |
| 2. Dataset Acquisition | ✅ | ✅ | `step2_dataset_acquisition*.md` |
| 3. Data Cleaning | ✅ | ✅ | `step3_data_cleaning*.md` |
| 4. Label Construction | ✅ | ✅ | `step4_label_construction*.md` |
| 5. Topic Construction | ✅ | ✅ | `step5_topic_construction*.md` |
| 6. Time Slicing | ✅ | N/A (no timestamps) | `step6_time_slicing_weibo.md` |
| 7. Topic Category Tagging | ✅ (LDA, 24 clusters) | ✅ (direct mapping) | `step7_topic_category_tagging.md` |
| 8. Train/Val/Test Split | ✅ | ✅ | `step8_train_val_test_split.md` |

**All 8 pipeline steps complete for both datasets.**

---

## 5. Final deliverable files

data/processed/
├── weibo_ced/
│ ├── original_microblog_cleaned.json
│ ├── rumor_repost_cleaned.json
│ ├── non_rumor_repost_cleaned.json
│ ├── excluded_original_ids.txt
│ ├── guided_labels_computed.json
│ ├── topics_constructed.json
│ ├── topics_time_sliced.json
│ ├── topics_with_categories_lda.json
│ ├── topics_split_train.json
│ ├── topics_split_val.json
│ └── topics_split_test.json
└── fakenewsnet/
├── politifact_edges_cleaned.txt
├── gossipcop_edges_cleaned.txt
├── politifact_guided_labels_computed.json
├── gossipcop_guided_labels_computed.json
├── politifact_topics_with_categories.json
├── gossipcop_topics_with_categories.json
├── politifact_split_{train,val,test}.json
└── gossipcop_split_{train,val,test}.json


All processing scripts are in `scripts/`, one per pipeline step,
matching the step numbering in `docs/`.

---

## 6. Suggested next steps (outside this scope)

1. Whoever builds TA-Louvain/AG2vec/GC-GRU can consume the
   `*_split_{train,val,test}.json` files directly.
2. Chinese NLP tooling (e.g., a Chinese BERT model) would be needed
   if deeper textual stance analysis (paper's Definition 3) is
   required beyond what's already computed.
3. If exact-scale FakeNewsNet reproduction becomes critical, revisit
   with paid Twitter/X API access as the only remaining path.
4. If topic categorization needs further refinement, the paper's
   "semantic tags" data source (not available to us) would likely be
   the highest-value addition — more valuable than further LDA
   cluster-count tuning, which was found to have diminishing/negative
   returns past 24 clusters (see Step 7 docs).