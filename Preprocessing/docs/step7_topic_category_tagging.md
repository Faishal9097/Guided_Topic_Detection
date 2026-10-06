# Step 7: Topic Category Tagging

**Date:** 18-09-2026
**Author:** Adrija Pal
**Status:** Completed (initially deferred, revisited and finalized)

## Background

This step was initially deferred (see Step 4-6 documentation timeline)
since topic categorization is not a dependency for model training
(TA-Louvain/AG2vec/GC-GRU) — it is only used by the paper for a
post-hoc evaluation analysis (Section V-B, Table X, Figure 12), run
*after* the model is trained. It was revisited and completed once the
core pipeline (Steps 1-6, 8) was finished.

## Approach

Per the paper (Section V-B): topics grouped into politics /
entertainment / health / social_events.

### FakeNewsNet — direct source-based mapping
The paper states: *"topics were mapped to political or entertainment
categories based on their sources (PolitiFact and Gossipcop)."* This
is a deterministic mapping, not a classification task:
- Politifact → politics (314 topics, 100%)
- Gossipcop → entertainment (5,464 topics, 100%)

No ambiguity, no limitations — matches the paper exactly.

### Weibo — LDA topic modeling
The paper used *"combined semantic tags and LDA modeling."* The
"semantic tags" component (likely internal platform metadata) is not
available to us, so LDA modeling alone was used, in two stages:

1. **Stage 1 (LDA training)**: Chinese text segmented via `jieba`,
   vectorized (CountVectorizer), LDA trained to discover topic
   clusters.
2. **Stage 2 (manual cluster labeling)**: Each discovered cluster's
   top words manually inspected and mapped to one of the 4 target
   categories, or "other" if no clean fit.

## Iteration history

Three configurations were tried, in order of increasing effort:

| Method | Config | "Other" % | Health category found? |
|---|---|---|---|
| Keyword-based (initial approach) | Manual keyword lists | 69.1% | Yes, but weak (6.0%) |
| LDA, 12 clusters | vocab=2000, min_df=5 | 58.3% | No — missing entirely |
| **LDA, 24 clusters (final)** | vocab=3000, min_df=3, max_iter=30 | **41.8%** | **Yes (11.4%)** |

Increasing LDA cluster count from 12 to 24 (plus tightening `min_df`
from 5→3 and expanding vocabulary from 2,000→3,000 features) allowed
finer-grained topics — including a genuine health cluster (containing
the explicit keyword "健康"/health) — to be isolated instead of being
absorbed into larger, vaguer clusters. Further increases beyond 24
clusters were considered but not pursued: diminishing returns were
expected (each doubling of cluster count yielded a shrinking "other"
percentage but increasing risk of unreliable manual interpretation on
smaller, noisier clusters), and this step is non-blocking for model
training — see "Sufficiency assessment" below.

## Cluster → category mapping (final, 24-cluster version)

24 discovered clusters were manually inspected and mapped based on
their top-word content. Full word lists in script run history; mapping
logic in `scripts/lda_apply_mapping_weibo.py`. Notable/borderline
calls:
- **Cluster 19** → health (high confidence — explicit "健康" keyword
  present)
- **Cluster 2** → health (medium confidence — toothpaste/chemical
  product-safety scare)
- **Cluster 23** → health (low confidence/borderline — pesticide
  scandal involving government and companies; could arguably be
  politics instead)
- **Cluster 12** → entertainment (borderline — movie/box-office
  content, but also contains a political term "钓鱼岛"/Diaoyu Islands;
  mapped by dominant theme)
- **Cluster 6** → social_events (borderline — contains a health-
  adjacent word "生病"/sick, but "chengguan" conflict theme dominates)
- **Cluster 11** → other (low confidence — too mixed to assign
  cleanly: nation/school/actor/villager)

## Final results

**Weibo** (LDA-based, n=3,300):
| Category | Count | % |
|---|---|---|
| Other | 1,379 | 41.8% |
| Social events | 1,066 | 32.3% |
| Health | 375 | 11.4% |
| Entertainment | 360 | 10.9% |
| Politics | 120 | 3.6% |

**FakeNewsNet**: 100% direct mapping (no ambiguity) —
Politifact=politics (314), Gossipcop=entertainment (5,464).

## Sufficiency assessment

This result is considered **sufficient** for the step's intended
downstream purpose (per-category evaluation analysis, if performed
later, matching the paper's Figure 12):
- All 4 target categories now have usable sample sizes (smallest,
  Politics, has 120 topics — comparable in scale to the paper's own
  Politifact subset of 157 samples per class).
- "Other" (41.8%) is not treated as a failure state: personal/generic
  social media content (goodnight messages, family updates, casual
  reflection) genuinely does not fit any topical category, and several
  clusters were clearly of this nature.
- Further cluster-count increases were evaluated as offering
  diminishing returns relative to effort, for a step that is
  non-blocking to model training.

## Known limitations
- No access to the paper's "semantic tags" component — LDA alone used
  for Weibo.
- 41.8% "other" remains the largest single category for Weibo, though
  significantly reduced from the initial 69.1%.
- Several cluster-to-category mappings involved subjective judgment on
  mixed-signal clusters (see mapping table above) — a source of
  potential misclassification at the margins.
- Category distribution is skewed toward social_events (32.3%) — may
  reflect a genuine dataset characteristic (social-event rumors common
  on Weibo) or residual clustering imprecision; cannot be fully
  disambiguated without the paper's original semantic tags.
- FakeNewsNet's category assignment is fully deterministic and matches
  the paper exactly — no limitations for that dataset.

## Output files

data/processed/weibo_ced/topics_with_categories_lda.json
data/processed/fakenewsnet/politifact_topics_with_categories.json
data/processed/fakenewsnet/gossipcop_topics_with_categories.json


## Scripts used
- `scripts/topic_category_tagging.py` — initial keyword-based approach
  (superseded, kept for reference/comparison)
- `scripts/lda_topic_modeling_weibo.py` — Stage 1: LDA training
  (final config: 24 clusters, vocab=3000, min_df=3)
- `scripts/lda_apply_mapping_weibo.py` — Stage 2: cluster-to-category
  mapping application