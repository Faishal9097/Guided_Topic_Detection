# Step 2: Dataset Acquisition — Weibo (CED)

**Date:** 17-09-2026
**Author:** Adrija Pal

## Source
- Repository: `github.com/thunlp/Chinese_Rumor_Dataset`
- Subfolder used: `CED_Dataset/`
- Associated paper: Song et al., "CED: Credible Early Detection of
  Social Media Rumors," arXiv:1811.04175 (2018) — cited as reference [38]
  in the target paper.

## Structure

data/raw/weibo_ced/
├── CED_Dataset/
│ ├── original-microblog/ (3,387 files — one per original post)
│ ├── rumor-repost/ (1,538 files — reply threads for rumor posts)
│ └── non-rumor-repost/ (1,849 files — reply threads for non-rumor posts)
├── README.md
└── rumors_v170613.json (separate 31,669-rumor metadata file, not
used — lacks propagation data)


## Field schema

**original-microblog/{post_id}.json:**
| Field | Type | Notes |
|---|---|---|
| text | string | Post content |
| user | object | See below |
| time | int (Unix ts) | Post time |
| likes / reposts / comments | int | Engagement counts |
| has_url, pics, source | misc | Not used by paper's formulas |

**user object:**
| Field | Maps to paper |
|---|---|
| followers | fans(ui) |
| friends | follow(ui) |
| gender | sex(ui) |
| messages | proxy for lifetime activity (used for Wi) |
| verified, verified_type | not used |
| ❌ age | **not present — known gap, see below** |

**rumor-repost/ and non-rumor-repost/ (identical schema):**
List of records per original post:
| Field | Notes |
|---|---|
| uid | User ID of reposter/commenter |
| parent | mid of parent reply (empty = direct reply to original post) |
| text | Comment text (empty = pure repost, no added comment) |
| mid | Unique ID of this reply |
| kids | IDs of nested replies (thread structure) |
| date | Timestamp, format `YYYY-MM-DD HH:MM:SS` |

## Verification against paper (Table III)

| Metric | Paper (Table III) | Found in this dataset | Match |
|---|---|---|---|
| Nonguided (non-rumor) | 1,538 | 1,538 | ✅ |
| Guided (rumor) | 1,849 | 1,849 | ✅ |
| Total | 3,387 | 3,387 | ✅ |
| Number of comments | 1,275,180 | 1,275,180 (483,617 + 791,563) | ✅ |

This confirms the dataset matches the exact version used in the paper.

## Known limitations / gaps

1. **Age field missing**: Weibo's public data never exposed user age.
   Definition 6 (`UserProp`) cannot include `age(ui)` for any user.
   This is likely a limitation the original paper faced too, since it's
   a platform-level restriction, not a dataset-specific one.

2. **Profile data coverage gap (major)**: Only users who authored an
   original post (3,387 users) have full profile data (followers,
   friends, gender). Of the 1,064,970 unique users who reposted or
   commented, only 3,387 (0.3%) have profile data available.
   - **Resolution**: `Influence(ui)` (Definition 2) will be computed
     directly where profile data exists; for the remaining 99.7% of
     users, a fallback value (dataset median Influence score) will be
     used. This will be documented wherever Influence() is computed
     in later stages.

3. **Malformed 'user' field**: 87 out of 3,387 original posts (2.6%)
   have a malformed `user` field (not a valid object). These will be
   flagged/handled in Step 3 (Cleaning) — either repaired if possible
   or excluded with logging.

4. **Language**: All text content is in Chinese. Structural/numeric
   fields (used for Definitions 1, 2, 4, 5) are unaffected. Textual
   analysis (Definition 3, and Table I's "textual narrative" dimension)
   will require Chinese NLP tooling in later pipeline stages — flagged
   for team.

## Raw exploration script
See `scripts/explore_weibo.py` for the script used to generate these
statistics.
## Additional attempt: KaiDMML/Kaggle full-scale labels (unsuccessful)

Attempted to source full-scale (1,056/22,140) labels via the
`mdepak/fakenewsnet` Kaggle mirror to supplement UPFD's smaller subset.

**Result**: this mirror actually contains the older, smaller FakeNewsNet
pilot release (Politifact + BuzzFeed only):
- Politifact: 240 articles (120 fake + 120 real) — smaller than UPFD (314)
- Gossipcop: not included at all
- `PolitifactUserFeature.mat`: (23,865 users × 154,348 features) — a
  sparse bag-of-words/TF-IDF style matrix, not raw follower/friend counts
  (same limitation as UPFD's profile features)

**Conclusion**: this source does not close the scale gap and offers no
usable raw social features beyond what UPFD already provides. Not used
further.

## Final decision

**UPFD (safe-graph/GNN-FakeNews) is used as the sole FakeNewsNet source**
for this pipeline:
- Politifact: 314 graphs (157 real / 157 fake)
- Gossipcop: 5,464 graphs (2,732 real / 2,732 fake)

This is confirmed as the best practically-obtainable substitute, given
that full-scale live data collection (matching the paper's 1,056/22,140)
requires paid Twitter/X API access, which is out of scope for this
project. This limitation and the two alternative sources investigated
(KaiDMML CSVs, Kaggle mirror) are documented above for transparency.