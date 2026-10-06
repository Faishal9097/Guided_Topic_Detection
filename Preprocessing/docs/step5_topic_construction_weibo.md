# Step 5: Topic Construction — Weibo (CED)

**Date:** 18-09-2026
**Author:** Adrija Pal

## Approach

Combined outputs from Steps 3 (cleaned data) and 4 (guided labels) into
unified per-topic records. Each topic = one original post + its full
repost/comment interaction thread + its computed guided/nonguided label.

## Important structural note

The original-microblog schema does not include a `uid` for the post
author (only profile fields like `followers`, `friends`, `gender`).
This means there is **no shared identifier system** linking original
posters to their appearances (if any) as commenters/reposters
elsewhere in the dataset. This confirms (rather than changes) the
Step 2/3 decision: full profile data is only usable for a topic's own
original poster; all reposting/commenting users require the
median-fallback treatment for Influence() computation, with no
possibility of cross-referencing profiles elsewhere in the corpus.

## Results

| Metric | Value |
|---|---|
| Total topics constructed | 3,300 |
| Skipped (no matching repost data) | 0 |
| Avg interactions per topic | 378.4 |
| Min interactions | 3 |
| Max interactions | 964 |
| Guided topics | 1,803 |
| Nonguided topics | 1,497 |

## Output structure

Each topic record in `topics_constructed.json` contains:
```json
{
  "topic_id": "...",
  "original_label": "rumor" | "non-rumor",
  "guided_label": "guided" | "nonguided",
  "combined_guided_score": float,
  "original_post": { text, time, user_profile, likes, reposts, comments },
  "interactions": [ { uid, text, date, parent, mid, has_profile_data } ],
  "num_interactions": int
}
```

## Output file
`data/processed/weibo_ced/topics_constructed.json`

## Script used
See `scripts/topic_construction_weibo.py`.