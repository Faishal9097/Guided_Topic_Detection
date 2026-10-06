"""
Step 4: Label Construction — Weibo (CED)

Implements the paper's Table I rubric directly, as three INDEPENDENT
binary (present/absent) dimension flags:

  1. Propagator structure   — abnormal account patterns, synchronized
                               bursts, centralized propagation
  2. Textual narrative      — one-sided sentiment, emotional/directive
                               language, templated/duplicate text
  3. Propagation dynamics   — multi-peak diffusion curve, explosive
                               growth, event-misalignment (not evaluable
                               here — no external event calendar exists
                               for this dataset, so this sub-component is
                               always recorded as None/"not_evaluable"
                               and does NOT contribute to the dimension
                               flag)

A dimension is flagged True if >=1 of its sub-components fires.
guided = 1 iff >= 2 of the 3 dimension flags are True.

IMPORTANT — this replaces the previous version of this script, which
computed continuous, normalized, equal-weighted scores and then forced
the guided/nonguided split to match the paper's published 1849/3387
(54.6%) ratio. That calibration was circular (it fit the *output* of
the labeling procedure to the paper's known answer rather than letting
the rubric decide). This version applies fixed, binary, data-derived
thresholds uniformly and reports whatever ratio falls out — see
`print_paper_comparison()`.

Data-derived thresholds: two sub-components (abnormal account posting
frequency, synchronized-burst account count) don't have a principled
fixed cutoff a priori, so their thresholds are derived from the
PERCENTILE of the relevant raw-metric's distribution across the whole
dataset (90th percentile, per the task spec), computed once up front
and then applied as a fixed number to every topic. This is different
from calibrating the *label ratio* to a known answer: here we are
deriving a statistical outlier threshold from the metric's own
distribution, which is what "abnormal"/"outlier" means operationally,
and it does not reference the paper's guided-count at all.

Known dataset limitation: the cleaned Weibo repost records
(uid, mid, text, date, parent — see clean_weibo.py) do not carry
follower/following counts for the reposting accounts. So sub-component
(a) "abnormal account patterns" is implemented using the *posting
frequency* proxy only (how many times the same account re-appears
in a single topic's thread) — the follower/following-ratio proxy
described in Table I is not computable from this schema and is
documented here, not silently skipped.
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

from scipy.signal import find_peaks

PROCESSED_DIR = Path("data/processed/weibo_ced")
OUTPUT_DIR = Path("data/processed/weibo_ced")

RANDOM_SEED = 42
VALIDATION_SAMPLE_SIZE = 300
PAPER_WEIBO_GUIDED_PCT = 91.7  # Table II reference point, rumor subset only

# ---------------------------------------------------------------------------
# Fixed thresholds (percentages / multiples). These are NOT fit to the
# paper's label counts — they're the rubric's own stated cutoffs.
# ---------------------------------------------------------------------------
CENTRALIZATION_SHARE_THRESHOLD = 0.20     # top account's share of topic replies
ONE_SIDED_SENTIMENT_SHARE_THRESHOLD = 0.70  # dominant-direction share of classified comments
EMOTIONAL_LANGUAGE_SHARE_THRESHOLD = 0.30
TEMPLATED_DUPLICATE_RATIO_THRESHOLD = 0.25
EXPLOSIVE_GROWTH_MULTIPLE = 3.0
SYNC_BURST_WINDOW_SECONDS = 300  # 5 minutes

# Percentile-derived thresholds (computed at runtime from the dataset's own
# distribution — see compute_global_thresholds()).
ABNORMAL_ACCOUNT_PERCENTILE = 90
SYNC_BURST_PERCENTILE = 90

# Minimum sample sizes below which a dimension's sub-components default to
# False ("not enough evidence to call it manipulative") rather than being
# silently imputed.
MIN_REPLIES_FOR_STRUCTURE = 5
MIN_REPLIES_FOR_NARRATIVE = 10
MIN_TIMESTAMPS_FOR_DYNAMICS = 10

# Chinese emotional / directive / rallying-language lexicon (simple
# substring lexicon — documented as a heuristic proxy, per Table I's
# "strongly emotional or directive language" sub-component).
EMOTIONAL_DIRECTIVE_LEXICON = [
    "转发", "扩散", "必须", "一定要", "求", "快看", "震惊", "愤怒", "可怕",
    "太惨", "谴责", "强烈", "呼吁", "赶紧", "大家注意", "警惕", "曝光",
    "人渣", "禽兽", "天理", "良心", "！！", "!!", "？？", "必转",
]

TEMPLATED_PHRASES = {"转发微博", "转发微博。", "轉發微博。", "轉發微博"}

# ---------------------------------------------------------------------------
# Sentiment backend
# ---------------------------------------------------------------------------
try:
    from snownlp import SnowNLP

    def sentiment_polarity(text: str):
        """Returns polarity in [0, 1], 1 = most positive, via SnowNLP."""
        if not text or not text.strip():
            return None
        try:
            return SnowNLP(text).sentiments
        except Exception:
            return None

    SENTIMENT_BACKEND = "snownlp"
except ImportError:
    # Fallback: a tiny hand-built polarity lexicon. This is a much weaker
    # signal than a trained classifier and is used ONLY because snownlp is
    # not installed in this environment (no network access to pip install
    # it here). Before running this in production, install snownlp
    # (`pip install snownlp`) so sentiment_polarity() above is used instead
    # of this fallback — the fallback is deliberately left in place (rather
    # than raising) so the rubric's *structure* and *narrative-duplication*
    # sub-components still run even without the dependency, but narrative
    # sentiment results produced by the fallback should be treated as low
    # confidence.
    _POS_WORDS = {"好", "棒", "支持", "加油", "感谢", "谢谢", "喜欢", "赞", "愿", "希望"}
    _NEG_WORDS = {"坏", "差", "愤怒", "可怕", "谴责", "恨", "骗", "假", "垃圾", "恶心", "可恨"}

    def sentiment_polarity(text: str):
        if not text or not text.strip():
            return None
        pos = sum(text.count(w) for w in _POS_WORDS)
        neg = sum(text.count(w) for w in _NEG_WORDS)
        if pos == 0 and neg == 0:
            return 0.5
        return pos / (pos + neg)

    SENTIMENT_BACKEND = "fallback_lexicon (snownlp not installed)"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    keep = [ch for ch in text if not ch.isspace() and ch not in "，。！？!?.,~～、"]
    return "".join(keep)


# ---------------------------------------------------------------------------
# Pass 1: raw per-topic metrics (needed before we can derive percentile
# thresholds for the two data-driven sub-components)
# ---------------------------------------------------------------------------

def compute_raw_structure_metrics(replies: list) -> dict:
    uids = [r.get("uid") for r in replies if r.get("uid")]
    total = len(replies)

    max_repeat_count = 0
    top_account_share = 0.0
    if uids:
        counts = Counter(uids)
        max_repeat_count = max(counts.values())
        top_account_share = max_repeat_count / total if total else 0.0

    timestamps_uids = sorted(
        [(parse_date(r.get("date")), r.get("uid")) for r in replies
         if parse_date(r.get("date")) and r.get("uid")],
        key=lambda x: x[0],
    )
    max_burst_distinct_accounts = 0
    window = []
    for ts, uid in timestamps_uids:
        window.append((ts, uid))
        window = [(t, u) for t, u in window
                  if (ts - t).total_seconds() <= SYNC_BURST_WINDOW_SECONDS]
        max_burst_distinct_accounts = max(max_burst_distinct_accounts, len(set(u for _, u in window)))

    return {
        "max_repeat_count": max_repeat_count,
        "top_account_share": top_account_share,
        "max_burst_distinct_accounts": max_burst_distinct_accounts,
        "n_replies": total,
    }


def compute_global_thresholds(raw_metrics_list: list) -> dict:
    """Derive the two percentile-based hard thresholds from the dataset's
    own distribution (only over topics with enough replies to be
    meaningful)."""
    eligible = [m for m in raw_metrics_list if m["n_replies"] >= MIN_REPLIES_FOR_STRUCTURE]

    repeat_counts = [m["max_repeat_count"] for m in eligible]
    burst_counts = [m["max_burst_distinct_accounts"] for m in eligible]

    abnormal_account_threshold = (
        float(np.percentile(repeat_counts, ABNORMAL_ACCOUNT_PERCENTILE)) if repeat_counts else 0.0
    )
    sync_burst_threshold = (
        float(np.percentile(burst_counts, SYNC_BURST_PERCENTILE)) if burst_counts else 0.0
    )

    return {
        "abnormal_account_threshold": abnormal_account_threshold,
        "sync_burst_threshold": sync_burst_threshold,
        "n_topics_used_for_thresholds": len(eligible),
    }


# ---------------------------------------------------------------------------
# Pass 2: per-topic binary flags
# ---------------------------------------------------------------------------

def evaluate_structure(replies: list, raw: dict, thresholds: dict) -> dict:
    evaluable = raw["n_replies"] >= MIN_REPLIES_FOR_STRUCTURE
    abnormal_account = evaluable and raw["max_repeat_count"] > thresholds["abnormal_account_threshold"]
    synchronized_burst = evaluable and raw["max_burst_distinct_accounts"] > thresholds["sync_burst_threshold"]
    centralized = evaluable and raw["top_account_share"] > CENTRALIZATION_SHARE_THRESHOLD

    return {
        "evaluable": evaluable,
        "abnormal_account_pattern": abnormal_account,
        "synchronized_burst": synchronized_burst,
        "centralized_propagation": centralized,
        "flag": bool(abnormal_account or synchronized_burst or centralized),
        # note: follower/following-ratio sub-component is not computable —
        # this dataset's repost records carry no profile data for reposters.
        "follower_following_ratio_evaluable": False,
    }


def evaluate_narrative(replies: list) -> dict:
    texts = [r.get("text", "") for r in replies]
    non_empty_texts = [t for t in texts if t and t.strip()]
    evaluable = len(texts) >= MIN_REPLIES_FOR_NARRATIVE

    one_sided = False
    emotional = False
    templated = False

    if evaluable:
        # (a) one-sided sentiment bias
        polarities = [p for p in (sentiment_polarity(t) for t in non_empty_texts) if p is not None]
        if len(polarities) >= MIN_REPLIES_FOR_NARRATIVE:
            frac_pos = sum(1 for p in polarities if p > 0.6) / len(polarities)
            frac_neg = sum(1 for p in polarities if p < 0.4) / len(polarities)
            one_sided = max(frac_pos, frac_neg) >= ONE_SIDED_SENTIMENT_SHARE_THRESHOLD

        # (b) emotional / directive language
        emo_count = sum(
            1 for t in texts if any(marker in t for marker in EMOTIONAL_DIRECTIVE_LEXICON)
        )
        emotional = (emo_count / len(texts)) > EMOTIONAL_LANGUAGE_SHARE_THRESHOLD

        # (c) templated / duplicate text (exact match on normalized text,
        # plus the known low-effort templated-repost phrase set)
        normalized = [normalize_text(t) if t.strip() else "" for t in texts]
        normalized = [t if t else "__EMPTY__" for t in normalized]
        for i, t in enumerate(texts):
            if t.strip() in TEMPLATED_PHRASES:
                normalized[i] = "__TEMPLATED__"
        counts = Counter(normalized)
        duplicate_count = sum(c for c in counts.values() if c > 1)
        templated_ratio = duplicate_count / len(texts)
        templated = templated_ratio > TEMPLATED_DUPLICATE_RATIO_THRESHOLD

    return {
        "evaluable": evaluable,
        "sentiment_backend": SENTIMENT_BACKEND,
        "one_sided_sentiment": one_sided,
        "emotional_directive_language": emotional,
        "templated_duplicate_text": templated,
        "flag": bool(one_sided or emotional or templated),
    }


def evaluate_dynamics(replies: list) -> dict:
    timestamps = sorted(parse_date(r.get("date")) for r in replies if parse_date(r.get("date")))
    evaluable = len(timestamps) >= MIN_TIMESTAMPS_FOR_DYNAMICS

    multiple_peaks = False
    explosive_growth = False

    if evaluable:
        t0 = timestamps[0]
        hours = [(t - t0).total_seconds() / 3600 for t in timestamps]
        max_hour = int(max(hours)) + 1
        bins = np.zeros(max_hour + 1)
        for h in hours:
            bins[int(h)] += 1

        # (a) multiple diffusion peaks
        if len(bins) >= 3:
            peaks, _ = find_peaks(bins)
            multiple_peaks = len(peaks) > 1

        # (b) explosive growth: any bin exceeds EXPLOSIVE_GROWTH_MULTIPLE x
        # the topic's own baseline (median of non-zero bins)
        nonzero = bins[bins > 0]
        baseline = float(np.median(nonzero)) if len(nonzero) else 0.0
        baseline = max(baseline, 1.0)
        explosive_growth = bool(bins.max() > EXPLOSIVE_GROWTH_MULTIPLE * baseline)

    return {
        "evaluable": evaluable,
        "multiple_peaks": multiple_peaks,
        "explosive_growth": explosive_growth,
        # No external real-world-event calendar is available for this
        # dataset, so this sub-component is explicitly marked as not
        # evaluable rather than defaulted to False. It does not
        # contribute to `flag` below.
        "event_misalignment": None,
        "flag": bool(multiple_peaks or explosive_growth),
    }


def label_topic(replies: list) -> dict:
    raw_structure = compute_raw_structure_metrics(replies)
    return raw_structure


def main():
    print("Loading cleaned repost data...")
    rumor_repost = load_json(PROCESSED_DIR / "rumor_repost_cleaned.json")
    non_rumor_repost = load_json(PROCESSED_DIR / "non_rumor_repost_cleaned.json")

    print(f"Sentiment backend in use: {SENTIMENT_BACKEND}")
    if SENTIMENT_BACKEND != "snownlp":
        print("  WARNING: snownlp is not installed in this environment; using a small")
        print("  hand-built lexicon fallback for the one-sided-sentiment sub-component.")
        print("  Install snownlp and re-run before treating narrative-dimension labels")
        print("  as final.")

    all_topics_raw = []  # [(post_id, original_label, replies, raw_structure_metrics)]
    for post_id, replies in rumor_repost.items():
        raw = compute_raw_structure_metrics(replies)
        all_topics_raw.append((post_id, "rumor", replies, raw))
    for post_id, replies in non_rumor_repost.items():
        raw = compute_raw_structure_metrics(replies)
        all_topics_raw.append((post_id, "non-rumor", replies, raw))

    print(f"Total topics: {len(all_topics_raw)}")

    # Derive the two data-driven hard thresholds once, from the whole
    # dataset's own distribution (NOT from the paper's label ratio).
    thresholds = compute_global_thresholds([raw for _, _, _, raw in all_topics_raw])
    print("\nData-derived thresholds:")
    print(f"  abnormal_account_threshold (repeat posts by one uid in a topic, "
          f"P{ABNORMAL_ACCOUNT_PERCENTILE}): {thresholds['abnormal_account_threshold']:.2f}")
    print(f"  sync_burst_threshold (distinct uids within {SYNC_BURST_WINDOW_SECONDS}s, "
          f"P{SYNC_BURST_PERCENTILE}): {thresholds['sync_burst_threshold']:.2f}")
    print(f"  (derived from {thresholds['n_topics_used_for_thresholds']} topics with "
          f">= {MIN_REPLIES_FOR_STRUCTURE} replies)")

    all_posts = []
    for post_id, original_label, replies, raw in all_topics_raw:
        structure = evaluate_structure(replies, raw, thresholds)
        narrative = evaluate_narrative(replies)
        dynamics = evaluate_dynamics(replies)

        flags = [structure["flag"], narrative["flag"], dynamics["flag"]]
        guided = sum(flags) >= 2

        # combined_score kept ONLY for backward compatibility with
        # topic_construction_weibo.py, which reads `combined_score` as a
        # diagnostic float. It is no longer used to derive guided_label
        # (that's now a direct >=2-of-3 binary vote) — it's just
        # (# flags true) / 3 for reference.
        combined_score = sum(flags) / 3

        all_posts.append({
            "post_id": post_id,
            "original_label": original_label,
            "structure": structure,
            "narrative": narrative,
            "dynamics": dynamics,
            "guided_label": "guided" if guided else "nonguided",
            "combined_score": combined_score,
        })

    guided_count = sum(1 for p in all_posts if p["guided_label"] == "guided")

    print("\n" + "=" * 60)
    print("RULE-BASED LABEL CONSTRUCTION SUMMARY (no calibration)")
    print("=" * 60)
    print(f"Total posts: {len(all_posts)}")
    pct_guided = guided_count / len(all_posts) * 100
    print(f"Guided: {guided_count} ({pct_guided:.1f}%)")
    print(f"Nonguided: {len(all_posts) - guided_count} ({100 - pct_guided:.1f}%)")

    crosstab = Counter((p["original_label"], p["guided_label"]) for p in all_posts)
    print("\nCross-tabulation (original label -> computed guided label):")
    for (orig, guided_lbl), count in crosstab.items():
        print(f"  {orig} -> {guided_lbl}: {count}")

    print("\nDimension flag rates (share of ALL topics where the dimension fired):")
    for dim in ("structure", "narrative", "dynamics"):
        rate = sum(1 for p in all_posts if p[dim]["flag"]) / len(all_posts) * 100
        evaluable_rate = sum(1 for p in all_posts if p[dim]["evaluable"]) / len(all_posts) * 100
        print(f"  {dim}: flagged {rate:.1f}% of all topics ({evaluable_rate:.1f}% were evaluable)")

    print("\n" + "-" * 60)
    print("COMPARISON TO PAPER'S PUBLISHED RATIO (informational only —")
    print("this run was NOT calibrated to match it):")
    print(f"  paper's guided share (Table III, full Weibo dataset): 54.6% (1849/3387)")
    print(f"  this rule-based reconstruction's guided share:         {pct_guided:.1f}%")
    print("-" * 60)

    output_path = OUTPUT_DIR / "guided_labels_computed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")

    run_table2_validation(all_posts)


def run_table2_validation(all_posts: list):
    """Validation step (NOT a calibration step): draw a random 300-sample
    from the rumor subset, using the SAME already-computed rubric labels,
    and report what fraction come out 'guided' as a sanity check against
    the paper's Table II figure (91.7% for Weibo). This does not feed back
    into the full-dataset labels above in any way."""
    print("\n" + "=" * 60)
    print("VALIDATION: Table II style reproduction (rumor subset, n=300 sample)")
    print("=" * 60)

    rumor_posts = [p for p in all_posts if p["original_label"] == "rumor"]
    if len(rumor_posts) < VALIDATION_SAMPLE_SIZE:
        print(f"  Only {len(rumor_posts)} rumor topics available; sampling all of them "
              f"instead of {VALIDATION_SAMPLE_SIZE}.")
        sample = rumor_posts
    else:
        rng = random.Random(RANDOM_SEED)
        sample = rng.sample(rumor_posts, VALIDATION_SAMPLE_SIZE)

    guided_in_sample = sum(1 for p in sample if p["guided_label"] == "guided")
    pct = guided_in_sample / len(sample) * 100 if sample else 0.0

    print(f"  Sample size: {len(sample)}")
    print(f"  Classified as intentionally guided: {guided_in_sample} ({pct:.1f}%)")
    print(f"  Paper's reported figure (Table II, Weibo): {PAPER_WEIBO_GUIDED_PCT}%")
    print(f"  Difference: {pct - PAPER_WEIBO_GUIDED_PCT:+.1f} percentage points")
    if abs(pct - PAPER_WEIBO_GUIDED_PCT) > 15:
        print("  -> Large gap: treat this as a signal to revisit thresholds, not as a")
        print("     reason to back-fit the full-dataset labels above.")

    validation_output = {
        "sample_size": len(sample),
        "guided_count": guided_in_sample,
        "guided_pct": pct,
        "paper_reference_pct": PAPER_WEIBO_GUIDED_PCT,
        "difference_pct_points": pct - PAPER_WEIBO_GUIDED_PCT,
        "random_seed": RANDOM_SEED,
        "sampled_post_ids": [p["post_id"] for p in sample],
    }
    output_path = OUTPUT_DIR / "table2_validation_weibo.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validation_output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved validation record to: {output_path}")


if __name__ == "__main__":
    main()
