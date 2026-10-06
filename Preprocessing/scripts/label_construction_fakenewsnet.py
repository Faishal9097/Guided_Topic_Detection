"""
Step 4: Label Construction — FakeNewsNet (UPFD)

Implements the paper's Table I rubric as binary (present/absent)
dimension flags, applied to whatever sub-components are actually
computable from the UPFD graph-feature release (A.txt, graph_labels.npy,
node_graph_id.npy, and the *.npz feature matrices). No calibration to
the paper's published counts is performed — see the summary printed at
the end of main().

DIMENSION-BY-DIMENSION, WHAT IS AND ISN'T EVALUABLE HERE, AND WHY
-------------------------------------------------------------------
1. Propagator structure
   (a) abnormal account patterns (posting frequency / follower-following
       ratio) — NOT EVALUABLE. `new_profile_feature.npz` contains an
       opaque numeric profile-embedding per node (UPFD does not document
       these columns as raw, individually interpretable follower/
       following counts), so there is no principled way to threshold a
       "follower/following ratio" from it without guessing which columns
       mean what. Rather than guess, this sub-component is recorded as
       None ("not evaluable") for every topic.
   (b) highly synchronized behavior (burst within a short time window) —
       NOT EVALUABLE. UPFD's graph release carries no timestamps.
   (c) centralized propagation structure — EVALUABLE. Computed exactly
       as before: max out-degree / total edges in the graph, now
       compared against a FIXED percentage threshold (not a normalized
       continuous score).
   => structure_flag = centralized_propagation (the only evaluable
      sub-component).

2. Textual narrative
   (a) one-sided sentiment bias, (b) emotional/directive language — both
       require the article/tweet's raw text. The standard UPFD graph
       release used by this pipeline (see clean_fakenewsnet.py) ships
       pre-computed feature matrices (TF-IDF-style content vectors,
       spaCy, BERT) but not raw text strings. This script WILL use real
       VADER sentiment/emotional-lexicon analysis on raw text if a raw
       text mapping file is found (see `try_load_raw_text()` — it looks
       for a few conventional filenames), but if none is found it marks
       both sub-components None ("not evaluable") rather than silently
       skipping or faking a score from the TF-IDF vectors.
   (c) templated/duplicate content — EVALUABLE without raw text: computed
       from pairwise cosine similarity of the existing content-feature
       vectors (near-duplicate content vectors are still a reasonable
       templating signal even without token-level text), now thresholded
       as a hard near-duplicate-pair ratio rather than an averaged
       continuous similarity score.
   => narrative_flag = one_sided_sentiment OR emotional_language OR
      templated_duplicate_content (using whichever sub-components are
      evaluable; if a raw-text mapping is supplied, all three
      contribute).

3. Propagation dynamics — NOT EVALUABLE AT ALL (no timestamps in UPFD).
   Recorded as None for every topic and excluded entirely from the
   guided-label vote, per the task's "apply the rule as >=2 of 2
   evaluable dimensions" instruction. This means FakeNewsNet's guided
   label is a genuinely weaker, 2-dimension approximation compared to
   Weibo's full 3-dimension rubric — this is a real, documented
   limitation of the dataset, not a bug.

RULE: guided = 1 iff BOTH evaluable dimensions (structure, narrative)
are flagged True (i.e. >= 2 of 2 evaluable dimensions — with only 2
dimensions evaluable, "at least 2" collapses to "both").
"""

import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import scipy.sparse as sp

BASE_DIR = Path("data/raw/fakenewsnet")
OUTPUT_DIR = Path("data/processed/fakenewsnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["politifact", "gossipcop"]

RANDOM_SEED = 42
VALIDATION_SAMPLE_SIZE = 300
PAPER_GUIDED_PCT = {"politifact": 96.0, "gossipcop": 98.3}  # Table II reference points

# Fixed thresholds — not fit to the paper's label counts.
CENTRALIZATION_OUT_DEGREE_SHARE_THRESHOLD = 0.20
NEAR_DUP_COSINE_SIM_THRESHOLD = 0.90
TEMPLATED_DUPLICATE_PAIR_RATIO_THRESHOLD = 0.25
ONE_SIDED_SENTIMENT_SHARE_THRESHOLD = 0.70
EMOTIONAL_LANGUAGE_SHARE_THRESHOLD = 0.30

MIN_NODES_FOR_STRUCTURE = 3
MIN_NODES_FOR_NARRATIVE = 3
CONTENT_SAMPLE_SIZE = 30  # cap pairwise-similarity cost on large graphs

EMOTIONAL_DIRECTIVE_LEXICON_EN = [
    "shocking", "breaking", "must see", "share this", "urgent", "exposed",
    "outrage", "disgusting", "unbelievable", "wake up", "they don't want you",
    "!!!", "retweet", "share now",
]

# ---------------------------------------------------------------------------
# Optional raw-text loading. UPFD's standard graph release (what this
# pipeline consumes per clean_fakenewsnet.py) does not include raw text.
# If the user has a separate raw-text mapping (node index -> text) sitting
# alongside the feature files, we'll use it for genuine VADER sentiment /
# lexicon analysis. Otherwise we say so explicitly and mark those
# sub-components not evaluable.
# ---------------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _vader = SentimentIntensityAnalyzer()

    def vader_compound(text: str):
        if not text or not text.strip():
            return None
        return _vader.polarity_scores(text)["compound"]

    VADER_AVAILABLE = True
except ImportError:
    _vader = None
    VADER_AVAILABLE = False

    def vader_compound(text: str):
        return None


def try_load_raw_text(folder: Path):
    """Looks for a conventional raw-text mapping file. Returns a
    {node_index: text} dict, or None if nothing is found. We do NOT
    fabricate this mapping from the TF-IDF/BERT feature matrices — those
    are not invertible to text."""
    candidate_names = ["node_text.json", "raw_text.json", "content_text.json"]
    for name in candidate_names:
        path = folder / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
    return None


def load_edge_list(folder: Path):
    a_file = folder / "A.txt"
    if not a_file.exists():
        a_file = folder / "A"
    edges = []
    with open(a_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                edges.append((int(parts[0].strip()), int(parts[1].strip())))
    return edges


# ---------------------------------------------------------------------------
# Dimension evaluators
# ---------------------------------------------------------------------------

def evaluate_structure(graph_id, node_graph_id, edges_by_src):
    node_indices = np.where(node_graph_id == graph_id)[0]
    evaluable = len(node_indices) >= MIN_NODES_FOR_STRUCTURE

    centralized = False
    out_degree_share = 0.0
    if evaluable:
        out_degrees = [len(edges_by_src.get(n, [])) for n in node_indices]
        total_edges = sum(out_degrees)
        if total_edges > 0:
            out_degree_share = max(out_degrees) / total_edges
            centralized = out_degree_share > CENTRALIZATION_OUT_DEGREE_SHARE_THRESHOLD
        else:
            evaluable = False  # no edges at all: nothing to assess

    return {
        "evaluable": evaluable,
        "abnormal_account_pattern": None,  # not evaluable, see module docstring
        "synchronized_burst": None,        # not evaluable, no timestamps in UPFD
        "centralized_propagation": centralized,
        "out_degree_share": out_degree_share,
        "flag": bool(centralized) if evaluable else False,
    }


def evaluate_narrative(graph_id, node_graph_id, content_features, raw_text_map):
    node_indices = np.where(node_graph_id == graph_id)[0]
    evaluable = len(node_indices) >= MIN_NODES_FOR_NARRATIVE

    templated = False
    one_sided = None
    emotional = None

    if evaluable:
        sample_indices = node_indices
        if len(sample_indices) > CONTENT_SAMPLE_SIZE:
            sample_indices = np.random.choice(sample_indices, CONTENT_SAMPLE_SIZE, replace=False)

        sub_features = content_features[sample_indices].toarray()
        norms = np.linalg.norm(sub_features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = sub_features / norms
        similarity_matrix = normalized @ normalized.T
        n = len(sample_indices)
        num_pairs = (n * (n - 1)) / 2
        if num_pairs > 0:
            upper = np.triu(similarity_matrix, k=1)
            near_dup_pairs = int(np.sum(upper > NEAR_DUP_COSINE_SIM_THRESHOLD))
            templated_ratio = near_dup_pairs / num_pairs
            templated = bool(templated_ratio > TEMPLATED_DUPLICATE_PAIR_RATIO_THRESHOLD)

        if raw_text_map is not None:
            texts = [raw_text_map.get(int(i)) for i in node_indices]
            texts = [t for t in texts if t and t.strip()]
            if texts and VADER_AVAILABLE:
                scores = [vader_compound(t) for t in texts]
                scores = [s for s in scores if s is not None]
                if scores:
                    frac_pos = sum(1 for s in scores if s > 0.3) / len(scores)
                    frac_neg = sum(1 for s in scores if s < -0.3) / len(scores)
                    one_sided = max(frac_pos, frac_neg) >= ONE_SIDED_SENTIMENT_SHARE_THRESHOLD

                    emo_count = sum(
                        1 for t in texts
                        if any(marker in t.lower() for marker in EMOTIONAL_DIRECTIVE_LEXICON_EN)
                    )
                    emotional = (emo_count / len(texts)) > EMOTIONAL_LANGUAGE_SHARE_THRESHOLD

    flag_components = [c for c in (one_sided, emotional, templated) if c is True]
    flag = bool(flag_components) if evaluable else False

    return {
        "evaluable": evaluable,
        "one_sided_sentiment": one_sided,   # None if no raw text available
        "emotional_directive_language": emotional,  # None if no raw text available
        "templated_duplicate_content": templated,
        "flag": flag,
    }


def process_dataset(name: str):
    print("\n" + "=" * 60)
    print(f"PROCESSING: {name.upper()}")
    print("=" * 60)

    folder = BASE_DIR / name
    edges = load_edge_list(folder)
    labels = np.load(folder / "graph_labels.npy")
    node_graph_id = np.load(folder / "node_graph_id.npy")
    content_features = sp.load_npz(folder / "new_content_feature.npz")

    raw_text_map = try_load_raw_text(folder)
    if raw_text_map is None:
        print("  No raw-text mapping found (looked for node_text.json / raw_text.json /")
        print("  content_text.json). one_sided_sentiment and emotional_directive_language")
        print("  will be recorded as 'not evaluable' (None) for this dataset — only the")
        print("  content-vector-based templated/duplicate sub-component will be used for")
        print("  the narrative dimension. See module docstring for details.")
    elif not VADER_AVAILABLE:
        print("  Raw-text mapping found, but vaderSentiment is not installed in this")
        print("  environment (no network access to pip install it here). Install")
        print("  vaderSentiment and re-run to get real sentiment/emotional-language flags;")
        print("  for now those two sub-components are recorded as not evaluable (None).")

    edges_by_src = {}
    for src, dst in edges:
        edges_by_src.setdefault(src, []).append(dst)

    num_graphs = len(labels)
    results = []

    print(f"Computing dimension flags for {num_graphs} graphs...")
    for graph_id in range(num_graphs):
        structure = evaluate_structure(graph_id, node_graph_id, edges_by_src)
        narrative = evaluate_narrative(graph_id, node_graph_id, content_features, raw_text_map)

        # Propagation dynamics: not evaluable at all for UPFD (no timestamps).
        dynamics = {
            "evaluable": False,
            "multiple_peaks": None,
            "explosive_growth": None,
            "event_misalignment": None,
            "flag": None,
        }

        # Only 2 of 3 dimensions are evaluable here -> "guided" requires
        # BOTH of them to be flagged (>= 2 of 2 evaluable dimensions).
        guided = bool(structure["flag"] and narrative["flag"])

        # combined_score kept ONLY for backward compatibility with
        # topic_construction_fakenewsnet.py (reads it as a diagnostic
        # float). No longer used to derive guided_label.
        combined_score = (int(structure["flag"]) + int(narrative["flag"])) / 2

        results.append({
            "graph_id": int(graph_id),
            "original_label": "fake" if labels[graph_id] == 1 else "real",
            "structure": structure,
            "narrative": narrative,
            "dynamics": dynamics,
            "guided_label": "guided" if guided else "nonguided",
            "combined_score": combined_score,
        })
        if graph_id % 500 == 0 and graph_id > 0:
            print(f"  Processed {graph_id}/{num_graphs}...")

    guided_count = sum(1 for r in results if r["guided_label"] == "guided")
    pct_guided = guided_count / num_graphs * 100

    print(f"\nGuided: {guided_count} / {num_graphs} ({pct_guided:.1f}%)")

    crosstab = Counter((r["original_label"], r["guided_label"]) for r in results)
    print("\nCross-tabulation (original label -> computed guided label):")
    for (orig, guided_lbl), count in crosstab.items():
        print(f"  {orig} -> {guided_lbl}: {count}")

    for dim in ("structure", "narrative"):
        rate = sum(1 for r in results if r[dim]["flag"]) / num_graphs * 100
        print(f"  {dim} flagged: {rate:.1f}% of all graphs")

    print("\n" + "-" * 60)
    print("COMPARISON TO PAPER'S PUBLISHED RATIO (informational only —")
    print("this run was NOT calibrated to match it):")
    print("  paper's guided share (UPFD is reported ~balanced in Table III)")
    print(f"  this rule-based reconstruction's guided share: {pct_guided:.1f}%")
    print("-" * 60)

    output_path = OUTPUT_DIR / f"{name}_guided_labels_computed.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")

    run_table2_validation(name, results)


def run_table2_validation(name: str, results: list):
    """Validation step (NOT calibration): random 300-sample from the fake
    subset, same already-computed rubric, reported against the paper's
    Table II figure for this dataset. Does not feed back into the
    full-dataset labels."""
    print(f"\nVALIDATION: Table II style reproduction ({name}, fake subset, n=300 sample)")

    fake_results = [r for r in results if r["original_label"] == "fake"]
    if len(fake_results) < VALIDATION_SAMPLE_SIZE:
        print(f"  Only {len(fake_results)} fake graphs available; sampling all of them "
              f"instead of {VALIDATION_SAMPLE_SIZE}.")
        sample = fake_results
    else:
        rng = random.Random(RANDOM_SEED)
        sample = rng.sample(fake_results, VALIDATION_SAMPLE_SIZE)

    guided_in_sample = sum(1 for r in sample if r["guided_label"] == "guided")
    pct = guided_in_sample / len(sample) * 100 if sample else 0.0
    paper_pct = PAPER_GUIDED_PCT.get(name)

    print(f"  Sample size: {len(sample)}")
    print(f"  Classified as intentionally guided: {guided_in_sample} ({pct:.1f}%)")
    print(f"  Paper's reported figure (Table II, {name}): {paper_pct}%")
    print(f"  Difference: {pct - paper_pct:+.1f} percentage points")
    if abs(pct - paper_pct) > 15:
        print("  -> Large gap: treat this as a signal to revisit thresholds, not as a")
        print("     reason to back-fit the full-dataset labels above.")

    validation_output = {
        "dataset": name,
        "sample_size": len(sample),
        "guided_count": guided_in_sample,
        "guided_pct": pct,
        "paper_reference_pct": paper_pct,
        "difference_pct_points": pct - paper_pct,
        "random_seed": RANDOM_SEED,
        "sampled_graph_ids": [r["graph_id"] for r in sample],
    }
    output_path = OUTPUT_DIR / f"{name}_table2_validation.json"
    with open(output_path, "w") as f:
        json.dump(validation_output, f, indent=2)
    print(f"  Saved validation record to: {output_path}")


def main():
    for name in DATASETS:
        process_dataset(name)


if __name__ == "__main__":
    main()
