"""
Compares Oracle vs ASR-accent results per question, categorizes ASR error types,
breaks down by question metadata, and identifies worst-case degradation patterns.

Usage:
    python evaluation/core/error_analysis.py \
        --results results/experiment_all+oracle_20260330_224620.json \
        --ground-truth dataset/2wikimultihopqa_1000/ground_truth.json \
        --asr-data data/accent_nbest_results_2wiki.json \
        --accent ng --cross-method
"""

import json
import re
import sys
import io
import difflib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from evaluation.core.metrics import f1_score  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCENTS = ["us", "in", "ph", "ng"]

# ASR cell -> oracle counterpart
ORACLE_COUNTERPART = {
    "A": "E",
    "B": "E",  # Naive retrieval
    "C": "F",
    "D": "F",  # HippoRAG
    "G": "G",
    "H": "H",  # IRCoT (oracle runs under accent="oracle")
    "I": "G",
    "J": "H",  # NbIRCoT vs oracle IRCoT
}

ASR_CELLS = {"A", "B", "C", "D", "G", "H", "I", "J"}

WER_BUCKETS = [
    (0.0, 0.05, "0-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.20, "10-20%"),
    (0.20, 1.01, "20%+"),
]

FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "can",
    "could",
    "and",
    "or",
    "but",
    "not",
    "no",
    "if",
    "than",
    "that",
    "which",
    "who",
    "whom",
    "this",
    "these",
    "those",
    "it",
    "its",
    "with",
    "by",
    "from",
    "as",
    "into",
    "about",
    "between",
    "what",
    "where",
    "when",
    "how",
    "why",
    "whose",
}

METHOD_LABELS = {
    "A": "Naive+top1",
    "B": "Naive+nbest",
    "C": "HippoRAG+top1",
    "D": "HippoRAG+nbest",
    "E": "Oracle+Naive",
    "F": "Oracle+HippoRAG",
    "G": "IRCoT+Naive",
    "H": "IRCoT+HippoRAG",
    "I": "NbIRCoT+Naive",
    "J": "NbIRCoT+HippoRAG",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuestionComparison:
    qid: str
    original_question: str
    asr_question: str
    wer: float
    oracle_answer: str
    accent_answer: str
    ground_truth: str
    oracle_f1: float
    accent_f1: float
    category: str  # both_correct, degradation, both_wrong, accent_better
    f1_drop: float
    question_type: str
    difficulty: str
    num_hops: int
    error_types: List[str] = field(default_factory=list)
    corrupted_entities: List[str] = field(default_factory=list)
    accent_cell: str = ""
    oracle_cell: str = ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_result_file(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_result_files(paths: List[str]) -> Dict:
    """Load and merge multiple result files into one combined structure."""
    merged = {
        "config": {},
        "accent_summaries": {},
        "accent_detailed_results": {},
    }
    for p in paths:
        data = load_result_file(p)
        # Handle multi-accent format
        if "accent_detailed_results" in data:
            src = data["accent_detailed_results"]
            sums = data.get("accent_summaries", {})
        # Handle old single-accent format (summaries/detailed_results with cells at top)
        elif "detailed_results" in data and "summaries" in data:
            # Convert to multi-accent format: put under a pseudo-accent
            # Detect accent from config or filename
            src = {}
            sums = {}
            cfg = data.get("config", {})
            accents = cfg.get("accents", ["unknown"])
            for acc in accents:
                src[acc] = data["detailed_results"]
                sums[acc] = data["summaries"]
        else:
            continue

        if not merged["config"]:
            merged["config"] = data.get("config", {})

        for accent, cells in src.items():
            if accent not in merged["accent_detailed_results"]:
                merged["accent_detailed_results"][accent] = {}
                merged["accent_summaries"][accent] = {}
            if isinstance(cells, dict):
                for cell, results in cells.items():
                    merged["accent_detailed_results"][accent][cell] = results
            if accent in sums and isinstance(sums[accent], dict):
                for cell, summary in sums[accent].items():
                    merged["accent_summaries"][accent][cell] = summary

    return merged


def load_ground_truth(path: str) -> Dict[str, Dict]:
    """Load ground truth and normalize schema across datasets."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    gt = {}
    for qid, entry in raw.items():
        gt[qid] = {
            "answer": entry.get("answer", ""),
            "answer_aliases": entry.get("answer_aliases", []),
            "type": entry.get("type", ""),
            "level": entry.get("level", ""),
            "num_hops": entry.get("num_hops", 0),
        }
        # Musique: no type field, use num_hops
        if not gt[qid]["type"] and gt[qid]["num_hops"]:
            gt[qid]["type"] = f"{gt[qid]['num_hops']}hop"
        if not gt[qid]["type"]:
            gt[qid]["type"] = "unknown"
        if not gt[qid]["level"]:
            gt[qid]["level"] = "unknown"
    return gt


def load_asr_data(path: str) -> Dict[str, Dict]:
    """Load accent_nbest_results and index by question ID."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {item["id"]: item for item in raw}


# ---------------------------------------------------------------------------
# Oracle result extraction
# ---------------------------------------------------------------------------


def get_oracle_results(result_data: Dict, oracle_cell: str) -> Dict[str, Dict]:
    """Extract oracle cell results indexed by qid."""
    src = result_data["accent_detailed_results"]

    # Try "oracle" accent first
    if "oracle" in src and oracle_cell in src["oracle"]:
        return {r["id"]: r for r in src["oracle"][oracle_cell]}

    # Fall back: oracle cells E/F are forward-filled into accent entries
    for accent in ACCENTS:
        if accent in src and oracle_cell in src[accent]:
            return {r["id"]: r for r in src[accent][oracle_cell]}

    return {}


# ---------------------------------------------------------------------------
# ASR error categorization
# ---------------------------------------------------------------------------


def extract_named_entities(text: str) -> Set[str]:
    """Extract capitalized phrases as entity proxies."""
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return set(entities)


def extract_numbers(text: str) -> Set[str]:
    """Extract numbers and date patterns."""
    nums = re.findall(r"\b\d+(?:st|nd|rd|th)?\b", text)
    return set(nums)


def compute_word_diff(original: str, asr: str) -> List[Tuple[str, str, str]]:
    """Word-level diff: returns list of (op, orig_word, asr_word)."""
    orig_words = original.split()
    asr_words = asr.split()
    sm = difflib.SequenceMatcher(None, orig_words, asr_words)
    diffs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                diffs.append(("equal", orig_words[i1 + k], asr_words[j1 + k]))
        elif op == "replace":
            max_len = max(i2 - i1, j2 - j1)
            for k in range(max_len):
                ow = orig_words[i1 + k] if (i1 + k) < i2 else ""
                aw = asr_words[j1 + k] if (j1 + k) < j2 else ""
                diffs.append(("replace", ow, aw))
        elif op == "delete":
            for k in range(i1, i2):
                diffs.append(("delete", orig_words[k], ""))
        elif op == "insert":
            for k in range(j1, j2):
                diffs.append(("insert", "", asr_words[k]))
    return diffs


def categorize_asr_errors(original: str, asr_transcription: str, wer: float) -> Dict:
    """Categorize ASR errors for a single question."""
    entities = extract_named_entities(original)
    numbers = extract_numbers(original)
    entity_words = set()
    for ent in entities:
        entity_words.update(ent.split())
    number_words = set()
    for num in numbers:
        number_words.add(num)

    diffs = compute_word_diff(original, asr_transcription)

    error_types = set()
    corrupted_entities = []
    corrupted_numbers = []
    function_word_changes = []
    content_word_changes = 0
    total_changes = 0

    for op, ow, aw in diffs:
        if op == "equal":
            continue
        total_changes += 1
        ow_clean = re.sub(r"[^\w]", "", ow)
        # Check entity
        if ow_clean in entity_words or ow in entity_words:
            error_types.add("entity_corruption")
            corrupted_entities.append(f"{ow} -> {aw}")
        # Check number
        elif ow_clean in number_words:
            error_types.add("number_corruption")
            corrupted_numbers.append(f"{ow} -> {aw}")
        # Check function word
        elif ow.lower() in FUNCTION_WORDS:
            function_word_changes.append((ow, aw))
        else:
            content_word_changes += 1

    if wer > 0.20 and content_word_changes > 0:
        error_types.add("severe_garbling")
    if function_word_changes and not error_types:
        error_types.add("function_word_noise")

    if not error_types and total_changes > 0:
        error_types.add("other_content_change")

    return {
        "error_types": sorted(error_types),
        "corrupted_entities": corrupted_entities,
        "corrupted_numbers": corrupted_numbers,
        "function_word_changes": function_word_changes,
        "total_changes": total_changes,
        "content_word_changes": content_word_changes,
    }


# ---------------------------------------------------------------------------
# Per-question comparison
# ---------------------------------------------------------------------------


def classify_comparison(
    oracle_f1: float, accent_f1: float, f1_threshold: float = 0.5
) -> str:
    oracle_ok = oracle_f1 >= f1_threshold
    accent_ok = accent_f1 >= f1_threshold
    if oracle_ok and accent_ok:
        return "both_correct"
    elif oracle_ok and not accent_ok:
        return "degradation"
    elif not oracle_ok and not accent_ok:
        return "both_wrong"
    else:  # accent ok but oracle not
        return "accent_better"


def compare_all_questions(
    result_data: Dict,
    accent: str,
    accent_cell: str,
    ground_truth: Dict,
    asr_data: Optional[Dict] = None,
    f1_threshold: float = 0.5,
) -> List[QuestionComparison]:
    """Compare oracle vs accent for every question in a cell."""
    oracle_cell = ORACLE_COUNTERPART.get(accent_cell)
    if not oracle_cell:
        return []

    oracle_by_id = get_oracle_results(result_data, oracle_cell)
    accent_results = (
        result_data["accent_detailed_results"].get(accent, {}).get(accent_cell, [])
    )

    comparisons = []
    for ar in accent_results:
        qid = ar["id"]
        orc = oracle_by_id.get(qid)
        if orc is None:
            continue

        gt_entry = ground_truth.get(qid, {})
        gt_answer = gt_entry.get("answer", ar.get("ground_truth", ""))

        oracle_f1 = f1_score(orc["answer"], gt_answer)
        accent_f1_val = f1_score(ar["answer"], gt_answer)
        wer_val = ar.get("wer", 0.0) or 0.0

        category = classify_comparison(oracle_f1, accent_f1_val, f1_threshold)

        original = ar.get("original", orc.get("query", ""))
        asr_question = ar.get("query", "")

        # Error categorization for degradation cases
        error_info = {"error_types": [], "corrupted_entities": []}
        if category == "degradation" and original and asr_question:
            error_info = categorize_asr_errors(original, asr_question, wer_val)

        comparisons.append(
            QuestionComparison(
                qid=qid,
                original_question=original,
                asr_question=asr_question,
                wer=wer_val,
                oracle_answer=orc["answer"],
                accent_answer=ar["answer"],
                ground_truth=gt_answer,
                oracle_f1=oracle_f1,
                accent_f1=accent_f1_val,
                category=category,
                f1_drop=oracle_f1 - accent_f1_val,
                question_type=gt_entry.get("type", "unknown"),
                difficulty=gt_entry.get("level", "unknown"),
                num_hops=gt_entry.get("num_hops", 0),
                error_types=error_info["error_types"],
                corrupted_entities=error_info.get("corrupted_entities", []),
                accent_cell=accent_cell,
                oracle_cell=oracle_cell,
            )
        )

    return comparisons


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------


def wer_bucket(wer: float) -> str:
    for low, high, label in WER_BUCKETS:
        if low <= wer < high:
            return label
    return "20%+"


def _bucket_stats(items: List[QuestionComparison]) -> Dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    cats = defaultdict(int)
    for c in items:
        cats[c.category] += 1
    return {
        "n": n,
        "both_correct": cats["both_correct"],
        "degradation": cats["degradation"],
        "both_wrong": cats["both_wrong"],
        "accent_better": cats["accent_better"],
        "degradation_rate": cats["degradation"] / n,
        "avg_oracle_f1": sum(c.oracle_f1 for c in items) / n,
        "avg_accent_f1": sum(c.accent_f1 for c in items) / n,
        "avg_wer": sum(c.wer for c in items) / n,
    }


def breakdown_by_metadata(comparisons: List[QuestionComparison]) -> Dict:
    by_type = defaultdict(list)
    by_level = defaultdict(list)
    by_wer = defaultdict(list)
    by_hops = defaultdict(list)

    for c in comparisons:
        by_type[c.question_type].append(c)
        by_level[c.difficulty].append(c)
        by_wer[wer_bucket(c.wer)].append(c)
        if c.num_hops:
            by_hops[f"{c.num_hops}hop"].append(c)

    return {
        "by_question_type": {k: _bucket_stats(v) for k, v in sorted(by_type.items())},
        "by_difficulty": {k: _bucket_stats(v) for k, v in sorted(by_level.items())},
        "by_wer_bucket": {k: _bucket_stats(v) for k, v in sorted(by_wer.items())},
        "by_num_hops": {k: _bucket_stats(v) for k, v in sorted(by_hops.items())},
    }


def error_type_distribution(comparisons: List[QuestionComparison]) -> Dict[str, int]:
    counts = defaultdict(int)
    degradation_cases = [c for c in comparisons if c.category == "degradation"]
    for c in degradation_cases:
        for et in c.error_types:
            counts[et] += 1
        if not c.error_types:
            counts["no_asr_error_detected"] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Cross-method analysis
# ---------------------------------------------------------------------------


def cross_method_analysis(
    result_data: Dict,
    accent: str,
    ground_truth: Dict,
    f1_threshold: float = 0.5,
) -> Dict:
    """Compare degradation patterns across retrieval methods."""
    pairs_to_check = [
        ("A", "E", "Naive+top1"),
        ("C", "F", "HippoRAG+top1"),
        ("B", "E", "Naive+nbest"),
        ("D", "F", "HippoRAG+nbest"),
    ]

    # Also check IRCoT cells if present
    accent_cells = set(result_data["accent_detailed_results"].get(accent, {}).keys())
    if "G" in accent_cells:
        pairs_to_check.append(("G", "G", "IRCoT+Naive"))
    if "H" in accent_cells:
        pairs_to_check.append(("H", "H", "IRCoT+HippoRAG"))
    if "I" in accent_cells:
        pairs_to_check.append(("I", "G", "NbIRCoT+Naive"))
    if "J" in accent_cells:
        pairs_to_check.append(("J", "H", "NbIRCoT+HippoRAG"))

    results = []
    comparisons_by_cell = {}
    for acell, ocell, label in pairs_to_check:
        if acell not in accent_cells:
            continue
        comps = compare_all_questions(
            result_data, accent, acell, ground_truth, f1_threshold=f1_threshold
        )
        if not comps:
            continue
        comparisons_by_cell[acell] = comps
        n = len(comps)
        cats = defaultdict(int)
        f1_drops = []
        for c in comps:
            cats[c.category] += 1
            if c.category == "degradation":
                f1_drops.append(c.f1_drop)
        results.append(
            {
                "accent_cell": acell,
                "oracle_cell": ocell,
                "method": label,
                "n": n,
                "degradation": cats["degradation"],
                "degradation_rate": cats["degradation"] / n if n else 0,
                "both_correct": cats["both_correct"],
                "both_wrong": cats["both_wrong"],
                "accent_better": cats["accent_better"],
                "avg_f1_drop": sum(f1_drops) / len(f1_drops) if f1_drops else 0,
                "avg_accent_f1": sum(c.accent_f1 for c in comps) / n if n else 0,
                "avg_oracle_f1": sum(c.oracle_f1 for c in comps) / n if n else 0,
            }
        )

    # Compare Naive vs HippoRAG resilience
    naive_top1 = next((r for r in results if r["accent_cell"] == "A"), None)
    hippo_top1 = next((r for r in results if r["accent_cell"] == "C"), None)
    resilience = {}
    if naive_top1 and hippo_top1:
        resilience = {
            "naive_degradation_rate": naive_top1["degradation_rate"],
            "hipporag_degradation_rate": hippo_top1["degradation_rate"],
            "more_resilient": "HippoRAG"
            if hippo_top1["degradation_rate"] < naive_top1["degradation_rate"]
            else "Naive",
        }

    return {"pairs": results, "resilience": resilience}


# ---------------------------------------------------------------------------
# Worst-case extraction
# ---------------------------------------------------------------------------


def extract_worst_cases(
    comparisons: List[QuestionComparison],
    top_n: int = 20,
    asr_data: Optional[Dict] = None,
) -> List[Dict]:
    degradation = [c for c in comparisons if c.category == "degradation"]
    degradation.sort(key=lambda c: c.f1_drop, reverse=True)
    worst = degradation[:top_n]

    cases = []
    for i, c in enumerate(worst):
        case = {
            "rank": i + 1,
            "qid": c.qid,
            "original_question": c.original_question,
            "asr_transcription": c.asr_question,
            "wer": round(c.wer, 4),
            "oracle_answer": c.oracle_answer,
            "accent_answer": c.accent_answer,
            "ground_truth": c.ground_truth,
            "oracle_f1": round(c.oracle_f1, 4),
            "accent_f1": round(c.accent_f1, 4),
            "f1_drop": round(c.f1_drop, 4),
            "question_type": c.question_type,
            "error_types": c.error_types,
            "corrupted_entities": c.corrupted_entities,
        }
        # Add all accent transcriptions if available
        if asr_data and c.qid in asr_data:
            item = asr_data[c.qid]
            accents_data = item.get("accents", {})
            case["all_accent_transcriptions"] = {
                acc: accents_data[acc]["top1"] for acc in ACCENTS if acc in accents_data
            }
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_summary_table(
    accent: str,
    cell: str,
    oracle_cell: str,
    comparisons: List[QuestionComparison],
    breakdowns: Dict,
    error_dist: Dict,
):
    n = len(comparisons)
    if n == 0:
        print(f"  No data for {accent}/{cell}")
        return

    cats = defaultdict(int)
    for c in comparisons:
        cats[c.category] += 1
    avg_wer = sum(c.wer for c in comparisons) / n
    avg_oracle_f1 = sum(c.oracle_f1 for c in comparisons) / n
    avg_accent_f1 = sum(c.accent_f1 for c in comparisons) / n

    print(f"\n{'=' * 70}")
    print(
        f"  Accent: {accent.upper()}  |  Cell: {cell} ({METHOD_LABELS.get(cell, cell)}) vs {oracle_cell} ({METHOD_LABELS.get(oracle_cell, oracle_cell)})"
    )
    print(
        f"  N={n}  |  Avg WER={avg_wer:.1%}  |  Oracle F1={avg_oracle_f1:.3f}  |  Accent F1={avg_accent_f1:.3f}"
    )
    print(f"{'=' * 70}")

    print("\n  Category Distribution:")
    print(
        f"    Both Correct:  {cats['both_correct']:3d} ({cats['both_correct'] / n:.1%})"
    )
    print(
        f"    Degradation:   {cats['degradation']:3d} ({cats['degradation'] / n:.1%})  <-- ASR caused failure"
    )
    print(f"    Both Wrong:    {cats['both_wrong']:3d} ({cats['both_wrong'] / n:.1%})")
    print(
        f"    Accent Better: {cats['accent_better']:3d} ({cats['accent_better'] / n:.1%})"
    )

    if error_dist:
        degrad_n = cats["degradation"]
        print(f"\n  Error Types (in {degrad_n} degradation cases):")
        for et, count in error_dist.items():
            print(
                f"    {et:30s}  {count:3d} ({count / degrad_n:.0%})"
                if degrad_n
                else f"    {et}: {count}"
            )

    # Question type breakdown
    by_type = breakdowns.get("by_question_type", {})
    if by_type and len(by_type) > 1:
        print("\n  By Question Type:")
        print(
            f"    {'Type':<20s} {'N':>5s} {'Degrad':>7s} {'Rate':>7s} {'OracF1':>8s} {'AccF1':>8s}"
        )
        for t, stats in by_type.items():
            if stats["n"] == 0:
                continue
            print(
                f"    {t:<20s} {stats['n']:5d} {stats['degradation']:7d} {stats['degradation_rate']:7.1%} {stats['avg_oracle_f1']:8.3f} {stats['avg_accent_f1']:8.3f}"
            )

    # WER bucket breakdown
    by_wer = breakdowns.get("by_wer_bucket", {})
    if by_wer:
        print("\n  By WER Bucket:")
        print(
            f"    {'WER':<10s} {'N':>5s} {'Degrad':>7s} {'Rate':>7s} {'OracF1':>8s} {'AccF1':>8s}"
        )
        for bucket_label in ["0-5%", "5-10%", "10-20%", "20%+"]:
            stats = by_wer.get(bucket_label, {"n": 0})
            if stats["n"] == 0:
                continue
            print(
                f"    {bucket_label:<10s} {stats['n']:5d} {stats['degradation']:7d} {stats['degradation_rate']:7.1%} {stats['avg_oracle_f1']:8.3f} {stats['avg_accent_f1']:8.3f}"
            )

    print()


def print_cross_method_table(accent: str, cross: Dict):
    pairs = cross.get("pairs", [])
    if not pairs:
        return
    print(f"\n{'=' * 70}")
    print(f"  Cross-Method Comparison: {accent.upper()}")
    print(f"{'=' * 70}")
    print(
        f"  {'Method':<22s} {'N':>5s} {'Degrad':>7s} {'Rate':>7s} {'AvgDrop':>8s} {'OracF1':>8s} {'AccF1':>8s}"
    )
    for p in pairs:
        print(
            f"  {p['method']:<22s} {p['n']:5d} {p['degradation']:7d} {p['degradation_rate']:7.1%} {p['avg_f1_drop']:8.3f} {p['avg_oracle_f1']:8.3f} {p['avg_accent_f1']:8.3f}"
        )

    res = cross.get("resilience", {})
    if res:
        print(
            f"\n  Resilience: {res.get('more_resilient', '?')} is more resilient to ASR errors"
        )
        print(f"    Naive degradation rate:    {res['naive_degradation_rate']:.1%}")
        print(f"    HippoRAG degradation rate: {res['hipporag_degradation_rate']:.1%}")
    print()


def print_worst_cases(cases: List[Dict], max_print: int = 10):
    if not cases:
        return
    print(f"\n{'=' * 70}")
    print(f"  Top {min(max_print, len(cases))} Worst Degradation Cases")
    print(f"{'=' * 70}")
    for c in cases[:max_print]:
        print(f"\n  #{c['rank']} (F1 drop: {c['f1_drop']:.3f}, WER: {c['wer']:.1%})")
        print(f"    Type: {c['question_type']}")
        print(f"    Original:  {c['original_question'][:120]}")
        print(f"    ASR:       {c['asr_transcription'][:120]}")
        print(f"    GT:        {c['ground_truth']}")
        print(f"    Oracle:    {c['oracle_answer']}")
        print(f"    Accent:    {c['accent_answer']}")
        if c["error_types"]:
            print(f"    Errors:    {', '.join(c['error_types'])}")
        if c["corrupted_entities"]:
            print(f"    Corrupted: {'; '.join(c['corrupted_entities'][:5])}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for Spoken Multi-hop RAG"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Path(s) to experiment result JSON(s)",
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground_truth.json"
    )
    parser.add_argument(
        "--asr-data", default=None, help="Path to accent_nbest_results JSON (optional)"
    )
    parser.add_argument(
        "--accent", default="ng", help="Accent to analyze: us, in, ph, ng, or 'all'"
    )
    parser.add_argument(
        "--cell",
        default=None,
        help="Specific ASR cell to analyze (default: all present)",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.5,
        help="F1 threshold for correctness classification",
    )
    parser.add_argument(
        "--top-n", type=int, default=20, help="Number of worst-case examples"
    )
    parser.add_argument(
        "--cross-method", action="store_true", help="Include cross-method comparison"
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load data
    if len(args.results) == 1:
        result_data = load_result_file(args.results[0])
        # Ensure multi-accent format
        if "accent_detailed_results" not in result_data:
            result_data = merge_result_files(args.results)
    else:
        result_data = merge_result_files(args.results)

    ground_truth = load_ground_truth(args.ground_truth)
    asr_data = load_asr_data(args.asr_data) if args.asr_data else None

    # Determine accents
    available_accents = set(result_data["accent_detailed_results"].keys()) - {"oracle"}
    if args.accent == "all":
        accents_to_analyze = [a for a in ACCENTS if a in available_accents]
    else:
        accents_to_analyze = [args.accent] if args.accent in available_accents else []

    if not accents_to_analyze:
        print(
            f"ERROR: No matching accents found. Available: {sorted(available_accents)}"
        )
        return

    # Run analysis
    all_output = {"config": result_data.get("config", {}), "analyses": {}}

    for accent in accents_to_analyze:
        accent_cells_available = set(
            result_data["accent_detailed_results"][accent].keys()
        )
        cells_to_analyze = (
            [args.cell] if args.cell else sorted(accent_cells_available & ASR_CELLS)
        )

        for cell in cells_to_analyze:
            if cell not in accent_cells_available:
                continue
            oracle_cell = ORACLE_COUNTERPART.get(cell)
            if not oracle_cell:
                continue

            comparisons = compare_all_questions(
                result_data, accent, cell, ground_truth, asr_data, args.f1_threshold
            )
            if not comparisons:
                continue

            breakdowns = breakdown_by_metadata(comparisons)
            error_dist = error_type_distribution(comparisons)
            worst = extract_worst_cases(comparisons, args.top_n, asr_data)

            # Print
            print_summary_table(
                accent, cell, oracle_cell, comparisons, breakdowns, error_dist
            )
            print_worst_cases(worst, max_print=min(5, args.top_n))

            # Collect for JSON
            key = f"{accent}/{cell}"
            all_output["analyses"][key] = {
                "accent": accent,
                "accent_cell": cell,
                "oracle_cell": oracle_cell,
                "total": len(comparisons),
                "categories": {
                    cat: sum(1 for c in comparisons if c.category == cat)
                    for cat in [
                        "both_correct",
                        "degradation",
                        "both_wrong",
                        "accent_better",
                    ]
                },
                "avg_wer": sum(c.wer for c in comparisons) / len(comparisons),
                "avg_oracle_f1": sum(c.oracle_f1 for c in comparisons)
                / len(comparisons),
                "avg_accent_f1": sum(c.accent_f1 for c in comparisons)
                / len(comparisons),
                "error_type_counts": error_dist,
                "breakdowns": breakdowns,
                "worst_cases": worst,
                "all_comparisons": [asdict(c) for c in comparisons],
            }

    # Cross-method analysis
    if args.cross_method:
        all_output["cross_method"] = {}
        for accent in accents_to_analyze:
            cross = cross_method_analysis(
                result_data, accent, ground_truth, args.f1_threshold
            )
            all_output["cross_method"][accent] = cross
            print_cross_method_table(accent, cross)

    # Save
    if args.output:
        out_path = args.output
    else:
        out_path = f"error_analysis_{args.accent}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved detailed analysis to {out_path}")


if __name__ == "__main__":
    main()
