"""
Evaluation metrics for spoken multi-hop QA.

Reuses DualRAG's evaluation logic + adds ASR-specific metrics (WER, CER).
"""

import re
from typing import Dict, List, Optional


# ---- QA Metrics (reused from new/benchmarks/metrics.py) ----


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Check if normalized prediction matches ground truth."""
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def substring_match(predicted: str, ground_truth: str) -> bool:
    """Check if ground truth is a substring of prediction."""
    return normalize_answer(ground_truth) in normalize_answer(predicted)


def f1_score(predicted: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(predicted).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_qa_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Compute aggregate QA metrics."""
    n = len(predictions)
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "substring_match": 0.0, "n": 0}

    em_total = sum(exact_match(p, g) for p, g in zip(predictions, ground_truths))
    f1_total = sum(f1_score(p, g) for p, g in zip(predictions, ground_truths))
    sub_total = sum(substring_match(p, g) for p, g in zip(predictions, ground_truths))

    return {
        "em": em_total / n,
        "f1": f1_total / n,
        "substring_match": sub_total / n,
        "n": n,
    }


# ---- ASR-specific Metrics ----


def word_error_rate(hypothesis: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER) using edit distance.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=words in reference.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def entity_error_rate(
    hypothesis: str,
    reference: str,
    entities: Optional[List[str]] = None,
) -> Dict:
    """
    Measure how well entities in the reference are preserved in the ASR output.

    If entities are not provided, extract capitalized multi-word phrases from reference.

    Returns dict with entity_recall, entity_precision, missed_entities.
    """
    if entities is None:
        # Simple heuristic: extract capitalized words (likely entities)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", reference)
        entities = list(set(entities))

    if not entities:
        return {
            "entity_recall": 1.0,
            "entity_precision": 1.0,
            "missed_entities": [],
            "total_entities": 0,
        }

    hyp_lower = hypothesis.lower()
    found = 0
    missed = []

    for entity in entities:
        if entity.lower() in hyp_lower:
            found += 1
        else:
            missed.append(entity)

    recall = found / len(entities) if entities else 1.0

    return {
        "entity_recall": recall,
        "missed_entities": missed,
        "found_entities": found,
        "total_entities": len(entities),
    }


def compute_asr_metrics(
    transcriptions: List[str],
    references: List[str],
) -> Dict:
    """Compute aggregate ASR metrics."""
    n = len(transcriptions)
    if n == 0:
        return {"wer": 0.0, "avg_entity_recall": 0.0, "n": 0}

    wer_total = sum(word_error_rate(t, r) for t, r in zip(transcriptions, references))
    entity_results = [
        entity_error_rate(t, r) for t, r in zip(transcriptions, references)
    ]
    entity_recall_total = sum(er["entity_recall"] for er in entity_results)

    return {
        "wer": wer_total / n,
        "avg_entity_recall": entity_recall_total / n,
        "n": n,
    }


# ---- Combined Report ----


def full_evaluation_report(
    predictions: List[str],
    ground_truths: List[str],
    transcriptions: Optional[List[str]] = None,
    original_texts: Optional[List[str]] = None,
) -> Dict:
    """
    Generate a complete evaluation report.

    Args:
        predictions: generated answers
        ground_truths: gold answers
        transcriptions: ASR outputs (optional, for WER)
        original_texts: original text questions (optional, for WER)
    """
    report = {"qa": compute_qa_metrics(predictions, ground_truths)}

    if transcriptions and original_texts:
        report["asr"] = compute_asr_metrics(transcriptions, original_texts)

    return report
