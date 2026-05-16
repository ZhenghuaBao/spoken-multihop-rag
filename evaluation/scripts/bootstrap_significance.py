"""
Paired bootstrap significance test on per-question F1 scores.

Tests two claims:
  1. Oracle vs accent gap is significantly > 0 (each method, each accent).
  2. |Δ F1| of complex methods (IRCoT+Hippo) is significantly larger than
     |Δ F1| of simple methods (Naive RAG) -- i.e., amplification is real.

Usage:
    python evaluation/bootstrap_significance.py \
        --results results/2wiki_1000.json \
        --n-resamples 10000
"""

import argparse
import json
import sys
import io
import random
from typing import List, Tuple

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

ACCENTS = ["us", "in", "ph", "ng"]
ASR_TO_ORACLE = {
    "A": "E",
    "B": "E",
    "C": "F",
    "D": "F",
    "G": "G",
    "H": "H",
    "I": "G",
    "J": "H",
}
METHOD_LABEL = {"A": "Naive", "C": "HippoRAG", "G": "IRCoT+N", "H": "IRCoT+H"}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_per_question_f1(detailed: dict, accent: str, cell: str) -> dict:
    """Returns {qid: f1}."""
    if accent not in detailed or cell not in detailed[accent]:
        return {}
    return {r["id"]: r.get("f1", 0.0) for r in detailed[accent][cell]}


def get_oracle_f1(detailed: dict, oracle_cell: str) -> dict:
    """Returns {qid: oracle_f1}, falling back to forward-filled accent if needed."""
    if "oracle" in detailed and oracle_cell in detailed["oracle"]:
        return {r["id"]: r.get("f1", 0.0) for r in detailed["oracle"][oracle_cell]}
    for acc in ACCENTS:
        if acc in detailed and oracle_cell in detailed[acc]:
            return {r["id"]: r.get("f1", 0.0) for r in detailed[acc][oracle_cell]}
    return {}


def paired_bootstrap_pvalue(
    diffs: List[float],
    n_resamples: int,
    null_value: float = 0.0,
    rng: random.Random = None,
) -> Tuple[float, float, float]:
    """One-sided paired bootstrap p-value testing whether mean(diffs) > null_value.

    Returns (mean_diff, p_value, ci_low, ci_high) - 95% CI via percentile.
    """
    if rng is None:
        rng = random.Random(42)
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0.0, 0.0
    mean = sum(diffs) / n
    centered = [d - mean for d in diffs]  # H0: mean = 0

    extreme = 0
    means = []
    for _ in range(n_resamples):
        sample = [centered[rng.randrange(n)] for _ in range(n)]
        sm = sum(sample) / n
        means.append(sm)
        if abs(sm) >= abs(mean - null_value):
            extreme += 1
    p = extreme / n_resamples

    # 95% CI for the actual mean (resample original diffs without centering)
    ci_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        ci_means.append(sum(sample) / n)
    ci_means.sort()
    lo = ci_means[int(0.025 * n_resamples)]
    hi = ci_means[int(0.975 * n_resamples)]
    return mean, p, lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    data = load(args.results)
    detailed = data["accent_detailed_results"]

    # ---------- Test 1: each (method, accent) Oracle--accent gap ----------
    print("=" * 70)
    print(
        f"TEST 1: Oracle--accent F1 gap (paired bootstrap, "
        f"n_resamples={args.n_resamples})"
    )
    print("=" * 70)
    print(
        f"{'Cell':<6}{'Method':<14}{'Accent':<8}{'gap':>9}{'CI_low':>9}"
        f"{'CI_high':>9}{'p-value':>11}"
    )
    print("-" * 70)

    for cell in ["A", "C", "G", "H"]:
        oracle_cell = ASR_TO_ORACLE[cell]
        oracle_f1 = get_oracle_f1(detailed, oracle_cell)
        if not oracle_f1:
            continue
        for accent in ACCENTS:
            accent_f1 = get_per_question_f1(detailed, accent, cell)
            if not accent_f1:
                continue
            common = set(oracle_f1.keys()) & set(accent_f1.keys())
            diffs = [oracle_f1[q] - accent_f1[q] for q in common]
            mean, pval, lo, hi = paired_bootstrap_pvalue(
                diffs, args.n_resamples, rng=rng
            )
            print(
                f"{cell:<6}{METHOD_LABEL.get(cell, cell):<14}"
                f"{accent.upper():<8}{mean:>9.4f}{lo:>9.4f}{hi:>9.4f}"
                f"{pval:>11.4g}"
            )

    # ---------- Test 2: amplification (|Δ| larger for complex methods) ------
    print()
    print("=" * 70)
    print("TEST 2: Amplification (|Δ_complex| > |Δ_naive|)")
    print("=" * 70)
    print(
        f"{'Pair':<28}{'Accent':<8}{'mean Δ|gap|':>13}"
        f"{'CI_low':>9}{'CI_high':>9}{'p-value':>11}"
    )
    print("-" * 70)

    pairs = [("A", "C"), ("A", "G"), ("A", "H"), ("C", "H")]
    for naive_cell, complex_cell in pairs:
        naive_oracle = get_oracle_f1(detailed, ASR_TO_ORACLE[naive_cell])
        complex_oracle = get_oracle_f1(detailed, ASR_TO_ORACLE[complex_cell])
        for accent in ACCENTS:
            naive_acc = get_per_question_f1(detailed, accent, naive_cell)
            complex_acc = get_per_question_f1(detailed, accent, complex_cell)
            if not (naive_oracle and complex_oracle and naive_acc and complex_acc):
                continue
            common = (
                set(naive_oracle.keys())
                & set(complex_oracle.keys())
                & set(naive_acc.keys())
                & set(complex_acc.keys())
            )
            # |Δ| per question = |oracle_f1 - accent_f1|
            naive_abs = [abs(naive_oracle[q] - naive_acc[q]) for q in common]
            complex_abs = [abs(complex_oracle[q] - complex_acc[q]) for q in common]
            diffs = [c - n for n, c in zip(naive_abs, complex_abs)]
            mean, pval, lo, hi = paired_bootstrap_pvalue(
                diffs, args.n_resamples, rng=rng
            )
            label = f"{METHOD_LABEL[naive_cell]}->{METHOD_LABEL[complex_cell]}"
            print(
                f"{label:<28}{accent.upper():<8}{mean:>13.4f}{lo:>9.4f}"
                f"{hi:>9.4f}{pval:>11.4g}"
            )


if __name__ == "__main__":
    main()
