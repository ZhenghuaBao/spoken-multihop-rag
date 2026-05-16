"""
Compute degradation rate per WER bin per method.

For each (method, accent) pair, classify questions into WER bins
(0-5%, 5-10%, 10-20%, 20%+) and report the fraction that degrade
(Oracle correct, accent incorrect).

Usage:
    python evaluation/scripts/wer_stratified_degradation.py \
        --results results/2wiki_1000.json \
        --accent ng
"""

import argparse
import json
import sys
import io
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.core.metrics import f1_score  # noqa: E402

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)


WER_BINS = [
    (0.0, 0.05, "0-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.20, "10-20%"),
    (0.20, 1.01, "20%+"),
]

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

METHOD_LABELS = {
    "A": "Naive RAG",
    "C": "HippoRAG2",
    "G": "IRCoT+Naive",
    "H": "IRCoT+Hippo",
}


def wer_bin(wer: float) -> str:
    for lo, hi, label in WER_BINS:
        if lo <= wer < hi:
            return label
    return "20%+"


def get_results(detailed, accent, cell):
    if accent in detailed and cell in detailed[accent]:
        return {r["id"]: r for r in detailed[accent][cell]}
    return {}


def get_oracle_results(detailed, oracle_cell):
    if "oracle" in detailed and oracle_cell in detailed["oracle"]:
        return {r["id"]: r for r in detailed["oracle"][oracle_cell]}
    for acc in ["us", "in", "ph", "ng"]:
        if acc in detailed and oracle_cell in detailed[acc]:
            return {r["id"]: r for r in detailed[acc][oracle_cell]}
    return {}


def compute_bin_degradation(detailed, accent, cell, f1_threshold=0.5):
    """For each WER bin, compute fraction of questions that
    degrade (Oracle F1 >= threshold AND accent F1 < threshold)."""
    oracle_cell = ASR_TO_ORACLE[cell]
    oracle = get_oracle_results(detailed, oracle_cell)
    accent_results = get_results(detailed, accent, cell)
    if not oracle or not accent_results:
        return {}

    bin_n = defaultdict(int)
    bin_degraded = defaultdict(int)

    for qid, ar in accent_results.items():
        if qid not in oracle:
            continue
        gt = ar.get("ground_truth", oracle[qid].get("ground_truth", ""))
        oracle_f1 = f1_score(oracle[qid]["answer"], gt)
        accent_f1 = f1_score(ar["answer"], gt)
        wer = ar.get("wer", 0.0) or 0.0
        wer_bin_label = wer_bin(wer)
        bin_n[wer_bin_label] += 1
        if oracle_f1 >= f1_threshold and accent_f1 < f1_threshold:
            bin_degraded[wer_bin_label] += 1

    return {b: (bin_degraded[b], bin_n[b]) for b in [w[2] for w in WER_BINS]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--accent", default="ng")
    args = p.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    detailed = data["accent_detailed_results"]

    print(f"\nWER bin degradation rates ({args.accent.upper()})")
    print("=" * 70)
    print(f"{'Bin':<14}", end="")
    for cell in ["A", "C", "G", "H"]:
        print(f"{METHOD_LABELS[cell]:>14s}", end="")
    print(f"{'n':>6s}")
    print("-" * 70)

    rows = {}
    for cell in ["A", "C", "G", "H"]:
        rows[cell] = compute_bin_degradation(detailed, args.accent, cell)

    for lo, hi, label in WER_BINS:
        print(f"{label:<14}", end="")
        n = 0
        for cell in ["A", "C", "G", "H"]:
            r = rows[cell].get(label, (0, 0))
            pct = r[0] / r[1] * 100 if r[1] else 0
            print(f"{pct:>13.1f}%", end="")
            n = max(n, r[1])
        print(f"{n:>6d}")

    # LaTeX-ready snippet
    print()
    print("=== LaTeX snippet ===")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(
        r"\textbf{WER wer_bin_label} & \textbf{Naive} & \textbf{HippoRAG2} "
        r"& \textbf{IRCoT+N} & \textbf{IRCoT+H} \\"
    )
    print(r"\midrule")
    for lo, hi, label in WER_BINS:
        n = max((rows[c].get(label, (0, 0))[1] for c in "ACGH"), default=0)
        line = f"{label:<8} ($n{{=}}{n}$) "
        for cell in ["A", "C", "G", "H"]:
            r = rows[cell].get(label, (0, 0))
            pct = r[0] / r[1] * 100 if r[1] else 0
            line += f" & {pct:.1f}"
        line += r" \\"
        print(line)
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Python list for figure regeneration
    print()
    print("=== Python lists for generate_figures.py ===")
    for cell in ["A", "C", "G", "H"]:
        lst = []
        for lo, hi, label in WER_BINS:
            r = rows[cell].get(label, (0, 0))
            pct = r[0] / r[1] * 100 if r[1] else 0
            lst.append(round(pct, 1))
        print(f"{METHOD_LABELS[cell]:<14}: {lst}")


if __name__ == "__main__":
    main()
