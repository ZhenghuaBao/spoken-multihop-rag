"""
Extract N-best Decoding table for the paper.

Reads result files containing cells A (Naive top-1), B (Naive N-best),
C (HippoRAG top-1), D (HippoRAG N-best), and oracle E/F. Produces a
LaTeX-formatted table comparing top-1 vs N-best F1 per accent, with
recovery percentages.

Usage:
    python evaluation/nbest_table.py \
        --results results/experiment_all+oracle_20260330_224620.json \
        --accent ng
"""

import argparse
import json
import sys
import io

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell_avg_f1(detailed: dict, accent: str, cell: str) -> float:
    """Get avg F1 for accent/cell from detailed_results, or -1 if missing."""
    if accent not in detailed or cell not in detailed[accent]:
        return -1.0
    rows = detailed[accent][cell]
    if not rows:
        return -1.0
    return sum(r.get("f1", 0.0) for r in rows) / len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        required=True,
        help="Result file with cells A, B, C, D plus oracle E/F",
    )
    p.add_argument("--accent", default="ng")
    args = p.parse_args()

    data = load(args.results)
    detailed = data["accent_detailed_results"]

    # Oracle baseline (avg of US Naive Oracle / HippoRAG Oracle)
    oracle_naive = cell_avg_f1(detailed, "oracle", "E")
    oracle_hippo = cell_avg_f1(detailed, "oracle", "F")

    # ASR top-1
    a_top1 = cell_avg_f1(detailed, args.accent, "A")
    c_top1 = cell_avg_f1(detailed, args.accent, "C")
    g_top1 = cell_avg_f1(detailed, args.accent, "G")  # IRCoT+Naive
    h_top1 = cell_avg_f1(detailed, args.accent, "H")  # IRCoT+Hippo

    # ASR N-best
    b_nbest = cell_avg_f1(detailed, args.accent, "B")
    d_nbest = cell_avg_f1(detailed, args.accent, "D")
    i_nbest = cell_avg_f1(detailed, args.accent, "I")  # NbIRCoT+Naive
    j_nbest = cell_avg_f1(detailed, args.accent, "J")  # NbIRCoT+Hippo

    rows = [
        ("Naive RAG", a_top1, b_nbest, oracle_naive),
        ("HippoRAG2", c_top1, d_nbest, oracle_hippo),
        ("IRCoT+Naive", g_top1, i_nbest, oracle_naive),
        ("IRCoT+Hippo", h_top1, j_nbest, oracle_hippo),
    ]

    print("\n=== N-best Decoding Table ===")
    print(f"Accent: {args.accent.upper()}")
    print()
    print(
        f"{'Method':<14s} {'top-1 F1':>9s} {'Δtop1':>8s} {'NB F1':>8s} "
        f"{'ΔNB':>8s} {'Recovery':>10s}"
    )
    print("-" * 60)

    for name, top1, nbest, oracle in rows:
        if top1 < 0 or oracle < 0:
            print(f"{name:<14s}  (data missing)")
            continue
        gap_top1 = top1 - oracle
        if nbest >= 0:
            gap_nbest = nbest - oracle
            recovery = (
                (1 - abs(gap_nbest) / abs(gap_top1)) * 100 if gap_top1 != 0 else 0.0
            )
            print(
                f"{name:<14s} {top1:9.3f} {gap_top1:+8.3f} {nbest:8.3f} "
                f"{gap_nbest:+8.3f} {recovery:9.1f}%"
            )
        else:
            print(f"{name:<14s} {top1:9.3f} {gap_top1:+8.3f}    (no N-best data)")

    # LaTeX output
    print()
    print("=== LaTeX ===\n")
    print(r"\begin{table}[t]")
    print(
        r"\caption{N-best Decoding results on \textsc{Dataset} under "
        + f"{args.accent.upper()} speech. Recovery is the fraction of the "
        + "Oracle--accent F1 gap closed by N-best.}"
    )
    print(r"\label{tab:nbest}")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(
        r" & \multicolumn{2}{c}{\textbf{top-1}} & \multicolumn{2}{c}{\textbf{N-best}} & \\"
    )
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"\textbf{Method} & F1 & $\Delta$ & F1 & $\Delta$ & \textbf{Recov.} \\")
    print(r"\midrule")
    for name, top1, nbest, oracle in rows:
        if top1 < 0 or oracle < 0:
            continue
        gap_top1 = top1 - oracle
        if nbest >= 0:
            gap_nbest = nbest - oracle
            recovery = (
                (1 - abs(gap_nbest) / abs(gap_top1)) * 100 if gap_top1 != 0 else 0.0
            )
            print(
                f"{name:<14s} & {top1:.3f} & ${gap_top1:+.3f}$ & "
                f"{nbest:.3f} & ${gap_nbest:+.3f}$ & {recovery:.1f}\\% \\\\"
            )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
