"""
Produce the Error-Type Distribution table (paper Table 2): per-accent
percentages of each error label within degradation cases, plus the
matching LaTeX block.

Thin wrapper over the JSONs produced by ``core/error_analysis.py``.
No labelling logic lives here -- this script only aggregates the
``error_type_counts`` field that error_analysis.py already emitted.

Usage:
    python evaluation/scripts/error_type_distribution.py \\
        --error-analysis results/error_analysis_2wiki_all.json \\
        --cell A \\
        --dataset-name 2WikiMultiHopQA
"""

import argparse
import io
import json
import sys

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

# Label order: entity-first, paper-table order
LABEL_ORDER = [
    ("entity_corruption", "Entity corruption"),
    ("severe_garbling", "Severe garbling"),
    ("number_corruption", "Number/date corrupt."),
    ("function_word_noise", "Function-word noise"),
    ("other_content_change", "Other content change"),
]

ACCENTS = ["us", "in", "ph", "ng"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--error-analysis",
        required=True,
        help="JSON produced by core/error_analysis.py (e.g. "
        "results/error_analysis_2wiki_all.json)",
    )
    p.add_argument(
        "--cell",
        default="A",
        help="Method cell to report (A=Naive, C=HippoRAG2, "
        "G=IRCoT+Naive, H=IRCoT+HippoRAG2). Default: A.",
    )
    p.add_argument(
        "--dataset-name",
        default="Dataset",
        help="Display name used in the LaTeX caption (e.g. 2WikiMultiHopQA)",
    )
    args = p.parse_args()

    with open(args.error_analysis, encoding="utf-8") as f:
        data = json.load(f)
    analyses = data["analyses"]
    n_questions = data.get("config", {}).get("num_questions", "?")

    # For each accent, collect (degradation_count, label_counts) for the cell
    per_accent = {}
    for accent in ACCENTS:
        key = f"{accent}/{args.cell}"
        if key not in analyses:
            continue
        sub = analyses[key]
        deg = sub.get("categories", {}).get("degradation", 0)
        etc = sub.get("error_type_counts", {})
        per_accent[accent] = (deg, etc)

    if not per_accent:
        print(
            f"No analyses found for cell {args.cell} in {args.error_analysis}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Plain-text table
    print("\n=== Error type distribution within degradation cases ===")
    print(f"  Source: {args.error_analysis}")
    print(f"  Cell:   {args.cell}    Dataset: {args.dataset_name}")
    print(f"  Per-accent pool: n={n_questions} questions\n")

    header = f"{'Error type':<22}" + "".join(
        f"{acc.upper():>9}" for acc in ACCENTS if acc in per_accent
    )
    print(header)
    print("-" * len(header))

    for key, label in LABEL_ORDER:
        row = f"{label:<22}"
        for acc in ACCENTS:
            if acc not in per_accent:
                continue
            deg, etc = per_accent[acc]
            c = etc.get(key, 0)
            pct = (c / deg * 100) if deg > 0 else 0.0
            row += f"{pct:>8.1f}%"
        print(row)
    print("-" * len(header))
    row = f"{'# degradation cases':<22}"
    for acc in ACCENTS:
        if acc not in per_accent:
            continue
        deg, _ = per_accent[acc]
        row += f"{deg:>9d}"
    print(row)

    # LaTeX
    print("\n=== LaTeX ===\n")
    print(r"\begin{table}[t]")
    print(
        r"\caption{Distribution of error types within degradation cases on "
        f"{args.dataset_name} under cell {args.cell}, with $n{{={n_questions}}}$ "
        r"questions per accent. Percentages may exceed 100\% because labels "
        r"are not mutually exclusive. Entity corruption dominates across all "
        r"four accents.}"
    )
    print(r"\label{tab:error_types}")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    n_cols = sum(1 for acc in ACCENTS if acc in per_accent)
    print(r"\begin{tabular}{l" + "c" * n_cols + "}")
    print(r"\toprule")
    head = r"\textbf{Error type}"
    for acc in ACCENTS:
        if acc in per_accent:
            head += rf" & \textbf{{{acc.upper()}}}"
    head += r" \\"
    print(head)
    print(r"\midrule")
    for key, label in LABEL_ORDER:
        line = f"{label:<22}"
        for acc in ACCENTS:
            if acc not in per_accent:
                continue
            deg, etc = per_accent[acc]
            c = etc.get(key, 0)
            pct = (c / deg * 100) if deg > 0 else 0.0
            line += f" & {pct:.0f}\\%"
        line += r" \\"
        print(line)
    print(r"\midrule")
    line = "# degradation cases     "
    for acc in ACCENTS:
        if acc not in per_accent:
            continue
        deg, _ = per_accent[acc]
        line += f" & {deg}"
    line += r" \\"
    print(line)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
