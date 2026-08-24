"""Entity-corruption rate within degradation cases, per method and accent
(paper Table 8, Appendix "Error-Type Distributions Across Benchmarks and
Methods").

Verifies that the dominance of entity corruption is not specific to Naive
RAG: the same rule-based categorization is applied to the degradation cases
of all four RAG configurations, on all three benchmarks.

For each (benchmark, method, accent) cell this reports

    entity_corruption_count / degradation_count

as a percentage, with the degradation count in parentheses. Cases can carry
multiple labels, so the percentages within a cell need not sum to 100.

Inputs are the cross-method error-analysis JSONs, produced by
    python evaluation/core/error_analysis.py --cross-method ...
one per benchmark.

Usage:
    python evaluation/scripts/cross_method_entity_table.py

Paths default to the repository-relative locations and can be overridden
with --hotpotqa / --2wiki / --musique.
"""

import argparse
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent  # repository root

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

# Display order matches paper Table 8.
CELLS = [
    ("A", "Naive RAG"),
    ("G", "IRCoT+Naive"),
    ("C", "HippoRAG2"),
    ("H", "IRCoT+HippoRAG2"),
]
ACCENTS = ["us", "in", "ph", "ng"]


def entity_share(analyses: dict, accent: str, cell: str):
    """Return (percentage, n_degradation) for one method/accent cell."""
    sub = analyses.get(f"{accent}/{cell}")
    if sub is None:
        return None, 0
    degradation = sub.get("categories", {}).get("degradation", 0)
    entity = sub.get("error_type_counts", {}).get("entity_corruption", 0)
    if not degradation:
        return None, 0
    return round(entity / degradation * 100), degradation


def load_analyses(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["analyses"]


def main():
    default_dir = _ROOT / "results"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hotpotqa",
        type=Path,
        default=default_dir / "error_analysis_hotpotqa_all_methods.json",
    )
    p.add_argument(
        "--2wiki",
        dest="twowiki",
        type=Path,
        default=default_dir / "error_analysis_2wiki_all_methods.json",
    )
    p.add_argument(
        "--musique",
        type=Path,
        default=default_dir / "error_analysis_musique_all_methods.json",
    )
    args = p.parse_args()

    datasets = [
        ("HotpotQA", args.hotpotqa),
        ("2WikiMultiHopQA", args.twowiki),
        ("MuSiQue", args.musique),
    ]

    missing = [(name, path) for name, path in datasets if not path.exists()]
    if missing:
        lines = ["Missing cross-method error-analysis input(s):", ""]
        lines += [f"  {name}: {path}" for name, path in missing]
        lines += [
            "",
            "Produce them with evaluation/core/error_analysis.py --cross-method.",
            "See the 'Datasets' section of README.md.",
        ]
        raise SystemExit("\n".join(lines))

    rows = []
    for name, path in datasets:
        analyses = load_analyses(path)
        for cell, label in CELLS:
            cells = [entity_share(analyses, a, cell) for a in ACCENTS]
            rows.append((name, label, cells))

    # Plain-text view
    header = f"{'Dataset':<18}{'Method':<18}" + "".join(
        f"{a.upper():>13}" for a in ACCENTS
    )
    print()
    print(header)
    print("-" * len(header))
    last = None
    for name, label, cells in rows:
        shown = name if name != last else ""
        last = name
        line = f"{shown:<18}{label:<18}"
        for pct, n in cells:
            line += f"{'--':>13}" if pct is None else f"{f'{pct}% ({n})':>13}"
        print(line)

    # LaTeX view, matching paper Table 8
    print("\n=== LaTeX ===\n")
    print(r"\begin{table*}[t]")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{llcccc}")
    print(r"    \toprule")
    print(
        r"    \textbf{Dataset} & \textbf{Method} & \textbf{US} & \textbf{IN} "
        r"& \textbf{PH} & \textbf{NG} \\"
    )
    print(r"    \midrule")
    for i, (name, label, cells) in enumerate(rows):
        if i % len(CELLS) == 0:
            if i:
                print(r"    \midrule")
            print(r"    \multirow{%d}{*}{%s}" % (len(CELLS), name))
        vals = " & ".join(
            "--" if pct is None else rf"{pct}\% ({n})" for pct, n in cells
        )
        print(rf"    & {label:<16}& {vals} \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(
        r"  \caption{Entity-corruption rate within degradation cases for each "
        r"RAG method" + "\n"
        r"and accent, with the number of degradation cases in parentheses. "
        r"Cases can" + "\n" + r"carry multiple labels.}"
    )
    print(r"  \label{tab:cross_method_entity}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
