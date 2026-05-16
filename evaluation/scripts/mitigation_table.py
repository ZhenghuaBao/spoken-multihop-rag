"""
Produce the full Mitigation table (paper Table 3): top-1 baseline,
N-best Decoding column, Phonetic Entity Correction column, recovery
percentages, and the matching LaTeX block.

Requires up to three result files:
  --baseline-results  top-1 baseline (cells A, C, G, H + oracle E/F)
  --nbest-results     N-best Decoding variants (cells B, D, I, J);
                      may be the same file as --baseline-results if
                      both top-1 and N-best cells were run together
  --phonetic-results  same-format file produced by re-running run_2x2
                      with --accent-json pointed at the
                      phonetic-corrected query JSON

Each mitigation column is independent: N-best comes from cells B/D/I/J
(top-k union over decoded hypotheses); Phonetic comes from cells
A/C/G/H in the phonetic-corrected file (single corrected query through
the same retriever as the top-1 baseline).

Usage:
    python evaluation/scripts/mitigation_table.py \\
        --baseline-results results/2wiki_1000.json \\
        --nbest-results results/2wiki_1000_nbest.json \\
        --phonetic-results results/2wiki_1000_phonetic_v2_corr.json \\
        --accent ng
"""

import argparse
import io
import json
import sys

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell_avg_f1(detailed: dict, accent: str, cell: str) -> float:
    """Average F1 for accent/cell from accent_detailed_results, or -1 if missing."""
    if accent not in detailed or cell not in detailed[accent]:
        return -1.0
    rows = detailed[accent][cell]
    if not rows:
        return -1.0
    return sum(r.get("f1", 0.0) for r in rows) / len(rows)


def recovery(top1: float, mitig: float, oracle: float) -> float:
    """Fraction of the Oracle--top-1 gap closed by mitigation, in percent."""
    gap_top1 = top1 - oracle
    gap_mitig = mitig - oracle
    if gap_top1 == 0:
        return 0.0
    return (1 - abs(gap_mitig) / abs(gap_top1)) * 100


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline-results",
        required=True,
        help="Top-1 baseline JSON (cells A/C/G/H + oracle E/F)",
    )
    p.add_argument(
        "--nbest-results",
        default=None,
        help="N-best Decoding JSON (cells B/D/I/J); defaults to "
        "--baseline-results if both top-1 and N-best are in one file",
    )
    p.add_argument(
        "--phonetic-results",
        required=True,
        help="Same-format JSON from re-running run_2x2 with phonetic-corrected queries",
    )
    p.add_argument("--accent", default="ng")
    args = p.parse_args()

    base = load(args.baseline_results)["accent_detailed_results"]
    nbest_path = args.nbest_results or args.baseline_results
    nbest = load(nbest_path)["accent_detailed_results"]
    phon = load(args.phonetic_results)["accent_detailed_results"]

    # Per-method rows: (label, top-1 cell, n-best cell, phonetic cell, oracle cell)
    # IRCoT methods (G, H) use their own oracle cells under accent="oracle",
    # not cells E/F which are the single-shot Naive/HippoRAG oracles.
    methods = [
        ("Naive RAG", "A", "B", "A", "E"),
        ("HippoRAG2", "C", "D", "C", "F"),
        ("IRCoT+Naive", "G", "I", "G", "G"),
        ("IRCoT+Hippo", "H", "J", "H", "H"),
    ]

    print(f"\n=== Mitigation Table (Accent: {args.accent.upper()}) ===\n")
    header = (
        f"{'Method':<14}{'top-1':>9}{'N-best':>9}{'Phon':>9}"
        f"{'Δtop1':>9}{'ΔN':>9}{'ΔPh':>9}{'RecN':>9}{'RecPh':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, c1, cn, cp, oc in methods:
        oracle = cell_avg_f1(base, "oracle", oc)
        top1 = cell_avg_f1(base, args.accent, c1)
        nbest_f1 = cell_avg_f1(nbest, args.accent, cn)
        phonetic = cell_avg_f1(phon, args.accent, cp)
        if top1 < 0 or oracle < 0:
            print(f"{name:<14} (data missing)")
            continue
        d1 = top1 - oracle
        dn = nbest_f1 - oracle if nbest_f1 >= 0 else None
        dp = phonetic - oracle if phonetic >= 0 else None
        rn = recovery(top1, nbest_f1, oracle) if nbest_f1 >= 0 else None
        rp = recovery(top1, phonetic, oracle) if phonetic >= 0 else None

        def fmt(x, w=9, sign=False, pct=False):
            if x is None:
                return f"{'—':>{w}}"
            if pct:
                return f"{x:>{w - 1}.1f}%"
            return f"{x:+{w}.3f}" if sign else f"{x:{w}.3f}"

        print(
            f"{name:<14}"
            f"{fmt(top1)}{fmt(nbest_f1)}{fmt(phonetic)}"
            f"{fmt(d1, sign=True)}{fmt(dn, sign=True)}{fmt(dp, sign=True)}"
            f"{fmt(rn, pct=True)}{fmt(rp, pct=True)}"
        )
        rows.append((name, top1, nbest_f1, phonetic, oracle, d1, dn, dp, rn, rp))

    # LaTeX
    print("\n=== LaTeX ===\n")
    print(r"\begin{table}[t]")
    print(
        r"\caption{Mitigation table on \textsc{Dataset} under "
        + f"{args.accent.upper()} speech. N-best Decoding ($N=5$) and Phonetic "
        + "Entity Correction are each applied before retrieval; Recovery is "
        + "the fraction of the Oracle--top-1 F1 gap closed by the mitigation.}"
    )
    print(r"\label{tab:mitigation}")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(
        r" & \multicolumn{1}{c}{\textbf{top-1}}"
        r" & \multicolumn{2}{c}{\textbf{N-best Decoding}}"
        r" & \multicolumn{2}{c}{\textbf{Phonetic Correction}}"
        r" & & \\"
    )
    print(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    print(r"\textbf{Method} & F1 & F1 & Rec. & F1 & Rec. & & \\")
    print(r"\midrule")
    for name, top1, nb, ph, _, _, _, _, rn, rp in rows:
        nb_s = f"{nb:.3f}" if nb >= 0 else "—"
        ph_s = f"{ph:.3f}" if ph >= 0 else "—"
        rn_s = f"{rn:.1f}\\%" if rn is not None else "—"
        rp_s = f"{rp:.1f}\\%" if rp is not None else "—"
        print(f"{name:<14} & {top1:.3f} & {nb_s} & {rn_s} & {ph_s} & {rp_s} & & \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
