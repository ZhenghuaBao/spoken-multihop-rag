"""
Hop-count breakdown analysis for MuSiQue.

MuSiQue ground truth contains a `num_hops` field (2, 3, or 4). This script
groups F1 by num_hops and reports degradation per accent and method, to
demonstrate that more hops -> larger gap (more entities to corrupt).

Usage:
    python evaluation/hop_count_breakdown.py \
        --results results/musique_1000.json \
        --ground-truth ../dataset/musique_1000/ground_truth.json
"""

import argparse
import json
import sys
import io
from collections import defaultdict

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

ACCENTS = ["us", "in", "ph", "ng"]
METHOD_CELL = {
    "Naive RAG": "A",
    "HippoRAG": "C",
    "IRCoT+Naive": "G",
    "IRCoT+Hippo": "H",
}
ASR_TO_ORACLE = {"A": "E", "C": "F", "G": "G", "H": "H"}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_per_q_f1(detailed: dict, accent: str, cell: str) -> dict:
    if accent not in detailed or cell not in detailed[accent]:
        return {}
    return {r["id"]: r.get("f1", 0.0) for r in detailed[accent][cell]}


def get_oracle_f1(detailed: dict, oracle_cell: str) -> dict:
    if "oracle" in detailed and oracle_cell in detailed["oracle"]:
        return {r["id"]: r.get("f1", 0.0) for r in detailed["oracle"][oracle_cell]}
    for acc in ACCENTS:
        if acc in detailed and oracle_cell in detailed[acc]:
            return {r["id"]: r.get("f1", 0.0) for r in detailed[acc][oracle_cell]}
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--ground-truth", required=True)
    args = p.parse_args()

    data = load(args.results)
    gt = load(args.ground_truth)
    detailed = data["accent_detailed_results"]

    # Group qids by num_hops
    qids_by_hop = defaultdict(list)
    for qid, entry in gt.items():
        n = entry.get("num_hops", 0)
        if n > 0:
            qids_by_hop[n].append(qid)

    print("\nQuestions per hop count:")
    for n_hops in sorted(qids_by_hop.keys()):
        print(f"  {n_hops}-hop: {len(qids_by_hop[n_hops])}")

    # Per-method, per-hop breakdown of Oracle/NG/gap
    for method_name, cell in METHOD_CELL.items():
        oracle_cell = ASR_TO_ORACLE[cell]
        oracle_f1 = get_oracle_f1(detailed, oracle_cell)

        print()
        print("=" * 64)
        print(f"  {method_name}  (cell {cell} vs oracle {oracle_cell})")
        print("=" * 64)
        header = f"{'Hops':<8}{'n':>5}{'Oracle':>9}"
        for acc in ACCENTS:
            header += f"{acc.upper():>9}"
        header += f"{'Δ NG':>9}"
        print(header)
        print("-" * 64)

        for n_hops in sorted(qids_by_hop.keys()):
            qids = qids_by_hop[n_hops]
            row_qids = [q for q in qids if q in oracle_f1]
            if not row_qids:
                continue
            o_avg = sum(oracle_f1[q] for q in row_qids) / len(row_qids)
            row = f"{n_hops}-hop   {len(row_qids):>5}{o_avg:>9.3f}"
            ng_avg = None
            for acc in ACCENTS:
                acc_f1 = get_per_q_f1(detailed, acc, cell)
                vals = [acc_f1[q] for q in row_qids if q in acc_f1]
                avg = sum(vals) / len(vals) if vals else 0.0
                row += f"{avg:>9.3f}"
                if acc == "ng":
                    ng_avg = avg
            gap = (ng_avg - o_avg) if ng_avg is not None else 0
            row += f"{gap:>9.3f}"
            print(row)

    # Compact LaTeX-ready summary table (NG only, all methods)
    print()
    print("=" * 64)
    print("  Compact summary: NG vs Oracle by hop count")
    print("=" * 64)
    print(f"{'Hops':<8}{'n':>5}", end="")
    for m in METHOD_CELL:
        print(f"{m + ' Δ':>16}", end="")
    print()
    print("-" * 64)
    for n_hops in sorted(qids_by_hop.keys()):
        qids = qids_by_hop[n_hops]
        line = f"{n_hops}-hop   {len(qids):>5}"
        for method_name, cell in METHOD_CELL.items():
            oracle_cell = ASR_TO_ORACLE[cell]
            oracle_f1 = get_oracle_f1(detailed, oracle_cell)
            ng_f1 = get_per_q_f1(detailed, "ng", cell)
            common = [q for q in qids if q in oracle_f1 and q in ng_f1]
            if not common:
                line += f"{'-':>16}"
                continue
            o = sum(oracle_f1[q] for q in common) / len(common)
            n = sum(ng_f1[q] for q in common) / len(common)
            line += f"{(n - o):>16.3f}"
        print(line)


if __name__ == "__main__":
    main()
