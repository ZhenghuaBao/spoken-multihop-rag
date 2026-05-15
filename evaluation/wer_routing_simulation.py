"""
WER-threshold routing simulation on 2WikiMultiHopQA NG.

Strategy: route queries with WER >= 20% to Naive RAG (dense, robust),
          others stay on HippoRAG2 (better on clean text).
Compare against always-HippoRAG2 and always-Naive baselines.

Numbers reported per question, all from 2wiki_1000.json.
"""

import json
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
with open(_ROOT / "results" / "2wiki_1000.json", encoding="utf-8") as f:
    d = json.load(f)

# Per-question F1 maps
naive_results = d["accent_detailed_results"]["ng"]["A"]
hippo_results = d["accent_detailed_results"]["ng"]["C"]

# Build a routing-friendly list of (qid, wer, naive_f1, hippo_f1)
naive_by_id = {r["id"]: r for r in naive_results}
hippo_by_id = {r["id"]: r for r in hippo_results}

rows = []
for qid in naive_by_id:
    if qid not in hippo_by_id:
        continue
    n = naive_by_id[qid]
    h = hippo_by_id[qid]
    wer = n.get("wer", 0) or 0
    rows.append((qid, wer, n.get("f1", 0), h.get("f1", 0)))

n = len(rows)
print(f"N = {n} questions\n")


def avg(lst):
    return sum(lst) / len(lst) if lst else 0


# Baselines
naive_f1 = avg([r[2] for r in rows])
hippo_f1 = avg([r[3] for r in rows])
print(f"Always-Naive RAG F1:   {naive_f1:.4f}")
print(f"Always-HippoRAG2 F1:   {hippo_f1:.4f}")
print()

# Routing strategies at various thresholds
for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    routed_f1 = []
    n_to_naive = 0
    for _, wer, n_f1, h_f1 in rows:
        if wer >= thresh:
            routed_f1.append(n_f1)
            n_to_naive += 1
        else:
            routed_f1.append(h_f1)
    rf1 = avg(routed_f1)
    pct_naive = n_to_naive / n * 100
    print(
        f"Route to Naive if WER >= {thresh:.0%}: "
        f"F1 = {rf1:.4f}  "
        f"(\u0394 vs always-Hippo: {rf1 - hippo_f1:+.4f}, "
        f"{pct_naive:.1f}% routed to Naive)"
    )

# Oracle ceiling: per-question pick whichever method wins
oracle_route_f1 = avg([max(n_f1, h_f1) for _, _, n_f1, h_f1 in rows])
print(
    f"\nOracle routing (per-q best):  {oracle_route_f1:.4f}  "
    f"({oracle_route_f1 - hippo_f1:+.4f} over always-Hippo)"
)
