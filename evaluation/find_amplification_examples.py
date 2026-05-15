"""Find error amplification examples:
1. Oracle + US correct
2. NG/IN/PH transcription error
3. Error causes RAG failure
4. Preferably HippoRAG/IRCoT fails but Naive still partially succeeds
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import f1_score  # noqa: E402

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_results(src, accent, cell):
    if accent in src and cell in src[accent]:
        return {r["id"]: r for r in src[accent][cell]}
    return {}


def analyze_dataset(ds_name, result_path, asr_path):
    data = load(result_path)
    asr_raw = load(asr_path)
    asr = {item["id"]: item for item in asr_raw}
    src = data["accent_detailed_results"]

    # Build index for all accent/cell combos
    cells_needed = ["A", "C", "G", "H"]
    idx = {}
    for acc in ["oracle", "us", "in", "ph", "ng"]:
        for cell in ["E", "F", "G", "H"] if acc == "oracle" else cells_needed:
            idx[(acc, cell)] = index_results(src, acc, cell)

    def get_f1(accent, cell, qid, gt):
        rec = idx.get((accent, cell), {}).get(qid)
        if rec is None:
            return -1
        return f1_score(rec["answer"], gt)

    def get_ans(accent, cell, qid):
        rec = idx.get((accent, cell), {}).get(qid)
        return rec["answer"] if rec else "N/A"

    all_qids = set(idx[("oracle", "E")].keys()) & set(idx[("us", "A")].keys())

    found = []
    for qid in sorted(all_qids):
        gt = idx[("oracle", "E")][qid]["ground_truth"]

        # Oracle correct?
        orc_E = get_f1("oracle", "E", qid, gt)
        orc_F = get_f1("oracle", "F", qid, gt)
        orc_G = get_f1("oracle", "G", qid, gt)
        orc_H = get_f1("oracle", "H", qid, gt)
        if max(orc_E, orc_F) < 0.5:
            continue

        # US correct on Naive?
        us_A = get_f1("us", "A", qid, gt)
        if us_A < 0.5:
            continue

        us_C = get_f1("us", "C", qid, gt)

        for bad_acc, bad_label in [("ng", "NG"), ("in", "IN"), ("ph", "PH")]:
            if qid not in idx.get((bad_acc, "A"), {}):
                continue

            ba = get_f1(bad_acc, "A", qid, gt)
            bc = get_f1(bad_acc, "C", qid, gt)
            bg = get_f1(bad_acc, "G", qid, gt)
            bh = get_f1(bad_acc, "H", qid, gt)

            patterns = []

            # Naive ok but HippoRAG fails
            if ba >= 0.3 and bc >= 0 and bc < 0.2:
                patterns.append(
                    f"Naive({bad_label})={ba:.2f} OK but HippoRAG({bad_label})={bc:.2f} FAIL"
                )

            # Naive ok but IRCoT fails
            if ba >= 0.3 and bg >= 0 and bg < 0.2:
                patterns.append(
                    f"Naive({bad_label})={ba:.2f} OK but IRCoT+N({bad_label})={bg:.2f} FAIL"
                )
            if ba >= 0.3 and bh >= 0 and bh < 0.2:
                patterns.append(
                    f"Naive({bad_label})={ba:.2f} OK but IRCoT+H({bad_label})={bh:.2f} FAIL"
                )

            # All NG cells fail but US all ok
            if ba < 0.3 and bc < 0.3 and us_A >= 0.5 and us_C >= 0.5:
                patterns.append(
                    f"ALL {bad_label} fail (A={ba:.2f},C={bc:.2f}) but US ok (A={us_A:.2f},C={us_C:.2f})"
                )

            if not patterns:
                continue

            asr_info = asr.get(qid, {}).get("accents", {})
            us_top1 = asr_info.get("us", {}).get("top1", "?")
            bad_top1 = asr_info.get(bad_acc, {}).get("top1", "?")
            wer = idx[(bad_acc, "A")][qid].get("wer", 0)

            found.append(
                {
                    "qid": qid,
                    "accent": bad_label,
                    "gt": gt,
                    "original": idx[(bad_acc, "A")][qid].get("original", ""),
                    "us_asr": us_top1,
                    "bad_asr": bad_top1,
                    "wer": wer,
                    "patterns": patterns,
                    "f1": {
                        "Oracle+Naive": orc_E,
                        "Oracle+Hippo": orc_F,
                        "Oracle+IRCoT+N": orc_G,
                        "Oracle+IRCoT+H": orc_H,
                        "US+Naive": us_A,
                        "US+Hippo": us_C,
                        f"{bad_label}+Naive": ba,
                        f"{bad_label}+Hippo": bc,
                        f"{bad_label}+IRCoT+N": bg,
                        f"{bad_label}+IRCoT+H": bh,
                    },
                    "answers": {
                        "oracle_E": get_ans("oracle", "E", qid),
                        "oracle_F": get_ans("oracle", "F", qid),
                        "us_A": get_ans("us", "A", qid),
                        "us_C": get_ans("us", "C", qid),
                        f"{bad_label}_A": get_ans(bad_acc, "A", qid),
                        f"{bad_label}_C": get_ans(bad_acc, "C", qid),
                        f"{bad_label}_G": get_ans(bad_acc, "G", qid),
                        f"{bad_label}_H": get_ans(bad_acc, "H", qid),
                    },
                }
            )

    found.sort(key=lambda x: (-len(x["patterns"]), -x["wer"]))

    print(f"\n{'=' * 80}")
    print(f"  {ds_name}: Found {len(found)} error amplification examples")
    print(f"{'=' * 80}")

    for i, ex in enumerate(found[:20]):
        print(f"\n  --- Example {i + 1} [{ex['accent']}] (WER={ex['wer']:.1%}) ---")
        print(f"  QID: {ex['qid']}")
        print(f"  GT:  {ex['gt']}")
        print(f"  Original: {ex['original'][:120]}")
        print(f"  US ASR:   {ex['us_asr'][:120]}")
        print(f"  {ex['accent']} ASR:   {ex['bad_asr'][:120]}")
        print(f"  Patterns: {'; '.join(ex['patterns'])}")
        f1s = ex["f1"]
        print("  F1: " + " | ".join(f"{k}={v:.2f}" for k, v in f1s.items() if v >= 0))
        ans = ex["answers"]
        print(f"  Oracle(E)={ans['oracle_E']}  US(A)={ans['us_A']}")
        print(
            f"  {ex['accent']}(A)={ans[ex['accent'] + '_A']}  {ex['accent']}(C)={ans[ex['accent'] + '_C']}  {ex['accent']}(G)={ans.get(ex['accent'] + '_G', '?')}  {ex['accent']}(H)={ans.get(ex['accent'] + '_H', '?')}"
        )

    return found


all_found = {}

# 2Wiki
f = analyze_dataset(
    "2WikiMultiHopQA",
    "results/experiment_all+oracle_20260330_224620.json",
    "data/accent_nbest_results_2wiki.json",
)
all_found["2wiki"] = f

# MuSiQue
f = analyze_dataset(
    "MuSiQue",
    "results/musique_all.json",
    "data/accent_nbest_results_musique.json",
)
all_found["musique"] = f

# Summary
print(f"\n{'=' * 80}")
print("  SUMMARY")
print(f"{'=' * 80}")
for ds, examples in all_found.items():
    print(f"  {ds}: {len(examples)} examples")
    by_accent = {}
    for ex in examples:
        by_accent.setdefault(ex["accent"], []).append(ex)
    for acc, exs in sorted(by_accent.items()):
        print(f"    {acc}: {len(exs)}")

# Save all
with open("amplification_examples.json", "w", encoding="utf-8") as f:
    json.dump(all_found, f, indent=2, ensure_ascii=False, default=str)
print("\nSaved to amplification_examples.json")
