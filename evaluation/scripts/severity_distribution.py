"""
Severity-distribution and conditional-analysis script for paper Section V-G.

Reproduces:
  (1) Severity tier distribution: bucket each gold entity by character edit
      distance between the entity and its closest contiguous span in the NG
      ASR transcript.
        bucket bounds: 0 (exact), 1-3 (soft), 4-7 (moderate), 8+ (severe)

  (2) Conditional analysis: partition modified queries by whether phonetic
      correction moves the query closer to / farther from the gold by
      character Levenshtein distance, then report mean F1 delta on Naive
      and HippoRAG2.

Inputs:
  - Synth 2Wiki gold + NG: data/2wiki_spoken/accent_nbest_results_2wiki.json
  - Phonetic-corrected NG queries: data/2wiki_spoken/accent_nbest_results_2wiki_phonetic_v2_full.json
  - Per-question F1 (orig NG): results/2wiki_1000.json
  - Per-question F1 (phonetic-corrected NG): results/2wiki_1000_phonetic_v2_corr.json

Output: results/severity_and_conditional_analysis.json

Usage:
    python evaluation/scripts/severity_distribution.py

All input/output paths default to the repository-relative locations above
and can be overridden with --synth-ng / --corrected / --baseline-results /
--phonetic-results / --output.
"""

import argparse
import json
import sys
import io
from pathlib import Path

import spacy

_ROOT = Path(__file__).resolve().parent.parent.parent  # repository root

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)


def lev(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def closest_span_dist(entity: str, hyp: str) -> int:
    """Minimum character edit distance between entity and any contiguous
    span of length len(entity) +/- 2 in hypothesis. Case-insensitive."""
    e = entity.lower()
    h = hyp.lower()
    if e in h:
        return 0
    L = len(e)
    best = L
    for w_len in (L, L - 1, L + 1, L - 2, L + 2):
        if w_len <= 0:
            continue
        for i in range(len(h) - w_len + 1):
            d = lev(e, h[i : i + w_len])
            if d < best:
                best = d
            if best == 0:
                return 0
    return best


def severity_bucket(d: int) -> str:
    if d == 0:
        return "exact"
    if d <= 3:
        return "soft"
    if d <= 7:
        return "moderate"
    return "severe"


def run_severity(nlp, synth_ng_path):
    print(f"\n[Severity] loading {synth_ng_path}")
    with open(synth_ng_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  n questions: {len(data)}")

    counts = {"exact": 0, "soft": 0, "moderate": 0, "severe": 0}
    n_entities = 0

    for i, item in enumerate(data):
        ref = item.get("question", "")
        hyp = item.get("accents", {}).get("ng", {}).get("top1", "")
        if not ref or not hyp:
            continue
        doc = nlp(ref)
        for ent in doc.ents:
            d = closest_span_dist(ent.text, hyp)
            counts[severity_bucket(d)] += 1
            n_entities += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(data)}]")

    corrupted_total = counts["soft"] + counts["moderate"] + counts["severe"]
    return {
        "n_entities": n_entities,
        "counts": counts,
        "rates_over_total": {k: v / max(n_entities, 1) for k, v in counts.items()},
        "n_corrupted": corrupted_total,
        "rates_over_corrupted": {
            k: counts[k] / max(corrupted_total, 1)
            for k in ["soft", "moderate", "severe"]
        },
        "bucket_bounds": {
            "exact": "0",
            "soft": "1-3",
            "moderate": "4-7",
            "severe": "8+",
        },
    }


def run_conditional(orig_path, corr_path, main_path, phonetic_rag_path):
    """Partition modified queries by whether correction moves closer/farther
    from gold (character Levenshtein, case-insensitive on full query),
    then report mean F1 delta for Naive (Cell A) and HippoRAG2 (Cell C)."""
    print("\n[Conditional] loading queries...")
    with open(orig_path, encoding="utf-8") as f:
        orig = json.load(f)
    with open(corr_path, encoding="utf-8") as f:
        corr = json.load(f)
    with open(main_path, encoding="utf-8") as f:
        main = json.load(f)
    with open(phonetic_rag_path, encoding="utf-8") as f:
        pho = json.load(f)

    orig_by_id = {it["id"]: it for it in orig}
    corr_by_id = {it["id"]: it for it in corr}

    closer = set()
    farther = set()
    unchanged = set()
    unmodified = 0
    for qid, o in orig_by_id.items():
        c = corr_by_id.get(qid)
        if c is None:
            continue
        o_ng = o["accents"].get("ng", {}).get("top1", "")
        c_ng = c["accents"].get("ng", {}).get("top1", "")
        gold = o.get("question", "")
        if o_ng == c_ng:
            unmodified += 1
            continue
        d_o = lev(o_ng.lower(), gold.lower())
        d_c = lev(c_ng.lower(), gold.lower())
        if d_c < d_o:
            closer.add(qid)
        elif d_c > d_o:
            farther.add(qid)
        else:
            unchanged.add(qid)

    n_modified = len(closer) + len(farther) + len(unchanged)
    print(f"  modified: {n_modified}/{len(orig_by_id)}")
    print(f"  closer/farther/unchanged: {len(closer)}/{len(farther)}/{len(unchanged)}")

    def fmap(j, accent, cell):
        return {
            r["id"]: r.get("f1", 0) for r in j["accent_detailed_results"][accent][cell]
        }

    orig_naive = fmap(main, "ng", "A")
    orig_hippo = fmap(main, "ng", "C")
    pho_naive = fmap(pho, "ng", "A")
    pho_hippo = fmap(pho, "ng", "C")

    def avg_delta(ids, base_map, post_map):
        deltas = []
        for qid in ids:
            if qid in base_map and qid in post_map:
                deltas.append(post_map[qid] - base_map[qid])
        return sum(deltas) / len(deltas) if deltas else 0, len(deltas)

    def avg_pair(ids, base_map, post_map):
        b, p = [], []
        for qid in ids:
            if qid in base_map and qid in post_map:
                b.append(base_map[qid])
                p.append(post_map[qid])
        return (sum(b) / len(b) if b else 0, sum(p) / len(p) if p else 0, len(b))

    out = {
        "n_total_queries": len(orig_by_id),
        "n_unmodified": unmodified,
        "n_modified": n_modified,
        "modification_rate": n_modified / max(len(orig_by_id), 1),
        "partition_counts": {
            "closer": len(closer),
            "farther": len(farther),
            "unchanged": len(unchanged),
        },
        "partition_rates_over_modified": {
            "closer": len(closer) / max(n_modified, 1),
            "farther": len(farther) / max(n_modified, 1),
            "unchanged": len(unchanged) / max(n_modified, 1),
        },
    }

    for label, ids in [
        ("closer", closer),
        ("farther", farther),
        ("unchanged", unchanged),
    ]:
        d_naive, n_n = avg_delta(ids, orig_naive, pho_naive)
        d_hippo, n_h = avg_delta(ids, orig_hippo, pho_hippo)
        b_h, p_h, _ = avg_pair(ids, orig_hippo, pho_hippo)
        b_n, p_n, _ = avg_pair(ids, orig_naive, pho_naive)
        out[f"{label}_subset"] = {
            "n_naive": n_n,
            "n_hippo": n_h,
            "naive_mean_delta": d_naive,
            "hippo_mean_delta": d_hippo,
            "naive_baseline_F1": b_n,
            "naive_post_F1": p_n,
            "hippo_baseline_F1": b_h,
            "hippo_post_F1": p_h,
        }

    return out


def _parse_args():
    _data = _ROOT / "data" / "2wiki_spoken"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--synth-ng",
        type=Path,
        default=_data / "accent_nbest_results_2wiki.json",
        help="2WikiMultiHopQA accent transcripts (gold question + accents.ng.top1)",
    )
    p.add_argument(
        "--corrected",
        type=Path,
        default=_data / "accent_nbest_results_2wiki_phonetic_v2_full.json",
        help="Phonetic-corrected NG queries",
    )
    p.add_argument(
        "--baseline-results",
        type=Path,
        default=_ROOT / "results" / "2wiki_1000.json",
        help="Per-question F1 for the original NG condition",
    )
    p.add_argument(
        "--phonetic-results",
        type=Path,
        default=_ROOT / "results" / "2wiki_1000_phonetic_v2_corr.json",
        help="Per-question F1 after phonetic correction",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "results" / "severity_and_conditional_analysis.json",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    synth_ng_path = args.synth_ng
    corr_path = args.corrected
    main_path = args.baseline_results
    pho_path = args.phonetic_results
    output_path = args.output

    required = {
        "--synth-ng": synth_ng_path,
        "--corrected": corr_path,
        "--baseline-results": main_path,
        "--phonetic-results": pho_path,
    }
    for label, path in required.items():
        if not path.exists():
            raise SystemExit(
                f"Missing input {label}: {path}\n"
                "Transcripts and per-question results are not bundled with "
                "the code. See the 'Datasets' section of README.md."
            )

    print("Loading spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")

    severity = run_severity(nlp, synth_ng_path)
    conditional = run_conditional(synth_ng_path, corr_path, main_path, pho_path)

    out = {
        "methodology": {
            "severity": "spaCy NER on gold question, then closest-span "
            "char Levenshtein to NG hypothesis (window L-2..L+2). "
            "Buckets: 0=exact, 1-3=soft, 4-7=moderate, 8+=severe.",
            "conditional": "Partition modified queries by full-query "
            "character Levenshtein to gold (case-insensitive). "
            "Closer = corrected dist < orig dist; "
            "Farther = corrected dist > orig dist; "
            "Unchanged = equal.",
        },
        "severity_distribution": severity,
        "conditional_analysis": conditional,
        "sources": {
            "synth_ng_questions": str(synth_ng_path),
            "phonetic_corrected_ng_queries": str(corr_path),
            "main_ng_per_question_f1": str(main_path),
            "phonetic_ng_per_question_f1": str(pho_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("SEVERITY DISTRIBUTION")
    print("=" * 72)
    s = severity
    n = s["n_entities"]
    print(f"Total entities: {n}")
    for k in ["exact", "soft", "moderate", "severe"]:
        c = s["counts"][k]
        print(f"  {k:9s} ({s['bucket_bounds'][k]:>4s}): {c} ({c / n * 100:.1f}%)")
    nc = s["n_corrupted"]
    print(f"\nOf {nc} corrupted (= soft+moderate+severe):")
    for k in ["soft", "moderate", "severe"]:
        c = s["counts"][k]
        print(f"  {k:9s}: {c} ({c / nc * 100:.1f}%)")

    print()
    print("=" * 72)
    print("CONDITIONAL ANALYSIS")
    print("=" * 72)
    c = conditional
    print(f"Total queries: {c['n_total_queries']}")
    print(f"  unmodified:   {c['n_unmodified']}")
    print(f"  modified:     {c['n_modified']} ({c['modification_rate'] * 100:.1f}%)")
    pr = c["partition_rates_over_modified"]
    pc = c["partition_counts"]
    print(f"  closer:       {pc['closer']} ({pr['closer'] * 100:.1f}%)")
    print(f"  farther:      {pc['farther']} ({pr['farther'] * 100:.1f}%)")
    print(f"  unchanged:    {pc['unchanged']} ({pr['unchanged'] * 100:.1f}%)")
    for label in ["closer", "farther"]:
        sub = c[f"{label}_subset"]
        print(f"\n  {label} subset (n={sub['n_hippo']}):")
        print(
            f"    HippoRAG2: F1 {sub['hippo_baseline_F1']:.4f} -> "
            f"{sub['hippo_post_F1']:.4f}, "
            f"delta = {sub['hippo_mean_delta']:+.4f}"
        )
        print(
            f"    Naive RAG: F1 {sub['naive_baseline_F1']:.4f} -> "
            f"{sub['naive_post_F1']:.4f}, "
            f"delta = {sub['naive_mean_delta']:+.4f}"
        )

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
