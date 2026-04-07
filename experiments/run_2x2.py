"""
2x2x4 Experiment: ASR (top-1 vs N-best) x Retrieval (Naive RAG vs HippoRAG) x Accent

              | Oracle    | ASR top-1 | ASR N-best |
  Naive RAG   |     E     |     A     |     B      |
  HippoRAG    |     F     |     C     |     D      |

Usage:
    python experiments/run_2x2.py --cells E F
"""

import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List

# Setup path
_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _ROOT.parent  # DualRAG root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from retrieval.naive_rag import NaiveRAG  # noqa: E402
from evaluation.metrics import exact_match, f1_score  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ACCENTS = ["us", "in", "ph", "ng"]


def load_accent_results(path: Path) -> List[Dict]:
    """Load accent_nbest_results.json (entries with .accents.{us,in,ph,ng})."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {path}")
    return data


def load_ground_truth(path: Path) -> Dict[str, Dict]:
    """Load ground truth answers from ground_truth.json."""
    with open(path, encoding="utf-8") as f:
        gt = json.load(f)
    print(f"Loaded ground truth for {len(gt)} queries")
    return gt


def load_corpus_documents(docs_dir: Path) -> List[str]:
    """Load corpus documents from text files."""
    docs = []
    for doc_file in sorted(docs_dir.glob("*.txt")):
        with open(doc_file, encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                docs.append(content)
    print(f"Loaded {len(docs)} documents from {docs_dir}")
    return docs


def _sanitize_text(text: str) -> str:
    """Remove non-printable / non-UTF8 characters that break OpenAI API calls."""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = "".join(c for c in text if c.isprintable() or c == " ")
    return text.strip()


def prepare_oracle_transcriptions(asr_data: List[Dict]) -> List[Dict]:
    """Use original text as query (no ASR). Upper-bound baseline."""
    results = []
    for entry in asr_data:
        q = _sanitize_text(entry["question"])
        results.append(
            {
                "id": entry["id"],
                "original_text": q,
                "transcribed_text": q,  # oracle = original
            }
        )
    return results


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are a precise answer extraction assistant.

CRITICAL RULES:
1. Answer with ONLY the exact information requested
2. NO explanations, NO context, NO extra words
3. Be as concise as possible
4. If the answer is a name, give ONLY the name
5. If the answer is a number, give ONLY the number
6. If the answer is yes/no, give ONLY yes or no

Extract the answer from the context and respond with ONLY the answer."""


def generate_answer_openai(
    context: str, query: str, model: str = "gpt-4o-mini"
) -> Dict:
    """Generate answer using OpenAI API (same prompt for fair comparison)."""
    from openai import OpenAI

    client = OpenAI()

    context = _sanitize_text(context)
    query = _sanitize_text(query)

    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (be extremely concise):",
            },
        ],
        temperature=0,
        max_tokens=50,
        stream=True,
        stream_options={"include_usage": True},
    )

    parts = []
    ttft = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if not parts:
                ttft = time.time() - start
            parts.append(chunk.choices[0].delta.content)
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    return {
        "answer": "".join(parts).strip(),
        "generation_time": time.time() - start,
        "ttft": ttft,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Experiment cells: Oracle baseline
# ---------------------------------------------------------------------------


def run_cell_E(
    transcriptions_oracle: List[Dict],
    ground_truth: Dict,
    naive_rag: NaiveRAG,
    top_k: int = 10,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell E: Naive RAG + Oracle (original text, no ASR)."""
    print("\n" + "=" * 60)
    print("CELL E: Naive RAG + Oracle (text query)")
    print("=" * 60)

    results = []
    for i, t in enumerate(transcriptions_oracle):
        query = t["transcribed_text"]  # = original_text
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")

        ret = naive_rag.retrieve_top1(query, top_k=top_k)

        context = "\n\n".join(ret["docs"][:top_k])
        gen = generate_answer_openai(context, query, model=llm_model)

        em = exact_match(gen["answer"], gt)
        f1 = f1_score(gen["answer"], gt)

        results.append(
            {
                "id": qid,
                "query": query,
                "original": t["original_text"],
                "answer": gen["answer"],
                "ground_truth": gt,
                "em": em,
                "f1": f1,
                "wer": 0.0,
                "retrieval_time": ret["retrieval_time"],
                "generation_time": gen["generation_time"],
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            print(
                f"  [{i + 1}/{len(transcriptions_oracle)}] F1={avg_f1:.3f} EM={avg_em:.3f}"
            )

    return results


def run_cell_F(
    transcriptions_oracle: List[Dict],
    ground_truth: Dict,
    hipporag,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell F: HippoRAG + Oracle (original text, no ASR)."""
    print("\n" + "=" * 60)
    print("CELL F: HippoRAG + Oracle (text query)")
    print("=" * 60)

    results = []
    for i, t in enumerate(transcriptions_oracle):
        query = t["transcribed_text"]  # = original_text
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")

        ret = hipporag.retrieve_top1(query)

        context = "\n\n".join(ret["docs"])
        gen = generate_answer_openai(context, query, model=llm_model)

        em = exact_match(gen["answer"], gt)
        f1 = f1_score(gen["answer"], gt)

        results.append(
            {
                "id": qid,
                "query": query,
                "original": t["original_text"],
                "answer": gen["answer"],
                "ground_truth": gt,
                "em": em,
                "f1": f1,
                "wer": 0.0,
                "retrieval_time": ret["retrieval_time"],
                "generation_time": gen["generation_time"],
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            print(
                f"  [{i + 1}/{len(transcriptions_oracle)}] F1={avg_f1:.3f} EM={avg_em:.3f}"
            )

    return results


# ---------------------------------------------------------------------------
# Summary & output
# ---------------------------------------------------------------------------


def summarize_cell(name: str, results: List[Dict]) -> Dict:
    """Compute summary statistics for one cell."""
    n = len(results)
    if n == 0:
        return {"cell": name, "n": 0}

    return {
        "cell": name,
        "n": n,
        "em": sum(r["em"] for r in results) / n,
        "f1": sum(r["f1"] for r in results) / n,
        "wer": sum(r["wer"] for r in results) / n,
        "avg_retrieval_time": sum(r["retrieval_time"] for r in results) / n,
        "avg_generation_time": sum(r["generation_time"] for r in results) / n,
    }


def print_results_table(summaries: Dict[str, Dict]) -> None:
    """Print the results table."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for cell_name in ["E", "F"]:
        s = summaries.get(cell_name)
        if s and s["n"] > 0:
            print(f"  Cell {cell_name}: F1={s['f1']:.3f} EM={s['em']:.3f} (n={s['n']})")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run 2x2x4 spoken multi-hop QA experiment"
    )

    # Data paths
    parser.add_argument(
        "--accent-json",
        type=str,
        default=str(_ROOT / "data" / "accent_nbest_results.json"),
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=str(
            _PROJECT_ROOT / "dataset" / "hotpotqa_1000_hf" / "ground_truth.json"
        ),
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(_PROJECT_ROOT / "dataset" / "hotpotqa_1000_hf" / "documents"),
    )

    # Experiment config
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["E", "F"],
        help="Which cells to run (e.g., --cells E F)",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--sample", type=int, default=None, help="Only run first N questions"
    )

    # Index paths
    parser.add_argument(
        "--hipporag-dir",
        type=str,
        default=str(_PROJECT_ROOT / "hipporag_outputs" / "hotpotqa"),
    )
    parser.add_argument(
        "--naive-index",
        type=str,
        default=str(
            _PROJECT_ROOT
            / "vector_store"
            / "conventional_vector_chunk400_hotpotqa_1000_hf"
        ),
    )
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cells_to_run = set(c.upper() for c in args.cells)
    need_naive = bool(cells_to_run & {"E"})
    need_hipporag = bool(cells_to_run & {"F"})

    print("=" * 60)
    print("SPOKEN MULTI-HOP QA: ORACLE BASELINE")
    print("=" * 60)
    print(f"Cells: {sorted(cells_to_run)}")
    print(f"LLM: {args.llm_model}")

    # --- Load data ---
    asr_data = load_accent_results(Path(args.accent_json))
    ground_truth = load_ground_truth(Path(args.ground_truth))
    docs = load_corpus_documents(Path(args.docs_dir))

    if args.sample:
        asr_data = asr_data[: args.sample]
        print(f"  Using first {args.sample} questions only")

    transcriptions_oracle = prepare_oracle_transcriptions(asr_data)

    # --- Load indices ---
    naive_rag = None
    hipporag = None

    if need_naive:
        print("\n--- Loading Naive RAG index ---")
        index_path = args.naive_index
        naive_rag = NaiveRAG(index_path=index_path)
        if Path(index_path).exists() and not args.force_reindex:
            naive_rag.load_index(index_path)
        else:
            naive_rag.index_documents(docs, save_path=index_path)

    if need_hipporag:
        print("\n--- Loading HippoRAG index ---")
        from retrieval.hipporag import SpokenHippoRAG

        hipporag = SpokenHippoRAG(
            save_dir=args.hipporag_dir,
            llm_model=args.llm_model,
        )
        hipporag.initialize(docs, force_reindex=args.force_reindex)

    # --- Run cells ---
    all_results = {}
    summaries = {}

    if "E" in cells_to_run:
        results = run_cell_E(
            transcriptions_oracle,
            ground_truth,
            naive_rag,
            top_k=args.top_k,
            llm_model=args.llm_model,
        )
        all_results["E"] = results
        summaries["E"] = summarize_cell("E", results)

    if "F" in cells_to_run:
        results = run_cell_F(
            transcriptions_oracle, ground_truth, hipporag, llm_model=args.llm_model
        )
        all_results["F"] = results
        summaries["F"] = summarize_cell("F", results)

    print_results_table(summaries)

    # --- Save ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(
        _ROOT / f"results/experiment_oracle_{timestamp}.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "cells": sorted(cells_to_run),
                    "llm_model": args.llm_model,
                    "top_k": args.top_k,
                    "num_questions": len(asr_data),
                    "timestamp": timestamp,
                },
                "summaries": summaries,
                "detailed_results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
