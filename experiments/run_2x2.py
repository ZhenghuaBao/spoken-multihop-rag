"""
2x2x4 Experiment: ASR (top-1 vs N-best) x Retrieval (Naive RAG vs HippoRAG) x Accent

              | Oracle    | ASR top-1 | ASR N-best |
  Naive RAG   |     E     |     A     |     B      |
  IRCoT+Naive |     -     |     G     |     -      |
  HippoRAG    |     F     |     C     |     D      |
  IRCoT+Hippo |     -     |     H     |     -      |

Usage:
    python experiments/run_2x2.py --cells E F              # Oracle only
    python experiments/run_2x2.py --cells A C --accent us  # ASR top-1, US accent
    python experiments/run_2x2.py --cells A B C D --accent ng  # All Naive/HippoRAG cells
    python experiments/run_2x2.py --cells G H --accent us  # IRCoT, US accent
"""

import json
import math
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
from evaluation.metrics import exact_match, f1_score, word_error_rate  # noqa: E402


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


def prepare_top1_transcriptions(asr_data: List[Dict], accent: str) -> List[Dict]:
    """Extract top-1 transcriptions for a specific accent."""
    results = []
    for entry in asr_data:
        acc = entry["accents"][accent]
        results.append(
            {
                "id": entry["id"],
                "original_text": _sanitize_text(entry["question"]),
                "transcribed_text": _sanitize_text(acc["top1"]),
            }
        )
    return results


def prepare_nbest_transcriptions(asr_data: List[Dict], accent: str) -> List[Dict]:
    """Extract N-best transcriptions for a specific accent."""
    results = []
    for entry in asr_data:
        acc = entry["accents"][accent]
        hypotheses = []
        for i, hyp in enumerate(acc["nbest"]):
            score = hyp.get("avg_logprob", 0.0)
            hypotheses.append(
                {
                    "text": hyp["text"],
                    "score": score,
                    "rank": i,
                    "temperature": hyp.get("temperature", 0.0),
                }
            )

        # Compute normalized scores (softmax over avg_logprob)
        if len(hypotheses) > 1:
            scores = [h["score"] for h in hypotheses]
            max_s = max(scores)
            if all(s == scores[0] for s in scores):
                for h in hypotheses:
                    h["normalized_score"] = 1.0 / len(hypotheses)
            else:
                exp_scores = [math.exp(s - max_s) for s in scores]
                total = sum(exp_scores)
                for h, es in zip(hypotheses, exp_scores):
                    h["normalized_score"] = es / total
        elif hypotheses:
            hypotheses[0]["normalized_score"] = 1.0

        results.append(
            {
                "id": entry["id"],
                "original_text": entry["question"],
                "best_text": acc["top1"],
                "hypotheses": hypotheses,
                "num_hypotheses": len(hypotheses),
            }
        )
    return results


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
# Experiment cells: ASR top-1
# ---------------------------------------------------------------------------


def run_cell_A(
    transcriptions_top1: List[Dict],
    ground_truth: Dict,
    naive_rag: NaiveRAG,
    top_k: int = 10,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell A: Naive RAG + ASR top-1."""
    print("\n" + "=" * 60)
    print("CELL A: Naive RAG + ASR top-1")
    print("=" * 60)

    results = []
    for i, t in enumerate(transcriptions_top1):
        query = t["transcribed_text"]
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")

        ret = naive_rag.retrieve_top1(query, top_k=top_k)

        context = "\n\n".join(ret["docs"][:top_k])
        gen = generate_answer_openai(context, query, model=llm_model)

        em = exact_match(gen["answer"], gt)
        f1 = f1_score(gen["answer"], gt)
        wer = word_error_rate(query, t["original_text"])

        results.append(
            {
                "id": qid,
                "query": query,
                "original": t["original_text"],
                "answer": gen["answer"],
                "ground_truth": gt,
                "em": em,
                "f1": f1,
                "wer": wer,
                "retrieval_time": ret["retrieval_time"],
                "generation_time": gen["generation_time"],
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            print(
                f"  [{i + 1}/{len(transcriptions_top1)}] F1={avg_f1:.3f} EM={avg_em:.3f}"
            )

    return results


def run_cell_C(
    transcriptions_top1: List[Dict],
    ground_truth: Dict,
    hipporag,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell C: HippoRAG + ASR top-1."""
    print("\n" + "=" * 60)
    print("CELL C: HippoRAG + ASR top-1")
    print("=" * 60)

    results = []
    for i, t in enumerate(transcriptions_top1):
        query = t["transcribed_text"]
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")

        ret = hipporag.retrieve_top1(query)

        context = "\n\n".join(ret["docs"])
        gen = generate_answer_openai(context, query, model=llm_model)

        em = exact_match(gen["answer"], gt)
        f1 = f1_score(gen["answer"], gt)
        wer = word_error_rate(query, t["original_text"])

        results.append(
            {
                "id": qid,
                "query": query,
                "original": t["original_text"],
                "answer": gen["answer"],
                "ground_truth": gt,
                "em": em,
                "f1": f1,
                "wer": wer,
                "retrieval_time": ret["retrieval_time"],
                "generation_time": gen["generation_time"],
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            print(
                f"  [{i + 1}/{len(transcriptions_top1)}] F1={avg_f1:.3f} EM={avg_em:.3f}"
            )

    return results


# ---------------------------------------------------------------------------
# Experiment cells: ASR N-best
# ---------------------------------------------------------------------------


def run_cell_B(
    transcriptions_nbest: List[Dict],
    ground_truth: Dict,
    naive_rag: NaiveRAG,
    nbest_strategy: str = "union",
    top_k: int = 10,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell B: Naive RAG + ASR N-best."""
    print("\n" + "=" * 60)
    print(f"CELL B: Naive RAG + ASR N-best (strategy={nbest_strategy})")
    print("=" * 60)

    retrieval_fn = {
        "union": naive_rag.retrieve_nbest_union,
        "weighted": naive_rag.retrieve_nbest_weighted,
        "concat": naive_rag.retrieve_nbest_concat,
    }[nbest_strategy]

    results = []
    for i, t in enumerate(transcriptions_nbest):
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")
        hypotheses = t["hypotheses"]
        best_query = t["best_text"]

        ret = retrieval_fn(hypotheses, top_k=top_k)

        context = "\n\n".join(ret["docs"][:top_k])
        gen = generate_answer_openai(context, best_query, model=llm_model)

        em = exact_match(gen["answer"], gt)
        f1 = f1_score(gen["answer"], gt)
        wer = word_error_rate(best_query, t["original_text"])

        results.append(
            {
                "id": qid,
                "query": best_query,
                "original": t["original_text"],
                "num_hypotheses": len(hypotheses),
                "answer": gen["answer"],
                "ground_truth": gt,
                "em": em,
                "f1": f1,
                "wer": wer,
                "retrieval_time": ret["retrieval_time"],
                "generation_time": gen["generation_time"],
                "strategy": nbest_strategy,
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            print(
                f"  [{i + 1}/{len(transcriptions_nbest)}] F1={avg_f1:.3f} EM={avg_em:.3f}"
            )

    return results


def run_cell_D(
    transcriptions_nbest: List[Dict],
    ground_truth: Dict,
    hipporag,
    nbest_strategy: str = "union",
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell D: HippoRAG + ASR N-best."""
    print("\n" + "=" * 60)
    print(f"CELL D: HippoRAG + ASR N-best (strategy={nbest_strategy})")
    print("=" * 60)

    retrieval_fn = {
        "union": hipporag.retrieve_nbest_union,
        "concat": hipporag.retrieve_nbest_concat,
    }[nbest_strategy]

    results = []
    for i, t in enumerate(transcriptions_nbest):
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")
        hypotheses = t["hypotheses"]
        best_query = t["best_text"]

        ret = retrieval_fn(hypotheses)

        context = "\n\n".join(ret["docs"])
        gen = generate_answer_openai(context, best_query, model=llm_model)

        em = exact_match(gen["answer"], gt)
        f1 = f1_score(gen["answer"], gt)
        wer = word_error_rate(best_query, t["original_text"])

        results.append(
            {
                "id": qid,
                "query": best_query,
                "original": t["original_text"],
                "num_hypotheses": len(hypotheses),
                "answer": gen["answer"],
                "ground_truth": gt,
                "em": em,
                "f1": f1,
                "wer": wer,
                "retrieval_time": ret["retrieval_time"],
                "generation_time": gen["generation_time"],
                "strategy": nbest_strategy,
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            print(
                f"  [{i + 1}/{len(transcriptions_nbest)}] F1={avg_f1:.3f} EM={avg_em:.3f}"
            )

    return results


# ---------------------------------------------------------------------------
# IRCoT (Interleaving Retrieval with Chain-of-Thought)
# ---------------------------------------------------------------------------

IRCOT_COT_PROMPT = """You are a reasoning assistant that helps answer multi-hop questions step by step.

Question: {question}

Retrieved Information:
{context}

Reasoning so far:
{cot_so_far}

Based on the retrieved information, write ONE brief reasoning sentence that makes progress toward answering the question. Then, if more information is needed, suggest a specific search query to retrieve the missing information.

Format your response EXACTLY as:
Reasoning: <one sentence of reasoning>
Search: <next search query, or DONE if you have enough information to answer>"""


def run_ircot_loop(
    query: str,
    retrieve_fn,
    llm_model: str = "gpt-4o-mini",
    max_steps: int = 3,
    top_k: int = 5,
) -> Dict:
    """
    Generic IRCoT loop. Works with any retriever that accepts (query, top_k) or (query).

    Returns dict with docs, cot_chain, retrieval_queries, total_retrieval_time, num_steps.
    """
    from openai import OpenAI

    client = OpenAI()

    query = _sanitize_text(query)

    total_retrieval_time = 0.0

    # Step 0: initial retrieval
    start = time.time()
    try:
        initial = retrieve_fn(query, top_k=top_k)
    except TypeError:
        initial = retrieve_fn(query)
    total_retrieval_time += time.time() - start

    collected_docs = {}
    for doc in initial.get("docs", []):
        h = hash(doc)
        if h not in collected_docs:
            collected_docs[h] = doc

    cot_sentences = []
    retrieval_queries = [query]

    for step in range(max_steps):
        context = _sanitize_text("\n\n".join(list(collected_docs.values())[:10]))
        cot_so_far = "\n".join(cot_sentences) if cot_sentences else "None yet."

        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": IRCOT_COT_PROMPT.format(
                            question=query,
                            context=context[:3000],
                            cot_so_far=cot_so_far,
                        ),
                    }
                ],
                temperature=0,
                max_tokens=150,
            )
            cot_text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    IRCoT step {step + 1} failed: {e}")
            break

        reasoning_line = ""
        search_query = ""
        for line in cot_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("reasoning:"):
                reasoning_line = line[len("reasoning:") :].strip()
            elif line.lower().startswith("search:"):
                search_query = line[len("search:") :].strip()

        if reasoning_line:
            cot_sentences.append(reasoning_line)

        if not search_query or search_query.upper() == "DONE":
            break

        search_query = _sanitize_text(search_query)

        retrieval_queries.append(search_query)
        start = time.time()
        try:
            new_results = retrieve_fn(search_query, top_k=top_k)
        except TypeError:
            new_results = retrieve_fn(search_query)
        total_retrieval_time += time.time() - start

        for doc in new_results.get("docs", []):
            h = hash(doc)
            if h not in collected_docs:
                collected_docs[h] = doc

    cot_chain = "\n".join(f"- {s}" for s in cot_sentences)
    final_docs = list(collected_docs.values())[:10]
    final_context = ""
    if cot_chain:
        final_context += f"Reasoning chain:\n{cot_chain}\n\n"
    final_context += "Supporting documents:\n" + "\n\n".join(final_docs)

    return {
        "context": final_context,
        "docs": final_docs,
        "cot_sentences": cot_sentences,
        "retrieval_queries": retrieval_queries,
        "total_retrieval_time": total_retrieval_time,
        "num_steps": len(cot_sentences),
        "num_docs": len(collected_docs),
    }


def run_cell_G(
    transcriptions: List[Dict],
    ground_truth: Dict,
    naive_rag: NaiveRAG,
    top_k: int = 5,
    max_steps: int = 3,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell G: IRCoT + Naive RAG + ASR top-1 (or oracle)."""
    print("\n" + "=" * 60)
    print(f"CELL G: IRCoT + Naive RAG (max_steps={max_steps})")
    print("=" * 60)

    results = []
    for i, t in enumerate(transcriptions):
        query = t["transcribed_text"]
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")

        ircot = run_ircot_loop(
            query,
            naive_rag.retrieve_top1,
            llm_model=llm_model,
            max_steps=max_steps,
            top_k=top_k,
        )

        gen = generate_answer_openai(ircot["context"], query, model=llm_model)

        em = exact_match(gen["answer"], gt)
        f1 = f1_score(gen["answer"], gt)
        wer = word_error_rate(query, t["original_text"])

        results.append(
            {
                "id": qid,
                "query": query,
                "original": t["original_text"],
                "answer": gen["answer"],
                "ground_truth": gt,
                "em": em,
                "f1": f1,
                "wer": wer,
                "retrieval_time": ircot["total_retrieval_time"],
                "generation_time": gen["generation_time"],
                "num_steps": ircot["num_steps"],
                "num_docs": ircot["num_docs"],
                "retrieval_queries": ircot["retrieval_queries"],
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            print(f"  [{i + 1}/{len(transcriptions)}] F1={avg_f1:.3f} EM={avg_em:.3f}")

    return results


def run_cell_H(
    transcriptions: List[Dict],
    ground_truth: Dict,
    hipporag,
    max_steps: int = 3,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell H: IRCoT + HippoRAG + ASR top-1 (or oracle)."""
    print("\n" + "=" * 60)
    print(f"CELL H: IRCoT + HippoRAG (max_steps={max_steps})")
    print("=" * 60)

    results = []
    for i, t in enumerate(transcriptions):
        query = t["transcribed_text"]
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")

        ircot = run_ircot_loop(
            query,
            hipporag.retrieve_top1,
            llm_model=llm_model,
            max_steps=max_steps,
        )

        gen = generate_answer_openai(ircot["context"], query, model=llm_model)

        em = exact_match(gen["answer"], gt)
        f1 = f1_score(gen["answer"], gt)
        wer = word_error_rate(query, t["original_text"])

        results.append(
            {
                "id": qid,
                "query": query,
                "original": t["original_text"],
                "answer": gen["answer"],
                "ground_truth": gt,
                "em": em,
                "f1": f1,
                "wer": wer,
                "retrieval_time": ircot["total_retrieval_time"],
                "generation_time": gen["generation_time"],
                "num_steps": ircot["num_steps"],
                "num_docs": ircot["num_docs"],
                "retrieval_queries": ircot["retrieval_queries"],
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            print(f"  [{i + 1}/{len(transcriptions)}] F1={avg_f1:.3f} EM={avg_em:.3f}")

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


def print_results_table(summaries: Dict[str, Dict], accent: str = "") -> None:
    """Print the results table."""
    label = f" [{accent.upper()}]" if accent else ""
    print("\n" + "=" * 60)
    print(f"RESULTS{label}")
    print("=" * 60)

    for cell_name in ["E", "F", "A", "G", "B", "C", "H", "D"]:
        s = summaries.get(cell_name)
        if s and s["n"] > 0:
            wer_str = f" WER={s['wer']:.4f}" if s["wer"] > 0 else ""
            print(
                f"  Cell {cell_name}: F1={s['f1']:.3f} EM={s['em']:.3f}{wer_str} (n={s['n']})"
            )

    # Degradation comparisons
    e, a, c = summaries.get("E", {}), summaries.get("A", {}), summaries.get("C", {})
    f_cell = summaries.get("F", {})
    if e.get("n") and a.get("n"):
        diff = a["f1"] - e["f1"]
        print(f"  A vs E (Naive ASR vs Oracle):       F1 diff = {diff:+.3f}")
    if f_cell.get("n") and c.get("n"):
        diff = c["f1"] - f_cell["f1"]
        print(f"  C vs F (HippoRAG ASR vs Oracle):    F1 diff = {diff:+.3f}")
    if a.get("n") and c.get("n"):
        diff = c["f1"] - a["f1"]
        print(f"  C vs A (HippoRAG vs Naive, top-1):  F1 diff = {diff:+.3f}")
    b, d = summaries.get("B", {}), summaries.get("D", {})
    if a.get("n") and b.get("n"):
        diff = b["f1"] - a["f1"]
        print(f"  B vs A (N-best vs top-1, Naive):    F1 diff = {diff:+.3f}")
    if c.get("n") and d.get("n"):
        diff = d["f1"] - c["f1"]
        print(f"  D vs C (N-best vs top-1, HippoRAG): F1 diff = {diff:+.3f}")

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
        "--accent",
        type=str,
        default="us",
        help="Which accent to run: us, in, ph, ng",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["E", "F"],
        help="Which cells to run (e.g., --cells E F A B C D)",
    )
    parser.add_argument(
        "--nbest-strategy",
        type=str,
        default="union",
        choices=["union", "weighted", "concat"],
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--ircot-steps", type=int, default=3, help="Max IRCoT reasoning steps"
    )
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
    need_naive = bool(cells_to_run & {"A", "B", "E", "G"})
    need_hipporag = bool(cells_to_run & {"C", "D", "F", "H"})

    print("=" * 60)
    print("SPOKEN MULTI-HOP QA: 2x2 EXPERIMENT")
    print("=" * 60)
    print(f"Accent: {args.accent}")
    print(f"Cells: {sorted(cells_to_run)}")
    print(f"LLM: {args.llm_model}")

    # --- Load data ---
    asr_data = load_accent_results(Path(args.accent_json))
    ground_truth = load_ground_truth(Path(args.ground_truth))
    docs = load_corpus_documents(Path(args.docs_dir))

    if args.sample:
        asr_data = asr_data[: args.sample]
        print(f"  Using first {args.sample} questions only")

    # Prepare transcriptions
    transcriptions_oracle = None
    transcriptions_top1 = None
    transcriptions_nbest = None

    if cells_to_run & {"E", "F"}:
        transcriptions_oracle = prepare_oracle_transcriptions(asr_data)
    if cells_to_run & {"A", "C", "G", "H"}:
        transcriptions_top1 = prepare_top1_transcriptions(asr_data, args.accent)
    if cells_to_run & {"B", "D"}:
        transcriptions_nbest = prepare_nbest_transcriptions(asr_data, args.accent)

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

    if "A" in cells_to_run:
        results = run_cell_A(
            transcriptions_top1,
            ground_truth,
            naive_rag,
            top_k=args.top_k,
            llm_model=args.llm_model,
        )
        all_results["A"] = results
        summaries["A"] = summarize_cell("A", results)

    if "C" in cells_to_run:
        results = run_cell_C(
            transcriptions_top1, ground_truth, hipporag, llm_model=args.llm_model
        )
        all_results["C"] = results
        summaries["C"] = summarize_cell("C", results)

    if "B" in cells_to_run:
        results = run_cell_B(
            transcriptions_nbest,
            ground_truth,
            naive_rag,
            nbest_strategy=args.nbest_strategy,
            top_k=args.top_k,
            llm_model=args.llm_model,
        )
        all_results["B"] = results
        summaries["B"] = summarize_cell("B", results)

    if "D" in cells_to_run:
        results = run_cell_D(
            transcriptions_nbest,
            ground_truth,
            hipporag,
            nbest_strategy=args.nbest_strategy,
            llm_model=args.llm_model,
        )
        all_results["D"] = results
        summaries["D"] = summarize_cell("D", results)

    if "G" in cells_to_run:
        results = run_cell_G(
            transcriptions_top1,
            ground_truth,
            naive_rag,
            top_k=args.top_k,
            max_steps=args.ircot_steps,
            llm_model=args.llm_model,
        )
        all_results["G"] = results
        summaries["G"] = summarize_cell("G", results)

    if "H" in cells_to_run:
        results = run_cell_H(
            transcriptions_top1,
            ground_truth,
            hipporag,
            max_steps=args.ircot_steps,
            llm_model=args.llm_model,
        )
        all_results["H"] = results
        summaries["H"] = summarize_cell("H", results)

    print_results_table(summaries, accent=args.accent)

    # --- Save ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    accent_tag = args.accent
    output_path = args.output or str(
        _ROOT / f"results/experiment_{accent_tag}_{timestamp}.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "accent": args.accent,
                    "cells": sorted(cells_to_run),
                    "nbest_strategy": args.nbest_strategy,
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
