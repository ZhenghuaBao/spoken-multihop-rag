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
    python experiments/run_2x2.py --cells E F A B C D --accent all  # Full table
    python experiments/run_2x2.py --cells E F A B C D G H I J --accent all+oracle
"""

import json
import math
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

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
# N-best seeded IRCoT (cells I, J)
# ---------------------------------------------------------------------------


def run_ircot_loop_nbest(
    query: str,
    hypotheses: List[Dict],
    retrieve_nbest_fn,
    retrieve_top1_fn,
    llm_model: str = "gpt-4o-mini",
    max_steps: int = 3,
    top_k: int = 5,
) -> Dict:
    """IRCoT with N-best seeded initial retrieval.

    Step 0: retrieve using N-best hypotheses (robust to entity errors).
    Steps 1+: retrieve using top-1 reformulated query from CoT reasoning.
    """
    from openai import OpenAI

    client = OpenAI()

    query = _sanitize_text(query)

    # --- Step 0: N-best initial retrieval ---
    start = time.time()
    try:
        initial = retrieve_nbest_fn(hypotheses, top_k=top_k)
    except TypeError:
        initial = retrieve_nbest_fn(hypotheses)
    total_retrieval_time = time.time() - start

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
            new_results = retrieve_top1_fn(search_query, top_k=top_k)
        except TypeError:
            new_results = retrieve_top1_fn(search_query)
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


def run_cell_I(
    transcriptions: List[Dict],
    ground_truth: Dict,
    naive_rag: NaiveRAG,
    nbest_strategy: str = "union",
    top_k: int = 5,
    max_steps: int = 3,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell I: N-best seeded IRCoT + Naive RAG."""
    print("\n" + "=" * 60)
    print(
        f"CELL I: N-best IRCoT + Naive RAG (max_steps={max_steps}, strategy={nbest_strategy})"
    )
    print("=" * 60)

    retrieve_nbest_fn = {
        "union": naive_rag.retrieve_nbest_union,
        "weighted": naive_rag.retrieve_nbest_weighted,
        "concat": naive_rag.retrieve_nbest_concat,
    }.get(nbest_strategy, naive_rag.retrieve_nbest_union)

    results = []
    for i, t in enumerate(transcriptions):
        query = t["best_text"]
        hypotheses = t["hypotheses"]
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")

        ircot = run_ircot_loop_nbest(
            query,
            hypotheses,
            retrieve_nbest_fn=retrieve_nbest_fn,
            retrieve_top1_fn=naive_rag.retrieve_top1,
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


def run_cell_J(
    transcriptions: List[Dict],
    ground_truth: Dict,
    hipporag,
    top_k: int = 5,
    max_steps: int = 3,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Cell J: N-best seeded IRCoT + HippoRAG."""
    print("\n" + "=" * 60)
    print(f"CELL J: N-best IRCoT + HippoRAG (max_steps={max_steps})")
    print("=" * 60)

    results = []
    for i, t in enumerate(transcriptions):
        query = t["best_text"]
        hypotheses = t["hypotheses"]
        qid = t["id"]
        gt = ground_truth.get(qid, {}).get("answer", "")

        ircot = run_ircot_loop_nbest(
            query,
            hypotheses,
            retrieve_nbest_fn=hipporag.retrieve_nbest_union,
            retrieve_top1_fn=hipporag.retrieve_top1,
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


def print_2x2_table(summaries: Dict[str, Dict], accent: str = "") -> None:
    """Print the results table for one accent."""
    label = f" [{accent.upper()}]" if accent else ""
    print("\n" + "=" * 90)
    print(f"RESULTS{label}")
    print("=" * 90)

    has_oracle = (
        summaries.get("E", {}).get("n", 0) > 0 or summaries.get("F", {}).get("n", 0) > 0
    )

    header = f"{'':20s}"
    if has_oracle:
        header += f" | {'Oracle':20s}"
    header += f" | {'ASR top-1':20s} | {'ASR N-best':20s}"
    print(header)
    print("-" * 90)

    for retriever, oracle_cell, cells in [
        ("Naive RAG", "E", ("A", "B")),
        ("IRCoT+Naive", None, ("G",)),
        ("HippoRAG", "F", ("C", "D")),
        ("IRCoT+Hippo", None, ("H",)),
    ]:
        row = f"{retriever:20s}"
        if has_oracle:
            if oracle_cell:
                s = summaries.get(oracle_cell)
                if s and s["n"] > 0:
                    row += f" | F1={s['f1']:.3f} EM={s['em']:.3f}"
                else:
                    row += f" | {'(not run)':20s}"
            else:
                row += f" | {'':20s}"
        for cell_name in cells:
            s = summaries.get(cell_name)
            if s and s["n"] > 0:
                row += f" | F1={s['f1']:.3f} EM={s['em']:.3f}"
            else:
                row += f" | {'(not run)':20s}"
        if len(cells) == 1:
            row += f" | {'':20s}"
        print(row)

    print("-" * 90)

    for cell_name in ["A", "G", "B", "C", "H", "D"]:
        s = summaries.get(cell_name)
        if s and s["n"] > 0 and s["wer"] > 0:
            print(f"  Cell {cell_name} avg WER: {s['wer']:.4f}")

    e, f_cell = summaries.get("E", {}), summaries.get("F", {})
    a, b, c, d = [summaries.get(x, {}) for x in "ABCD"]

    if e.get("n") and a.get("n"):
        diff = a["f1"] - e["f1"]
        print(f"  A vs E (Naive ASR vs Oracle):       F1 diff = {diff:+.3f}")
    if f_cell.get("n") and c.get("n"):
        diff = c["f1"] - f_cell["f1"]
        print(f"  C vs F (HippoRAG ASR vs Oracle):    F1 diff = {diff:+.3f}")
    if a.get("n") and c.get("n"):
        diff = c["f1"] - a["f1"]
        print(f"  C vs A (HippoRAG vs Naive, top-1):  F1 diff = {diff:+.3f}")
    if c.get("n") and d.get("n"):
        diff = d["f1"] - c["f1"]
        print(f"  D vs C (N-best vs top-1, HippoRAG): F1 diff = {diff:+.3f}")
    if a.get("n") and b.get("n"):
        diff = b["f1"] - a["f1"]
        print(f"  B vs A (N-best vs top-1, Naive):    F1 diff = {diff:+.3f}")


def print_accent_comparison(all_summaries: Dict[str, Dict[str, Dict]]) -> None:
    """Print cross-accent comparison table."""
    print("\n" + "=" * 80)
    print("CROSS-ACCENT COMPARISON")
    print("=" * 80)

    accents = sorted(all_summaries.keys())
    header = f"{'Cell':8s}"
    for acc in accents:
        header += f" | {acc.upper():^18s}"
    print(header)
    print("-" * 80)

    for cell_name in ["E", "F", "A", "G", "B", "C", "H", "D", "I", "J"]:
        if cell_name in ("E", "F"):
            s = None
            for acc in accents:
                s = all_summaries.get(acc, {}).get(cell_name)
                if s and s["n"] > 0:
                    break
            if not s or s.get("n", 0) == 0:
                continue
            label = "E (Naive Oracle)" if cell_name == "E" else "F (Hippo Oracle)"
            row = f"{label:8s}"
            row += f" | F1={s['f1']:.3f} EM={s['em']:.3f}"
            for _ in accents[1:]:
                row += f" | {'(same)':^18s}"
            print(row)
        else:
            row = f"{cell_name:8s}"
            for acc in accents:
                s = all_summaries.get(acc, {}).get(cell_name)
                if s and s["n"] > 0:
                    row += f" | F1={s['f1']:.3f} EM={s['em']:.3f}"
                else:
                    dash = "\u2014"
                    row += f" | {dash:^18s}"
            print(row)

    print("-" * 80)
    row = f"{'WER':8s}"
    for acc in accents:
        for cell_name in "ABCD":
            s = all_summaries.get(acc, {}).get(cell_name)
            if s and s["n"] > 0:
                row += f" | {s['wer']:^18.4f}"
                break
        else:
            dash = "\u2014"
            row += f" | {dash:^18s}"
    print(row)


# ---------------------------------------------------------------------------
# Per-accent orchestrator
# ---------------------------------------------------------------------------


def run_cells_for_accent(
    accent: str,
    asr_data: List[Dict],
    ground_truth: Dict,
    cells_to_run: set,
    naive_rag,
    hipporag,
    args,
    oracle_done: bool = False,
    existing_results: Optional[Dict] = None,
    existing_summaries: Optional[Dict] = None,
    save_fn=None,
) -> tuple:
    """Run all requested cells for one accent. Returns (all_results, summaries).

    save_fn(results, summaries) is called after each cell completes (checkpoint).
    existing_results / existing_summaries allow resuming from a prior checkpoint.
    """
    need_top1 = bool(cells_to_run & {"A", "C", "G", "H"}) and accent != "oracle"
    need_nbest = bool(cells_to_run & {"B", "D", "I", "J"}) and accent != "oracle"
    need_oracle = (bool(cells_to_run & {"E", "F"}) and not oracle_done) or (
        bool(cells_to_run & {"G", "H"}) and accent == "oracle"
    )

    print(f"\n{'#' * 70}")
    print(f"# ACCENT: {accent.upper()}")
    print(f"{'#' * 70}")

    all_results = dict(existing_results) if existing_results else {}
    summaries = dict(existing_summaries) if existing_summaries else {}
    if all_results:
        print(f"  Resuming from checkpoint: done = {sorted(all_results.keys())}")

    transcriptions_top1 = None
    transcriptions_nbest = None
    transcriptions_oracle = None

    if need_top1:
        transcriptions_top1 = prepare_top1_transcriptions(asr_data, accent)
    if need_nbest:
        transcriptions_nbest = prepare_nbest_transcriptions(asr_data, accent)
    if need_oracle:
        transcriptions_oracle = prepare_oracle_transcriptions(asr_data)

    def maybe_run(cell_name, run_fn):
        """Run cell if not already in checkpoint, then save."""
        if cell_name in all_results:
            print(f"  [SKIP] Cell {cell_name} (checkpoint)")
            return
        result = run_fn()
        all_results[cell_name] = result
        summaries[cell_name] = summarize_cell(cell_name, result)
        if save_fn:
            save_fn(all_results, summaries)

    if need_oracle and "E" in cells_to_run:
        maybe_run(
            "E",
            lambda: run_cell_E(
                transcriptions_oracle,
                ground_truth,
                naive_rag,
                top_k=args.top_k,
                llm_model=args.llm_model,
            ),
        )

    if need_oracle and "F" in cells_to_run:
        maybe_run(
            "F",
            lambda: run_cell_F(
                transcriptions_oracle, ground_truth, hipporag, llm_model=args.llm_model
            ),
        )

    if "A" in cells_to_run and accent != "oracle":
        maybe_run(
            "A",
            lambda: run_cell_A(
                transcriptions_top1,
                ground_truth,
                naive_rag,
                top_k=args.top_k,
                llm_model=args.llm_model,
            ),
        )

    if "B" in cells_to_run and accent != "oracle":
        maybe_run(
            "B",
            lambda: run_cell_B(
                transcriptions_nbest,
                ground_truth,
                naive_rag,
                nbest_strategy=args.nbest_strategy,
                top_k=args.top_k,
                llm_model=args.llm_model,
            ),
        )

    if "C" in cells_to_run and accent != "oracle":
        maybe_run(
            "C",
            lambda: run_cell_C(
                transcriptions_top1,
                ground_truth,
                hipporag,
                llm_model=args.llm_model,
            ),
        )

    if "D" in cells_to_run and accent != "oracle":
        maybe_run(
            "D",
            lambda: run_cell_D(
                transcriptions_nbest,
                ground_truth,
                hipporag,
                nbest_strategy=args.nbest_strategy,
                llm_model=args.llm_model,
            ),
        )

    if "G" in cells_to_run:
        ircot_transcriptions = (
            transcriptions_oracle if accent == "oracle" else transcriptions_top1
        )
        maybe_run(
            "G",
            lambda: run_cell_G(
                ircot_transcriptions,
                ground_truth,
                naive_rag,
                top_k=args.top_k,
                max_steps=args.ircot_steps,
                llm_model=args.llm_model,
            ),
        )

    if "H" in cells_to_run:
        ircot_transcriptions = (
            transcriptions_oracle if accent == "oracle" else transcriptions_top1
        )
        maybe_run(
            "H",
            lambda: run_cell_H(
                ircot_transcriptions,
                ground_truth,
                hipporag,
                max_steps=args.ircot_steps,
                llm_model=args.llm_model,
            ),
        )

    if "I" in cells_to_run and accent != "oracle":
        maybe_run(
            "I",
            lambda: run_cell_I(
                transcriptions_nbest,
                ground_truth,
                naive_rag,
                nbest_strategy=args.nbest_strategy,
                top_k=args.top_k,
                max_steps=args.ircot_steps,
                llm_model=args.llm_model,
            ),
        )

    if "J" in cells_to_run and accent != "oracle":
        maybe_run(
            "J",
            lambda: run_cell_J(
                transcriptions_nbest,
                ground_truth,
                hipporag,
                top_k=args.top_k,
                max_steps=args.ircot_steps,
                llm_model=args.llm_model,
            ),
        )

    print_2x2_table(summaries, accent)
    return all_results, summaries


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
        default="all",
        help="Which accent(s) to run: us, in, ph, ng, all, or all+oracle",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="Which cells to run (e.g., --cells E F A B C D G H I J)",
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
    if args.accent == "all":
        accents_to_run = ACCENTS
    elif args.accent == "all+oracle":
        accents_to_run = ["oracle"] + ACCENTS
    else:
        accents_to_run = [args.accent]
    need_naive = bool(cells_to_run & {"A", "B", "E", "G", "I"})
    need_hipporag = bool(cells_to_run & {"C", "D", "F", "H", "J"})

    print("=" * 70)
    print("SPOKEN MULTI-HOP QA: 2x2x4 EXPERIMENT")
    print("=" * 70)
    print(f"Accents: {accents_to_run}")
    print(f"Cells: {sorted(cells_to_run)}")
    print(f"N-best strategy: {args.nbest_strategy}")
    print(f"LLM: {args.llm_model}")

    # --- Determine output path EARLY (required for checkpointing) ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    accent_tag = args.accent if args.accent != "all" else "all"
    output_path = args.output or str(
        _ROOT / f"results/experiment_{accent_tag}_{timestamp}.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint if output file already exists ---
    all_accent_results: Dict[str, Dict] = {}
    all_accent_summaries: Dict[str, Dict] = {}
    oracle_results: Dict = {}
    oracle_summaries_ckpt: Dict = {}

    if Path(output_path).exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            all_accent_results = ckpt.get("accent_detailed_results", {})
            all_accent_summaries = ckpt.get("accent_summaries", {})
            for acc_res in all_accent_results.values():
                for cell in ("E", "F"):
                    if cell in acc_res and cell not in oracle_results:
                        oracle_results[cell] = acc_res[cell]
            for acc_sum in all_accent_summaries.values():
                for cell in ("E", "F"):
                    if cell in acc_sum and cell not in oracle_summaries_ckpt:
                        oracle_summaries_ckpt[cell] = acc_sum[cell]
            print(f"\nCheckpoint loaded from {output_path}")
            done_accents = [a for a, r in all_accent_results.items() if r]
            print(f"  Accents with data: {done_accents}")
        except Exception as e:
            print(f"Warning: could not load checkpoint ({e}), starting fresh")
            all_accent_results = {}
            all_accent_summaries = {}

    oracle_done = bool(oracle_results)

    config: Dict = {
        "accents": accents_to_run,
        "cells": sorted(cells_to_run),
        "nbest_strategy": args.nbest_strategy,
        "llm_model": args.llm_model,
        "top_k": args.top_k,
        "num_questions": 0,
        "timestamp": timestamp,
    }

    def save_checkpoint():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": config,
                    "accent_summaries": all_accent_summaries,
                    "accent_detailed_results": all_accent_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  [checkpoint -> {Path(output_path).name}]")

    # --- Load data ---
    asr_data = load_accent_results(Path(args.accent_json))
    ground_truth = load_ground_truth(Path(args.ground_truth))
    docs = load_corpus_documents(Path(args.docs_dir))
    config["num_questions"] = len(asr_data)

    if args.sample:
        asr_data = asr_data[: args.sample]
        print(f"  Using first {args.sample} questions only")

    # --- Load indices (once, shared across accents) ---
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

    # --- Run per accent ---
    for accent in accents_to_run:

        def make_save_fn(acc: str):
            def _save(results, summaries):
                all_accent_results[acc] = results
                all_accent_summaries[acc] = summaries
                save_checkpoint()

            return _save

        results, summaries = run_cells_for_accent(
            accent,
            asr_data,
            ground_truth,
            cells_to_run,
            naive_rag,
            hipporag,
            args,
            oracle_done=oracle_done,
            existing_results=all_accent_results.get(accent, {}),
            existing_summaries=all_accent_summaries.get(accent, {}),
            save_fn=make_save_fn(accent),
        )
        all_accent_results[accent] = results
        all_accent_summaries[accent] = summaries

        if not oracle_done and (summaries.get("E") or summaries.get("F")):
            oracle_done = True
            oracle_results = {k: v for k, v in results.items() if k in ("E", "F")}
            oracle_summaries_ckpt = {
                k: v for k, v in summaries.items() if k in ("E", "F")
            }

    # Forward-fill oracle results into all accent entries
    if oracle_done and oracle_results:
        for acc in accents_to_run:
            for cell in ("E", "F"):
                if cell not in all_accent_summaries.get(acc, {}):
                    if cell in oracle_summaries_ckpt:
                        all_accent_summaries.setdefault(acc, {})[cell] = (
                            oracle_summaries_ckpt[cell]
                        )
                        all_accent_results.setdefault(acc, {})[cell] = oracle_results[
                            cell
                        ]

    # --- Cross-accent comparison ---
    if len(accents_to_run) > 1:
        print_accent_comparison(all_accent_summaries)

    # --- Final save ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "accent_summaries": all_accent_summaries,
                "accent_detailed_results": all_accent_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
