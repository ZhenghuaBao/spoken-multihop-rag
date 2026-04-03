"""
Load spoken HotpotQA dataset from Feng et al. (2025).

Dataset: the-bird-F/HotpotQA_RGBzh_speech
Contains audio recordings of HotpotQA questions + original text context.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

DATASET_NAME = "the-bird-F/HotpotQA_RGBzh_speech"
OUTPUT_DIR = Path(__file__).parent


def load_spoken_hotpotqa(
    num_samples: int = 200,
    seed: int = 42,
    split: str = "test",
) -> List[Dict]:
    """
    Load spoken HotpotQA dataset from HuggingFace.

    Returns list of dicts with keys:
        - id: str
        - question_text: str (original text question)
        - question_audio: audio array + sampling_rate
        - answer: str
        - type: str (bridge / comparison)
        - context: list of (title, sentences) for retrieval corpus
        - supporting_facts: dict with title + sent_id
    """
    print(f"Loading {DATASET_NAME} split={split}...")
    ds = load_dataset(DATASET_NAME, split=split)
    print(f"Total samples: {len(ds)}")

    # Sample subset
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    sampled = ds.select(indices)
    print(f"Sampled {len(sampled)} questions")

    records = []
    for entry in sampled:
        record = {
            "id": entry["id"],
            "question_text": entry["question"],
            "question_audio": entry.get("question_audio", None),
            "answer": entry["answer"],
            "type": entry.get("type", "unknown"),
            "context": entry.get("context", {}),
            "supporting_facts": entry.get("supporting_facts", {}),
        }
        records.append(record)

    return records


def extract_corpus_documents(records: List[Dict]) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Extract retrieval corpus and ground truth from loaded records.

    Returns:
        docs: list of document strings (one per paragraph)
        ground_truth: dict mapping id -> {answer, type, supporting_facts}
    """
    docs = []
    seen_titles = set()
    ground_truth = {}

    for record in records:
        # Build ground truth
        ground_truth[record["id"]] = {
            "answer": record["answer"],
            "type": record["type"],
            "supporting_facts": record["supporting_facts"],
        }

        # Extract context paragraphs as documents
        context = record.get("context", {})
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        for title, sentences in zip(titles, sentences_list):
            if title not in seen_titles:
                text = f"# {title}\n\n" + " ".join(sentences)
                docs.append(text)
                seen_titles.add(title)

    print(f"Extracted {len(docs)} unique documents from {len(records)} records")
    return docs, ground_truth


def save_dataset(
    records: List[Dict],
    output_dir: Optional[Path] = None,
    suffix: str = "200",
):
    """Save dataset to disk for offline use (strips audio to save space)."""
    output_dir = output_dir or OUTPUT_DIR / f"spoken_hotpotqa_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save queries (without audio - audio is loaded on the fly)
    queries = []
    for r in records:
        queries.append(
            {
                "id": r["id"],
                "question": r["question_text"],
                "type": r["type"],
            }
        )

    with open(output_dir / "test_queries.json", "w", encoding="utf-8") as f:
        json.dump(
            {"queries": queries, "total_queries": len(queries)},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save ground truth
    _, ground_truth = extract_corpus_documents(records)
    with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    # Save documents
    docs_dir = output_dir / "documents"
    docs_dir.mkdir(exist_ok=True)
    docs, _ = extract_corpus_documents(records)
    for i, doc in enumerate(docs):
        with open(docs_dir / f"doc_{i:04d}.txt", "w", encoding="utf-8") as f:
            f.write(doc)

    print(f"Saved dataset to {output_dir}")
    print(f"  Queries: {len(queries)}")
    print(f"  Documents: {len(docs)}")
    print(f"  Ground truth: {len(ground_truth)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load spoken HotpotQA dataset")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save to disk")
    args = parser.parse_args()

    records = load_spoken_hotpotqa(num_samples=args.samples, seed=args.seed)

    if args.save:
        save_dataset(records, suffix=str(args.samples))
    else:
        # Preview
        print("\nSample record:")
        r = records[0]
        print(f"  ID: {r['id']}")
        print(f"  Question: {r['question_text']}")
        print(f"  Answer: {r['answer']}")
        print(f"  Type: {r['type']}")
        has_audio = r["question_audio"] is not None
        print(f"  Has audio: {has_audio}")
