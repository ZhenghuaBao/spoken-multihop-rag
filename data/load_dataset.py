"""
Load a spoken multi-hop QA dataset from HuggingFace Hub.

This script assumes a HuggingFace dataset entry shape compatible with
the HotpotQA-format columns: ``id``, ``question``, ``answer``,
``type``, ``context`` (with ``title`` + ``sentences`` subkeys),
``supporting_facts``, and optionally ``question_audio``. Datasets
with a different schema will need their parsing adapted in
``extract_corpus_documents``.

Usage:
    python data/load_dataset.py \\
        --hf-dataset the-bird-F/HotpotQA_RGBzh_speech \\
        --split test \\
        --samples 1000 \\
        --output-dir data/hotpotqa_spoken_1000 \\
        --save
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


def load_spoken_qa(
    hf_dataset: str,
    num_samples: int = 200,
    seed: int = 42,
    split: str = "test",
) -> List[Dict]:
    """Load a spoken QA dataset from HuggingFace.

    Args:
        hf_dataset: HuggingFace dataset identifier
            (e.g. ``the-bird-F/HotpotQA_RGBzh_speech``).
        num_samples: number of questions to sample uniformly at random.
        seed: RNG seed for reproducible sampling.
        split: dataset split to load.

    Returns:
        List of dicts with keys ``id``, ``question_text``,
        ``question_audio``, ``answer``, ``type``, ``context``,
        ``supporting_facts``.
    """
    print(f"Loading {hf_dataset} split={split}...")
    ds = load_dataset(hf_dataset, split=split)
    print(f"Total samples: {len(ds)}")

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
    """Extract retrieval corpus + ground truth from loaded records.

    Returns:
        docs: list of document strings (one per unique title).
        ground_truth: dict mapping qid -> answer/type/supporting_facts.
    """
    docs = []
    seen_titles = set()
    ground_truth = {}

    for record in records:
        ground_truth[record["id"]] = {
            "answer": record["answer"],
            "type": record["type"],
            "supporting_facts": record["supporting_facts"],
        }

        # Extract context paragraphs as documents (HotpotQA-style schema:
        # context is a dict with "title" and "sentences" lists).
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
    output_dir: Path,
):
    """Save dataset to disk for offline use (strips audio to save space)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = [
        {
            "id": r["id"],
            "question": r["question_text"],
            "type": r["type"],
        }
        for r in records
    ]
    with open(output_dir / "test_queries.json", "w", encoding="utf-8") as f:
        json.dump(
            {"queries": queries, "total_queries": len(queries)},
            f,
            indent=2,
            ensure_ascii=False,
        )

    docs, ground_truth = extract_corpus_documents(records)
    with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    docs_dir = output_dir / "documents"
    docs_dir.mkdir(exist_ok=True)
    for i, doc in enumerate(docs):
        with open(docs_dir / f"doc_{i:04d}.txt", "w", encoding="utf-8") as f:
            f.write(doc)

    print(f"Saved dataset to {output_dir}")
    print(f"  Queries: {len(queries)}")
    print(f"  Documents: {len(docs)}")
    print(f"  Ground truth: {len(ground_truth)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load (default: test)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to draw (default 200)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the dataset (omit to preview only)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the dataset to disk under --output-dir",
    )
    args = parser.parse_args()

    records = load_spoken_qa(
        hf_dataset=args.hf_dataset,
        num_samples=args.samples,
        seed=args.seed,
        split=args.split,
    )

    if args.save:
        if not args.output_dir:
            raise SystemExit("--save requires --output-dir")
        save_dataset(records, Path(args.output_dir))
    else:
        print("\nSample record:")
        r = records[0]
        print(f"  ID: {r['id']}")
        print(f"  Question: {r['question_text']}")
        print(f"  Answer: {r['answer']}")
        print(f"  Type: {r['type']}")
        print(f"  Has audio: {r['question_audio'] is not None}")
