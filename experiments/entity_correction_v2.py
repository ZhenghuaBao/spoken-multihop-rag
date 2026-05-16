import os
import json
from openai import OpenAI
from rapidfuzz import process, fuzz
from tqdm import tqdm

client = OpenAI()


# ── 1. Load KB entities ──────────────────────────────
def load_kb_entities(docs_dir: str) -> list:
    entities = []
    for f in os.listdir(docs_dir):
        if f.endswith(".txt"):
            entity = f.replace(".txt", "").replace("_", " ")
            entities.append(entity)
    return entities


# ── 2. Extract entities from query ───────────────────
def extract_entities(text: str) -> list:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Extract all named entities (people, places, organizations, works, species, etc.) from this question.
Return only a JSON list of strings, nothing else.

Question: {text}
Entities:""",
            }
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []


# ── 3. Match against KB entity ───────────────────────
def match_entity(asr_entity: str, kb_entities: list, threshold: int = 70) -> str:
    matches = process.extract(
        asr_entity,
        kb_entities,
        scorer=fuzz.ratio,
        limit=1,
    )
    if matches and matches[0][1] >= threshold:
        return matches[0][0]
    return asr_entity  # keep original if no match passes threshold


# ── 4. Correct query ─────────────────────────────────
def correct_query(asr_query: str, kb_entities: list, threshold: int = 70) -> dict:
    # Extract entities from ASR query
    asr_entities = extract_entities(asr_query)

    corrected = asr_query
    corrections = []

    for ent in asr_entities:
        matched = match_entity(ent, kb_entities, threshold)
        if matched != ent:
            corrected = corrected.replace(ent, matched)
            corrections.append(
                {
                    "original": ent,
                    "corrected": matched,
                }
            )

    return {
        "original": asr_query,
        "corrected": corrected,
        "entities_found": asr_entities,
        "corrections": corrections,
    }


# ── 5. Main experiment ───────────────────────────────
def run_correction(
    nbest_path: str,
    docs_dir: str,
    accent: str = "ng",
    threshold: int = 70,
    output_path: str = "data/corrected_queries_v2.json",
):
    kb_entities = load_kb_entities(docs_dir)
    print(f"Loaded {len(kb_entities)} KB entities")

    with open(nbest_path) as f:
        data = json.load(f)

    results = []
    for sample in tqdm(data, desc=f"Correcting {accent}"):
        if accent not in sample["accents"]:
            continue

        asr_top1 = sample["accents"][accent]["top1"]

        correction = correct_query(asr_top1, kb_entities, threshold)

        results.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "asr_top1": asr_top1,
                "corrected": correction["corrected"],
                "corrections": correction["corrections"],
            }
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Stats
    changed = sum(1 for r in results if r["corrections"])
    print(f"\nModified: {changed}/{len(results)}")

    # Examples
    print("\nExamples:")
    count = 0
    for r in results:
        if r["corrections"] and count < 3:
            print(f"\n  Original:  {r['question']}")
            print(f"  ASR:       {r['asr_top1']}")
            print(f"  Corrected: {r['corrected']}")
            print(f"  Changes:   {r['corrections']}")
            count += 1

    return results


if __name__ == "__main__":
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parent.parent
    _PROJECT_ROOT = _ROOT.parent

    run_correction(
        nbest_path=str(_ROOT / "data" / "accent_nbest_results.json"),
        docs_dir=str(_PROJECT_ROOT / "dataset" / "hotpotqa_1000_hf" / "documents"),
        accent="ng",
        threshold=70,
        output_path=str(_ROOT / "data" / "corrected_ng_queries_v2.json"),
    )
