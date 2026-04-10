# method/run_entity_correction.py

import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()


def correct_entities_basic(asr_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""This question was transcribed from accented speech and may contain entity recognition errors (proper nouns, names, places may be misspelled).

Correct any potential entity errors while keeping the question structure exactly the same. If unsure, keep the original.

Transcription: {asr_text}
Corrected:""",
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def run_correction_experiment(
    nbest_results_path: str,
    accent: str = "ng",
    output_path: str = "data/corrected_queries.json",
):
    # 加载 N-best results
    with open(nbest_results_path, encoding="utf-8") as f:
        data = json.load(f)

    corrected = []

    for sample in tqdm(data, desc=f"Correcting {accent} queries"):
        if accent not in sample["accents"]:
            continue

        asr_top1 = sample["accents"][accent]["top1"]
        original = sample["question"]

        # 跳过空的
        if not asr_top1:
            corrected.append(
                {
                    "id": sample["id"],
                    "question": original,
                    "answer": sample["answer"],
                    "asr_top1": asr_top1,
                    "corrected": asr_top1,
                    "accent": accent,
                }
            )
            continue

        corrected_text = correct_entities_basic(asr_top1)

        corrected.append(
            {
                "id": sample["id"],
                "question": original,
                "answer": sample["answer"],
                "asr_top1": asr_top1,
                "corrected": corrected_text,
                "accent": accent,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(corrected)} corrected queries to {output_path}")

    # 快速统计有多少被修改了
    changed = sum(1 for s in corrected if s["corrected"] != s["asr_top1"])
    print(
        f"Modified: {changed}/{len(corrected)} ({changed / len(corrected) * 100:.1f}%)"
    )

    # 看几个例子
    print("\nExamples of corrections:")
    count = 0
    for s in corrected:
        if s["corrected"] != s["asr_top1"] and count < 3:
            print(f"\n  Original:  {s['question']}")
            print(f"  ASR top-1: {s['asr_top1']}")
            print(f"  Corrected: {s['corrected']}")
            count += 1

    return corrected


if __name__ == "__main__":
    run_correction_experiment(
        nbest_results_path="data/accent_nbest_results.json",
        accent="ng",
        output_path="data/corrected_ng_queries.json",
    )
