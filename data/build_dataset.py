"""
Build a spoken-question dataset from a text-question JSON file using
neural TTS across multiple English accent voices.

Generic over any QA dataset whose queries JSON has the shape:
    {"queries": [{"id": ..., "question": ...}, ...]}
or simply a flat list of such dicts.

Usage:
    python data/build_dataset.py \\
        --questions /path/to/test_queries.json \\
        --output-dir data/<dataset>_spoken \\
        --n 1000
"""

import asyncio
import json
from pathlib import Path

import edge_tts

# Accented voices (Edge TTS).
VOICES = {
    "us": "en-US-JennyNeural",
    "in": "en-IN-NeerjaNeural",
    "ph": "en-PH-RosaNeural",
    "ng": "en-NG-EzinneNeural",
}


async def text_to_speech(
    text: str, output_path: str, voice: str = "en-US-JennyNeural", max_retries: int = 3
):
    # Sanitize: remove non-printable chars, ensure non-empty
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = "".join(c for c in text if c.isprintable() or c == " ")
    text = text.strip()
    if not text:
        text = "empty question"
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return
        except edge_tts.exceptions.NoAudioReceived:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                print(
                    f"  WARNING: TTS failed after {max_retries} retries for: {text[:80]}"
                )
                raise


async def add_speech_to_questions(
    questions_json: str,
    output_dir: str,
    n: int = 200,
):
    with open(questions_json, encoding="utf-8") as f:
        data = json.load(f)

    questions = data["queries"] if isinstance(data, dict) else data
    questions = questions[:n]

    audio_dir = Path(output_dir) / "audio"

    # Create per-accent subdirectories
    for accent in VOICES:
        (audio_dir / accent).mkdir(parents=True, exist_ok=True)

    spoken_questions = []

    for i, q in enumerate(questions):
        if i % 10 == 0:
            print(f"Generating speech {i}/{n}...")

        audio_paths = {}
        skipped = 0
        for accent, voice in VOICES.items():
            audio_path = audio_dir / accent / f"{q['id']}.wav"
            if audio_path.exists():
                skipped += 1
            else:
                try:
                    await text_to_speech(q["question"], str(audio_path), voice=voice)
                except Exception as e:
                    print(f"  ERROR at i={i} id={q['id']} accent={accent}: {e}")
                    print(f"  Question text: {q['question'][:120]}")
                    raise
            audio_paths[accent] = str(audio_path)
        if skipped == len(VOICES) and i % 100 == 0:
            print(f"  [{i}/{n}] skipped (audio exists)")

        spoken_questions.append(
            {
                **q,
                "audio_paths": audio_paths,
            }
        )

    output_path = Path(output_dir) / "spoken_questions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spoken_questions, f, indent=2, ensure_ascii=False)

    print(
        f"Done. {len(spoken_questions)} questions x {len(VOICES)} accents ({', '.join(VOICES)}) saved to {output_path}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to test_queries.json with the dataset's text questions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write audio/ and spoken_questions.json into",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of questions to synthesize (default 200)",
    )
    args = parser.parse_args()
    asyncio.run(
        add_speech_to_questions(
            questions_json=args.questions,
            output_dir=args.output_dir,
            n=args.n,
        )
    )
