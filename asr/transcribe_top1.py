"""
ASR top-1 transcription using Whisper large-v3.

Takes audio from the spoken dataset and produces
single-best transcription for each question.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import whisper
import numpy as np


class WhisperTop1Transcriber:
    """Whisper top-1 transcription wrapper."""

    def __init__(self, model_name: str = "large-v3", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper {model_name} on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        print("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """
        Transcribe a single audio array.

        Args:
            audio: numpy array of audio samples (float32, mono)
            sr: sample rate (Whisper expects 16kHz)

        Returns:
            dict with:
                - text: str (best transcription)
                - language: str
                - duration: float (audio length in seconds)
                - transcription_time: float (wall-clock time)
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed (Whisper expects 16kHz)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        start = time.time()
        result = self.model.transcribe(
            audio,
            language="en",
            fp16=(self.device == "cuda"),
        )
        elapsed = time.time() - start

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "en"),
            "duration": len(audio) / sr,
            "transcription_time": elapsed,
        }


def transcribe_dataset(
    records: List[Dict],
    model_name: str = "large-v3",
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Transcribe all spoken questions in the dataset.

    Args:
        records: list of dicts from load_dataset.py (must contain question_audio)
        model_name: Whisper model size
        output_path: optional path to save transcription results

    Returns:
        list of dicts with id, original_text, transcribed_text, wer info
    """
    transcriber = WhisperTop1Transcriber(model_name=model_name)
    results = []

    print(f"\nTranscribing {len(records)} questions (top-1)...")
    total_time = 0.0

    for i, record in enumerate(records):
        audio_data = record.get("question_audio")
        if audio_data is None:
            print(f"  [{i + 1}/{len(records)}] SKIP: no audio for {record['id']}")
            results.append(
                {
                    "id": record["id"],
                    "original_text": record["question_text"],
                    "transcribed_text": record["question_text"],  # fallback to text
                    "wer": 0.0,
                    "duration": 0.0,
                    "transcription_time": 0.0,
                    "skipped": True,
                }
            )
            continue

        # Extract audio array and sample rate
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]

        result = transcriber.transcribe(audio_array, sr=sr)
        total_time += result["transcription_time"]

        results.append(
            {
                "id": record["id"],
                "original_text": record["question_text"],
                "transcribed_text": result["text"],
                "language": result["language"],
                "duration": result["duration"],
                "transcription_time": result["transcription_time"],
                "skipped": False,
            }
        )

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i + 1}/{len(records)}] {result['text'][:60]}... ({result['transcription_time']:.2f}s)"
            )

    print(f"\nTop-1 transcription complete. Total time: {total_time:.1f}s")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-serializable fields for saving
        save_data = [{k: v for k, v in r.items()} for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"Saved transcriptions to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.load_dataset import load_spoken_hotpotqa

    parser = argparse.ArgumentParser(description="Whisper top-1 transcription")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument(
        "--output", type=str, default="results/transcriptions_top1.json"
    )
    args = parser.parse_args()

    records = load_spoken_hotpotqa(num_samples=args.samples)
    output = Path(__file__).parent.parent / args.output
    transcribe_dataset(records, model_name=args.model, output_path=output)
