"""
ASR N-best hypotheses extraction using Whisper large-v3.

Extracts multiple transcription hypotheses via beam search
for use in robust retrieval strategies.

Key approach: use Whisper's beam search to get top-N hypotheses,
then downstream retrieval can:
  (A) Retrieve separately per hypothesis, union results
  (B) Ensemble embeddings weighted by beam score
  (C) Concatenate hypotheses as a single enriched query
"""

import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np


class WhisperNBestTranscriber:
    """Whisper N-best transcription using beam search."""

    def __init__(
        self,
        model_name: str = "large-v3",
        beam_size: int = 5,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.beam_size = beam_size

        print(
            f"Loading Whisper {model_name} on {self.device} (beam_size={beam_size})..."
        )
        import whisper

        self.model = whisper.load_model(model_name, device=self.device)
        print("Whisper model loaded.")

    def transcribe_nbest(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> Dict:
        """
        Transcribe audio and return N-best hypotheses.

        Args:
            audio: numpy array of audio samples (float32, mono)
            sr: sample rate

        Returns:
            dict with:
                - hypotheses: list of {text, score, normalized_score}
                - best: str (top-1 text)
                - duration: float
                - transcription_time: float
        """
        import whisper

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        start = time.time()

        # Use Whisper's internal decoding with beam search
        # whisper.decode() gives access to beam search results
        audio_padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_padded).to(self.device)

        options = whisper.DecodingOptions(
            language="en",
            beam_size=self.beam_size,
            fp16=(self.device == "cuda"),
            without_timestamps=True,
        )

        elapsed = time.time() - start

        # Extract hypotheses from the result
        # whisper.decode returns a single DecodingResult for the best beam
        # To get N-best, we use the internal _decode_with_fallback approach
        hypotheses = self._extract_nbest(mel, options)

        return {
            "hypotheses": hypotheses,
            "best": hypotheses[0]["text"] if hypotheses else "",
            "duration": len(audio) / sr,
            "transcription_time": elapsed,
        }

    def _extract_nbest(self, mel: torch.Tensor, options) -> List[Dict]:
        """
        Extract N-best hypotheses using Whisper's beam search internals.

        Falls back to single hypothesis + perturbation if beam extraction
        is not directly supported.
        """
        import whisper

        hypotheses = []

        # Primary: use beam search with temperature variations
        # Whisper's beam search returns the best sequence; to get diverse
        # hypotheses we run multiple passes with slight temperature variation
        temperatures = [0.0, 0.2, 0.4, 0.6, 0.8][: self.beam_size]

        for i, temp in enumerate(temperatures):
            try:
                opts = whisper.DecodingOptions(
                    language="en",
                    beam_size=max(self.beam_size, 3),
                    fp16=(self.device == "cuda"),
                    without_timestamps=True,
                    temperature=temp,
                )
                result = whisper.decode(self.model, mel, opts)
                text = result.text.strip()

                # Compute a pseudo-score based on avg_logprob
                score = getattr(result, "avg_logprob", -1.0)

                # Avoid exact duplicates
                if not any(h["text"] == text for h in hypotheses):
                    hypotheses.append(
                        {
                            "text": text,
                            "score": float(score),
                            "rank": len(hypotheses),
                            "temperature": temp,
                        }
                    )
            except Exception as e:
                print(f"  Warning: beam {i} failed: {e}")
                continue

        if not hypotheses:
            # Ultimate fallback: greedy decode
            result = whisper.decode(self.model, mel, options)
            hypotheses.append(
                {
                    "text": result.text.strip(),
                    "score": float(getattr(result, "avg_logprob", -1.0)),
                    "rank": 0,
                    "temperature": 0.0,
                }
            )

        # Normalize scores to probabilities
        if len(hypotheses) > 1:
            scores = [h["score"] for h in hypotheses]
            max_score = max(scores)
            exp_scores = [math.exp(s - max_score) for s in scores]
            total = sum(exp_scores)
            for h, es in zip(hypotheses, exp_scores):
                h["normalized_score"] = es / total
        else:
            hypotheses[0]["normalized_score"] = 1.0

        # Sort by score (highest first)
        hypotheses.sort(key=lambda h: h["score"], reverse=True)
        for i, h in enumerate(hypotheses):
            h["rank"] = i

        return hypotheses


def transcribe_dataset_nbest(
    records: List[Dict],
    model_name: str = "large-v3",
    beam_size: int = 5,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Transcribe all spoken questions with N-best hypotheses.

    Returns list of dicts with id, original_text, hypotheses, best_text.
    """
    transcriber = WhisperNBestTranscriber(
        model_name=model_name,
        beam_size=beam_size,
    )
    results = []

    print(f"\nTranscribing {len(records)} questions (N-best, beam={beam_size})...")
    total_time = 0.0

    for i, record in enumerate(records):
        audio_data = record.get("question_audio")
        if audio_data is None:
            print(f"  [{i + 1}/{len(records)}] SKIP: no audio for {record['id']}")
            results.append(
                {
                    "id": record["id"],
                    "original_text": record["question_text"],
                    "hypotheses": [
                        {
                            "text": record["question_text"],
                            "score": 0.0,
                            "normalized_score": 1.0,
                            "rank": 0,
                        }
                    ],
                    "best_text": record["question_text"],
                    "duration": 0.0,
                    "transcription_time": 0.0,
                    "skipped": True,
                }
            )
            continue

        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]

        result = transcriber.transcribe_nbest(audio_array, sr=sr)
        total_time += result["transcription_time"]

        results.append(
            {
                "id": record["id"],
                "original_text": record["question_text"],
                "hypotheses": result["hypotheses"],
                "best_text": result["best"],
                "num_hypotheses": len(result["hypotheses"]),
                "duration": result["duration"],
                "transcription_time": result["transcription_time"],
                "skipped": False,
            }
        )

        if (i + 1) % 10 == 0 or i == 0:
            n_hyp = len(result["hypotheses"])
            print(
                f"  [{i + 1}/{len(records)}] {n_hyp} hypotheses | best: {result['best'][:50]}..."
            )

    print(f"\nN-best transcription complete. Total time: {total_time:.1f}s")
    print(
        f"Avg hypotheses per question: {sum(r['num_hypotheses'] for r in results if not r.get('skipped')) / max(1, sum(1 for r in results if not r.get('skipped'))):.1f}"
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved N-best transcriptions to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.load_dataset import load_spoken_hotpotqa

    parser = argparse.ArgumentParser(description="Whisper N-best transcription")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--output", type=str, default="results/transcriptions_nbest.json"
    )
    args = parser.parse_args()

    records = load_spoken_hotpotqa(num_samples=args.samples)
    output = Path(__file__).parent.parent / args.output
    transcribe_dataset_nbest(
        records, model_name=args.model, beam_size=args.beam_size, output_path=output
    )
