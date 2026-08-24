"""
Nigerian English ASR Validation.

Validates that our TTS Nigerian-accented speech produces ASR error profiles
consistent with real Nigerian English speakers. Uses the public
benjaminogbonna/nigerian_accented_english_dataset (parquet, ~3400 clips,
self-reported Nigerian accent).

We bypass torchcodec by setting decode=False on the audio column and
decoding mp3 bytes manually with librosa.

Usage:
    python evaluation/scripts/real_speech_validation.py \
        --n-samples 500 \
        --output results/nigerian_validation.json
"""

import argparse
import json
import sys
import io
import re
import statistics
from pathlib import Path
from io import BytesIO

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)


def _import_deps():
    import torch  # noqa: F401
    from datasets import load_dataset, Audio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import jiwer
    import librosa
    import soundfile as sf  # noqa: F401

    return (
        load_dataset,
        Audio,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        jiwer,
        librosa,
    )


# Reference TTS WER from main experiments
TTS_NG_WER = {
    "HotpotQA": 0.145,
    "2WikiMultiHopQA": 0.171,
    "MuSiQue": 0.080,
}


ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def count_corrupted_entities(reference: str, hypothesis: str) -> int:
    ref_entities = set(ENTITY_RE.findall(reference))
    hyp_lower = hypothesis.lower()
    return sum(1 for e in ref_entities if e.lower() not in hyp_lower)


def decode_audio_bytes(audio_bytes: bytes, target_sr: int = 16000):
    """Decode mp3/wav bytes using librosa (no torchcodec needed)."""
    import librosa

    audio, sr = librosa.load(BytesIO(audio_bytes), sr=target_sr, mono=True)
    return audio, sr


def run_validation(
    n_samples: int,
    output_path: str,
    model_id: str = "openai/whisper-large-v3",
    dataset_name: str = "benjaminogbonna/nigerian_accented_english_dataset",
    sampling_rate: int = 16000,
):
    (
        load_dataset,
        Audio,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        jiwer,
        librosa,
    ) = _import_deps()
    import torch

    # Combine train+val+test, then sample
    print(f"Loading {dataset_name}...")
    splits_to_use = ["train", "validation", "test"]
    samples = []
    for split in splits_to_use:
        try:
            ds = load_dataset(dataset_name, split=split)
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=sampling_rate, decode=False)
            )
            for item in ds:
                samples.append(item)
                if len(samples) >= n_samples:
                    break
        except Exception as e:
            print(f"  skip split={split}: {e}")
        if len(samples) >= n_samples:
            break

    print(f"Collected {len(samples)} samples.")
    if not samples:
        print("ERROR: no samples loaded.")
        return

    # Load Whisper
    print(f"Loading {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    per_sample, wers, corrupt_rates = [], [], []

    for i, item in enumerate(samples):
        audio_bytes = item["audio"]["bytes"]
        if audio_bytes is None:
            continue
        try:
            audio, sr = decode_audio_bytes(audio_bytes, target_sr=sampling_rate)
        except Exception as e:
            print(f"  skip sample {i}: decode error {e}")
            continue

        ref = (item.get("sentence") or "").strip()
        if not ref:
            continue

        inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                language="en",
                task="transcribe",
                num_beams=1,
                do_sample=False,
            )
        hyp = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        try:
            wer = jiwer.wer(ref, hyp)
        except ValueError:
            wer = 1.0
        n_corrupt = count_corrupted_entities(ref, hyp)
        n_ref_entities = len(set(ENTITY_RE.findall(ref)))
        corrupt_rate = n_corrupt / max(n_ref_entities, 1) if n_ref_entities else 0.0

        per_sample.append(
            {
                "id": item.get("client_id", f"sample_{i}"),
                "accent": item.get("accent", ""),
                "reference": ref,
                "hypothesis": hyp,
                "wer": wer,
                "n_ref_entities": n_ref_entities,
                "n_corrupted_entities": n_corrupt,
                "entity_corruption_rate": corrupt_rate,
            }
        )
        wers.append(wer)
        corrupt_rates.append(corrupt_rate)

        if (i + 1) % 25 == 0:
            print(
                f"  [{i + 1}/{len(samples)}] avg WER: "
                f"{statistics.mean(wers):.3f}, "
                f"avg entity-err: {statistics.mean(corrupt_rates):.3f}"
            )

    summary = {
        "n_samples": len(per_sample),
        "model_id": model_id,
        "dataset": dataset_name,
        "wer_mean": statistics.mean(wers) if wers else 0,
        "wer_median": statistics.median(wers) if wers else 0,
        "wer_std": statistics.stdev(wers) if len(wers) > 1 else 0,
        "wer_min": min(wers) if wers else 0,
        "wer_max": max(wers) if wers else 0,
        "entity_corruption_rate_mean": statistics.mean(corrupt_rates)
        if corrupt_rates
        else 0,
        "tts_ng_wer_reference": TTS_NG_WER,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "per_sample": per_sample},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("  Real Nigerian English ASR Validation Summary")
    print("=" * 70)
    print(f"  Dataset:                  {summary['dataset']}")
    print(f"  Samples:                  {summary['n_samples']}")
    print(f"  Real NG WER (mean):       {summary['wer_mean']:.3f}")
    print(f"  Real NG WER (median):     {summary['wer_median']:.3f}")
    print(f"  Real NG WER (std):        {summary['wer_std']:.3f}")
    print(
        f"  Real NG WER range:        [{summary['wer_min']:.3f}, "
        f"{summary['wer_max']:.3f}]"
    )
    print(f"  Entity corruption rate:   {summary['entity_corruption_rate_mean']:.3f}")
    print()
    print("  TTS NG WER (HotpotQA):       0.145")
    print("  TTS NG WER (2WikiMultiHopQA): 0.171")
    print("  TTS NG WER (MuSiQue):         0.080")
    print()

    real_wer = summary["wer_mean"]
    tts_min = min(TTS_NG_WER.values())
    tts_max = max(TTS_NG_WER.values())
    if tts_min - 0.05 <= real_wer <= tts_max + 0.10:
        print(
            "  Verdict: TTS NG WER is in a comparable range to real "
            "Nigerian English speakers. The TTS-based evaluation is "
            "a reasonable proxy."
        )
    else:
        print(
            f"  Verdict: real NG WER ({real_wer:.3f}) is OUTSIDE TTS "
            f"range [{tts_min:.3f}, {tts_max:.3f}]. Discuss in "
            "limitations."
        )
    print(f"\nSaved to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument(
        "--dataset", default="benjaminogbonna/nigerian_accented_english_dataset"
    )
    p.add_argument("--model-id", default="openai/whisper-large-v3")
    p.add_argument("--output", default="results/nigerian_validation.json")
    args = p.parse_args()

    run_validation(
        n_samples=args.n_samples,
        dataset_name=args.dataset,
        model_id=args.model_id,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
