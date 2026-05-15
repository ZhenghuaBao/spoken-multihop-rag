"""
SeamlessM4T-v2 transcription script for L40 server.

Usage on server:
    pip install transformers>=4.40 torchaudio librosa sentencepiece
    python transcribe_seamless_server.py \
        --audio-dir ./ng_audio \
        --output ./ng_seamless_transcripts.json \
        --batch-size 1

Outputs JSON: {qid: {"text": "...", "duration_s": ...}, ...}
"""

import argparse
import json
import time
from pathlib import Path

import librosa
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model


def load_audio(path: Path, target_sr: int = 16000):
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio, sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True, help="Directory of .wav files")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument(
        "--model-id",
        default="facebook/seamless-m4t-v2-large",
        help="HF model id",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Limit N files (for smoke test)"
    )
    args = p.parse_args()

    audio_dir = Path(args.audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))
    if args.limit:
        wav_files = wav_files[: args.limit]
    print(f"Found {len(wav_files)} wav files in {audio_dir}")

    print(f"Loading {args.model_id} on {args.device}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = SeamlessM4Tv2Model.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Resume from existing output if present
    out_path = Path(args.output)
    results = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} already done")

    t_start = time.time()
    for i, wav in enumerate(wav_files):
        qid = wav.stem
        if qid in results:
            continue

        try:
            audio, sr = load_audio(wav)
            if audio is None or len(audio) == 0:
                raise ValueError("empty audio")
        except Exception as e:
            print(f"  [SKIP] {qid}: {type(e).__name__}: {e}")
            results[qid] = {"text": "", "duration_s": 0.0, "skipped": True}
            continue

        inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt").to(
            args.device
        )
        if args.device == "cuda":
            inputs = {
                k: v.half() if v.dtype == torch.float32 else v
                for k, v in inputs.items()
            }

        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                tgt_lang="eng",
                generate_speech=False,
            )[0].tolist()

        text = processor.decode(tokens[0], skip_special_tokens=True)
        results[qid] = {"text": text, "duration_s": len(audio) / sr}

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(wav_files) - i - 1)
            print(
                f"[{i + 1}/{len(wav_files)}] elapsed={elapsed:.0f}s "
                f"eta={eta:.0f}s | last: {text[:60]!r}"
            )
            # Checkpoint every 25
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDone. {len(results)} transcripts saved to {out_path}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
