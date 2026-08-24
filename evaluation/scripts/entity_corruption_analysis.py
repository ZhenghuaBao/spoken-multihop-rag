"""
Entity-corruption analysis for paper Section III-A.

Reproduces the validity-of-TTS-based-evaluation numbers:
  - utterance-level: fraction of utterances with >=1 corrupted entity
  - entity-instance level: fraction of gold entities altered
  - per-word-token level: entity-words vs non-entity-words corruption rates

Methodology (matches paper text):
  - spaCy NER (en_core_web_sm) on gold reference
  - alphanumeric-normalized substring miss in ASR hypothesis
    (lowercase + strip non-alphanumeric; entity is "altered" if its
    normalized form is NOT a contiguous substring of normalized hyp)

Inputs:
  - Real NG: results/nigerian_validation.json per_sample[*].{reference,hypothesis}
  - Synth NG: data/2wiki_spoken/accent_nbest_results_2wiki.json
              [*].{question, accents.ng.top1}

Output: results/entity_corruption_analysis.json

Usage:
    python evaluation/scripts/entity_corruption_analysis.py

All input/output paths default to the repository-relative locations above
and can be overridden with --real-ng / --synth-ng / --output.
"""

import argparse
import json
import sys
import io
from pathlib import Path

import spacy

_ROOT = Path(__file__).resolve().parent.parent.parent  # repository root

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)


def norm_alphanum(s: str) -> str:
    """Lowercase, keep only alphanumeric characters."""
    return "".join(c for c in s.lower() if c.isalnum())


def is_entity_corrupted(entity: str, hyp: str) -> bool:
    """True iff normalized entity is NOT a contiguous substring of normalized hyp."""
    en = norm_alphanum(entity)
    hn = norm_alphanum(hyp)
    if not en:
        return False
    return en not in hn


def find_entity_spans(reference: str, entities):
    """Return character-position spans (start, end) of each entity occurrence in reference.
    Uses spaCy's span info for exactness.
    """
    spans = []
    for ent in entities:
        spans.append((ent.start_char, ent.end_char))
    return spans


def analyze_pair(reference: str, hypothesis: str, nlp):
    """Per-utterance analysis. Returns dict with counts:
    n_entities, n_corrupted_entities,
    n_entity_words, n_corrupted_entity_words,
    n_nonentity_words, n_corrupted_nonentity_words.
    """
    doc = nlp(reference)
    entities = list(doc.ents)
    hn = norm_alphanum(hypothesis)

    # Entity-instance level
    n_entities = len(entities)
    n_corrupted_entities = sum(
        1 for ent in entities if is_entity_corrupted(ent.text, hypothesis)
    )

    # Per-word-token level: each constituent word of every entity counted separately
    n_entity_words = 0
    n_corrupted_entity_words = 0
    for ent in entities:
        for w in ent.text.split():
            wn = norm_alphanum(w)
            if not wn:
                continue
            n_entity_words += 1
            if wn not in hn:
                n_corrupted_entity_words += 1

    # Non-entity words: whitespace-tokenize ref, exclude any word whose
    # character span overlaps an entity span
    occupied = [False] * len(reference)
    for ent in entities:
        for k in range(ent.start_char, ent.end_char):
            occupied[k] = True

    n_nonentity_words = 0
    n_corrupted_nonentity_words = 0
    pos = 0
    for word in reference.split():
        wpos = reference.find(word, pos)
        if wpos < 0:
            pos += len(word) + 1
            continue
        in_ent = any(occupied[wpos : wpos + len(word)])
        pos = wpos + len(word)
        if in_ent:
            continue  # already counted as entity word above
        wn = norm_alphanum(word)
        if not wn:
            continue
        n_nonentity_words += 1
        if wn not in hn:
            n_corrupted_nonentity_words += 1

    return {
        "n_entities": n_entities,
        "n_corrupted_entities": n_corrupted_entities,
        "n_entity_words": n_entity_words,
        "n_corrupted_entity_words": n_corrupted_entity_words,
        "n_nonentity_words": n_nonentity_words,
        "n_corrupted_nonentity_words": n_corrupted_nonentity_words,
    }


def aggregate(samples):
    """Aggregate per-utterance counts into corpus-level statistics."""
    samples_with_entities = [s for s in samples if s["n_entities"] >= 1]
    n_utt_total = len(samples)
    n_utt_with_ent = len(samples_with_entities)
    n_utt_corrupt = sum(
        1 for s in samples_with_entities if s["n_corrupted_entities"] >= 1
    )
    total_ent = sum(s["n_entities"] for s in samples_with_entities)
    total_corr = sum(s["n_corrupted_entities"] for s in samples_with_entities)
    total_ew = sum(s["n_entity_words"] for s in samples_with_entities)
    total_cew = sum(s["n_corrupted_entity_words"] for s in samples_with_entities)
    total_new = sum(s["n_nonentity_words"] for s in samples_with_entities)
    total_cnew = sum(s["n_corrupted_nonentity_words"] for s in samples_with_entities)

    return {
        "n_utterances": n_utt_total,
        "n_utterances_with_entity": n_utt_with_ent,
        "n_utterances_corrupted": n_utt_corrupt,
        "utterance_level_rate": n_utt_corrupt / max(n_utt_with_ent, 1),
        "n_entity_instances": total_ent,
        "n_corrupted_entity_instances": total_corr,
        "entity_instance_rate": total_corr / max(total_ent, 1),
        "n_entity_word_tokens": total_ew,
        "n_corrupted_entity_word_tokens": total_cew,
        "entity_word_rate": total_cew / max(total_ew, 1),
        "n_nonentity_word_tokens": total_new,
        "n_corrupted_nonentity_word_tokens": total_cnew,
        "nonentity_word_rate": total_cnew / max(total_new, 1),
    }


def run_real_ng(nlp, path):
    print(f"\n[Real-NG] loading {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["per_sample"]
    print(f"  n samples: {len(samples)}")

    per_utt = []
    for i, s in enumerate(samples):
        ref = s.get("reference", "")
        hyp = s.get("hypothesis", "")
        if not ref or not hyp:
            continue
        per_utt.append(analyze_pair(ref, hyp, nlp))
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(samples)}]")

    return aggregate(per_utt)


def run_synth_ng(nlp, path):
    print(f"\n[Synth-NG] loading {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  n questions: {len(data)}")

    per_utt = []
    for i, item in enumerate(data):
        ref = item.get("question", "")
        hyp = item.get("accents", {}).get("ng", {}).get("top1", "")
        if not ref or not hyp:
            continue
        per_utt.append(analyze_pair(ref, hyp, nlp))
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(data)}]")

    return aggregate(per_utt)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--real-ng",
        type=Path,
        default=_ROOT / "results" / "nigerian_validation.json",
        help="Output of evaluation/scripts/real_speech_validation.py",
    )
    p.add_argument(
        "--synth-ng",
        type=Path,
        default=_ROOT / "data" / "2wiki_spoken" / "accent_nbest_results_2wiki.json",
        help="2WikiMultiHopQA accent transcripts (gold question + accents.ng.top1)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "results" / "entity_corruption_analysis.json",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    real_ng_path = args.real_ng
    synth_ng_path = args.synth_ng
    output_path = args.output

    for label, path in (("--real-ng", real_ng_path), ("--synth-ng", synth_ng_path)):
        if not path.exists():
            raise SystemExit(
                f"Missing input {label}: {path}\n"
                "Speech transcripts are not bundled with the code. See the "
                "'Datasets' section of README.md."
            )

    print("Loading spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")

    real = run_real_ng(nlp, real_ng_path)
    synth = run_synth_ng(nlp, synth_ng_path)

    # Compute synth/real ratios
    ratios = {
        "utterance_level": (
            synth["utterance_level_rate"] / real["utterance_level_rate"]
        )
        if real["utterance_level_rate"]
        else 0,
        "entity_instance": (
            synth["entity_instance_rate"] / real["entity_instance_rate"]
        )
        if real["entity_instance_rate"]
        else 0,
        "entity_word": (synth["entity_word_rate"] / real["entity_word_rate"])
        if real["entity_word_rate"]
        else 0,
        "nonentity_word": (synth["nonentity_word_rate"] / real["nonentity_word_rate"])
        if real["nonentity_word_rate"]
        else 0,
    }

    out = {
        "methodology": {
            "ner": "spaCy en_core_web_sm",
            "corruption_test": "alphanumeric-normalized substring miss",
            "entity_word_definition": "each constituent word of each entity-instance counted separately",
            "nonentity_word_definition": "whitespace-tokenized words whose char span does not overlap any spaCy entity span",
        },
        "real_ng": real,
        "synth_ng": synth,
        "synth_over_real_ratios": ratios,
        "sources": {
            "real_ng_per_sample": str(real_ng_path),
            "synth_ng_questions": str(synth_ng_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"\nReal NG ({real['n_utterances']} utterances):")
    print(
        f"  with entity:       {real['n_utterances_with_entity']}/"
        f"{real['n_utterances']}"
    )
    print(
        f"  utterance-level:   {real['n_utterances_corrupted']}/"
        f"{real['n_utterances_with_entity']} = "
        f"{real['utterance_level_rate'] * 100:.1f}%"
    )
    print(
        f"  entity-instance:   {real['n_corrupted_entity_instances']}/"
        f"{real['n_entity_instances']} = "
        f"{real['entity_instance_rate'] * 100:.1f}%"
    )
    print(
        f"  entity-word:       {real['n_corrupted_entity_word_tokens']}/"
        f"{real['n_entity_word_tokens']} = "
        f"{real['entity_word_rate'] * 100:.1f}%"
    )
    print(
        f"  non-entity-word:   {real['n_corrupted_nonentity_word_tokens']}/"
        f"{real['n_nonentity_word_tokens']} = "
        f"{real['nonentity_word_rate'] * 100:.1f}%"
    )

    print(f"\nSynth NG ({synth['n_utterances']} questions):")
    print(
        f"  with entity:       {synth['n_utterances_with_entity']}/"
        f"{synth['n_utterances']}"
    )
    print(
        f"  utterance-level:   {synth['n_utterances_corrupted']}/"
        f"{synth['n_utterances_with_entity']} = "
        f"{synth['utterance_level_rate'] * 100:.1f}%"
    )
    print(
        f"  entity-instance:   {synth['n_corrupted_entity_instances']}/"
        f"{synth['n_entity_instances']} = "
        f"{synth['entity_instance_rate'] * 100:.1f}%"
    )
    print(
        f"  entity-word:       {synth['n_corrupted_entity_word_tokens']}/"
        f"{synth['n_entity_word_tokens']} = "
        f"{synth['entity_word_rate'] * 100:.1f}%"
    )
    print(
        f"  non-entity-word:   {synth['n_corrupted_nonentity_word_tokens']}/"
        f"{synth['n_nonentity_word_tokens']} = "
        f"{synth['nonentity_word_rate'] * 100:.1f}%"
    )

    print("\nSynth/Real ratios:")
    print(f"  utterance-level:   {ratios['utterance_level']:.2f}x")
    print(f"  entity-instance:   {ratios['entity_instance']:.2f}x")
    print(f"  entity-word:       {ratios['entity_word']:.2f}x")
    print(f"  non-entity-word:   {ratios['nonentity_word']:.2f}x")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
