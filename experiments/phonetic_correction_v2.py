"""
Phonetic-aware Entity Correction (v2).

Improvements over v1 (phonetic_correction.py):
  1. Word-level Double Metaphone with bag-of-codes Jaccard match
     (instead of buggy whole-string concat).
  2. Optional spaCy NER for candidate entity extraction (robust to
     Whisper's unstable casing).
  3. Optional corpus-body entity extraction (instead of only document
     titles), recovers entities that appear inside passages.
  4. Word-boundary-safe replacement (prevents substring overlap).
  5. Tunable thresholds: phonetic Jaccard min and edit-distance min,
     with separate stricter threshold for full-corpus fallback.

Usage:
    python experiments/phonetic_correction_v2.py \\
        --asr-data data/2wiki_spoken/accent_nbest_results_2wiki.json \\
        --docs-dir dataset/2wikimultihopqa_1000/documents \\
        --accent ng \\
        --jaccard 0.4 --edit-thresh 75 --fallback-thresh 85 \\
        --use-spacy \\
        --output data/2wiki_spoken/accent_nbest_results_2wiki_phonetic_v2.json
"""

import argparse
import json
import os
import sys
import io
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz, process
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)


# ---------------------------------------------------------------------------
# Phonetic encoder: Double Metaphone (per word)
# ---------------------------------------------------------------------------
try:
    from metaphone import doublemetaphone

    def word_code(word: str) -> str:
        primary, secondary = doublemetaphone(word)
        return primary or secondary or ""
except ImportError:

    def word_code(word: str) -> str:
        # Soundex fallback (lossier than DMP)
        word = re.sub(r"[^A-Za-z]", "", word).upper()
        if not word:
            return ""
        first = word[0]
        m = {
            **dict.fromkeys("BFPV", "1"),
            **dict.fromkeys("CGJKQSXZ", "2"),
            **dict.fromkeys("DT", "3"),
            "L": "4",
            **dict.fromkeys("MN", "5"),
            "R": "6",
        }
        encoded = "".join(m.get(c, "") for c in word[1:])
        out = first
        for c in encoded:
            if not out or c != out[-1]:
                out += c
        return (out + "0000")[:4]


def entity_codes(entity: str) -> Tuple[str, ...]:
    """Per-word phonetic code tuple. Stable across word permutations."""
    words = [w for w in re.split(r"\s+", entity.strip()) if w]
    return tuple(word_code(w) for w in words if word_code(w))


def codes_jaccard(q_codes: Tuple[str, ...], e_codes: Tuple[str, ...]) -> float:
    """Jaccard similarity over phonetic-code multisets."""
    q_set, e_set = set(q_codes), set(e_codes)
    if not q_set or not e_set:
        return 0.0
    return len(q_set & e_set) / len(q_set | e_set)


# ---------------------------------------------------------------------------
# Candidate entity extraction
# ---------------------------------------------------------------------------
COMMON_LEADING = {
    "Which",
    "What",
    "Where",
    "When",
    "Who",
    "How",
    "Why",
    "The",
    "Are",
    "Is",
    "Was",
    "Were",
    "Do",
    "Does",
    "Did",
    "Can",
    "Could",
    "Would",
    "Will",
    "Should",
    "May",
    "Might",
}
NER_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "WORK_OF_ART",
    "EVENT",
    "PRODUCT",
    "NORP",
}

ENTITY_RE = re.compile(r"\b[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*\b")


def regex_extract(text: str) -> List[str]:
    matches = ENTITY_RE.findall(text)
    out, seen = [], set()
    for m in matches:
        ws = m.split()
        if len(ws) == 1 and ws[0] in COMMON_LEADING:
            continue
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def spacy_extract(text: str, nlp) -> List[str]:
    """spaCy NER. More robust to Whisper's unstable casing."""
    doc = nlp(text)
    out, seen = [], set()
    for e in doc.ents:
        if e.label_ in NER_LABELS:
            t = e.text.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


# ---------------------------------------------------------------------------
# Corpus entity sources
# ---------------------------------------------------------------------------
def load_title_entities(docs_dir: str) -> List[str]:
    out = []
    for f in os.listdir(docs_dir):
        if f.endswith(".txt"):
            out.append(f.replace(".txt", "").replace("_", " "))
    return out


def load_body_entities(docs_dir: str, nlp, max_files: int = None) -> Set[str]:
    """Run spaCy NER over document bodies to expand the entity index."""
    print("  Extracting body entities via spaCy NER...")
    files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]
    if max_files:
        files = files[:max_files]
    body_ents: Set[str] = set()
    for f in tqdm(files, desc="  body NER"):
        with open(os.path.join(docs_dir, f), encoding="utf-8", errors="ignore") as fp:
            text = fp.read()[:50000]  # cap to keep NER tractable
        try:
            doc = nlp(text)
            for e in doc.ents:
                if e.label_ in NER_LABELS and 2 <= len(e.text) <= 80:
                    body_ents.add(e.text.strip())
        except Exception:
            continue
    return body_ents


# ---------------------------------------------------------------------------
# Phonetic index
# ---------------------------------------------------------------------------
def build_phonetic_index(entities: List[str]) -> Dict[str, List[str]]:
    """Map each phonetic code -> list of entities containing that code."""
    idx: Dict[str, List[str]] = defaultdict(list)
    for ent in entities:
        for c in set(entity_codes(ent)):
            idx[c].append(ent)
    return idx


# ---------------------------------------------------------------------------
# Word-boundary-safe replacement
# ---------------------------------------------------------------------------
def safe_replace(text: str, old: str, new: str) -> Tuple[str, int]:
    pattern = re.compile(r"\b" + re.escape(old) + r"\b", re.IGNORECASE)
    return pattern.subn(new, text, count=1)


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------
def phonetic_match(
    query_entity: str,
    phonetic_idx: Dict[str, List[str]],
    all_entities: List[str],
    jaccard_min: float,
    edit_thresh: int,
    fallback_thresh: int,
) -> Tuple[str, float, str]:
    """Return (best_match, score, source) or ("", 0, "none")."""
    q_codes = entity_codes(query_entity)
    if not q_codes:
        return "", 0.0, "none"

    # Step 1: collect candidates whose entity codes overlap with q_codes
    candidate_set: Set[str] = set()
    for c in set(q_codes):
        candidate_set.update(phonetic_idx.get(c, []))

    # Step 2: rank candidates by Jaccard on phonetic codes
    scored = []
    for ent in candidate_set:
        e_codes = entity_codes(ent)
        j = codes_jaccard(q_codes, e_codes)
        if j >= jaccard_min:
            scored.append((ent, j))
    scored.sort(key=lambda x: -x[1])

    if scored:
        # Step 3a: among phonetic candidates, pick best by edit distance
        top_candidates = [s[0] for s in scored[:50]]
        best, score, _ = process.extractOne(
            query_entity, top_candidates, scorer=fuzz.ratio
        )
        if score >= edit_thresh:
            return best, score, "phonetic"

    # Step 3b: fallback - global edit distance with stricter threshold
    best, score, _ = process.extractOne(query_entity, all_entities, scorer=fuzz.ratio)
    if score >= fallback_thresh:
        return best, score, "fallback"

    return "", 0.0, "none"


def correct_query(
    asr_query: str,
    phonetic_idx: Dict[str, List[str]],
    all_entities: List[str],
    jaccard_min: float,
    edit_thresh: int,
    fallback_thresh: int,
    extractor,
) -> Dict:
    asr_entities = extractor(asr_query)
    corrected = asr_query
    corrections = []

    # Replace longer entities first to avoid substring conflicts
    for ent in sorted(asr_entities, key=lambda x: -len(x)):
        match, score, source = phonetic_match(
            ent,
            phonetic_idx,
            all_entities,
            jaccard_min,
            edit_thresh,
            fallback_thresh,
        )
        if not match or match.lower() == ent.lower():
            continue
        corrected, n = safe_replace(corrected, ent, match)
        if n > 0:
            corrections.append((ent, match, score, source))

    return {
        "corrected": corrected,
        "entities_found": asr_entities,
        "corrections": corrections,
    }


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def build_corrected_asr_data(
    asr_data_path: str,
    docs_dir: str,
    accent: str,
    output_path: str,
    jaccard_min: float,
    edit_thresh: int,
    fallback_thresh: int,
    use_spacy: bool,
    expand_corpus: bool,
) -> None:
    title_entities = load_title_entities(docs_dir)
    print(f"Loaded {len(title_entities)} title entities from {docs_dir}")

    extractor = regex_extract
    nlp = None
    body_entities: Set[str] = set()

    if use_spacy or expand_corpus:
        try:
            import spacy

            print("  loading spaCy en_core_web_sm...")
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("  en_core_web_sm not installed. Install with:")
                print("    python -m spacy download en_core_web_sm")
                raise
            if use_spacy:

                def extractor(t):
                    return spacy_extract(t, nlp)

                print("  using spaCy NER for candidate extraction")
            if expand_corpus:
                body_entities = load_body_entities(docs_dir, nlp)
                print(f"  added {len(body_entities)} body entities from corpus")
        except ImportError:
            print("  spaCy not installed; falling back to regex extraction")

    all_entities = list(set(title_entities) | body_entities)
    print(f"Total corpus entities: {len(all_entities)}")
    phonetic_idx = build_phonetic_index(all_entities)
    print(f"Phonetic index: {len(phonetic_idx)} codes")

    with open(asr_data_path, encoding="utf-8") as f:
        data = json.load(f)

    n_modified, n_total, n_corrections = 0, 0, 0
    source_counts: Dict[str, int] = defaultdict(int)

    for item in tqdm(data, desc=f"Correcting {accent}"):
        if accent not in item.get("accents", {}):
            continue
        asr_top1 = item["accents"][accent]["top1"]
        if not asr_top1 or not isinstance(asr_top1, str):
            continue

        result = correct_query(
            asr_top1,
            phonetic_idx,
            all_entities,
            jaccard_min,
            edit_thresh,
            fallback_thresh,
            extractor,
        )
        n_total += 1
        if result["corrected"] != asr_top1:
            n_modified += 1
            n_corrections += len(result["corrections"])
            for _, _, _, src in result["corrections"]:
                source_counts[src] += 1

        item["accents"][accent]["top1"] = result["corrected"]
        if "nbest" in item["accents"][accent] and item["accents"][accent]["nbest"]:
            item["accents"][accent]["nbest"][0]["text"] = result["corrected"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    pct = n_modified / n_total * 100 if n_total else 0
    print(f"\nPhonetic v2 stats for accent={accent}:")
    print(f"  Total queries:          {n_total}")
    print(f"  Modified queries:       {n_modified} ({pct:.1f}%)")
    print(f"  Total entity replaced:  {n_corrections}")
    for src, cnt in source_counts.items():
        print(f"    via {src}: {cnt}")
    print(f"  Saved to: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asr-data", required=True)
    p.add_argument("--docs-dir", required=True)
    p.add_argument("--accent", default="ng")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--jaccard",
        type=float,
        default=0.4,
        help="Min phonetic-code Jaccard for candidate set",
    )
    p.add_argument(
        "--edit-thresh",
        type=int,
        default=75,
        help="Min edit-distance ratio within phonetic pool",
    )
    p.add_argument(
        "--fallback-thresh",
        type=int,
        default=85,
        help="Stricter min ratio for full-corpus fallback",
    )
    p.add_argument(
        "--use-spacy",
        action="store_true",
        help="Use spaCy NER for candidate extraction",
    )
    p.add_argument(
        "--expand-corpus",
        action="store_true",
        help="Expand corpus entities via spaCy NER on document bodies",
    )
    args = p.parse_args()

    build_corrected_asr_data(
        asr_data_path=args.asr_data,
        docs_dir=args.docs_dir,
        accent=args.accent,
        output_path=args.output,
        jaccard_min=args.jaccard,
        edit_thresh=args.edit_thresh,
        fallback_thresh=args.fallback_thresh,
        use_spacy=args.use_spacy,
        expand_corpus=args.expand_corpus,
    )


if __name__ == "__main__":
    main()
