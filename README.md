# Spoken Multi-hop RAG

Evaluation suite for studying how upstream ASR errors propagate through
multi-hop retrieval-augmented generation (RAG) architectures. Covers
three multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue),
four English accents (US, IN, PH, NG) synthesized via neural TTS, and
four RAG configurations (Naive RAG, HippoRAG2, IRCoT+Naive,
IRCoT+HippoRAG2).

## Directory layout

- `asr/` — Speech transcription scripts (Whisper-large-v3,
  SeamlessM4T-v2-large).
- `data/` — Dataset build/load utilities. Audio and large JSON dumps
  are gitignored; regenerate via the build scripts.
- `evaluation/` — Metrics, error categorization, statistical
  significance tests, validation against real accented speech.
- `experiments/` — Main 2x2 experiment runner (`run_2x2.py`) plus
  mitigation experiments (N-best rescoring, phonetic entity
  correction).
- `retrieval/` — Retrieval wrappers (dense, HippoRAG2).
- `results/` — Per-cell evaluation outputs. Gitignored; large.
- `logs/` — Run logs. Gitignored.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

API access: set `OPENAI_API_KEY` (for `gpt-4o-mini` answer generation
and `text-embedding-3-small` embeddings) in your environment or in a
local `.env.local`.

GPU: Whisper-large-v3 and SeamlessM4T-v2-large need ~12 GB of VRAM for
inference. The HippoRAG2 index loads ~4 GB into RAM (parquet).

## Reproducing the main results

Run the full $48$-cell main table (3 datasets × 4 methods × 4 accents
+ Oracle), per dataset:

```bash
python experiments/run_2x2.py \
    --cells A C G H \
    --accent all+oracle \
    --accent-json data/hotpotqa_spoken/accent_nbest_results_hotpotqa.json \
    --ground-truth ../dataset/hotpotqa_1000_hf/ground_truth.json \
    --output results/hotpotqa_1000.json
```

Each run produces a JSON with per-question F1/EM/WER and per-cell
summaries.

## Reproducing analyses

| Paper claim | Script |
|---|---|
| Paired bootstrap p-values + 95% CI | `evaluation/bootstrap_significance.py` |
| WER-stratified degradation (Fig. wer_threshold) | `evaluation/wer_bucket_per_method.py` |
| WER-threshold routing experiment | `evaluation/wer_routing_simulation.py` |
| Entity corruption rate (real NG vs synth) | `evaluation/entity_corruption_analysis.py` |
| Real Nigerian English validation | `evaluation/nigerian_validation.py` |
| Phonetic severity + conditional analysis | `evaluation/severity_distribution.py` |
| Hop-count breakdown on MuSiQue | `evaluation/hop_count_breakdown.py` |
| N-best rescoring table | `evaluation/nbest_table.py` |
| Cascade case-study examples | `evaluation/find_amplification_examples.py` |

## Mitigations

```bash
# Phonetic entity correction (Table 3)
python experiments/phonetic_correction_v2.py \
    --asr-data data/2wiki_spoken/accent_nbest_results_2wiki.json \
    --docs-dir ../dataset/2wikimultihopqa_1000/documents \
    --accent ng --use-spacy \
    --jaccard 0.4 --edit-thresh 75 --fallback-thresh 85 \
    --output data/2wiki_spoken/accent_nbest_results_2wiki_phonetic_v2.json
```

N-best rescoring is built into `run_2x2.py` (use cells B, D, I, J for
the N-best variants of A, C, G, H respectively).

## Cross-system ASR

```bash
python asr/transcribe_seamless_server.py \
    --audio-dir data/2wiki_spoken/audio/ng \
    --output data/2wiki_spoken/ng_seamless_transcripts.json
```

## Citation

If you use this code or findings, please cite:

```bibtex
@inproceedings{TODO_citekey,
  title     = {TODO: paper title},
  author    = {TODO: authors},
  booktitle = {TODO: venue},
  year      = {TODO},
  url       = {TODO: arxiv / anthology URL}
}
```
