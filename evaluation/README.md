# Evaluation

Two subpackages:

- **`core/`** — Imported by the experiment runner. Defines F1/EM/WER
  and the rule-based ASR error categorization used by every analysis
  downstream.
- **`scripts/`** — One-shot analyses. Each script produces one table or
  figure in the paper. Run from the project root with the invocations
  below.

## `core/`

| File | Purpose |
|---|---|
| `metrics.py` | Token-level F1, Exact Match, Word Error Rate, entity-recall helpers. Imported by `run_2x2.py`. |
| `error_analysis.py` | Per-question Oracle vs accent comparison + rule-based error categorization (entity / number / function-word / severe garbling / other content change). Produces the JSON behind Table 2; run with `--cross-method` it also produces the per-method JSON behind Table 8. |

## `scripts/` — one script per paper artifact

| Paper artifact | Script | One-line invocation |
|---|---|---|
| Table 1 stats ($p$-values, 95% CIs, amplification contrast) | `bootstrap_significance.py` | `python evaluation/scripts/bootstrap_significance.py --results results/2wiki_1000.json --n-resamples 10000` |
| Table 2 (Error-type distribution within degradation cases, per accent) | `error_type_distribution.py` | `python evaluation/scripts/error_type_distribution.py --error-analysis results/error_analysis_2wiki_all.json --cell A --dataset-name 2WikiMultiHopQA` |
| Table 8 (entity corruption per method and accent) | `cross_method_entity_table.py` | `python evaluation/scripts/cross_method_entity_table.py` |
| Appendix E (entity corruption, real NG vs synthetic NG) | `entity_corruption_analysis.py` | `python evaluation/scripts/entity_corruption_analysis.py` |
| Figure 3 (degradation stratified by WER bin) | `wer_stratified_degradation.py` | `python evaluation/scripts/wer_stratified_degradation.py --results results/2wiki_1000.json --accent ng` |
| Table 3 (Mitigation: top-1 / N-best Decoding / Phonetic Correction, with recoveries) | `mitigation_table.py` | `python evaluation/scripts/mitigation_table.py --baseline-results results/2wiki_1000.json --nbest-results results/2wiki_1000_nbest.json --phonetic-results results/2wiki_1000_phonetic_v2_corr.json --accent ng` |
| Appendix E (real-speech WER; feeds the real-vs-synthetic comparison) | `real_speech_validation.py` | `python evaluation/scripts/real_speech_validation.py --n-samples 500 --output results/nigerian_validation.json` |

## Notes

- All `--results` paths refer to JSONs produced by
  `experiments/run_2x2.py`; see the top-level `README.md` for how to
  generate them.
- `cross_method_entity_table.py` and `entity_corruption_analysis.py` run
  with no arguments because every path defaults to its
  repository-relative location. Pass `--help` to see the flags for
  pointing them elsewhere. Both fail immediately, naming the missing flag
  and path, if an input is absent.
- `cross_method_entity_table.py` needs one
  `error_analysis_<benchmark>_all_methods.json` per benchmark, each
  produced by `core/error_analysis.py --cross-method`. It prints both a
  plain-text table and the LaTeX for Table 8.
- `bootstrap_significance.py` takes ~30 s for $n{=}10{,}000$
  resamples on 1000 questions; CI numbers are deterministic given the
  default seed (`42`).
- `real_speech_validation.py` is the only script that requires a GPU
  (loads Whisper-large-v3 for re-transcribing the public real-NG
  corpus); the rest are CPU-only.
- `mitigation_table.py` produces the full Table 3 in a single
  invocation: N-best columns come from the N-best results file
  (cells B/D/I/J) and Phonetic columns come from the
  phonetic-corrected results file (cells A/C/G/H re-run on
  phonetic-corrected queries).
