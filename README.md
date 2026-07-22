# Speech Signals Complement LLMs for Predicting Interpersonal Attraction in Speed Dating

Code and reproducibility materials for the ICMI 2026 paper *Speech Signals
Complement LLMs for Predicting Interpersonal Attraction in Speed Dating*.

This directory contains the code submitted with the paper. The maintained
repository is available at
<https://github.com/yurikomium/speech-llm-complementarity>.

## 1. Overview

This release provides the core training, fusion, evaluation, and numerical
coupling-analysis code, as well as the exact executable zero-shot LLM prompts
used in the experiments. Because the speed-dating corpus is available only under a
data-use agreement, this is not a self-contained reproduction package: the
original data, derived features, and model/API outputs are not redistributed.

### What this release can reproduce

With authorized corpus access and the external inputs documented in Section 3,
the released code covers supervised single-modality training, zero-shot LLM
inference, conversion to a common out-of-fold prediction format, score-level
fusion, aggregate evaluation metrics, and the numerical participant-level
coupling analysis. Without restricted inputs, the offline unit and synthetic
smoke tests still exercise the filename and direction semantics, split
validation, metric definitions, held-out-fold fusion protocol, and coupling
output contract.

### What this release cannot reproduce by itself

The package cannot regenerate the paper's numerical results from public inputs
alone. It does not contain the restricted corpus, transcripts, audio, upstream
feature artifacts, trained checkpoints, or saved LLM/API responses.

## 2. Setup

Python 3.10 or 3.11 is recommended. Other versions may work but are not
tested.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On systems where `python` already invokes Python 3, it may be used instead of
`python3` in the first command. After activation, the examples below use the
virtual environment's `python`.

Run the commands below from the repository root. Unless explicitly marked as
external, every `data/...` and `log/...` path is resolved under that root.

The offline core tests do not access the corpus or any paid API:

```bash
python -m unittest discover -s tests -v
```

The synthetic smoke test additionally generates clearly labeled artificial
LLM and HuBERT out-of-fold prediction CSVs, runs weighted fusion for all four
Session x direction conditions, checks held-out-fold exclusion and output
contracts, and passes the resulting predictions to the coupling analysis:

```bash
bash scripts/smoke_test.sh
```

It uses a temporary directory, normally finishes in under one minute, and
does not reproduce any numerical result from the paper. Set
`CODE_RELEASE_PYTHON=/path/to/python` to select a specific Python executable.

### Environment variables

| Variable              | Required for              | Notes                                                        |
|-----------------------|---------------------------|--------------------------------------------------------------|
| `SD_DATA_ROOT`        | `convert_csv_to_pt.py`    | Base for conventional feature-CSV discovery; `--input-csv` overrides it. |
| `SD_AUDIO_DIR`        | `preprocessing.py`        | Default raw-audio base; `--audio-dir` overrides it. |
| `ANTHROPIC_API_KEY`   | `predict_liking.py`, `predict_liking_sess1.py` | Claude API key.                              |
| `OPENAI_API_KEY`      | `predict_liking_gpt54.py`, `predict_liking_gpt_audio.py` | OpenAI API key.                   |
| `GOOGLE_API_KEY`      | `predict_liking_gemini.py`, `predict_liking_gemma.py`   | Google Generative AI key.         |

API keys must be provided via environment variables; none are hard-coded.
The training, fusion, evaluation, and analysis scripts do not call paid APIs.
`SD_DATA_ROOT` and `SD_AUDIO_DIR` affect only the scripts named in the table;
the prompt scripts use the repository-local paths specified in Section 3.
Each received provider response is retained, including responses that fail
parsing or validation before a retry. Gemma capability-probe responses are
stored separately in `probe_<condition>.jsonl`.

### Experimental settings recorded in the release

The identifiers and parameters encoded in the released scripts are listed
below.

| Component | Released setting |
|---|---|
| Python | 3.10 or 3.11 |
| Supervised training | seed 42; 25-fold Leave-One-Group-Out CV; validation CCC for early stopping |
| Speech/text encoders | frozen `facebook/hubert-large-ll60k` (1024 dimensions) and `sentence-transformers/sentence-t5-large` (768 dimensions) |
| Primary LLM | `claude-sonnet-4-6`, Condition A: reasoning off, temperature 0, forced structured tool output |
| Claude sensitivity | Condition B: adaptive thinking with structured output |
| Other text LLMs | `gpt-5.4`, temperature 0; `gemma-3-12b-it`, temperature 0 |
| Direct-input MLLMs | `gemini-2.5-flash`, temperature 0 and thinking budget 0; `gpt-audio-mini`, temperature 0 |
| Primary fusion | weighted average; first-predictor weight in `{0.00, 0.05, ..., 1.00}`; per-participant CCC on pooled held-out predictions from all non-test folds |

The canonical prompt text is stored directly in `prompts/*.py`. Dependency
ranges are recorded in `requirements.txt`; they are compatibility bounds, not
a bit-for-bit snapshot of a provider's hosted model. `requirements-test.txt`
contains the smaller dependency set used by the offline checks.

## 3. Restricted inputs and runtime path contract

No runtime data is included; `data/` contains only `README.md`. Authorized
users must supply the restricted inputs described in the paper's *Dataset and
Task Definition* section.

| Component | Required input | Path or override | Output |
|---|---|---|---|
| CV splits | fold-assignment CSV | `--pairs-csv` | `data/cv_folds/` |
| Sentence-T5 / HuBERT | transcripts and WAV files | `--transcript-dir`, `--audio-dir` | `data/preprocessed/` |
| BERT / openSMILE / J-LIWC | per-utterance feature CSV | `--input-csv` | `data/preprocessed_<modality>/sess<N>/` |
| LLM / MLLM inference | prepared JSONL and, for T+A, audio inputs | fixed paths under `data/interim/llm_keyutts/` | model-specific directories under the same path |
| Training and fusion | `.pt`, CV splits, and prediction CSVs | command-line arguments | `log/modality_comparison/` or `--output-dir` |

Transcript, WAV, and old-style `.pt` filenames use:

```text
<kaisu>_<session>_<conv>_<room>_<female_id>_<male_id>_<suffix>
```

The second underscore-separated field is always `session`; the first is
`kaisu` and the third is `conv`. WAV files are read from
`<SD_AUDIO_DIR>/<kaisu>/<room>/`. Converted modality files use
`<pair_key>_<speaker>.pt`.

Prompt scripts expect `windows_full_624_sess1.jsonl` and
`windows_full_624.jsonl` under `data/interim/llm_keyutts/`; Gemma uses copies
under `all_624_llm_top3_v1/`. T+A additionally requires Gemini's
`audio_uris.json` or GPT-audio's `<pair_key>_sess<N>.mp3` files. Their
preparation is outside this release. JSONL records contain `pair_key` and
`windows[].utterances[]` (`start`, `speaker`, and `text`); the Gemini manifest
maps `<pair_key>_sess<N>` to a provider file URI.

Minimal preprocessing examples:

```bash
python src/preprocessing.py make_cvfold \
    --pairs-csv /path/to/fold_assignments_full.csv \
    --split-dir data/cv_folds

python src/preprocessing.py feature_extraction \
    --pairs-csv /path/to/fold_assignments_full.csv \
    --transcript-dir /path/to/transcript_csvs \
    --audio-dir /path/to/raw_audio \
    --output-dir data/preprocessed

python src/convert_csv_to_pt.py --modality bert --session 2 \
    --input-csv /path/to/bert_per_utterance_624.csv.gz \
    --pairs-csv /path/to/fold_assignments_full.csv
```

## 4. Running the released pipeline

The commands below document how the released components were used. Exact
reproduction on the original corpus requires the restricted inputs listed in
Section 3.

### LLM inference (Table 1 LLM rows)

Run each entry point for every value listed in its reported argument set:

| Model / input | Entry point | Reported arguments |
|---|---|---|
| Claude Sonnet 4.6, Session 2 | `prompts/predict_liking.py` | `--condition A` or `B`; `--direction m_to_f` or `f_to_m` |
| Claude Sonnet 4.6, Session 1 | `prompts/predict_liking_sess1.py` | `--condition A` or `B`; `--direction m_to_f` or `f_to_m` |
| GPT-5.4, T | `prompts/predict_liking_gpt54.py` | `--session 1` or `2`; both directions |
| Gemini 2.5 Flash, T / T+A | `prompts/predict_liking_gemini.py` | `--modality T` or `TA`; both sessions and directions |
| GPT-audio-mini, T+A | `prompts/predict_liking_gpt_audio.py` | both sessions and directions |
| Gemma 3 12B-IT, T | `prompts/predict_liking_gemma.py` | both sessions and directions |

For example:

```bash
python prompts/predict_liking.py --condition A --direction m_to_f
```

For scripts supporting both sessions, pass `--session 1` or `--session 2`;
all scripts use `--direction m_to_f` or `--direction f_to_m`.

All LLM scripts accept `--smoke` for a 2-pair limited run. This option still
calls the selected API and may incur cost; use the offline unit tests in
Section 2 for a no-API check. Predictions are written under
`data/interim/llm_keyutts/<model>_direct_liking/`.

Convert a saved JSONL to the common prediction CSV schema before fusion:

```bash
python src/llm_jsonl_to_predictions.py \
    --input-jsonl /path/to/claude_outputs.jsonl \
    --pairs-csv data/document/fold_assignments_full.csv \
    --direction f_to_m \
    --session 2 \
    --output-csv /path/to/claude_oof.csv
```

The converter removes duplicate-pair records and constant-score rater groups
using the same direction-, session-, and fold-group-specific rule as the
paper. It also requires every canonical pair to be present. The resulting row
counts are 604/619 pairs for F2M/M2F in Session 1 and 619/619 in Session 2.

### Supervised single-modality training and supervised fusion baselines

After the `.pt` files described in Section 3 are in place:

```bash
bash scripts/run_modality_comparison.sh --session 2 --skip-convert
bash scripts/run_modality_comparison.sh --session 1 --skip-convert
```

Omit `--skip-convert` only when all three external feature CSVs for that
session are present in the conventional `SD_DATA_ROOT` layout. The shell
orchestrator does not accept a different `--input-csv` for each modality; use
the explicit conversion commands in Section 3 first when the CSVs live
elsewhere.

This runs `main_single_modal.py` for each supervised predictor (Sentence-T5,
BERT, J-LIWC, HuBERT, and openSMILE) under
25-fold Leave-One-Group-Out CV, then runs supervised-only late-fusion
comparisons. Each predictor pools only the rating-side participant's own
utterances, as specified in the paper. Outputs land under
`log/modality_comparison/`.

The paper's primary Claude + HuBERT fusion is run directly with
`nway_fusion.py`, after both out-of-fold prediction CSVs are available:

```bash
python src/nway_fusion.py \
    --pred-csvs llm:/path/to/claude_oof.csv \
                hubert:/path/to/hubert_oof.csv \
    --methods weighted_avg \
    --speaker female \
    --output-dir /path/to/fusion_output
```

### Fusion weight optimization

`nway_fusion.py` supports unweighted and weighted averaging. The main paper uses the
weighted-average method with a 21-point grid search on per-participant CCC
over `w in {0.00, 0.05, ..., 1.00}`. For each test fold, the weight is selected
using the pooled out-of-fold predictions from all remaining folds, then applied
to that test fold without further tuning.

Fusion inputs must have unique `pair_key` values and consistent target, fold,
and participant metadata. `y_true` differences up to `1e-6` are accepted only
to accommodate six-decimal CSV serialization.

The participant-disjoint sensitivity protocol additionally removes every
selection-pool row involving a test-fold participant:

```bash
python src/nway_fusion.py \
    --pred-csvs llm:/path/to/claude_oof.csv \
                hubert:/path/to/hubert_oof.csv \
    --methods weighted_avg \
    --speaker female \
    --strict-participant-disjoint \
    --output-dir /path/to/out_strict
```

### Evaluation metrics

`src/evaluation_metrics.py` exposes `compute_metrics(df, id_col)` returning
participant-macro MAE / Pearson r / CCC, together with
`pairwise_accuracy(df, id_col, partner_col)` and the tie-aware
`top1_accuracy(df, id_col, partner_col)`. These functions expect a DataFrame
with `y_true`, `y_pred`, and participant/partner identifier columns.

### Participant-level association analysis (main-text rho values)

After single-modality CSVs (`results_{modality}.csv`) and the 2-way
fusion CSV (`fusion_weighted_avg.csv`) are produced, the per-participant
association between HuBERT speech-predictor accuracy and fusion gain over LLM
is computed by:

```bash
python src/coupling_analysis.py \
    --results-dir /path/to/log/modality_comparison \
    --output-dir /path/to/out
```

The script expects, under `--results-dir`, the directory layout
produced by `main_single_modal.py` + `nway_fusion.py`:

```
singles/{session}/{direction}/{mod}/results_{mod}.csv
fusion_2way/{session}/{direction}/{llm}+{speech}/fusion_{method}.csv
```

with CSV columns `pair_key, y_true, y_pred, female_id, male_id`. The
modality names for the LLM and speech predictor and the fusion method are
configurable (`--llm-name`, `--speech-name`, `--fusion-method`);
defaults are `llm`, `hubert`, and `weighted_avg`, matching the main
paper's bimodal fusion (LLM + HuBERT) with the weighted-average grid
search. The analysis script reports, for each condition, Spearman
`rho(r_HuBERT, r_Fusion - r_LLM)` across participants with a
Fisher-z 95% CI.

## 5. Paper-to-code map

| Paper item | Released entry point | Output or coverage | Important boundary |
|---|---|---|---|
| Figure 1, prediction pipeline | `prompts/*.py`, `src/main_single_modal.py`, `src/llm_jsonl_to_predictions.py`, `src/nway_fusion.py` | LLM score, HuBERT score, and fold-wise weighted fusion paths | Conceptual figure source is not included |
| Table 1, supervised single-modality and late-fusion rows | `src/main_single_modal.py`, `src/training.py`, `scripts/run_modality_comparison.sh` | `results_<modality>.csv` and score-level fusion outputs | Corpus features and checkpoints are not included; the utterance-level early-fusion runner is outside this release's core scope |
| Table 1, LLM/MLLM rows | `prompts/*.py`, then `src/llm_jsonl_to_predictions.py` | Common out-of-fold prediction CSV | Requires corpus access and, for hosted models, paid APIs; saved responses are not included |
| Table 1, primary LLM + HuBERT fusion | `src/nway_fusion.py --methods weighted_avg` | `fusion_weighted_avg.csv` plus selected weights | Requires both source prediction CSVs |
| Table 2, aggregate PW and Top-1 | `src/evaluation_metrics.py` | Participant-macro PW and tie-aware Top-1 | Rank-distance bins and significance-test reporting are not included |
| Table 3, incremental variance | Not included | None | Analysis code and derived participant-level artifacts are outside this release's core scope |
| Figure 2, participant-level association | `src/coupling_analysis.py` | Numerical Spearman rho and Fisher-z CI | Plot-generation code is not included |

## 6. Repository layout

```
speech-llm-complementarity/
├── .github/
│   └── workflows/tests.yml         # offline CI
├── .gitignore                       # excludes restricted/generated data
├── README.md                       # this file
├── LICENSE                         # MIT License for original software code
├── THIRD_PARTY_NOTICES.md          # excluded third-party materials
├── requirements.txt
├── requirements-test.txt           # smaller offline-test dependency set
├── CITATION.cff                    # citation metadata
├── data/
│   └── README.md                   # restricted-data notice; runtime paths are in Section 3
├── src/
│   ├── evaluation_metrics.py       # per-participant r / CCC / PW / Top-1
│   ├── main_single_modal.py        # single-modality training + CV
│   ├── nway_fusion.py              # score-level late fusion
│   ├── training.py                 # training loop
│   ├── preprocessing.py            # transcript/audio preprocessing + CV folds
│   ├── convert_csv_to_pt.py        # feature CSV -> .pt converter
│   ├── llm_jsonl_to_predictions.py # LLM JSONL -> common prediction CSV
│   └── coupling_analysis.py        # per-participant coupling rho + CI
├── tests/
│   ├── test_core.py                # filename, metric, and fusion-protocol tests
│   └── synthetic_smoke.py          # offline fusion + coupling E2E smoke
├── prompts/
│   ├── _api_response.py            # raw-response preservation helpers
│   ├── predict_liking.py           # Claude Sonnet, Session 2
│   ├── predict_liking_sess1.py     # Claude Sonnet, Session 1
│   ├── predict_liking_gpt54.py     # GPT-5.4 (text-only)
│   ├── predict_liking_gemini.py    # Gemini 2.5 Flash (T / TA)
│   ├── predict_liking_gpt_audio.py # gpt-audio-mini (T+A)
│   └── predict_liking_gemma.py     # Gemma 3 12B-IT (text-only)
└── scripts/
    ├── run_modality_comparison.sh  # supervised baseline orchestration
    └── smoke_test.sh               # one-command offline synthetic smoke
```

## 7. Notes and limitations

### Known limitations

- Hosted LLM outputs may differ after provider-side updates even with the same
  model identifier and temperature. API availability and pricing can also
  change.
- A full GPU retraining may not be bitwise deterministic across hardware and
  library versions.

### Prompt fidelity

The `prompts/*.py` scripts are the canonical record of the exact executable
Japanese prompts, including the full Q1--Q13 wording. The paper's appendix,
*LLM Prompt Details*, documents the prompt structure, English translation,
and placeholder substitutions while abbreviating the questionnaire item list
by citation. The English translation is documentation only and was not used as
model input in the reported experiments.

## 8. License

The original software code in this repository is licensed under the MIT
License. See [`LICENSE`](LICENSE).

This license does not grant rights to third-party materials. In particular,
the questionnaire item wording included in `prompts/`, which is based on
Rubin's Liking Scale and its Japanese translation cited in the accompanying
paper, is not relicensed under the MIT License. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The research corpus, transcripts, audio, derived data, model weights, and API
outputs are not included in this repository and are not covered by this
license.

## 9. Citation

If you use this software, cite the accompanying ICMI 2026 paper. Machine-readable
metadata and the current author order are provided in [`CITATION.cff`](CITATION.cff).
The paper DOI is
[`10.1145/3776574.3831151`](https://doi.org/10.1145/3776574.3831151).
An archival artifact DOI will be added after it is issued.
