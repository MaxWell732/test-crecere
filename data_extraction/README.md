# collections-ai-vs-human — audio extraction pipeline

Audio → validated tables for 100 censored Colombian-Spanish debt-collection calls (50 human agents, 50 AI agent).
Specification: `collections_ai_vs_human_HANDOFF.md`. Extraction only — no analysis.

**Everything lives inside this directory:** the interpreter (`.cache/python`), uv cache (`.cache/uv`), models
(`.cache/huggingface`), temp files (`.cache/tmp`), data (`data/`, git-ignored). The raw audio is read from
`../data/raw/` and copied into `data/raw/`; nothing outside this directory is written.

## Setup

```bash
source scripts/env.sh          # always first: confines interpreter / caches to ./.cache
uv sync                        # Python 3.12 env from uv.lock
uv run extract check-env       # §5.3/§5.4 checks (GPU, ctranslate2, packages, HF token)
uv run pytest                  # unit tests on synthetic data
```

## Stages

| Command | Stage | Needs |
|---|---|---|
| `uv run extract download` | copy + sha256-verify raw WAVs | — |
| `uv run extract inventory` | E0 IDs, QC, censor detection, manifest | — |
| `uv run extract preprocess` | E1 16 kHz model copy | E0 |
| `uv run extract vad` | E2 Silero VAD, SNR | E1 |
| `uv run extract asr` | E3 faster-whisper large-v3 int8 (GPU) | E2 |
| `uv run extract diarize` | E4 pyannote (GPU) | E2 + **H2** |
| `uv run extract turns` | E5 turns, backchannels, interruptions | E3, E4 |
| `uv run extract roles-render` | E6 role inputs → `data/interim/llm_inputs/roles/` | E5 |
| *(Claude Code annotates roles)* | `data/interim/annotations/roles/*.json` | **H4** codebook approved |
| `uv run extract roles-validate` · `roles-review-sheet` | E6 checks + `data/interim/review/roles_review.csv` | annotations |
| *(The reviewer fills the review sheet — H5)* | `approved` Y/N, `corrected_role`, `accept_suspect`, `note` | |
| `uv run extract canonical` | E7 canonical JSON + transcripts → `llm_inputs/calls/` | H5 |
| `uv run extract features-all` | E8a–d | E7 |
| `uv run extract validation-select` | E10 dev/gold/WER ids + templates | E0 |
| *(Claude Code: pass A / pass B)* · `annotate-validate --pass a\|b\|rerun_a` · `annotate-status` | E9 | H4 |
| `uv run extract objection-qa` | E8d QA | pass A |
| *(The reviewer: gold annotations H6, WER references H7)* | `data/validation/` | |
| `uv run extract validate` | E10 metrics + reliability decisions | H5, H6, H7, E9 |
| `uv run extract tables` | E11 tables + data dictionary → `data/processed/` | all |

Shortcuts: `uv run extract audio-all` (download → E5, each stage in its own process) and `uv run extract features-all`.
Every stage is cached per call (config hash); `--calls C001,C002` restricts, `--force` recomputes.
Smoke test (§5.5): `scripts/smoke_test.sh asr diarize turns`.

## Human checkpoints

- **H2** Hugging Face: create a read token, accept the terms of `pyannote/speaker-diarization-community-1`,
  `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then run `! scripts/hf_login.sh`
  (the token is stored in `.cache/huggingface/`; answer **n** to "add as git credential").
- **H3** (optional) listen to the example censor spans listed in `docs/data_notes.md`.
- **H4** approve `configs/codebook.md`. **H5** role review. **H6** gold annotations. **H7** WER references.

Parameters: `configs/pipeline.yaml`. Dictionary/validation: `configs/variables.yaml`.
Facts about the data: `docs/data_notes.md`. Deviations from the spec: `docs/decisions.md`.
