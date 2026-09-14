# Build Spec: Audio Processing & Data Extraction — Collections Calls (AI vs Human)


**Purpose:** build specification for the extraction pipeline, run end to end on one local workstation.

**Scope:** ONLY processing the audio files and extracting data from them (audio → validated tables). No reports, no statistical analysis, no visualizations.

---

## 0. TL;DR

- **Input:** 100 WAV call recordings (50 human agents, 50 AI agent), Colombian Spanish bank debt collection, mono (agent + client mixed), 8 kHz telephony, **already censored** (PII removed from audio by the provider).
- **Output:** `calls`, `turns`, `objections`, `offers` tables + data dictionary + validation report.
- **Pipeline:** download → inventory/QC + censor detection → preprocessing → VAD → ASR (faster-whisper) → diarization (pyannote) → turns → roles (Claude Code + human review) → canonical JSON → deterministic features → interpretive annotation by Claude Code with verifiable quotes → validation against a human gold set → tables.
- **Zero paid services.** Everything runs locally with open models. The only LLM work is done by a Claude Code session. **No Anthropic API or other paid APIs.**

---

## 1. Rules for the building session

1. **Stay in scope:** extraction only. No analysis, reports or dashboards in this pipeline.
2. **No paid APIs or services.** No Anthropic API key, no OpenAI/Gemini/Deepgram/AssemblyAI. Local models + Claude Code only.
3. **Never commit data:** audio, transcripts, annotations, quotes and tables stay in `data/` (gitignored).
4. **Git:** code and configuration only; the remote repository is created by the project owner.
5. **Interactive/sudo commands** (apt install, `hf auth login`) are run by a person, not by the pipeline.
6. **GPU:** only one GPU stage at a time (4 GB VRAM). Each GPU stage runs as its own process and exits.
7. **Operational definitions and thresholds live in `configs/`.** Never change them silently. Any change → bump `configs/pipeline.yaml: version`, log it in `docs/decisions.md`, re-run affected stages.
8. **Blinding during annotation:** while annotating (E6, E9), read only files under `data/interim/llm_inputs/` plus `configs/`. Never open `data/private/`, `data/raw/`, or anything that maps `call_id` → group.
9. **Every stage is idempotent and resumable:** skip calls whose outputs exist with a matching config hash unless `--force`. One failing call must not stop the batch (log the error, continue, summarize at the end).
10. **Human-in-the-loop gates are blocking** (§4). Never fabricate human reviews or gold annotations.
11. **Verify, don't assume:** confirm facts about the data on first contact (§9.0) and record them in `docs/data_notes.md`.

---

## 2. Verified facts (checked 2026-09-12)

### 2.1 Data source
- The raw audio folder has one subfolder per group:
  - `audios_humanos_censurados` — **50 .wav** (human agents)
  - `audios_ia_censurados` — **50 .wav** (AI agent)
- **Filenames are UUIDs** (e.g. `0445c357-e465-49ae-8067-bfa91d11b532.wav`) → **no metadata in filenames** (no date, time, campaign or agent). Group is known only from the subfolder.
- **"censurados"** → personal data was censored in the audio. The censoring method (beep tone, digital silence, noise) is **unknown** and must be detected (E0).

### 2.2 Hardware and environment constraints
| Item | Constraint | Implication |
|---|---|---|
| Python | System Python too new for ML wheels | **Use uv-managed Python 3.12** |
| Env manager | uv | Env + lockfile manager |
| ffmpeg | Not installed | Not needed: soundfile decodes the WAVs (D-008) |
| GPU | **4 GB** VRAM, compute capability **6.1** (Pascal) | int8 only for CTranslate2; torch wheels must include `sm_61` |
| CPU | Multi-core | CPU fallback is viable |
| Disk | ~30 GB (models, caches, environment, data) | — |
| Hugging Face | Read token needed for pyannote | Created by a person (§4) |
| Annotator | Claude Code | — |

---

## 3. Locked decisions

| Topic | Decision |
|---|---|
| LLM work | **Claude Code session as annotator** (reads rendered transcripts, writes JSON). No API. |
| Variables | **Core table (§10.1) + full dictionary (§10.2)** |
| ASR | faster-whisper `large-v3`, `compute_type="int8"`, GPU; fallback `large-v3-turbo` and/or CPU int8 |
| Diarization | pyannote `speaker-diarization-community-1` (pyannote.audio 4.x); fallback `speaker-diarization-3.1` (pyannote.audio 3.x) |
| VAD | Silero VAD |
| Prosody | praat-parselmouth |
| Sentiment | pysentimiento (Spanish) |
| Embeddings | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` |
| Commitment detection | **Codebook-first:** Claude Code drafts `configs/codebook.md` from §10 → **the reviewer approves** → the same codebook guides Claude's annotation and the reviewer's gold annotation. `ptp_strength` computed in code. |
| Env | uv project, Python 3.12, `uv.lock` committed |
| Orchestration | Single CLI with cached stages. No Airflow/DVC/DB. |

---

## 4. Human-in-the-loop checkpoints (blocking)

| # | Reviewer does | When | Command / artifact |
|---|---|---|---|
| H1 | Install ffmpeg | Before setup verification | `! sudo apt install -y ffmpeg` |
| H2 | Hugging Face: create account (if needed); create a **read** token at https://huggingface.co/settings/tokens; accept model terms at https://huggingface.co/pyannote/speaker-diarization-community-1 (fallback also https://huggingface.co/pyannote/speaker-diarization-3.1 and https://huggingface.co/pyannote/segmentation-3.0); log in | Before E4 | `! uv run hf auth login` (older CLI: `huggingface-cli login`) |
| H3 | Confirm the censoring method by listening to 1–2 flagged spans, if E0 detection is ambiguous | After E0 | `docs/data_notes.md` |
| H4 | Approve `configs/codebook.md` (v1, and again after dev-set revision) | Before E6 annotation / before E9 full run | Reply "approved" or edit the file |
| H5 | **Role review** for 100% of calls | After E6 annotation, before E7 | Fill `data/interim/review/roles_review.csv` |
| H6 | **Gold annotation** (20 calls), without looking at Claude's outputs for those calls | Before E10 | Fill `data/validation/gold_annotations.csv` |
| H7 | **WER references:** correct the Whisper transcript of 4 calls (2 human, 2 AI) while listening | Before E10 | `data/validation/wer_refs/<call_id>.txt` |

---

## 5. Environment setup (step by step, with checks)

### 5.1 Project & Python
- Create the project directory (see §6); `git init`; `uv init --package --python 3.12`; add `.python-version` = 3.12.
- **Check:** `uv run python --version` → 3.12.x.

### 5.2 Dependencies (pyproject, locked with uv)
- **Core:** numpy, scipy, pandas, pyarrow, pyyaml, soundfile, soxr, tqdm, jsonschema, rapidfuzz, scikit-learn, jiwer, requests
- **Audio models:** torch, torchaudio (from the PyTorch CUDA index — §5.3), faster-whisper, pyannote.audio, silero-vad
- **Features:** praat-parselmouth, pysentimiento, sentence-transformers
- **Dev:** pytest, ruff

### 5.3 PyTorch on a Pascal GPU (sm_61)
- Configure a uv index for PyTorch **CUDA 12.6** wheels (`https://download.pytorch.org/whl/cu126`), which is a safe choice for Pascal GPUs.
- **Checks (all must pass):**
  - `torch.cuda.is_available()` is True.
  - `"sm_61"` is in `torch.cuda.get_arch_list()`.
  - A small tensor matmul on `cuda` runs without "no kernel image is available".
- **Fallback:** if the checks fail, run pyannote, pysentimiento and sentence-transformers on **CPU**. faster-whisper is independent of torch (CTranslate2).
- If `pyannote.audio` 4.x requires a torch version without a working Pascal wheel → pin `pyannote.audio>=3.3,<4` and use `speaker-diarization-3.1`. Record the choice in `docs/decisions.md`.

### 5.4 faster-whisper / CTranslate2 on GPU
- Needs cuBLAS 12 + cuDNN 9 runtime libraries. They come from pip packages (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12` 9.x; usually already pulled in by torch cu126).
- Export `LD_LIBRARY_PATH` to include the venv's `site-packages/nvidia/cublas/lib` and `site-packages/nvidia/cudnn/lib`. Put this in a wrapper (e.g. set in the CLI entry before importing ctranslate2, or a `scripts/env.sh`).
- **Checks:**
  - `ctranslate2.get_cuda_device_count()` ≥ 1.
  - `"int8"` is in `ctranslate2.get_supported_compute_types("cuda")`.
- **Use `compute_type="int8"`.** Do not use `float16` / `int8_float16` (Pascal has no efficient FP16).
- **Fallback:** `device="cpu"`, `compute_type="int8"`, `cpu_threads=8`; if too slow, use `large-v3-turbo`.

### 5.5 Smoke test (before any batch)
- Run E0 → E5 on **3 calls** (mixed groups).
- Check that:
  - `nvidia-smi` shows VRAM < 4 GB during ASR.
  - Outputs are written.
  - The rendered transcript is readable Spanish.
  - Speakers look plausible.
- Record the measured real-time factor for ASR and diarization in `docs/data_notes.md` (a technical fact, not a plan).

---

## 6. Repository layout

```
collections-ai-vs-human/
├── pyproject.toml, uv.lock, .python-version, .gitignore, README.md
├── configs/
│   ├── pipeline.yaml            # all parameters + version
│   ├── codebook.md              # operational definitions (reviewer-approved)
│   ├── variables.yaml           # single source for data dictionary + output validation
│   ├── lexicons.yaml            # friction, robot/human-request, politeness, masking, hallucination blacklist
│   └── schemas/
│       ├── roles.schema.json
│       ├── pass_a.schema.json
│       └── pass_b.schema.json
├── src/extraction/
│   ├── cli.py                   # entry point: `uv run extract <stage> [--calls C001,C002] [--force]`
│   ├── paths.py, config.py, logging_utils.py, cache.py
│   ├── download.py              # stage: download
│   ├── inventory.py             # E0 (+ censor detection)
│   ├── preprocess.py            # E1
│   ├── vad.py                   # E2
│   ├── asr.py                   # E3
│   ├── diarize.py               # E4
│   ├── turns.py                 # E5
│   ├── roles.py                 # E6 (render inputs, validate annotations, build review sheet, merge review)
│   ├── canonical.py             # E7 (+ transcript rendering, masking)
│   ├── features/
│   │   ├── dynamics.py          # E8a
│   │   ├── prosody.py           # E8b
│   │   ├── lexical.py           # E8c (friction, lexicons, sentiment)
│   │   └── embeddings.py        # E8d
│   ├── annotation.py            # E9 (render inputs, validate outputs, verify quotes, progress)
│   ├── validation.py            # E10 (gold/dev selection, templates, κ, WER, reliability)
│   └── tables.py                # E11
├── tests/                        # unit tests on synthetic data (§12)
├── docs/
│   ├── data_notes.md            # facts found on first contact with data
│   └── decisions.md             # any change to definitions/thresholds/fallbacks
└── data/                         # GITIGNORED ENTIRELY
    ├── raw/{humanos,ia}/*.wav
    ├── private/id_map.csv       # uuid ↔ call_id ↔ group  (never read during annotation)
    ├── interim/
    │   ├── manifest.csv
    │   ├── censor/  audio_16k/  vad/  asr/  diarization/  turns/  canonical/  features/
    │   ├── llm_inputs/{roles,calls}/
    │   ├── annotations/{roles,pass_a,pass_b,rerun_pass_a}/  + progress.csv
    │   └── review/roles_review.csv
    ├── validation/  gold_ids.csv  dev_ids.csv  gold_annotations.csv  wer_refs/  validation_report.md  variable_reliability.csv
    ├── processed/  calls.parquet/.csv  turns.parquet  objections.parquet/.csv  offers.parquet/.csv  data_dictionary.csv
    └── logs/<stage>.log
```

**`.gitignore`:** `data/`, `.venv/`, `__pycache__/`, `*.wav`, `*.parquet`, `.env`.

**Stage metadata:** each stage writes `data/interim/<stage>/_stage.json` with config hash, package versions, model names, start/end timestamps, n_ok / n_failed.

**Convenience commands:**
- `uv run extract audio-all` → download…E5 in order.
- `uv run extract features-all` → E7-dependent deterministic features.

---

## 7. Configuration (`configs/pipeline.yaml`) — initial values

| Key | Value |
|---|---|
| `version` | 1 |
| `seed` | 20260912 |
| `audio.master_sr` / `audio.model_sr` | 8000 / 16000 |
| `audio.highpass_hz` / `audio.peak_dbfs` | 70 / −1.0 (model copy only; normalization ignores censor spans) |
| `censor.min_zero_run_ms` | 50 |
| `censor.tone_peak_ratio` / `censor.tone_min_ms` | 0.8 / 150 |
| `censor.noise_flatness` / `censor.noise_min_ms` | 0.5 / 150 |
| `vad.threshold` / `min_speech_ms` / `min_silence_ms` / `speech_pad_ms` | 0.5 / 250 / 300 / 100 |
| `asr.model` / `fallback_model` | large-v3 / large-v3-turbo |
| `asr.device` / `compute_type` / `cpu_threads` | cuda / int8 / 8 |
| `asr.language` / `beam_size` | es / 5 |
| `asr.word_timestamps` / `vad_filter` / `condition_on_previous_text` | true / true / **false** |
| `asr.compression_ratio_threshold` / `log_prob_threshold` / `no_speech_threshold` | 2.4 / −1.0 / 0.6 |
| `asr.initial_prompt` | "Buenos días, le hablo del área de cartera sobre su obligación en mora. Eh, sí, señor, la cuota vencida, el acuerdo de pago, pues, por PSE, Efecty o Nequi. Ajá, listo, centrales de riesgo." |
| `asr.low_conf_word_prob` | 0.5 |
| `diarization.pipeline` / `fallback_pipeline` | pyannote/speaker-diarization-community-1 / pyannote/speaker-diarization-3.1 |
| `diarization.min_speakers` / `max_speakers` | 1 / 3 |
| `turns.backchannel_max_s` / `backchannel_max_words` | 1.0 / 2 |
| `turns.interruption_min_overlap_s` | 0.5 |
| `turns.word_to_segment_max_gap_s` | 0.5 |
| `features.dead_air_min_s` | 3.0 |
| `features.monologue_min_s` | 30 |
| `features.mechanical_repetition_ratio` / `min_words` | 80 (rapidfuzz token_set_ratio) / 5 |
| `features.sentiment_edge_turns` / `min_words` | 3 / 3 |
| `prosody.f0_floor_hz` / `f0_ceiling_hz` / `st_ref_hz` / `min_segment_s` | 75 / 500 / 100 / 0.5 |
| `annotation.quote_match_min` | 90 (rapidfuzz partial_ratio after normalization) |
| `annotation.batch_size` | 10 |
| `annotation.roles_context_s` / `roles_context_turns` | 90 / 20 |
| `validation.gold_per_group` / `dev_per_group` / `wer_per_group` | 10 / 3 / 2 |
| `validation.kappa_keep` / `kappa_flag` / `exact_match_keep` | 0.6 / 0.4 / 0.8 |
| `quality.low_snr_db` / `high_clipping_pct` / `low_asr_conf` / `no_speech_ratio` / `censor_heavy_pct` | 10 / 1.0 / 0.6 / 0.05 / 20 |


---

## 9. Pipeline stages

```
download → E0 inventory+censor → E1 preprocess → E2 VAD → E3 ASR → E4 diarization → E5 turns
→ E6 roles (Claude annotates → the reviewer reviews) → E7 canonical JSON + rendered transcripts
→ E8 deterministic features ─┐
→ E9 annotation pass A / pass B (Claude Code) ─┤→ E10 validation → E11 tables
```

### 9.0 First contact with the data (record in `docs/data_notes.md`)
- **Encoding:** for every file, via soundfile `info` (subtype: PCM_16 / ULAW / ALAW…), sample rate, channels, duration. ffprobe once ffmpeg exists.
- **Duration distribution** per group.
- Any stereo files, sample rates ≠ 8000, silent/corrupt files.
- **Censoring method(s)** found (E0) and an example span per type.
- Number of speakers typically heard; whether AI calls include IVR/voicemail; any non-Spanish.
- If a fact contradicts this spec (e.g. stereo, 16 kHz) → adapt, log in `docs/decisions.md`.

### E0 · Inventory, QC, censor detection
- **Anonymous IDs:** shuffle the 100 files with `seed` → `C001…C100`. Write `data/private/id_map.csv` (uuid, group, call_id).
- **`data/interim/manifest.csv`** columns: call_id, sha256, duration_file_s, codec, sample_rate, channels, clipping_pct, level_dbfs, flags, exclusion_reason. **No group column.**
- **Censor detection** on the 8 kHz master (STFT, 32 ms window, 10 ms hop). Three types:
  - **zeros:** runs of exact-zero samples ≥ `min_zero_run_ms`.
  - **tone:** frames where the strongest spectral peak holds ≥ `tone_peak_ratio` of frame energy, same frequency bin ±1, for ≥ `tone_min_ms`.
  - **noise:** spectral flatness ≥ `noise_flatness` with energy above the file's median speech energy, for ≥ `noise_min_ms`.
- **Output:** `interim/censor/<call_id>.json` → `[{start, end, type}]`, plus `censored_s`, `n_censor_segments` in the manifest.
- If detection finds nothing in all files, or results look implausible → **H3** (the reviewer listens to a span to identify the method); tune thresholds; log it.
- **Exclusion** only for unusable files (corrupt, no audio at all). No-contact calls are kept: they are data.

### E1 · Preprocessing
- Decode to float32 mono. The **8 kHz master** stays unmodified (except decoding).
- **Model copy (16 kHz):**
  1. Replace censor spans with digital silence.
  2. High-pass 70 Hz (Butterworth, order 2, zero-phase).
  3. soxr resample to 16 kHz (HQ).
  4. Peak normalize to −1 dBFS (computed outside censor spans).
- Save as `interim/audio_16k/<call_id>.wav` (PCM_16).
- **No denoising. No trimming.**

### E2 · VAD
- Silero VAD on the 16 kHz copy with §7 params.
- **Output:** `interim/vad/<call_id>.json` → speech segments `[{start, end}]`.
- **SNR:** 10·log10(mean power in speech frames / mean power in non-speech frames), excluding censor spans and exact-zero regions. Uses the 8 kHz master aligned by time.
- `speech_ratio` = speech seconds / (duration − censored_s).

### E3 · ASR (GPU process 1)
- faster-whisper with §7 params. Load the model once for all calls.
- **Output:** `interim/asr/<call_id>.json`:
  - Segments: start, end, text, avg_logprob, no_speech_prob, compression_ratio.
  - Words: start, end, word, probability.
  - Metadata: model, fallback used.
- **Post-filters (logged, not silent):**
  - Drop segments with compression_ratio > threshold.
  - Drop segments whose words fall > 80% inside VAD non-speech or censor spans.
  - Drop exact matches to the hallucination blacklist in `lexicons.yaml`, e.g. "Subtítulos realizados por la comunidad de Amara.org", "Gracias por ver el video", "¡Suscríbete!".
  - Collapse the same n-gram (n ≥ 3) repeated ≥ 3 times consecutively.
  - Set flag `hallucination_removed` when anything is dropped.
- `asr_confidence` = duration-weighted mean word probability. `asr_low_conf_pct` = % words with probability < 0.5.
- **If CUDA OOM or unsupported:** retry that call with `fallback_model`, then CPU. Record it in `_stage.json`.

### E4 · Diarization (GPU process 2)
- **Always pass audio in memory** (`{"waveform": tensor(1, n), "sample_rate": 16000}`). This avoids torchcodec/ffmpeg decoding issues.
- `min_speakers=1`, `max_speakers=3`.
- **Output:** `interim/diarization/<call_id>.json`:
  - `regular`: segments with overlaps, `[{start, end, speaker}]`.
  - `exclusive`: one speaker at a time. If the pipeline doesn't provide exclusive output (3.1), derive it: overlapped time goes to the speaker whose covering segment is longer.
- Speaker labels normalized to `SPK_00`, `SPK_01`, `SPK_02`.
- **Speech per speaker** = regular segments ∩ VAD speech − censor spans. This removes diarization false alarms on noise.

### E5 · Fusion & turns
1. **Word → speaker:** max temporal overlap with exclusive segments; if no overlap, nearest segment within `word_to_segment_max_gap_s`; else `UNKNOWN`.
2. **Backchannel:** a speech stretch of speaker B that starts while A holds the floor, lasts ≤ 1.0 s and has ≤ 2 words. Marked `is_backchannel`; does not take the floor.
3. **Turn:** consecutive speech of one speaker until another speaker produces non-backchannel speech. Pauses and backchannels inside do not split the turn. Censor spans inside a turn stay in that turn as `[CENSURADO]`.
4. **Interruption:** B's non-backchannel speech starts while A is speaking, overlap ≥ 0.5 s, and A's speech ends before B's does (A yields). Overlap where A doesn't yield is counted in `overlap_s` only.
5. **Latency** for each floor transfer A→B without overlap: `B.start − A.end`. Transfers with overlap have no latency value.
- **Output:** `interim/turns/<call_id>.json` → `turns[]` (turn_id `T001…`, speaker, start, end, words, text, is_interruption, latency_s, backchannels[] inside the turn), `overlaps[]`, `interruptions[]`.

### E6 · Role assignment (highest-leverage stage — a flipped role flips every metric)
1. **Render** `llm_inputs/roles/<call_id>.txt`: first `roles_context_s` seconds or `roles_context_turns` turns, one line per turn: `[T003 | SPK_01 | 00:07.4–00:12.9] texto`. Plus a speaker summary (total seconds, n turns per speaker).
2. **Claude Code annotates** each file → `annotations/roles/<call_id>.json` (schema `roles.schema.json`):
   - `speaker_map`: `{SPK_00: AGENT|CLIENT|THIRD_PARTY|SYSTEM|UNKNOWN, …}`. Several speakers may map to the same role.
   - `evidence`: `[{speaker, turn_id, quote}]`.
   - `suspect_turns`: `[{turn_id, reason}]` — text inconsistent with its speaker, i.e. likely diarization error.
   - `confidence`: high|medium|low.
   - `_meta`.
   - **Role rules (put in codebook):**
     - AGENT = the collector, human or synthetic voice. The AI agent's TTS is AGENT, not SYSTEM.
     - SYSTEM = IVR menus, carrier messages, voicemail greetings, hold music with voice.
     - THIRD_PARTY = someone other than the debtor answering.
     - CLIENT = the debtor / account holder (use CLIENT even before identity is validated if it is the person being sought).
3. **Validate:** schema, turn_ids exist, quotes verified (§9-E9 quote rule).
4. **Build review sheet** `data/interim/review/roles_review.csv`. One row per call×speaker: call_id, speaker, proposed_role, confidence, seconds, 3 sample lines, suspect_turns, `approved` (Y/N), `corrected_role`, `note`.
5. **H5:** the reviewer reviews 100%.
6. **Merge:** E7 refuses to run for any call without `approved` filled. Corrections → flag `roles_manual_fix`. Accepted suspect-turn reassignments → flag `diar_corrected`, applied as explicit edits listed in the canonical JSON.

### E7 · Canonical JSON + rendered transcripts
- **`interim/canonical/<call_id>.json`** (no group field):
  - `call_id`
  - `audio`: duration, snr_db, clipping_pct, level_dbfs, speech_ratio, censored_s, n_censor_segments
  - `asr`: model, asr_confidence, asr_low_conf_pct
  - `speaker_map`
  - `turns[]`: turn_id, role, start, end, text, words[{w, start, end, prob}], backchannels[{role, start, end, text}], is_interruption, latency_s
  - `overlaps[]`, `interruptions[]`, `censor[]`, `nonspeech[]`
  - `edits[]` (manual/diarization corrections)
  - `flags[]`
  - `versions`
- **Rendered transcript** `llm_inputs/calls/<call_id>.txt`:
  - Header: call_id, conversation duration, speakers present.
  - One line per turn: `[T012 | AGENT | 00:41.2–00:47.9] texto … [CENSURADO] …`.
  - Backchannels inline as `(CLIENT: ajá)`.
  - Low-confidence words suffixed `[?]`.
  - **Masking** from `lexicons.yaml` (e.g. "asistente virtual", "inteligencia artificial", "soy una IA", "agente virtual") → `[AGENTE]`, applied to AGENT turns only.
- **Residual PII scan:** regex for ≥ 6-digit sequences, 10-digit numbers starting with 3, emails. Hits are listed in `data/logs/pii_scan.log` for the reviewer; transcripts stay local either way.

### E8 · Deterministic features (definitions in §10.2)
- **E8a dynamics** from canonical JSON only.
- **E8b prosody:**
  - Input: 8 kHz master, non-overlapped, non-censored speech ≥ 0.5 s per role.
  - Tool: parselmouth pitch (floor 75, ceiling 500).
  - Pitch: F0 → semitones re 100 Hz. Intensity in dB on voiced frames.
- **E8c lexical:**
  - Regex lexicons (`lexicons.yaml`) on turn text by role.
  - Mechanical repetition via rapidfuzz.
  - Sentiment: pysentimiento `sentiment` (lang `es`) on CLIENT turns with ≥ 3 words; score = P(POS) − P(NEG).
- **E8d embeddings:**
  - Model: MiniLM multilingual. Per call, the centroid of AGENT turn embeddings.
  - `agent_discourse_similarity` = mean cosine similarity to the centroids of the other calls **of the same group**. The group join happens in E11; E8d writes centroids only and the similarity is computed in E11.
  - QA: cluster objection quotes typed `other` (after E9) into `data/validation/objection_other_clusters.md`, to check the taxonomy.
- **Output:** `interim/features/<family>/<call_id>.json`.

### E9 · Interpretive annotation by Claude Code (pass A, then pass B)
- **Inputs:** `configs/codebook.md` (approved), `configs/schemas/pass_a.schema.json` / `pass_b.schema.json`, `llm_inputs/calls/<call_id>.txt`.
- **Dev iteration first:**
  1. Claude annotates the dev calls (`dev_ids.csv`, 3 + 3).
  2. The reviewer compares disagreements with their own reading.
  3. Codebook revised → v2 → **H4** approval.
  4. Then the full run.
- **Per-call procedure:**
  1. Read the whole transcript. Don't skim.
  2. Fill every field following the codebook.
  3. Quotes are copied verbatim from the cited turn.
  4. `null` = not mentioned; `"censored"` = mentioned but censored (`[CENSURADO]`); `"na"` = not applicable.
  5. Don't consult other calls' outputs while annotating.
- **Batches** of `batch_size`: re-read the codebook at the start of every batch (context may have been compacted). After each batch run `uv run extract annotate-validate --pass a` and fix failures before continuing.
- **Resumable:** skip calls with valid outputs. `annotations/progress.csv` holds call_id, pass, status, errors. Work continues across sessions and usage-limit resets because all state is on disk.
- **Output:** `annotations/pass_a/<call_id>.json`, `annotations/pass_b/<call_id>.json`.
- **`_meta` in every file:** call_id, annotator `claude-code`, model name as reported by the session, Claude Code version, codebook version (sha256 of `codebook.md`), annotated_at.
- **Validation checks (`annotate-validate`):**
  1. Valid JSON + schema (jsonschema), enums.
  2. Every `turn_id` exists in the canonical JSON.
  3. **Quote verification:** normalize both sides (casefold, strip punctuation and `[?]`, collapse spaces), then rapidfuzz `partial_ratio(quote, turn_text) ≥ 90`. On failure, set the field to `null` and add flag `quote_verification_failed` **only if** it cannot be fixed by re-reading.
  4. **Consistency rules:**
     - `ptp=false` ⇒ all `ptp_*` null and `final_outcome` ∉ {vague_promise, firm_promise}.
     - `contact_type ≠ account_holder` ⇒ `final_outcome` ∈ {no_contact, third_party}, `ptp=false`, script items `"na"` except greets/identifies_self/debt_disclosed_to_third_party.
     - `objections` empty ⇒ `objection_handling = "NA"`.
     - `ptp=false` ⇒ `recaps_agreement = "na"`.
     - Milestone turn_ids must be in chronological order where they exist.
- **Self-consistency rerun:** after the full run, re-annotate the 20 gold calls in a **fresh session**, without reading previous outputs → `annotations/rerun_pass_a/`.
- **Blindness probe:** pass A includes `blindness.guess_agent_type` (human|ai|unsure) + reason; its accuracy is computed in E10.

### E10 · Validation
- **Selection (seeded, stratified, disjoint):**
  - Dev: 3 human + 3 AI.
  - Gold: 10 + 10.
  - WER: 2 + 2, taken from gold.
  - Written to `data/validation/*.csv` (call_ids only; the group lookup stays private).
- **Gold template** `gold_annotations.csv` (H6), one row per gold call. Columns:
  - roles_ok
  - contact_type, identity_validated, final_outcome
  - ptp, ptp_amount, ptp_date_text, ptp_confirmed_by_client, agent_recapped
  - n_objections, objection_types (`;`-separated), n_offers_agent, first_anchor
  - the 10 script items, debt_disclosed_to_third_party
  - clarity, empathy, active_listening
- **Metrics:**
  - Role accuracy (from H5).
  - Cohen's κ for categorical/boolean; weighted κ (quadratic) for `final_outcome` and rubric scores.
  - Exact match for amounts/dates (ignoring "censored" rows; counted separately).
  - F1 for objection type sets; Spearman for counts.
  - Self-consistency κ (rerun vs first run).
  - Blindness probe accuracy.
  - **WER** with jiwer after normalization (lowercase, strip punctuation, numbers normalized consistently), plus accuracy on number/date tokens.
  - **Every metric overall AND by group.**
- **Decision per variable** → `variable_reliability.csv` (variable, metric, value, value_human, value_ai, n, decision):
  - κ ≥ 0.6 (or exact match ≥ 0.8), with no marked gap between groups → `keep`.
  - 0.4–0.6 → `flag`.
  - < 0.4 → `drop`.
  - Deterministic variables are `keep` by construction, but inherit `flag` if WER or role accuracy by group shows a gap.
- **Report:** `validation_report.md` (tables only, no charts).

### E11 · Tables
- Join `group` from `data/private/id_map.csv` (first time group enters any table).
- Build `calls`, `turns`, `objections`, `offers` (§10.3). Compute derived fields: `ptp_strength`, `pct_objections_resolved`, `script_checklist_score`, `time_to_*`, `agent_discourse_similarity`.
- Validate every column against `configs/variables.yaml` (name, dtype, allowed values, nullability).
- Write `data_dictionary.csv` (variable, table, description, type, allowed values, unit, source D/A/L, reliability decision from E10).
- Dropped variables are kept in the parquet with suffix `_unreliable` **or** removed, as set by `configs/pipeline.yaml: tables.keep_dropped` (default: remove from `calls`, keep in dictionary with decision `drop`).
- Row check: `calls` has one row per non-excluded audio (expected 100).

---

## 10. Output data specification

Source codes: **D** = deterministic code, **A** = acoustic, **L** = Claude Code annotation. "D over L" = computed in code from annotated fields.

### 10.1 Core variables (required)

| Group | Variable | Description | Example |
|---|---|---|---|
| ID & quality | call_id | Anonymous call ID | C037 |
| | group | Human or AI | ai |
| | snr_db, asr_confidence | Audio quality and transcription confidence (used as controls) | 18.2, 0.87 |
| | flags | Quality warnings (low audio, role corrected…) | low_snr |
| Context | product | Type of debt | credit card |
| | debt_amount | Amount mentioned in the call | 1,250,000 |
| | non_payment_reasons | Why the client hasn't paid (all reasons, primary first) | unemployment |
| Contact | contact_type | No answer / voicemail / wrong number / third party / account holder | account holder |
| | identity_validated | Agent confirmed they were speaking to the account holder | true |
| Outcome | final_outcome | Ordered scale from no contact → firm promise to pay | vague promise |
| | ptp | Promise to pay exists | true |
| | ptp_amount, ptp_date | Promise details | 400,000 · Friday |
| | ptp_strength | 0–3 score: amount + date + explicit confirmation | 2 |
| | agent_recapped | Agent summarized the agreement before closing | false |
| Negotiation | n_offers_agent, n_counteroffers_client | Back-and-forth volume | 2, 1 |
| | first_anchor | Who proposed the first number | agent |
| Objections | n_objections, pct_objections_resolved | Objection volume and resolution rate | 3, 67% |
| Conversation dynamics | conv_duration_s | First to last speech, not file length | 184 |
| | agent_talk_share | % of talk time by the agent | 72% |
| | agent_latency_median_s | Pause before the agent answers | 1.4 |
| | dead_air_count | Silences longer than 3 s mid-call | 2 |
| | interruptions_per_min (each direction) | Who cuts off whom | 0.8 / 0.3 |
| | yield_time_s | How long the agent keeps talking when interrupted | 2.1 |
| | max_agent_monologue_s | Longest uninterrupted agent speech | 41 |
| | time_to_debt_mention_s, time_to_first_offer_s, time_to_ptp_s | How fast the call reaches key moments | 22 · 65 · 140 |
| Friction | repeat_requests | "Sorry?", "Can you repeat?" | 3 |
| | incoherent_responses | Agent replies off-topic | 1 |
| Compliance | script_checklist_score | % of script items completed (identifies self, states amount, offers channels…) | 80% |
| | debt_disclosed_to_third_party | Privacy breach | false |
| Quality (LLM rubric, 1–5) | clarity, empathy, active_listening, objection_handling, professionalism | Anchored scores with justification | 4, 2, 2, 3, 5 |
| Voice & sentiment | agent_pitch_range_st | Voice expressiveness (low = monotone) | 3.1 |
| | client_escalation | Client pitch/energy rising over the call | +0.4 |
| | sentiment_start, sentiment_end, sentiment_delta | Client mood trajectory | −0.1 → −0.5 |

### 10.2 Full dictionary — `calls` table (core + extended, with exact definitions)

**Value conventions:**
- Booleans: `true` / `false`.
- `null` = not mentioned / not computable.
- `"censored"` = mentioned but censored.
- `"na"` = not applicable.
- Amounts: COP as numbers.
- Times: seconds from `conv_start` (first non-SYSTEM speech start) unless stated.

#### A. ID & quality (D)
| Variable | Type | Definition |
|---|---|---|
| `call_id` | str | C001…C100 |
| `group` | enum human/ai | From subfolder, joined in E11 |
| `duration_file_s` | float | File length |
| `snr_db` | float | E2 definition |
| `clipping_pct` | float | % samples at ≥ 0.999 full scale |
| `level_dbfs` | float | RMS level of speech frames |
| `speech_ratio` | float | E2 definition |
| `censored_s`, `n_censor_segments` | float/int | From E0 |
| `asr_confidence`, `asr_low_conf_pct` | float | E3 definitions |
| `n_speakers_detected` | int | Distinct diarization speakers |
| `flags` | list[str] | low_snr, high_clipping, low_asr_conf, no_speech, censor_heavy, one_speaker, three_speakers, hallucination_removed, roles_manual_fix, diar_corrected, quote_verification_failed, agent_spoke_first |

#### B. Contact (L)
| Variable | Type | Definition |
|---|---|---|
| `contact_type` | enum | `no_answer` (ringing/silence, no human) · `voicemail` · `ivr` · `wrong_number` · `hangup_before_identification` · `third_party` · `account_holder` · `unclear` |
| `identity_validated` | bool/na | Agent explicitly confirms the person is the account holder (name or ID confirmation) at any point |

#### C. Debt context (L)
| Variable | Type | Definition |
|---|---|---|
| `product` | enum | credit_card · personal_loan · vehicle_loan · mortgage · payroll_loan · microcredit · telecom_services · other · unknown |
| `debt_amount`, `overdue_amount` | number/null | Total debt / overdue amount stated (null if not stated or censored) |
| `non_payment_reasons` | list[enum] | All reasons stated, primary first: unemployment_income_loss · forgot · illness_calamity · over_indebtedness · disputes_charge · doesnt_recognize_debt_fraud · already_paid · payment_channel_problem · other; `not_stated` alone if none |

#### D. Outcome & promise to pay (L, D over L)
| Variable | Type | Definition |
|---|---|---|
| `final_outcome` | ordinal enum (0–7) | 0 `no_contact` < 1 `third_party` < 2 `holder_refuses` < 3 `no_agreement` < 4 `callback_scheduled` < 5 `claims_already_paid` < 6 `vague_promise` < 7 `firm_promise` |
| `ptp` | bool | Client (account holder) commits to pay, firm or vague |
| `ptp_type_commitment` | enum firm/vague/na | **firm** = explicit, unconditional ("sí, el viernes pago"); **vague** = conditional/uncertain ("voy a tratar", "apenas pueda") |
| `ptp_amount` | number/null | Amount promised (null if not stated or censored) |
| `ptp_date` | str/null/censored | Date as stated ("el viernes", "30 de septiembre") |
| `ptp_type` | enum full/partial/null | Relative to overdue amount |
| `ptp_confirmed_by_client` | bool/na | Client explicitly confirms the final terms |
| `agent_recapped` | bool/na | Agent restates amount and/or date and/or channel before closing |
| `callback_scheduled` | bool | A new contact is agreed |
| `ptp_strength` | int 0–3 / null | **D over L:** (ptp_amount is a number) + (ptp_date not null) + ptp_confirmed_by_client. null if ptp = false |

#### E. Negotiation (L, D over L)
| Variable | Type | Definition |
|---|---|---|
| `had_negotiation` | bool | Any payment option or amount/date is discussed |
| `n_offers_agent` | int | Rows in `offers` with proposed_by = agent |
| `n_counteroffers_client` | int | Rows in `offers` with proposed_by = client |
| `first_anchor` | enum agent/client/none | Who states the first amount or date proposal |

#### F. Objections (D over L)
| Variable | Type | Definition |
|---|---|---|
| `n_objections` | int | Rows in `objections` |
| `n_objections_resolved`, `n_objections_partial` | int | By `resolved` value |
| `pct_objections_resolved` | float/null | resolved = yes / n_objections; null if 0 |

#### G. Conversation dynamics (D; time_to_* are D over L)
| Variable | Type | Definition |
|---|---|---|
| `conv_duration_s` | float | Last non-SYSTEM speech end − conv_start |
| `agent_speech_s`, `client_speech_s` | float | Speech time per role (overlapped time counts for both; censor excluded) |
| `agent_talk_share` | float | agent_speech_s / (agent_speech_s + client_speech_s) |
| `silence_pct` | float | Non-speech, non-censored time within the conversation / (conv_duration_s − censored time) |
| `n_turns`, `turns_per_min` | int/float | Non-backchannel turns; per conversation minute |
| `mean_turn_s_agent`, `mean_turn_s_client` | float | Mean turn duration per role |
| `max_agent_monologue_s` | float | Longest AGENT turn |
| `n_monologues_30s` | int | AGENT turns ≥ 30 s |
| `agent_latency_median_s`, `agent_latency_p90_s` | float/null | Over CLIENT→AGENT floor transfers without overlap |
| `client_latency_median_s` | float/null | Over AGENT→CLIENT transfers |
| `agent_spoke_first` | bool | First non-SYSTEM speech is AGENT |
| `time_to_first_agent_voice_s` | float/null | End of the client's first utterance → start of the first AGENT turn; null if agent spoke first |
| `dead_air_count`, `dead_air_s` | int/float | Gaps with no speech from any role (censor excluded) ≥ 3.0 s, between conv_start and conv end |
| `interruptions_per_min_agent` | float | AGENT interrupts CLIENT, per conversation minute |
| `interruptions_per_min_client` | float | CLIENT interrupts AGENT, per conversation minute |
| `overlap_s` | float | Total overlapped speech time |
| `yield_time_s` | float/null | Median over CLIENT interruptions of AGENT: interruption start → end of agent speech |
| `backchannels_agent_per_min` | float | AGENT backchannels per minute |
| `words_per_min_agent`, `words_per_min_client` | float | Words / speech minutes per role |
| `time_to_identification_s` | float/null | Start time of the milestone turn cited in pass A |
| `time_to_debt_mention_s` | float/null | Same |
| `time_to_first_offer_s` | float/null | Same |
| `time_to_ptp_s` | float/null | Same |
| `closing_by` | enum agent/client_hangup/cut_unclear | L, checked against the last speaker (D) |

#### H. Friction & conversational failures
| Variable | Type | Src | Definition |
|---|---|---|---|
| `repeat_requests` | int | D | Lexicon hits (both roles): "¿perdón?", "¿cómo?", "¿qué?" as a full utterance, "no le/te entendí", "no le/te escuché", "¿me repite?", "¿puede repetir?", "no se escucha", "¿aló?" after the first 10 s |
| `repeat_requests_agent`, `repeat_requests_client` | int | D | Split by role |
| `mechanical_repetition` | bool | D | Any two AGENT turns with ≥ 5 words and token_set_ratio ≥ 80 |
| `agent_misunderstood` | int | L | AGENT turns that show failure to understand the client |
| `incoherent_responses` | int | L | AGENT turns that do not respond to the preceding client content |
| `client_expresses_annoyance` | bool | L | Explicit annoyance or anger |

#### I. Script & compliance (L; score D over L)
Each item is stored as `true` / `false` / `"na"`, with turn_id and quote in the annotation file.

| Variable | Definition |
|---|---|
| `greets` | Opening greeting |
| `identifies_self` | Agent states own name/role **and** the entity |
| `recording_notice` | Says the call is recorded |
| `validates_identity_before_disclosing` | Identity confirmed **before** any debt detail is revealed |
| `states_amount` | Amount owed or overdue stated |
| `states_due_date` | Due date or days overdue stated |
| `offers_payment_channels` | At least one channel named |
| `mentions_credit_bureaus` | Mentions reporting to credit bureaus (descriptive) |
| `recaps_agreement` | Restates the agreement (na if no ptp) |
| `polite_closing` | Courteous goodbye |
| `script_checklist_score` | % true over non-"na" items among the 10 above; null if contact_type ≠ account_holder |
| `debt_disclosed_to_third_party` | Agent reveals the debt's existence or amount to a non-account-holder (na if no third party) |
| `coercive_language` | Threats, intimidation, humiliation |

#### J. Rubric 1–5 (L, pass B)
Stored as `1–5` or `"NA"`. The annotation file holds a ≤ 25-word justification + turn_ids. Anchors for the codebook:

| Variable | 1 | 3 | 5 |
|---|---|---|---|
| `clarity` | Amount/date/next steps missing or confusing; client visibly confused | Main info given but incomplete or jargon-heavy | Amount, date, channel and next step explicit; client shows understanding |
| `empathy` | Ignores or dismisses the client's situation; pressure | Formulaic acknowledgment ("entiendo") without adapting | Specific acknowledgment and the proposal adapted to the situation |
| `active_listening` | Ignores client input; repeats already-answered questions | Responds to some points, misses others | References what the client said; relevant follow-ups; no repeated questions |
| `objection_handling` | Ignores objections or repeats the script | Addresses objections generically, not resolved | Addresses the root cause; fitting alternative; client moves forward. `NA` if no objections |
| `control_focus` | Conversation drifts or ends without direction | Reaches its purpose inefficiently | Guides to a concrete next step without pressure |
| `professionalism` | Rude, coercive or inappropriate | Correct but cold/mechanical | Courteous, respectful, appropriate register throughout |

#### K. Voice, sentiment, lexical, standardization
| Variable | Type | Src | Definition |
|---|---|---|---|
| `agent_pitch_range_st`, `client_pitch_range_st` | float/null | A | p90 − p10 of F0 in semitones (re 100 Hz) over valid segments |
| `agent_intensity_std_db` | float/null | A | Std of intensity (dB) on voiced frames |
| `client_escalation` | float/null | A | Slope (per unit normalized call time 0–1) of the per-turn mean of [z-scored semitone F0 + z-scored intensity] / 2 across CLIENT turns ≥ 1 s; null if < 4 such turns |
| `sentiment_start`, `sentiment_end` | float/null | D | Mean (P(POS) − P(NEG)) of the first / last 3 CLIENT turns with ≥ 3 words |
| `sentiment_delta`, `sentiment_min` | float/null | D | end − start; minimum turn score |
| `agent_questions` | int | D | AGENT sentences ending in "?" |
| `politeness_markers` | int | D | Lexicon: por favor, gracias, con gusto, disculpe, qué pena, le entiendo, muy amable |
| `lexical_richness_mattr` | float | D | MATTR (window 50 tokens) on AGENT text |
| `fillers_per_min` | float/null | D | Lexicon: eh, em, este, o sea, pues, bueno (as fillers) per AGENT speech minute; **kept only if validated against the WER references** |
| `agent_discourse_similarity` | float | D | E8d definition |
| `blindness_guess` | enum human/ai/unsure | L | From pass A (for validation only) |

### 10.3 Supporting tables

| Table | Grain | Columns |
|---|---|---|
| `turns` | 1 row per turn | call_id, turn_id, role, start_s, end_s, duration_s, n_words, is_interruption, latency_s, n_backchannels_received, asr_confidence, sentiment (CLIENT only), contains_censor. **Text stays only in canonical JSON.** |
| `objections` | 1 row per objection | call_id, objection_idx, turn_id, time_s, **type** (no_money · already_paid · call_later · doesnt_recognize_debt · wrong_amount · wants_discount · distrust_who_is_calling · annoyed_by_calls · asks_for_human · other), **agent_technique** (empathy_validation · clarification · alternative_offer · consequences · reschedule · ignores_repeats_script · escalates), response_turn_id, **resolved** (yes · partial · no). Resolution: yes = client moves forward (accepts, stops raising it, agrees to a next step); partial = acknowledged but client still hesitant or shifts to another objection; no = persists or the call ends on it |
| `offers` | 1 row per offer | call_id, offer_idx, turn_id, time_s, proposed_by (agent/client), type (full_payment · minimum_installment · partial_payment · refinancing_restructuring · discount_forgiveness · extra_time), amount (number/null/censored), date_text |

### 10.4 Annotation schemas (fields; implement as JSON Schema, `additionalProperties: false`)

**Evidence object:** `{turn_id: "T012", quote: "…"}`.

**`pass_a`:**
- `_meta`
- `contact`: contact_type, identity_validated, evidence[]
- `debt_context`: product, debt_amount, overdue_amount, non_payment_reasons[] (primary first), evidence[]
- `outcome`: final_outcome, ptp, ptp_type_commitment, ptp_amount, ptp_date, ptp_type, ptp_confirmed_by_client, agent_recapped, callback_scheduled, evidence[]
- `negotiation`: had_negotiation, first_anchor, offers[{turn_id, proposed_by, type, amount, date_text, quote}]
- `objections`: [{turn_id, type, quote, response_turn_id, agent_technique, resolved}]
- `compliance`: {item: {value, turn_id, quote}} for the 10 script items + debt_disclosed_to_third_party + coercive_language
- `milestones`: identification_turn_id, debt_mention_turn_id, first_offer_turn_id, ptp_turn_id (null if absent)
- `failures`: agent_misunderstood_turn_ids[], incoherent_response_turn_ids[], asks_if_robot{value, turn_id, quote}, asks_for_human{…}, client_expresses_annoyance{…}, abrupt_hangup, closing_by
- `blindness`: guess_agent_type, reason

**`pass_b`:**
- `_meta`
- `scores`: {clarity, empathy, active_listening, objection_handling, control_focus, professionalism}, each `{score: 1–5 | "NA", justification (≤ 25 words), turn_ids[]}`

**`roles`:** see E6.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No wheels / build errors for torch, ctranslate2, pyannote | Using system Python 3.14 | `uv` venv with Python 3.12 |
| `torch.cuda.is_available()` False, or "no kernel image is available for execution on the device" | Torch wheel lacks sm_61 | Reinstall from the cu126 index; else run torch stages on CPU |
| `Could not load library libcudnn_ops.so.9` / `libcublas.so.12` (faster-whisper) | CTranslate2 can't find CUDA libs | Add the venv's `nvidia/cublas/lib` and `nvidia/cudnn/lib` to `LD_LIBRARY_PATH` |
| "target device or backend do not support efficient int8_float16/float16" | Pascal GPU | `compute_type="int8"` |
| CUDA out of memory | large-v3 + another GPU process, or long file | Ensure no other GPU process (`nvidia-smi`); `large-v3-turbo`; CPU |
| pyannote 401/403 "gated repo" | Terms not accepted / not logged in | H2 with the same HF account |
| pyannote / torchcodec / ffmpeg decode error | Audio decoding backend | Pass the in-memory waveform dict |
| pyannote 4 incompatible with installed torch | Version constraints | `pyannote.audio>=3.3,<4` + `speaker-diarization-3.1` |
| pysentimiento/transformers conflicts with pyannote | Dependency pins | Separate uv environment for E8c/E8d (`envs/nlp`), run via its own `uv run` |
| soundfile can't open a file | Unusual codec | Decode with ffmpeg to PCM_16 (after H1) |
| Whisper text in silent/censored spans | Hallucination | Check censor zeroing (E1) + E3 post-filters |
| One speaker for a whole call | Diarization collapse | Check audio; retry with `min_speakers=2` if the transcript clearly has 2 voices; flag `one_speaker` |

---

## 12. Tests (pytest, synthetic data, no real audio)

- **Censor detector:** synthetic 1 kHz tone, zero runs and white-noise bursts inserted in noise → detected spans within ±20 ms.
- **Turn builder:** hand-built segment/word timelines → expected turns, backchannels, interruptions, latencies, yield time.
- **Dynamics:** dead air, talk share, monologues, per-minute rates on a known timeline.
- **Quote verifier:** exact, minor-variation and fabricated quotes → pass / pass / fail.
- **Annotation validator:** schema violations and each consistency rule trigger the expected errors.
- **`ptp_strength`, `script_checklist_score`, `pct_objections_resolved`:** table-driven cases including null / "censored" / "na".
- **Table validation:** a column outside `variables.yaml`, or a wrong dtype → error.

---

## 13. Deliberately discarded (and why)

- **Paid APIs** (Anthropic API, OpenAI, Gemini, cloud ASR) — the project uses no paid services; local models + Claude Code cover the need.
- **Acoustic speech-emotion recognition** — models trained on 16 kHz acted speech; no way to validate.
- **Jitter / shimmer / HNR** — invalidated by the telephony codec.
- **Denoising / audio "super-resolution"** — degrades ASR or invents information.
- **End-to-end audio LLM** — less auditable, unreliable timestamps.
- **Fine-tuning Whisper** — no labeled data.
- **Orchestrators / databases** — overkill for 100 files.
- **Metadata from filenames** — filenames are UUIDs (no date/campaign/agent), so `call_datetime` and `campaign` are not extractable.

---

## 14. Build order (technical dependencies) & definition of done

**Order:**
1. Environment (§5.1–5.4) + H1.
2. Download (§8).
3. §9.0 data notes.
4. E0 → E1 → E2.
5. H2.
6. Smoke test on 3 calls (§5.5).
7. E3 → E4 → E5 on all calls.
8. Draft codebook + schemas + `variables.yaml` → H4.
9. E6 annotation → H5 → merge.
10. E7.
11. E8a–E8c.
12. E9 dev → codebook v2 → H4.
13. E9 pass A all → pass B all → rerun gold pass A.
14. E8d + objection QA.
15. H6 + H7.
16. E10.
17. E11.

Unit tests are written alongside each stage.

**Definition of done:**
- `uv sync && uv run extract audio-all && …` reproduces E0–E8 from raw audio with cached stages. The E9 annotations are the only non-script step, and are stored with `_meta`.
- `data/processed/` contains `calls` (one row per usable audio, every §10.1 and §10.2 column or a documented drop), `turns`, `objections`, `offers`, `data_dictionary.csv`.
- `data/validation/validation_report.md` and `variable_reliability.csv` exist, with metrics overall and by group.
- All annotation files pass `annotate-validate` (schema, turn_ids, quotes, consistency).
- `pytest` passes.
- `docs/data_notes.md` and `docs/decisions.md` record every fact found and every deviation from this spec.
- Nothing under `data/` is tracked by git.
