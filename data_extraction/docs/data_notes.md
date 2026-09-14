# Data notes — facts found on first contact (spec §9.0)

Technical facts only (no analysis). Group-level figures are aggregates computed by code; no call-level group
assignment is shown here or anywhere outside `data/private/`.

## 1. Source & inventory (2026-09-12)

- 100 WAV files: 50 in `audios_humanos_censurados`, 50 in `audios_ia_censurados`. Filenames are UUIDs (no metadata).
- Delivered locally at `../raw_data/`; copied into `data/raw/{humanos,ia}/` with sha256 verification (100/100).
- Anonymous IDs `C001…C100`: seeded shuffle (seed 20260912); map in `data/private/id_map.csv`.
- **Encoding (soundfile, every file): WAV `PCM_16`, 8000 Hz, mono — identical in both groups.** No stereo, no other
  sample rate, no corrupt or silent file → 0 exclusions. (ffprobe not run: ffmpeg is not installed, D-008. The RIFF
  header scan and soundfile agree.)
- Clipping: effectively none (max 0.01 % of samples at ≥ 0.999 FS in any call).

### Duration per group (file length, seconds)

| group | min | p10 | p25 | median | p75 | p90 | max | total |
|---|---|---|---|---|---|---|---|---|
| human | 74.6 | 103.8 | 129.9 | 171.1 | 285.6 | 413.6 | 1233.2 | 3.31 h |
| ai | 60.3 | 81.9 | 94.4 | 149.0 | 314.2 | 462.4 | 561.6 | 3.05 h |

## 2. Censoring method (E0; decisions D-004, D-016)

- **Method: a digital 1000 Hz sine beep replaces censored audio.** Measured inside beep spans: interpolated frequency
  std 0.00 Hz, amplitude CV 0.000, 2nd harmonic −106 dB relative to the fundamental.
- v2 detection (`censor.active_types: [tone]`, 1000 ± 50 Hz, freq std ≤ 3 Hz): **1624 beep spans in 98 calls**
  (2 calls contain none). Span duration p10 0.48 s · median 0.78 s · p90 1.30 s · max 6.26 s. Max within-span
  frequency std 2.16 Hz.
- Censored time per call: human median 8.5 s (4.95 % of file; max 14.3 %); AI median 13.0 s (8.34 %; max 20.8 %).
  Beeps per call: human median 12 (max 44), AI median 15.5 (max 48). `censor_heavy` (> 20 %): 1 call.
- **No digital-silence or noise censoring.** 124 exact-zero runs in 32 calls (median 0.12 s, max 1.94 s; 8 at file
  start; 121 of 124 are > 0.1 s from any beep) → line dropouts/padding. 1 noise-like span (0.27 s). Kept as
  `diagnostic_spans`; E2 SNR excludes exact-zero frames.
- 2375 other tone-like candidates (mostly 125–550 Hz; frequency drift ~9 Hz, amplitude CV ~0.33) are voiced speech, not
  censoring. They are stored as diagnostics only.
- Example beep spans for optional H3 listening: **C001 4.62–5.32 s**, **C002 11.49–12.57 s**, **C003 17.16–18.98 s**
  (audio paths via `data/private/id_map.csv`).

## 3. Preprocessing & VAD (E1, E2)

- E1 peak-normalization gain (16 kHz model copy, outside censor spans): median −1.28 dB (range −5.1 to +2.2 dB).
- Silero VAD (§7 parameters) on the 16 kHz copy; SNR on the 8 kHz master (speech vs non-speech frames, censor and
  exact-zero frames excluded).

| group | SNR dB min | p10 | median | p90 | max | `low_snr` (< 10 dB) | speech_ratio min | median | speech-frame level dBFS median |
|---|---|---|---|---|---|---|---|---|---|
| human | 6.3 | 8.0 | 11.6 | 15.9 | 20.0 | 14 calls | 0.17 | 0.76 | −20.1 |
| ai | 7.5 | 12.8 | 18.6 | 28.7 | 41.4 | 4 calls | 0.30 | 0.71 | −17.6 |

- `no_speech` flags (speech_ratio < 0.05): 0.
- E2 runtime: ≈ 5 s per call on CPU (8 threads), including SNR on the master.

## 4. Machine / runtime facts

- 4 GB GPU, compute capability 6.1. torch 2.11.0+cu126: CUDA available; arch list
  `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90` (no literal `sm_61`), yet cuDNN 9.10 conv1d/conv2d/bi-LSTM/STFT execute on the
  GPU (D-007). ctranslate2 4.8.2: 1 CUDA device, compute types `float32, int8, int8_float32`.
- torchcodec cannot load without FFmpeg (warning at pyannote import); pyannote.audio 4.0.7 imports fine, and audio is
  passed in memory.
- **ASR smoke test (E3, 3 calls, 2 + 1 across groups):** faster-whisper large-v3, int8, CUDA, no fallback. Wall time
  35.1 s / 26.3 s / 37.0 s for 142 s / 156 s / 152 s of audio → **RTF 0.25 / 0.17 / 0.24** (model already loaded).
  **Peak VRAM 2973 MiB** of 4096 (99 % utilization). Detected language `es` (probability 1.0). Duration-weighted word
  confidence 0.91–0.96; low-confidence words 1.6–6.1 %. No post-filter removals on these calls.
  Transcripts are readable Spanish with occasional ASR errors (e.g. company names).
- **ASR full batch (E3, 100 calls):** 0 failures. Median RTF 0.22 (max 0.55), including model load and the
  20.6-min call. Median duration-weighted word confidence 0.93; no call below `low_asr_conf` (0.6). Detected language
  `es` for all 100. Post-filters removed text in 18 calls (20 segments mostly outside VAD speech or inside beeps, 3
  n-gram loops) → flag `hallucination_removed`. 4 calls ran out of CUDA memory on `large-v3` and fell back to
  `large-v3-turbo`; they were re-run to keep one model (D-019).
- Not everything is censored: amounts ("5 millones 360 mil pesos"), dates ("19 de julio") and some first names are
  audible in transcripts. The E7 PII regex scan logs digit-sequence hits to `data/logs/pii_scan.log`. Transcripts stay local.
- **Diarization smoke test (E4, same 3 calls):** `pyannote/speaker-diarization-community-1` on CUDA, in-memory
  waveforms; RTF 0.15 / 0.09 / 0.09; **peak VRAM 2087 MiB**. 2 speakers detected in each call, with an unbalanced
  talk split (one speaker holds 76–88 % of speech). pyannote 4 provides the exclusive annotation directly.
- Speaker-attribution errors in the smoke test:
  - Whole sentences clustered with the other speaker (C025 66.9–72.0 s and 138–150 s: agent sentences labeled as
    the client's cluster). Handled by E6 `suspect_turns` + H5; role inputs now show the whole call (D-022).
  - Late segment boundaries (~1 s), so the start of a sentence goes to the other speaker. 3.5–6.2 % of words were
    a minority speaker inside their own Whisper sentence. Sentence smoothing (D-021) reassigned 6 / 0 / 2 words and
    removed 1 false interruption (C021: 11 → 9 turns).
  - Short answers without a sentence boundary (e.g. "Sí, con él." inside the agent's diarized segment) cannot be
    fixed by smoothing.
- Speech with no ASR words ("wordless turns"): C021 ends with 7.5 s of agent speech that Whisper did not
  transcribe; C042 starts with 0.6 s (likely "Aló"). Their frequency is counted per call in `turns/<call_id>.json: stats`.
- Pipeline kept: community-1 (spec primary). A CPU comparison with 3.1 was stopped unfinished to free the CPU for the
  batch.

## 5. Diarization, turns and roles (E4–E6)

- **E4 batch:** 100/100 calls with `speaker-diarization-community-1` on CUDA, 0 failures, median RTF 0.09.
  Speakers detected: 2 in 64 calls, 3 in 35, 1 in 1.
- **Collapse:** in 15 calls one speaker holds ≥ 90 % of diarized speech. C063 is genuine: nobody answers and only
  the agent speaks. In the other 14 (C024, C028, C039, C046, C061, C066, C067, C069, C084, C085, C087, C092, C095,
  plus C055 at 87 %), agent and client speech are merged into one speaker. C020 (97 %) is merged too.
- **Re-diarization test (D-023, GPU):** community-1 min 1/max 3, community-1 `num_speakers=2` and 3.1 `num_speakers=2`
  on the 15 calls.
  - Overrides adopted for C020 and C092 (3.1) and for C039, C055, C061, C067 (community-1 `num_speakers=2`).
    Speech split after the override: C020 26/30 s, C039 95/24 s, C055 101/35 s, C061 15/5 s, C067 18/48 s, C092 42/51 s.
  - C066 and C087 are not a collapse: the client speaks little, and all variants give the same view.
  - No variant separated C024, C028, C046, C069, C084, C085, C095.
  - The text-marker separation score did not track readable separation (C046 scored 1.0 unseparated; C039 0.0 separated),
    so it was used only for ranking.
  - Sentences inside the re-diarized turns are still mixed at turn edges (openings especially).
- **Unfixed collapse calls in E6:** the merged speaker is `AGENT` when the merged client speech is only
  greetings/confirmations (C028, C069, C084, C085), and `UNKNOWN` when it carries substantive client content
  (C024, C046, C095). All are low confidence.
- **E5:** 100 turn files; 3 words left without a speaker in total; 188 turns without ASR words (speech Whisper did not
  transcribe) before the 6 re-diarized calls; sentence smoothing active (D-021).
- **Language:** all 100 calls transcribed as Spanish.
- **E6 role annotation (100 calls, all passing `roles-validate`):**
  - AGENT in 97 calls. The exceptions are C024, C046 and C095 (unfixed collapse, `UNKNOWN`).
  - CLIENT in 74 calls, THIRD_PARTY in 13 (relatives, wrong numbers / wrong person), SYSTEM (voicemail recorder menu)
    in 4, UNKNOWN speakers in 18.
  - The agent is split over more than one diarization speaker in 12 calls.
  - 74 suspect turns in 42 calls (clear diarization misattributions, proposed for reassignment in H5).
  - Confidence: high 54, medium 36, low 10.
- Review sheet for H5: `data/interim/review/roles_review.csv`, 231 rows (one per call × speaker) for 100 calls.

## 5b. H5, canonical transcripts and features (E7, E8)

- **H5 (D-024):** every role approved, and all suspect turns accepted. E7 then ran 100/100 calls.
  - Canonical flags: `diar_corrected` 42, `three_speakers` 35, `low_snr` 18, `hallucination_removed` 19,
    `agent_spoke_first` 52, `censor_heavy` 1, `one_speaker` 1.
  - The PII regex scan logged 3 lines (`data/logs/pii_scan.log`).
- **E8:** 100/100 calls in each stage, 0 failures.
  - dynamics 6 s, prosody (Praat) 39 s, embeddings 50 s.
  - lexical 352 s, including sentiment with `pysentimiento/robertuito-sentiment-analysis`.

## 5c. E9 dev round

- Dev calls C002, C046, C056, C062, C081, C090: pass A and pass B annotated with codebook v1. All 12 files pass
  `annotate-validate`.
- Material for human review:
  - `data/interim/review/dev_round_review.csv` (339 rows: Claude's value, plus empty reviewer-value / `note` columns).
  - `data/interim/review/dev_round_codebook_questions.md` (14 ambiguities, each with a proposed v2 rule).
- WER hypotheses (`data/validation/wer_refs/<call_id>.whisper.txt`) exist for all 4 WER calls.

## 5d. E9 full run, provisional E10, E11

- **E9 (codebook v2):** pass A and pass B for 100/100 calls, all passing `annotate-validate`.
  - Worked in batches of 10; for each call, pass A was written before pass B.
  - 0 `quote_verification_failed` flags. Pooled totals: 97 objections, 167 offers.
  - Blindness guesses: ai 49, human 46, unsure 5.
- **Objection QA:** 14 quotes typed `other` → `data/validation/objection_other_clusters.md`.
- **E10 (provisional, D-026):** run before H6 gold annotations and before the self-consistency rerun (0 gold rows, 0 reruns).
  - Role accuracy is 1.000 by construction: every H5 row was approved.
  - WER on the 4 H7 calls: 0.063 overall (human 0.060, ai 0.068); number/date token accuracy 0.885.
  - **Blindness probe accuracy: 0.950** (human 0.920, ai 0.980; 5 unsure). The annotator's guess matched the
    source group in almost every call, so annotation was not effectively blind to agent type (the transcript text itself
    carries the cues). Interpretive variables must be read with this in mind.
  - Variable decisions: keep 64, flag 68, drop 0. Every flag comes from missing gold/self-consistency data.
- **E11:** `data/processed/` tables written (calls 100, turns 1764, objections 97, offers 167) plus
  `data_dictionary.csv`. No variable dropped. They must be rebuilt after H6 and the rerun.

## 5e. Final E10 / E11 (after the partial H6 and the self-consistency rerun)

- **Inputs:**
  - Gold: 1 row (C004, D-027).
  - Self-consistency rerun: 20/20 gold calls, pass A, written in a separate Claude session; all valid.
- **Decisions (with `min_rows_decision: 10`, D-028):** keep 64, flag 70, drop 2.
  - Every `keep` is a deterministic variable. No interpretive variable reached `keep`: gold is too thin, and
    self-consistency alone is capped at `flag`.
  - Dropped for low rerun agreement (n = 20): `coercive_language` (κ 0.00) and `incoherent_responses` (Spearman −0.05).
    Both are rare events, and the two runs disagreed on them.
  - Self-consistency examples: `callback_scheduled`, `delinquency_stage`, `message_left`, `third_party_relation` κ 1.00.
    `closing_by` κ 0.79; `had_negotiation` κ 0.77 (group gap 0.38); `non_payment_reason` κ 0.70;
    `client_expresses_annoyance` κ 0.64. `milestone_ptp` exact 0.71; `debt_amount` exact 0.92 (group gap 0.25).
  - Undefined (no variance in the 20 calls): `product`, `ptp_channel`, `ptp_type`.
- **E11 final tables** in `data/processed/`: calls 100, turns 1764, objections 97, offers 167, plus `data_dictionary.csv`.
  The two dropped variables are removed from `calls`, and each variable's decision is in the data dictionary.
- **Environment incident:** the separate rerun session ran `uv` without `scripts/env.sh`, so `.venv` was rebuilt twice
  around a Python outside the project (a pre-existing user-level uv Python). The environment was rebuilt with
  the project-local interpreter before these runs, and all 59 tests pass.

## 6. Blinding log

- E6 role annotation: done by reading only `data/interim/llm_inputs/roles/*.txt` and `configs/` (codebook,
  schemas). No group lookup files were opened. Masked `[AGENTE]` spans were left uninterpreted. All E6 statistics above
  are pooled, not split by group.
- D-023 variant inspection (before E6 annotation of the 15 collapse calls): sentence-level views were printed from
  `data/interim/asr/` + diarization JSONs (no group files). The first printout was unmasked. Masking
  (`agent_masking`) targets AI self-descriptions only, so agent first names appear in role inputs as well. That matches
  the spec lexicon.
- E9 dev round: annotated by reading only `data/interim/llm_inputs/calls/<call_id>.txt`, `configs/codebook.md` and
  `configs/schemas/`. The validation id files were read through their `call_id` column only (they have no other
  column). `data/private/review_audio.csv` was not opened.
- E9 full run: same inputs only. For H7, a script wrote the WER audio paths to `data/private/wer_audio_paths.csv`
  without printing them. After all 100 annotations were finished, the provisional E10 report showed group-level
  aggregates (WER, blindness probe) and no per-call groups. The self-consistency rerun must therefore be done in a fresh
  session that has not seen this report.
- Build-phase exposure: group-level aggregates (above) only. Smoke-test calls (`data/interim/smoke_calls.txt`) were
  drawn 2 + 1 across groups without displaying which group is which. Their first ASR segments were displayed once, for
  the §5.5 readability check.
