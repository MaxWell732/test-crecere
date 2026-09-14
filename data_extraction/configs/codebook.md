# Codebook — Collections calls (AI vs human) · v3

> **Status: v3 (2026-09-13) = v2 minus the variables the reviewer removed from the tables (D-029).**
> v2 was APPROVED by the reviewer (H4, 2026-09-12) and used for the full E9 run. It added the 14 rules the reviewer accepted
> after the dev round (`data/interim/review/dev_round_codebook_questions.md`). E6 role annotations were made under v1;
> the role codes are unchanged in v2 and v3.
> The file's sha256 is the codebook version. Put it in every annotation's `_meta.codebook_sha256`
> (`sha256sum configs/codebook.md`). Any edit changes the hash and makes earlier annotations **stale**.

Changelog:
- v1 (2026-09-12): drafted by Claude Code from spec §9-E6, §9-E9 and §10.
- v2 (2026-09-12): dev-round rules, marked **[v2-n]** below:
  1. offers need a stated term;
  2. an unchanged prior agreement is not an offer;
  3. `written_off_legal` needs a current legal status;
  4. objections only from the account holder;
  5. agent backchannel quotes allowed for compliance items;
  6. `identifies_self` with a censored entity;
  7. coding from content in collapsed-speaker calls;
  8. installment amounts;
  9. settlement amounts are not balances;
  10. `ptp_confirmed_by_client` timing;
  11. `abrupt_hangup` only when visible;
  12. what counts as disclosure to a third party;
  13. wording of consistency rule 3;
  14. objection episodes.
  Consistency rule 10 (§4) is new.
- v3 (2026-09-13, D-029): removed `third_party_relation`, `message_left`, `delinquency_stage` (with [v2-3]),
  `days_past_due`, `ptp_days_until`, `ptp_channel`, `asks_if_robot`, `asks_for_human` and `abrupt_hangup` ([v2-11]
  keeps only its `closing_by` part). `non_payment_reason` + `non_payment_reasons_all` merged into
  `non_payment_reasons` (primary first). Consistency rule 4 removed (numbers kept); rule 11 is new. Existing
  annotations were migrated (fields dropped, reasons merged, no kept value changed).

---

## 0. Rules for whoever annotates

1. **Read only** `data/interim/llm_inputs/…`, `configs/codebook.md` and `configs/schemas/`. Never open `data/private/`,
   `data/raw/`, `data/processed/`, or anything that links a call_id to human/AI. Don't consult other calls' annotation
   files while annotating.
2. **Read the whole transcript.** Don't skim. Code from what is said, not from what "usually" happens.
3. **Quotes are copied verbatim** from the text of the cited turn line (the part after `] `). Use a contiguous stretch of
   3–30 words. You may include `[CENSURADO]`. Leave out inline backchannels `(ROLE: …)`. Leave out the `[?]` markers
   (the validator ignores them anyway). Validation: case, punctuation and `[?]` are ignored, then fuzzy
   `partial_ratio ≥ 90` against that turn.
   **[v2-5]** Exception, for compliance items only: when the agent's words appear only as a backchannel inside another
   speaker's line (e.g. `(AGENT: Buenas tardes.)`), quote the backchannel text with that line's `turn_id`. (The
   validator compares against the whole line, backchannels included.)
4. **Value conventions**
   - `null` = not mentioned in the call.
   - `"censored"` = mentioned, but the value is covered by `[CENSURADO]`.
   - `"na"` = not applicable (the question doesn't arise, e.g. `recaps_agreement` when there is no promise).
   - Booleans are `true` / `false`. `false` means it was applicable and did not happen.
   - Amounts are plain numbers in COP (`"cuatrocientos mil"` → `400000`; `"1,2 millones"` → `1200000`).
5. If a quote cannot be made to verify after re-reading (e.g. ASR garbled the words), set the quote and its `turn_id`
   to `null` and add `"quote_verification_failed"` to the top-level `flags`. This is a last resort.
6. `_meta`: `call_id`; `annotator: "claude-code"`; `model` as reported by the session; `claude_code_version`;
   `codebook_sha256`; `annotated_at` (ISO 8601 UTC).
7. Work in batches of 10. Re-read this codebook at the start of every batch. After each batch run
   `uv run extract annotate-validate --pass a` (or `b`, `roles`) and fix every error before continuing.

### Reading the transcript

`[T012 | AGENT | 00:41.2–00:47.9] texto … [CENSURADO] … (CLIENT: ajá) …`

- `word[?]` = low-confidence ASR word. Treat it with caution, but it is still evidence.
- `[CENSURADO]` = the provider censored audio there (usually names, IDs, phone numbers, amounts or dates).
- `[AGENTE]` = a masked phrase in which the agent described what it is. Do **not** try to guess what was masked.
- `(ROLE: text)` = a short backchannel by another speaker inside this turn.
- A short answer such as "Sí." or "Ajá" can appear as a backchannel inside the other speaker's turn. That's normal:
  read it as part of the dialogue.

---

## 1. Roles (E6) — `annotations/roles/<call_id>.json`

Input: `llm_inputs/roles/<call_id>.txt` (speaker labels `SPK_00…`, the first ≈ 90 s / 20 turns, plus the first turns
of any speaker who appears only later).

| Role | Definition | Typical cues |
|---|---|---|
| `AGENT` | The collector calling on behalf of the creditor, **human or synthetic voice**. A TTS/AI agent is `AGENT`, never `SYSTEM`. | Greets, names the entity, asks for the debtor by name, mentions the obligation, payment, channels. |
| `CLIENT` | The debtor / account holder being sought, **even before identity is validated**, if it is the person the agent is looking for. | Answers "¿aló?", confirms name, talks about their debt, payments, circumstances. |
| `THIRD_PARTY` | A person other than the debtor who answers or intervenes (relative, coworker, wrong-number person). | "No, él no está", "ella es mi mamá", "aquí no vive nadie con ese nombre". |
| `SYSTEM` | Non-conversational machine audio: IVR menus, carrier messages ("el número que usted marcó…"), voicemail greetings, recorded hold messages. | Fixed announcements, "deje su mensaje después del tono", menu options. |
| `UNKNOWN` | Speech that cannot be attributed (too little text, crosstalk, noise). | Use sparingly. |

Rules:
- Several diarization speakers may map to the same role (e.g. one agent split into `SPK_00` and `SPK_02`).
- `speaker_map` must contain **every** speaker label that appears in the file (including any that appear only in backchannels).
- `evidence`: at least one `{speaker, turn_id, quote}` per mapped speaker, whenever that speaker has a turn with text.
  The quote must come from that speaker's turn (or from their backchannel inside it).
- `suspect_turns`: turns whose text clearly belongs to another role, i.e. a likely diarization error. Example: a line
  saying "le hablo de parte del Banco…" labeled as the speaker you mapped to `CLIENT`. Give `reason` (≤ 300 chars) and
  `proposed_role`. Don't list turns that are just short or ambiguous.
- `confidence`: `high` = roles are obvious from explicit cues; `medium` = consistent but indirect cues; `low` = guesswork
  or heavy diarization problems.

---

## 2. Pass A (E9) — `annotations/pass_a/<call_id>.json`

Input: `llm_inputs/calls/<call_id>.txt` (roles already reviewed by a person). `turn_id`s refer to this file.

**[v2-7] Collapsed speakers.** If a turn's speaker is `UNKNOWN` because agent and client were merged, code from content
when the text clearly shows who says what. Add a line to `_meta.notes` saying so. Speaker-based features stay flagged
by E7.

### 2.1 `contact`

`contact_type` — pick the **most advanced** contact reached in the call:

| Value | Use when |
|---|---|
| `no_answer` | Ringing, silence or line noise; no human and no machine message. |
| `voicemail` | A voicemail greeting or recording answers (the agent may or may not leave a message). |
| `ivr` | An IVR / carrier menu or announcement, and no person is reached. |
| `wrong_number` | A person answers and states the debtor is unknown at this number. |
| `hangup_before_identification` | A person answers, but the call ends before it becomes clear who they are. |
| `third_party` | A person other than the debtor, who knows them or is related, speaks with the agent. |
| `account_holder` | The debtor speaks with the agent, whether or not identity was formally validated. |
| `unclear` | Cannot be determined from the transcript. |

- `identity_validated`: `true` if the agent explicitly confirms the person is the account holder at any point (asks for
  and gets confirmation of name or ID, e.g. "¿hablo con el señor [CENSURADO]?" – "Sí, señor"). `false` if contact was
  with a person but no explicit confirmation happened. `na` if no person was reached.
- `evidence`: 1–3 quotes supporting `contact_type` (and identity validation, if `true`).

### 2.2 `debt_context`

- `product`: `credit_card`, `personal_loan` (libre inversión, crédito de consumo), `vehicle_loan`, `mortgage` (hipotecario,
  leasing habitacional), `payroll_loan` (libranza), `microcredit`, `telecom_services` (plan celular, internet), `other`,
  `unknown` (not stated).
- `debt_amount` (total balance) and `overdue_amount` (amount due now / en mora): number, `"censored"` or `null`. If only one
  amount is stated and it isn't clear which it is, put it in `overdue_amount`.
  - **[v2-8]** Under an installment agreement, `overdue_amount` is the amount due now (the current installment).
  - **[v2-9]** Settlement, discounted or approved-payment amounts ("con el descuento quedaría en…", "la aprobación del
    pago total de…") never go in `debt_amount` / `overdue_amount`. They belong in `offers[].amount` and, if promised,
    in `ptp_amount`.
- `non_payment_reasons`: every reason stated, **primary first** (the one the client stresses most), each at most once:
  `unemployment_income_loss`, `forgot`, `illness_calamity`, `over_indebtedness`, `disputes_charge`,
  `doesnt_recognize_debt_fraud`, `already_paid`, `payment_channel_problem`, `other`.
  If no reason is given: `["not_stated"]` (never combined with other reasons).

### 2.3 `outcome`

`final_outcome` is ordinal. Choose the **highest** level the call reaches:

| Rank | Value | Definition |
|---|---|---|
| 0 | `no_contact` | No conversation with a person who could relay or act (no answer, voicemail, IVR, wrong number, hang-up before identification, unclear). |
| 1 | `third_party` | Spoke only with a third party. |
| 2 | `holder_refuses` | The account holder explicitly refuses to pay or to engage ("no voy a pagar", "no me vuelvan a llamar"). |
| 3 | `no_agreement` | The holder engages, but the call ends without a commitment or scheduled callback. |
| 4 | `callback_scheduled` | A concrete new contact is agreed (day/time or "llámeme el lunes"). |
| 5 | `claims_already_paid` | The holder states they already paid, and no new promise is made. |
| 6 | `vague_promise` | Conditional or uncertain commitment ("voy a tratar", "apenas me paguen", "a ver si el viernes"). |
| 7 | `firm_promise` | Explicit, unconditional commitment ("sí, el viernes pago los 400"). |

- `ptp`: `true` if the account holder commits to pay (firm or vague). Only the account holder can make a PTP.
- `ptp_type_commitment`: `firm` / `vague` if `ptp`; else `na`. Must match `final_outcome` (firm → `firm_promise`, vague → `vague_promise`).
- `ptp_amount`: number, `"censored"` or `null`. `ptp_date`: the date **as stated** ("el viernes", "30 de septiembre",
  "la otra semana"), `"censored"` or `null`.
- `ptp_type`: `full` if the promise covers the overdue amount (for an installment agreement, the current installment,
  **[v2-8]**); `partial` if less; `null` if unknown.
- `ptp_confirmed_by_client`: `true` if the client explicitly confirms the final terms after they are stated
  ("sí, listo, el viernes los 400"); `false` if not; `na` if no `ptp`. **[v2-10]** `true` when the client explicitly
  accepts at any point after the final amount **and** date have both been stated, even if the agent restates them
  later.
- `agent_recapped`: `true` if, before closing, the agent restates at least one of amount, date or channel of the
  agreement; `false` if not; `na` if no `ptp`.
- `callback_scheduled`: `true` if a new contact is agreed (independent of `final_outcome`).
- `evidence`: quotes for the outcome (and the promise, if any).

### 2.4 `negotiation`

- `had_negotiation`: `true` if any payment option, amount or date is discussed.
- `offers[]`: one entry per **distinct proposal** of amount, date, installment plan, discount, refinancing or extra time.
  A literal repetition of the same proposal is not a new offer.
  - **[v2-1]** An offer starts at the first turn that states at least one term: amount, percentage, date or number of
    installments. A generic announcement ("le tenemos una gran reducción", "una propuesta muy favorable") is not an
    offer.
  - **[v2-2]** Restating an agreement made in an earlier call without changing it is not an offer. Changing it (a new
    date, a single payment instead of installments, a different amount) is an offer.
  - `proposed_by`: `agent` / `client`.
  - `type`: `full_payment`, `minimum_installment` (pago mínimo / cuota), `partial_payment` (abono), `refinancing_restructuring`,
    `discount_forgiveness` (descuento, condonación de intereses), `extra_time` (plazo adicional without an amount).
  - `amount` (number / `"censored"` / `null`), `date_text` (as stated / `"censored"` / `null`), `turn_id`, `quote`.
- `first_anchor`: `proposed_by` of the earliest offer; `none` if there are no offers.

### 2.5 `objections[]`

A client statement that resists paying or continuing. One entry per objection **episode**. The same objection raised
again later, after the agent has moved on, is a new entry.

- **[v2-4]** Only the **account holder** raises objections. Complaints by a third party or a wrong-number person ("me
  llaman a toda hora", "yo no soy esa persona") are not objections. Code them in
  `failures.client_expresses_annoyance` when they are explicit.
- **[v2-14]** "Moved on" means the agent changed topic or changed the offer. If the client repeats the same objection
  while the agent insists on the same proposal, it is one entry: `turn_id` = the first statement, and `resolved`
  reflects the end state.

| `type` | Examples |
|---|---|
| `no_money` | "No tengo plata", "estoy sin trabajo" |
| `already_paid` | "Yo ya pagué eso" |
| `call_later` | "Estoy ocupado, llámeme luego" |
| `doesnt_recognize_debt` | "Yo no tengo esa deuda", "eso no es mío" |
| `wrong_amount` | "No debo tanto", "ese valor no es" |
| `wants_discount` | "¿Me hacen descuento?" as resistance |
| `distrust_who_is_calling` | "¿Y usted quién es?", "¿cómo sé que no es una estafa?" |
| `annoyed_by_calls` | "Me llaman todos los días" |
| `asks_for_human` | "Quiero hablar con una persona / un asesor" |
| `other` | Anything else (keep the quote precise; these are reviewed in QA) |

- `response_turn_id`: the agent turn that responds (`null` if none).
- `agent_technique` (the main one): `empathy_validation`, `clarification`, `alternative_offer`, `consequences`
  (reporting, interests, legal — stated descriptively or as pressure), `reschedule`, `ignores_repeats_script`, `escalates`
  (transfers, supervisor).
- `resolved`: `yes` = the client moves forward (accepts, stops raising it, agrees to a next step); `partial` = acknowledged,
  but the client remains hesitant or shifts to another objection; `no` = persists, or the call ends on it.

### 2.6 `compliance` — each item `{value, turn_id, quote}`

`value` ∈ `true` / `false` / `"na"`. When `true`, give the `turn_id` and `quote` that show it. When `false` or `na`,
use `null` for both.

| Item | `true` when | `na` when |
|---|---|---|
| `greets` | The agent opens with a greeting. | No agent speech. |
| `identifies_self` | The agent states own name **or** role **and** the entity (bank/company/agency). | No agent speech. |
| `recording_notice` | The agent says the call is recorded/monitored. | Not account_holder. |
| `validates_identity_before_disclosing` | Identity is confirmed **before** any debt detail (existence, amount, product). | Not account_holder, or no debt detail given. |
| `states_amount` | An amount owed or overdue is stated (even if censored). | Not account_holder. |
| `states_due_date` | A due date or the days/time overdue is stated. | Not account_holder. |
| `offers_payment_channels` | At least one payment channel is named. | Not account_holder. |
| `mentions_credit_bureaus` | Reporting to credit bureaus (centrales de riesgo, Datacrédito) is mentioned. | Not account_holder. |
| `recaps_agreement` | The agent restates the agreement before closing. | No `ptp`, or not account_holder. |
| `polite_closing` | A courteous goodbye by the agent. | Not account_holder. |
| `debt_disclosed_to_third_party` | The agent reveals the debt's existence or amount to a non-account-holder. | No third party in the call. |
| `coercive_language` | Threats, intimidation or humiliation (not a neutral statement of consequences). | No agent speech. |

- **[v2-6]** `identifies_self` is `true` when the agent states a name or a role/area **and** an entity reference is
  present, even if the entity name is `[CENSURADO]` ("Le hablo del [CENSURADO] área de embargos…").
- **[v2-12]** Disclosure to a third party = the agent names the existence of a debt/obligation, the creditor, or an
  amount. A generic "un proceso legal en curso" / "un asunto pendiente" alone is not a disclosure (`false`).
- **[v2-13]** If `contact_type ≠ account_holder`, the eight items from `recording_notice` to `polite_closing` are
  `"na"`. `greets`, `identifies_self`, `debt_disclosed_to_third_party` and `coercive_language` are coded normally.

### 2.7 `milestones`

The `turn_id` of the first turn where each happens, or `null`: `identification_turn_id` (identity confirmed),
`debt_mention_turn_id` (debt first mentioned), `first_offer_turn_id` (first offer), `ptp_turn_id` (the promise).
`debt_mention ≤ first_offer ≤ ptp` must be in chronological order. Identification is not constrained, because a
disclosure before identification is possible.

### 2.8 `failures`

- `agent_misunderstood_turn_ids`: agent turns that show a failure to understand the client (answers a different question,
  asks again for information just given, misreads a refusal as acceptance).
- `incoherent_response_turn_ids`: agent turns that do not respond to the client's preceding content at all.
- `client_expresses_annoyance`: `{value, turn_id, quote}`, where `value = true` on explicit annoyance or anger ("qué
  fastidio", "¡ya les dije!").
- `closing_by`: `agent` (the agent closes), `client_hangup` (the client ends it), `cut_unclear` (the call cuts or the ending is unclear).
  **[v2-11]** A call that ends on an unanswered agent question → `cut_unclear`.

### 2.9 `blindness`

`guess_agent_type`: `human` / `ai` / `unsure`, plus `reason` (≤ 400 chars). This is a blindness probe only. Answer
honestly from cues in the text; don't look for them on purpose.

---

## 3. Pass B (E9) — `annotations/pass_b/<call_id>.json`

Six scores, each `{score: 1–5 | "NA", justification (≤ 25 words), turn_ids[]}`. Anchors are given for 1 / 3 / 5; use 2 and 4
for intermediate cases. Score the **agent's** behavior. Use `"NA"` only where stated.

| Dimension | 1 | 3 | 5 | `NA` |
|---|---|---|---|---|
| `clarity` | Amount/date/next steps missing or confusing; client visibly confused | Main info given but incomplete or jargon-heavy | Amount, date, channel and next step explicit; client shows understanding | No conversation with account holder or third party |
| `empathy` | Ignores or dismisses the client's situation; pressure | Formulaic acknowledgment ("entiendo") without adapting | Specific acknowledgment and proposal adapted to the situation | Same |
| `active_listening` | Ignores client input; repeats already-answered questions | Responds to some points, misses others | References what the client said; relevant follow-ups; no repeated questions | Same |
| `objection_handling` | Ignores objections or repeats the script | Addresses objections generically, not resolved | Addresses the root cause; fitting alternative; client moves forward | **Required** when pass A has no objections |
| `control_focus` | Conversation drifts or ends without direction | Reaches its purpose inefficiently | Guides to a concrete next step without pressure | Same as clarity |
| `professionalism` | Rude, coercive or inappropriate | Correct but cold/mechanical | Courteous, respectful, appropriate register throughout | No agent speech |

---

## 4. Consistency rules (enforced by `annotate-validate`)

1. `ptp = false` ⇒ `ptp_amount`, `ptp_date`, `ptp_type` are `null`;
   `ptp_type_commitment`, `ptp_confirmed_by_client`, `agent_recapped` are `"na"`; `final_outcome ∉ {vague_promise, firm_promise}`;
   `compliance.recaps_agreement = "na"`.
2. `ptp = true` ⇒ `final_outcome` = `firm_promise` if commitment is `firm`, `vague_promise` if `vague`.
3. `contact_type ≠ account_holder` ⇒ `final_outcome ∈ {no_contact, third_party}`, `ptp = false`, and the script items are `"na"`
   except `greets` / `identifies_self`.
4. *(removed in v3 with `third_party_relation`, D-029)*
5. No objections ⇒ pass B `objection_handling = "NA"`; objections present ⇒ not `"NA"`.
6. `had_negotiation = false` ⇒ no offers. `first_anchor` = proposer of the earliest offer, or `none` when there are no offers.
7. `debt_mention_turn_id ≤ first_offer_turn_id ≤ ptp_turn_id` where present.
8. Every `turn_id` exists, and every quote verifies against its turn.
9. A compliance item or failure flag with `value = true` has a `turn_id` and `quote` (unless flagged `quote_verification_failed`).
10. **[v2-4]** `contact_type ≠ account_holder` ⇒ `objections` is empty.
11. **[v3]** `non_payment_reasons` is `["not_stated"]` alone, or stated reasons without `not_stated`.
