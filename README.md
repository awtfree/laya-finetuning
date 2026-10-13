# Laya, running locally

[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (Apache 2.0) is a
**non-autoregressive decision model**. You give it a *state* (text, an email, a ticket, any JSON) and
typed *questions*. It answers every question in **one encoder forward pass** with probabilities.
It never generates text.

| type | you give | you get |
|---|---|---|
| `choice` | `criteria: {label: description}` (or a list of labels) | `choice`, `probabilities`, `confidence` |
| `score` | `criteria: [level0, level1, ...]` | `score` (expected level), `probabilities` |
| `noul` | a yes/no question (optional `{"true":..., "false":...}`) | `noul` = P(true) |

The repo bundles three checkpoints, and this stack keeps all three resident on the GPU:

| checkpoint | encoder | params | context (options budget) | best at |
|---|---|---|---|---|
| `english` (repo root) | ModernBERT-large | 421M | 512 (192) | English text, guardrails, email triage |
| `multilingual` | mmBERT-base | 322M | 1024 (256) | 100+ languages |
| `typed-decisions` | ModernBERT-large | 421M | 1024 (256) | the 4 typed-decisions workflows it was fine-tuned on |

Requests go through the official `laya` SDK's `Router`. It detects the script and language in under a
millisecond and sends non-English text to `multilingual`. `typed-decisions` is only used when you ask for it.

This repo does two things:

1. **Run Laya locally** with an API and a UI, in Docker on your GPU. See [Start / stop](#start--stop) below.
2. **Fine-tune Laya on your own decisions**, evaluate the result against the base model, and serve it
   alongside the shipped checkpoints. See [Fine-tuning](#fine-tuning) and the full guide in
   [`docs/FINETUNING.md`](docs/FINETUNING.md).

### Requirements

- An NVIDIA GPU with Docker GPU access (`scripts/preflight.py` checks this). Serving all three checkpoints takes
  ~7 GB of GPU memory, including ~1.1 GB of CUDA graphs (`CUDA_GRAPHS=0` saves that); fine-tuning
  ModernBERT-large needs ~9 GB more. CPU-only works for serving (slowly), not for training.
- Docker with compose v2, ~5 GB of disk for weights, and ~10 GB for images.
- `python3` on the host for the helper scripts (stdlib only; they bootstrap their own dependencies).

### Layout

| path | what |
|---|---|
| `api/` | FastAPI server: `app.py` (laya SDK `Router` + published fine-tunes), `batching.py` (GPU scheduler: batching and request coalescing) |
| `frontend/` | Gradio UI |
| `train/` | fine-tuning image: `validate.py`, `train.py`, `calibrate.py`, `evaluate.py`, `publish.py`, `prepare_example.py`, `summarize.py`; `bakeoff.py`, `specialists.py`, `prepare_specialists.py` for the model comparison |
| `scripts/` | `download.py` (weights + update check), `smoke_test.py`, `test_batch_equivalence.py`, `test_scheduler.py`, `test_coalescing.py`, `load_test.py`, `preflight.py`, `reproduce_experiments.sh`; `bench/` holds the GPU/CPU micro-benchmarks behind docs/PERFORMANCE.md |
| `models/` | downloaded base weights (read-only in containers) |
| `data/` | your training data; `data/sample/` shows the format |
| `runs/` | training outputs, one directory per run |
| `finetuned/` | published fine-tunes; the API serves each subdirectory |
| `docs/FINETUNING.md` | data format, amounts, hardware, timings, measured results |
| `docs/USE_CASES.md` | what to build with each question type, combinations, hybrid patterns |
| `docs/DATA_PREP.md` | turning exports, votes and LLM-teacher labels into training JSONL; splits; tested recipes |
| `docs/PERFORMANCE.md` | every serving optimisation, the measurements behind it, and what each gained |
| `docs/BAKEOFF.md` | Laya vs popular zero-shot, fine-tuned and specialist models: does it live up to the hype? |
| `docs/BAKEOFF_NEXT_STEPS.md` | planned tests to make the bake-off fully fair (tuning budget, seeds, fine-tuned competitors) |

## Start / stop

First run on a new machine:

```bash
python3 scripts/preflight.py --workdir . --ports api=8001,frontend=7860   # GPU, Docker GPU access, ports, disk
cp .env.example .env                        # then set LOCAL_UID/LOCAL_GID (`id -u`/`id -g`) and ports if needed
python3 scripts/download.py convaiinnovations/laya   # 2.3 GB of weights -> models/, pinned in download.lock.json
```

Then:

```bash
docker compose up -d --build     # first build ~5 min; startup ~2 min (loads, warms, captures CUDA graphs)
python3 scripts/smoke_test.py    # 20 checks against the live stack (make test waits for startup)
docker compose logs -f api
docker compose down
```

`make help` lists the shortcuts (`make up`, `make test`, `make check`, ...).

| URL | what |
|---|---|
| http://localhost:7860 | Gradio UI |
| http://localhost:8001/docs | interactive API docs |
| http://localhost:8001/info | GPU, per-checkpoint device/dtype/latency, warnings |

The API is on **8001** because 8000 was already taken on this host. Change `API_PORT` in `.env`.

## API

```bash
curl -s localhost:8001/v1/decide -H 'content-type: application/json' -d '{
  "state": {"subject": "Duplicate charge", "body": "Billed twice for March. Refund today or we cancel."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "pricing", "other": "else"}},
    "urgency":    {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "critical"]},
    "churn_risk": {"type": "noul",  "instructions": "Does the user threaten to cancel?"}
  }}'
```

| endpoint | purpose |
|---|---|
| `POST /v1/decide` (alias `/v1/system_one`) | answer questions; optional `model` (`auto`/`english`/`multilingual`/`typed-decisions`/a published fine-tune), `lang`, `task` |
| `POST /v1/route` | which checkpoint would answer, and why, without running the model |
| `POST /v1/compare` | the same input on every checkpoint |
| `POST /v1/decide/batch` | many states (up to 1,024), one question set, truly batched on the GPU; same answers as `/v1/decide` per state |
| `GET /v1/presets` | the SDK's question sets (triage, email, guard, moderation, router) plus each fine-tune's trained questions |
| `GET /v1/preset_examples` | an example state for each fine-tune preset |
| `GET /health`, `/info`, `/v1/models` | introspection: GPU, per-checkpoint device/latency, fine-tunes with their eval results |

Errors: **422** for a malformed question (or options that do not fit the token budget), **400** for an unknown
`model`, **503** while loading or when the GPU queue is full (`QUEUE_MAX_SEQUENCES`; retry shortly), **504** if a
request waited longer than `REQUEST_TIMEOUT_S` (default 120 s), **500** if inference failed for that request's
input (see [Failures](#failures)).

- **No authentication.** Compose publishes ports 8001 and 7860 on all interfaces, so anything on your network can
  call the API and open the UI. For a shared machine, bind to localhost in `docker-compose.yml`
  (`"127.0.0.1:${API_PORT:-8001}:8000"`) or put an authenticating reverse proxy in front.
- **Concurrent calls are coalesced.** Parallel `/v1/decide` calls share GPU passes automatically (see
  [Batching and coalescing](#batching-and-coalescing)). A client that sends one request at a time gets no batching,
  so bulk work should still use `/v1/decide/batch`.
- **Not OpenAI-compatible.** Laya does not generate text, so there is no `/v1/chat/completions`. `/v1/system_one`
  has the same request and response shape as TypeSafe's Jev `system_one`.

### Batching and coalescing

Nothing touches the GPU directly. Every inference request becomes a *ticket* in one queue: a single
`/v1/decide`, each checkpoint of a `/v1/compare`, or each checkpoint group of a `/v1/decide/batch`. The ticket
holds the request's (state, question) sequences, already tokenised, and a future for the reply. One GPU worker
thread ([`api/batching.py`](api/batching.py)) repeatedly:

1. takes a round of queued rows for one checkpoint, up to 4 × `BATCH_TOKEN_BUDGET` tokens. The oldest ticket
   always gets at least a quarter, so big batches progress; the rest goes to the tickets with the fewest remaining
   rows, so single calls are not stuck behind a batch;
2. sorts the round by length and packs it into forward passes of at most `BATCH_TOKEN_BUDGET` padded tokens
   (default 8,192). Sorting cut padding from 43% to 13% of GPU work under mixed traffic;
3. hands every row back to the ticket it came from. When a ticket is complete its request wakes up,
   post-processes its own rows and responds.

It is opportunistic: a request arriving at an idle GPU runs immediately (~7 ms), and requests that
arrive while the GPU is busy share the next round. The event loop never blocks; tokenising and post-processing
run in the thread pool. Responses include `timing.queue_ms` (the wait for the GPU) and `timing.gpu_passes`.

**Throughput** (RTX 3090, English base, 3 questions per request, `make load-test`: N clients sending
`/v1/decide` back to back):

| clients | short messages: req/s (p50) | before coalescing | ~500-token states: req/s (p50) | before coalescing |
|---|---|---|---|---|
| 1 | 148 (6.7 ms) | 39 (26 ms) | 43 (24 ms) | 34 (28 ms) |
| 4 | 194 (21 ms) | 39 (102 ms) | 43 (94 ms) | 35 (114 ms) |
| 16 | 265 (60 ms) | 39 (407 ms) | 46 (333 ms) | 35 (455 ms) |
| 64 | 289 (214 ms) | 39 (1,618 ms) | 53 (1,172 ms) | 35 (1,835 ms) |
| 128 | 314 (389 ms) | 39 (3,229 ms) | 53 (2,377 ms) | 34 (3,662 ms) |

That is 8× for short messages and 1.6× for long states, with no errors at any level. The full history (GPU
batching, coalescing, length-sorted rounds, the CPU-side fixes and CUDA graphs below), with the measurements behind
each step, is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). The four CPU-side optimisations (each measured, and
each output identical to laya's):

- matmul weights stored in bf16 once at load, instead of autocast converting them on every pass (~400 kernel
  launches per pass, and 2.4 GB of VRAM across four checkpoints)
- question encodings cached per checkpoint, with each state tokenised once (720 → 57 µs per request)
- pass tensors built with numpy (2.4 → 0.17 ms per 60-row pass)
- responses serialised with orjson (94 → 1 µs)

**CUDA graphs for small passes.** An eager forward pass costs ~20 ms of Python kernel dispatch (~950 launches)
whatever its size, while a lone request's GPU work takes a few ms. So small passes replay pre-recorded CUDA
graphs: the launch sequence is recorded once per size bucket (rows × tokens) at startup and replayed with one
call. Each checkpoint measures its own eager cost and graph timings at startup, and routes a pass to a graph only
when that is faster; big, GPU-bound passes stay eager. A lone request went from 22 to 6.7 ms. Costs: ~40 s more
startup and ~1.1 GB of GPU memory. Results differ from `system_one` only by the same bf16 batch-shape noise as
batching (a solo request stays within 0.041, deterministically). `CUDA_GRAPHS=0` turns graphs off.

**One request, many states:** `/v1/decide/batch` puts all its states into the same queue, grouped by
checkpoint. Measured against one `/v1/decide` call per state from a single client:

| input size | N = 10 | N = 100 | N = 400+ | per state, batched |
|---|---|---|---|---|
| short messages (~110 tokens / state) | 5.9× faster | 8.1× | 8.7× | ~2.4 ms (~425 states/s) |
| typed-decisions states (~500 tokens / state) | 1.2× | 1.6× | ~1.75× | ~16.4 ms (~61 states/s) |

A single client that loops over items gets no coalescing, so bulk work should use the batch endpoint.

**Correctness.** `make test-coalescing` runs two layers of tests:

- **Scheduler unit tests** with a fake model that returns each row's own id. They cover isolation across 4,000
  tickets from 200 threads, per-checkpoint grouping, failure bisection, OOM splitting, the fatal path,
  cancellation, fairness both ways and backpressure.
- **A live leak test.** 1,000 concurrent requests (decide, batch, compare and invalid, across all checkpoints, each
  with unique question ids and labels) are checked against the SDK's own `system_one`. Keys, routing and token
  counts must match exactly. Within a batch, each state must match its own reference better than any sibling's
  (60,236 comparisons). Requests re-sent one at a time must match exactly (0.00000).

Values under concurrency differ slightly (median 0.0006, max 0.047 in probability) because bf16 kernels round
differently in differently padded batches. The SDK does the same on its own: one input, alone vs in a padded
batch, moved up to 0.13 for inputs whose probability is spread across neighbouring options.
`make test-batch` checks batch results against per-state `/v1/decide`.

### Failures

A failure affects only the request (or the state) that caused it:

- **Bad input** is rejected before queueing: 422 / 400.
- **A failing forward pass** (including CUDA out-of-memory) is split in half repeatedly until the failing rows are
  isolated. Other requests that shared the pass still get their results. A single `/v1/decide` whose row fails
  returns 500. In `/v1/decide/batch`, only the failing states get `{"error": ...}` in their slot (counted in
  `batching.errors`). In `/v1/compare`, a failing checkpoint gets an error entry.
- **A fatal CUDA error** (the context is unusable afterwards) fails everything queued with 503, and the process
  exits so Docker's `restart: unless-stopped` brings the API back with a fresh GPU context.
- **Overload:** a full queue returns 503 immediately, and a request that waits past `REQUEST_TIMEOUT_S` returns 504.
  Its remaining rows are skipped and late results are discarded.

## Question types

Every question has a `type`, `instructions` and (usually) `criteria`. Laya answers all of a request's questions in
one batched forward pass. It never generates text, so an answer is always one of the options you defined, with
probabilities. Pick the type by the *shape* of the answer you need:

| type | answer shape | you get | typical use |
|---|---|---|---|
| `noul` | yes / no | `noul` = P(true) | flags, guardrails, gates |
| `choice` | exactly one of N labels | `choice`, `probabilities`, `confidence` | routing, categorisation, next action |
| `score` | a level on an ordered scale | `score` (expected level), `probabilities` | severity, urgency, rubric marking |

Measured accuracy on the typed-decisions test set (400 cases; [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)):

| type | always-most-common-label baseline | English base, zero-shot | fine-tuned (`td-full`) |
|---|---|---|---|
| `noul` | 0.642 | 0.488 | **0.822** |
| `choice` | 0.418 | 0.285 | **0.718** |
| `score` (level exactly right) | 0.335 | 0.321 | **0.676** (mean error 0.32 levels) |

Zero-shot, the base checkpoint does worse than the baseline on every type. Fine-tuning is what makes these
reliable for a specific schema (see [Fine-tuning](#fine-tuning)).

**Ideas for what to build with each type**, how to combine them, ready-to-adapt question sets, hybrid patterns
with LLMs, and anti-patterns: [`docs/USE_CASES.md`](docs/USE_CASES.md).

### `noul`: yes / no

```json
"refund": {"type": "noul", "instructions": "Does the customer ask for money back?",
           "criteria": {"true": "explicitly asks for a refund or chargeback", "false": "no request for money back"}}
```
→ `{"type": "noul", "noul": 0.90, "confidence": 0.90}`

- **Output:** `noul` is P(true). `confidence` is `max(p, 1 − p)`, i.e. how far from a coin flip.
- **`criteria` is optional.** Without it, the options are "yes, the statement holds" / "no, the statement does not
  hold". Spelling out what counts as true and false helps with borderline cases.
- **Good for:** flags and gates (`is_phishing`, `needs_human`, `contains_pii`), guardrails (`jailbreak`,
  `prompt_injection`), and anything you will threshold. It is the most accurate type after fine-tuning (0.82).
- **Thresholds:** after hard-label calibration (the default for fine-tunes), P(true) = 0.9 means right about 90%
  of the time, so you can pick a cut-off from the precision you need. The shipped checkpoints are
  over-confident, so don't threshold them without calibrating first.
- **Limits:** one statement per question. "Is it urgent *and* billing-related?" is two questions. Zero-shot on
  an unfamiliar schema it was worse than always answering the majority (0.49 vs 0.64).

### `choice`: one of N labels

```json
"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
         "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors",
                      "sales": "pricing, quotes", "other": "none of the above"}}
```
→ `{"type": "choice", "choice": "billing", "probabilities": {"billing": 0.94, "technical": 0.03, ...}, "confidence": 0.78}`

- **Output:** `choice` is the most likely label, and `probabilities` sums to 1 over your labels. `confidence` is
  1 − normalised entropy (Jev-style), **not** the probability of the top label. To threshold "how sure is it
  about this answer", use `probabilities[choice]`.
- **`criteria`:** `{label: description}` or a plain list of labels. The descriptions are part of the input, and
  good ones matter more than label names.
- **Good for:** routing, categorisation, picking a next action or disposition, intent.
- **Single-label only.** The probabilities compete, so "which of these apply?" should be one `noul` per label.
- **Always offer an escape option** (`other`, `none of the above`). Otherwise every input is forced into one of
  your labels, confidently.
- **Option count:** best under ~20. All options share a 256-token budget (192 on the English base), each one cut at
  48 tokens, so beyond ~20 the label texts get squeezed. The model card measures 0.425 on 77-option Banking77.
  For large label sets, raise `head_max_len`, split the question into coarse and fine steps, or pre-filter with
  the SDK's `laya.shortlist_choice` / `predict_shortlist`. The shipped temperature for 11+ options is clamped as
  invalid, so treat confidence there as uncalibrated.
- **Measured:** 0.72 accuracy fine-tuned. The 8-option `disposition` was the weakest choice question (0.66).

### `score`: a level on an ordered scale (rubrics)

```json
"quality": {"type": "score", "instructions": "Mark the agent's reply against the rubric.",
            "criteria": ["1 - wrong or unhelpful; ignores the question",
                         "2 - partly answers; missing key steps",
                         "3 - correct and complete, but unclear or impersonal",
                         "4 - correct, complete, clear, and addresses the customer's tone"]}
```
→ `{"type": "score", "score": 2.4, "probabilities": {"0": 0.02, "1": 0.10, "2": 0.35, "3": 0.53}, "legend": {...}}`

- **`criteria` is the rubric:** a list of level descriptions, lowest first. The model reads them when marking.
- **Output:** `score` is the *expected* level (0-based), a decimal that can fall between levels. Add 1 for a scale
  starting at 1 (2.4 → 3.4 above). `probabilities` shows the spread; the most likely level (here "4") can differ
  from the rounded expectation. Use `score` for averages, rankings and dashboards, and the top level when you
  need one discrete mark.
- **Good for:** severity, urgency, risk, sentiment intensity, and rubric-style marking of answers, replies or
  documents.
- **Multi-criterion rubrics:** one `score` question per criterion (accuracy, clarity, tone…), combined with your own
  weights. All criteria are marked in the same pass.
- **Limits:**
  - It is the weakest type. The model card rates ordinal `score` the weakest primitive (SST-5 0.372), and
    fine-tuned here it had the lowest exact-level accuracy (0.68), with `urgency` at 0.61. But it is usually
    close: mean error 0.32 levels.
  - Keep descriptors short: levels share the same 256-token option budget, each cut at 48 tokens. 3–5 levels
    work best, and 10+ levels with long descriptors get truncated (`make validate` reports it).
  - Descriptors can be structured (`{"desc": ..., "example": ...}`, rendered as compact JSON), but that uses the
    budget faster than plain text.
- **If exact marks matter, fine-tune on marked examples** and keep the rubric wording fixed afterwards.

### Behaviour shared by all types

- **Input budget:** each question is encoded as `[type + instructions + options] + [state]`. The English base
  allows 512 tokens in total with 192 for the question, which leaves roughly 320–500 for the state depending on
  option length. Fine-tunes here use 1024 / 256. Longer states are cut from the end, so put what matters first or
  pre-clean it (the SDK's `laya.email_state` strips quoted threads and signatures).
- **Cost:** each question is its own sequence in the batch. Measured through the API on the 3090 (one
  typed-decisions state): 1 question 6 ms, 3 questions 15 ms, 5 questions 24 ms, 10 questions 36 ms. Small
  passes replay CUDA graphs; bigger ones are GPU-bound, at roughly 3–4 ms per extra question on a long state. For many states, use
  [`/v1/decide/batch`](#batching-and-coalescing); concurrent calls are coalesced automatically.
- **Wording is part of the model's input.** After fine-tuning, ask with the exact wording and options you trained
  on. The UI's `<model>: <workflow>` presets load them.
- **`action.act_probability`** (on every answer) comes from a separate "act vs escalate" head. This repo's
  fine-tuning does not train that head, so on fine-tuned models treat it as unreliable, and on the shipped
  checkpoints treat it as uncalibrated. For escalation decisions, prefer an explicit `noul` such as
  `needs_human`.
- **Nothing to parse and nothing to hallucinate,** but also no explanation. When you need a reason, ask an LLM
  afterwards, with Laya's decision as input.

## Fine-tuning

The base checkpoints are a starting point: on a real multi-workflow benchmark the English base scores *below* a
majority-label guess until fine-tuned. The `train` service does the whole loop in Docker on the same GPU:

```bash
make train-build                          # once
make validate DATA=/data/mine             # checks format, label balance, token budget
make train    DATA=/data/mine RUN=mine-v1 # ~1.3 min per 100 cases on an RTX 3090, ~9 GB VRAM
make evaluate RUN=mine-v1                 # held-out metrics vs the base and a majority baseline
make publish  RUN=mine-v1                 # served by the API/UI as model "mine-v1"
```

Data is JSONL: `{"state": ..., "gold": {"question": answer}}` with a shared `questions.json`. See
[`data/sample/`](data/sample) and the full guide, [`docs/FINETUNING.md`](docs/FINETUNING.md): what to collect,
format, how much, hardware, timings, what comes out, and how to serve it. To build that JSONL from helpdesk
exports, annotator votes or an LLM teacher, see [`docs/DATA_PREP.md`](docs/DATA_PREP.md).

**Measured here** ([`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)), on LocalLLaMA/typed-decisions (4 workflows,
400-case test set):

| training cases | 0 | 30 | 150 | 300 | 600 | 1,075 |
|---|---|---|---|---|---|---|
| accuracy | 0.360 | 0.514 | 0.591 | 0.657 | **0.735** | 0.733 |
| train time (RTX 3090) | – | 0.4 min | 2.0 min | 3.9 min | 8.2 min | 14.0 min |

Gains flatten at ~150 cases per question set. Hard-label calibration then brings ECE to 0.034.

**How it compares** ([`docs/BAKEOFF.md`](docs/BAKEOFF.md)): with 150 labelled cases Laya beats the other approaches
tested. With ~1,000, a plain fine-tune of its own encoder (ModernBERT-large + per-question heads) wins (0.781 vs
0.733), and it's faster to train and serve. Zero-shot, the free DeBERTa-v3-large NLI classifier beats it (0.539 vs
0.360). `make experiments`
reproduces every number (the dataset is fetched at a pinned revision, not stored in the repo). The published
example model is `td-full`; check it with `python3 scripts/smoke_test.py --stack stack.finetuned.json`.

### Auditing model-made labels

Labels a local model wrote (the Ollama teacher in `data/savic-review/teacher.jsonl`, the KV4 screens in
`data/kv4-screens/labels.jsonl`) are trained on only after Claude has checked a sample of them.
`scripts/label_audit.py` ties each audit to the label file's sha256. Any change to the file (a relabel, new
prospects) therefore needs a new audit before those labels are used again.

```bash
# 1. Draw a random sample, stratified by question, label value and agreement with the recorded decision
python -I scripts/label_audit.py sample --labels data/savic-review/teacher.jsonl --source teacher --n 30 --seed 7
# 2. The judge reads data/audits/<source>-<sha8>-blind.jsonl (no model labels in it) and writes
#    data/audits/<source>-<sha8>-answers.jsonl: {"key": ..., "label": true|false, "reason": "one line"}
# 3. Score it: agreement and Wilson 95% lower bound per question, appended to data/audits/registry.json
python -I scripts/label_audit.py score --source teacher --show-errors   # --min-agreement 0.85 --min-wilson-lb 0.70
```

A question passes when agreement is at least 0.85 and the Wilson lower bound is at least 0.70. With 30 rows,
that takes 26 agreements (0.867, lower bound 0.703). The sample is balanced across strata rather than proportional, so it gives
rare labels more weight than the full file does. The build scripts (`build_savic_review.py --split`,
`build_kv4_screens.py`) call `require_audit(source, path, question)`. They leave out any model-made labels with
no passing audit for the file as it is now, print a warning on stderr, and accept `--allow-unaudited` to
override. Human decisions and rationale-derived labels are not model-made and are not gated.

## Weights

- `./models/convaiinnovations__laya/` holds 2.3 GB for all three checkpoints plus the model card and assets. It is mounted **read-only** into the API container, and nothing is downloaded at runtime (`HF_HUB_OFFLINE=1`).
- The pinned revision is in `models/download.lock.json` (currently `1c5edc17a7ac`).

```bash
python3 scripts/download.py --check     # has upstream changed?
python3 scripts/download.py --update convaiinnovations/laya && docker compose restart api
python3 scripts/download.py --verify    # do local files match the lock?
```

## Measured on this host (RTX 3090 24 GB, driver 580, CUDA 12.6 image)

- **VRAM:** ~7.0 GB of device memory in use with the three shipped checkpoints and one published fine-tune:
  4.2 GB of weights in bf16, ~1.1 GB of CUDA graphs, and the rest allocator cache. Batching at the default
  8k-token budget adds under 1 GB of working memory.
- **Startup:** 53 s total (about 16–21 s per checkpoint, mostly encoder construction).
- **Latency:** ~7 ms for a short request with 3 questions on `english`, ~4.5 ms on `multilingual` (CUDA graphs),
  15 ms for 3 questions on a ~500-token state, and 36 ms with 10 (see [docs/PERFORMANCE.md](docs/PERFORMANCE.md)).
- **Startup:** ~111 s (loading, warm-up, and capturing ~200 CUDA graphs).
- **Throughput:** concurrent `/v1/decide` calls reach ~314 req/s on short messages and ~53 req/s on ~500-token
  states (39 and 35 without coalescing). `/v1/decide/batch` reaches ~425 states/s on short messages and ~61
  states/s on long ones. Single-request latency is ~7 ms (short input). See [Batching and coalescing](#batching-and-coalescing).

## Things worth knowing

- **Short Latin-script text defaults to English.** The SDK needs at least two function words, or diacritics,
  to call Latin-script text non-English. "Mein Konto wurde zweimal belastet" (4 words, 1 hit, no umlauts) routes to
  `english`, while the full German sentence routes to `multilingual`. Non-Latin scripts are always detected. If you
  know the language, pass `"lang": "de"` (or `"model": "multilingual"`).
- **Probabilities ship over-confident.** Per the model card, fit a temperature on your own data before trusting
  them (`make calibrate` does this for fine-tunes; the shipped checkpoints keep their own temperatures). The `choice:11+` temperature shipped with `english`/`typed-decisions` (0.10) is clamped to 0.5 by the SDK,
  so treat confidence on 11+-option questions as uncalibrated. `/info` reports this under `warnings`.
- **The base checkpoints are a base to fine-tune, not a zero-shot oracle** (0.36 on typed-decisions against a 0.45
  majority baseline). `typed-decisions` scores 0.77 only on the workflows it was trained on. See [Fine-tuning](#fine-tuning).
- **Many options share a fixed token budget** (192/256 tokens). Beyond about 20 options, labels get truncated and
  accuracy drops. Split the options hierarchically.
- **CPU fallback is surfaced.** The SDK silently falls back to CPU on OOM or an unsupported GPU. `/info` reports
  `all_on_gpu`, and the smoke test fails if it is false.
- **Other hardware:** pre-Ampere GPUs automatically use fp16. On CPU, set `DEVICE=cpu` and drop the
  `deploy:` GPU block (the model card quotes ~200–460 ms per call). To save memory, trim `PRELOAD`.
- The image pins `transformers==5.17.0` because the shipped encoder configs use the v5 format.
