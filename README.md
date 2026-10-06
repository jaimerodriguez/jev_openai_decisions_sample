# Pharmacy call-routing evaluation

## Introduction and use case

System One models are designed for fast, focused decisions, making them a useful way to route incoming prompts to the right agent or workflow. TypeSafe AI introduced Jev on September 15, 2026; two weeks later, OpenAI announced its Decisions API at DevDay. Both offer a way to turn unstructured input into answers that application code can act on directly. [Jev launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [OpenAI announcement](https://openai.com/index/devday-2026-recap/)

A pharmacy call is a practical example: is the caller checking a prescription's status, requesting a refill, seeking a pharmacist's advice, or submitting a new prescription as a medical provider? The selected route tells the application which workflow or human team should handle the call.

### How System One models work

You provide the context, a question, and a defined set of possible answers. For a **choice** question, those answers might be `rx_status`, `refill_request`, or `representative`, each with a description of when it applies. The API returns a selected value and probabilities over the available choices. Other question types support scores and yes/no judgments. [TypeSafe concepts](https://docs.typesafe.ai/introduction), [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)

The result is a typed decision, with no free-form response or written explanation to interpret. Your code still decodes and validates the JSON, then dispatches the selected route. Restricting the answers makes integration simpler; it does not guarantee that the chosen answer is correct. A well-defined routing policy and evaluation still matter.

## About this sample

This project is a quick Python example for calling Jev and OpenAI Decisions and comparing their **routing accuracy and client-observed API latency** on the same pharmacy-call transcripts. It supports Jev directly, through OpenRouter, and through the [reference article's](https://huggingface.co/blog/a2aprotocol/openai-decisions-api-vs-jev-a-practical-guide) gateway, plus an optional OpenAI Responses baseline using Structured Outputs.

The JSONL dataset can grow or change freely. Results include per-call timings, paired latency comparisons, classification metrics, and API failures. The sample evaluates transcript routing only; it does not handle live phone calls, access patient records, or submit prescriptions. Start with the commands below, then use the latency notes and abridged assessment to interpret the results.

## Sample

**Requirements:** Python 3.10+ and HTTPX for pooled HTTP connections.

### Setup and run

From this directory, create and activate a virtual environment, then install the pinned dependency:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the evaluator:

```sh
# Offline: inspect the selected dataset, with no model calls or invented scores.
python3 eval.py --dry-run

# Offline: contract, failure-path, scoring and end-to-end fixture tests.
python3 -m unittest -v

# Configure these in your shell or secret manager; never commit real keys.
export TYPESAFE_API_KEY='YOUR_TYPESAFE_KEY'
export OPENAI_API_KEY='YOUR_OPENAI_KEY'

# Live: one call per case to each provider, potentially billable.
python3 eval.py --providers jev openai-decisions

# Optional: reproduce the article's gateway path instead of direct TypeSafe.
export DECISIONS_API_KEY='YOUR_GATEWAY_KEY'
python3 eval.py --providers jev-gateway openai-decisions

# Extra endpoint: Jev via OpenRouter (uses a separate OpenRouter key).
# If OPENROUTER_API_KEY is already exported, no further setup is needed.
export OPENROUTER_API_KEY='YOUR_OPENROUTER_KEY'
python3 eval.py --providers jev-openrouter

# Compare OpenRouter Jev with the actual OpenAI Decisions API.
python3 eval.py --providers jev-openrouter openai-decisions

# Optional three-way comparison including the Responses baseline.
python3 eval.py --providers jev-openrouter openai-decisions openai-responses

# Offline: inspect all five integrations (five requests per case).
python3 eval.py --dry-run --providers jev jev-gateway jev-openrouter openai-decisions openai-responses

# Repeat to probe stability; repeats do not add distinct examples.
python3 eval.py --providers jev openai-decisions --repeat 3
```

**API contract checked October 6, 2026:** `openai-decisions` calls the public-beta `POST https://api.openai.com/v1/decisions` endpoint with `OPENAI_API_KEY`. The default comparison is `jev` versus `openai-decisions`. The Decisions model defaults to `gpt-6-luna`, currently the only model supported by that endpoint. `--decisions-model` controls it independently of the Responses baseline. Do not pass a Responses model to the Decisions endpoint. [Official Decisions guide](https://developers.openai.com/api/docs/guides/decisions)

The Decisions adapter sends `input`, a named `questions` array, and `choices` entries with `value`/`description`. Named answers and per-choice probability arrays are validated and normalized into the same scoring representation as Jev. Refusals, malformed answers, and incomplete answers count as failures with a human fallback disposition.

The optional OpenAI Responses baseline is the documented `gpt-4.1-mini-2025-04-14` snapshot, chosen as a reproducible baseline, not a claim about the best current model. Override with `--openai-model`. Direct Jev defaults to the documented `jev-latest` alias; use `--jev-model` with an account-supported fixed version for a stronger reproducibility guarantee. The gateway uses its documented `typesafe/jev-1.13` identifier.

The extra `jev-openrouter` provider uses `https://openrouter.ai/api/alpha/decisions`, `OPENROUTER_API_KEY`, and `typesafe/jev-1.13`; override its model independently with `--openrouter-model`. Missing or empty keys fail before network calls. It returns top-level `answers`, like direct TypeSafe, rather than the gateway envelope. Its dated returned model, upstream `provider`, request ID, and `usage.cost` (USD when reported) are preserved in `results.jsonl`. Cost is not aggregated into the summary or converted to gateway credits. All Jev integrations remain available. See the [OpenRouter Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial) for the alpha API contract.

Each run creates a new directory under `runs/`. `manifest.json` records endpoints, requested models, dataset case count, dataset/policy/code hashes, seed, repetitions, and timeout. Live `results.jsonl` records normalized predictions, returned model, latency, usage, and errors. `summary.json` reports accuracy including failures, valid-response accuracy, macro-F1, per-class precision/recall, confusion matrices, latency distributions, and paired disagreements. `latency.md` provides a comparison table; `latency.csv` provides per-call timings for further analysis. Usage sums include only valid responses; they are **not a bill**, since failed or uncertain calls can still incur charges. Gateway credits are not comparable to OpenAI tokens or dollars.

Requests are sequential and providers are interleaved in seeded random order. There are no automatic retries: this avoids quietly changing latency and charging for an uncertain gateway outcome. A 429 or timeout is recorded as an error. A gateway idempotency key is sent, but duplicate-key behavior is rejection, not cached response replay. Check provider history before rerunning. Missing keys fail before any calls. Errors return a nonzero exit status; classification mistakes remain measurements rather than process errors.

Dry runs contain no inference results. Tests use explicitly synthetic response fixtures; their scores are not model performance. Offline verification does not establish live provider performance; run with your own API keys to collect a benchmark.

### API latency measurement and comparison

```sh
# Compare Jev/OpenRouter and OpenAI Decisions with two warm-ups per provider.
# Warm-ups are extra billable calls; the default is zero.
python3 eval.py --providers jev-openrouter openai-decisions --repeat 3 --warmup 2

# Separate run to measure fresh connections instead of the default reuse policy.
python3 eval.py --providers jev-openrouter openai-decisions --repeat 3 --connection-mode fresh
```

Read `latency.md` for a quick comparison and `summary.json` for the detailed distributions. The terminal prints the same table. All timings are **client-observed**, including network, any gateway, service queueing and inference; they cannot isolate server inference time.

| Recorded field | Timing boundary |
|---|---|
| `api_latency_ns` | Immediately before HTTPX `send()` through complete response-body read; excludes request construction, JSON decoding, validation and file writes |
| `time_to_headers_ns` | Send start through receipt of parsed response headers; this is not exact time-to-first-byte or first token |
| `http_elapsed_ns` | Complete HTTP exchange, or elapsed time until a transport failure/timeout |
| `request_prepare_ns`, `json_decode_ns`, `validation_ns` | Separate local overhead measurements |
| `decision_latency_ns` | Payload construction through the validated answer or caught error; excludes client initialization and file writes |
| `latency_ms` | Legacy alias for total decision time in milliseconds, **not** the API-only metric |

The clock is `time.perf_counter_ns()`: elapsed wall-clock time with integer differences and no per-call rounding. Nanosecond units do not imply nanosecond physical accuracy. Python documents this as its high-resolution elapsed-time counter. [Python timing documentation](https://docs.python.org/3/library/time.html#time.perf_counter_ns)

Each provider has its own HTTPX client, HTTP/1.1, one connection at a time, and no automatic retries or redirects. Default `reuse` permits connection reuse; `fresh` disables idle connection retention. Server closure or 60-second idle expiry can still cause reconnects. Fresh calls include connection setup; DNS and TLS state may remain cached by the OS/runtime, so this is not a fully cold machine. Client/TLS-context initialization happens outside timed requests. Standard proxy/TLS environment settings are honored; the manifest records only whether proxy variables are configured, not their values. [HTTPX clients](https://www.python-httpx.org/advanced/clients/), [connection limits](https://www.python-httpx.org/advanced/resource-limits/)

Requests remain sequential with seeded randomized case/provider order and adjacent provider calls for each case. `--warmup N` sends N extra calls per provider using cycling dataset cases. Their `phase` is `warmup`; they remain in JSONL/CSV with usage/errors but are excluded from measured accuracy, latency distributions and paired comparisons. Warm-ups can also warm provider caches, not just connections. With zero warm-ups, the first connection setup is included. Warm-up errors are reported separately and cause a nonzero exit. The measured schedule does not change when the warm-up count changes.

`timing.api_ms_valid` is the primary distribution: completed **valid-answer** calls, whether correctly classified or not. It includes sample count, mean, sample standard deviation, min/max, and nearest-rank p50/p90/p95/p99. HTTP-2xx timings and failed-attempt elapsed times are separate. A timeout has no completed API latency and is never substituted with the timeout setting; its observed elapsed time is recorded. `--timeout` limits individual HTTP operations, not total call wall time. [HTTPX timeouts](https://www.python-httpx.org/advanced/timeouts/)

Paired comparisons match the same case ID and repetition, requiring valid timed answers on both sides. A positive **A−B** delta means B is faster; the A/B ratio exceeds 1 when B is faster. The mean case delta gives each distinct case equal weight after averaging its matched repetitions. Its exploratory 95% interval resamples cases with replacement (2,000 deterministic bootstrap draws); one distinct case produces no interval. Repeats are not independent new cases. These intervals describe the matched workload, not a production guarantee, and exclude failed pairs: assess accuracy, errors, timeout count, and the number of matched cases alongside speed.

Keep host, region, connection mode, warm-up count, dataset and model versions comparable. The manifest records connection mode, warm-up count, dataset and requested models, plus Python/HTTPX versions, clock metadata and timing schema version 2. Returned model IDs and token usage remain in each result. Inspect cache usage when comparing repeated prompts. Tail percentiles with small samples are unstable; the report flags fewer than 100 valid timed calls. Repeat separate runs at different times to assess service/network variability. No provider performance is inferred from offline fixtures.

### Single-call curl sample

With `OPENAI_API_KEY` already exported, run `bash openai_endpoint.sh` to classify the first case, or supply a transcript:

```sh
bash openai_endpoint.sh 'I requested my refill yesterday. Is it ready?'
```

The script rejects a missing or empty key, builds a `gpt-6-luna` request using the same Python routing policy, and sends it with curl. Use the same activated Python environment as the evaluator. It makes one live, potentially billable call and returns the API body; it does not perform a phone transfer.

### Routing contract and dataset

The four requested routes remain distinct: `rx_status`, `representative`, `refill_request`, `provider_new_rx`. Added routes are `rx_transfer`, `insurance_billing`, `general_info`, `clarify`, and `emergency`. These are proposed operational choices; pharmacy staff should approve the rubric and labels.

Emergency signals take priority over routine requests. Clinical questions and explicit human requests come next. A provider's role never substitutes for the actual intent. A refill request with no remaining refills enters a renewal-intake workflow; it does not authorize dispensing. New prescriptions and transfers are handed to authorized staff. Missing intent triggers clarification; missing patient identifiers alone do not obscure an otherwise clear intent.

Caller utterances and expected labels are in [cases.jsonl](cases.jsonl). Add or remove cases freely, or select another JSONL file with `--cases /path/to/cases.jsonl`. There is no fixed minimum or maximum size; a dataset must be nonempty. Each case requires a unique `id`, a nonempty `prompt`, and an `expected` label from the routing contract. `why` is optional documentation. A dataset may cover only a subset of routes. Ground-truth labels and explanations are never sent to any provider.

The total number of requests is **(case count × repetitions + warm-up calls per provider) × selected providers**. Counts and scores are calculated from the selected dataset; no test requires a fixed number of cases or coverage of every route. Provider quotas and available memory still apply.

`handoff()` returns an illustrative disposition only. Identity, authorization, pharmacy-system eligibility, and dispensing checks belong downstream. An API error falls back to a human disposition but still counts as a failure—and as a missed emergency when applicable. No confidence threshold silently changes predictions. Jev and OpenAI Decisions probabilities and confidence are retained; the Responses baseline does not invent equivalent probabilities. Brier scoring uses the returned distributions, not the confidence field. Do not assume confidence thresholds transfer between providers.

### Abridged assessment

**The routing policy is the central design decision.** A flawless API integration still misroutes calls if “refill” is treated as a keyword or “provider” as an intent. Keep clinical escalation separate from routine fulfillment and require explicit uncertainty handling. The emergency example reflects recognized warning signs, but this single case is not a validated triage system. [MedlinePlus](https://medlineplus.gov/ency/article/000844.htm)

| Option | Practical strengths | Limits / tradeoffs |
|---|---|---|
| Jev, direct TypeSafe | Documented Choice interface; full option distribution; compact integration | Confidence needs local validation; live alias can change; no pharmacy accuracy evidence here |
| Jev, article's gateway | Convenient shared interface and credits across models | Additional service dependency and data path; gateway latency/billing cannot be attributed solely to Jev |
| Jev, OpenRouter | Extra Jev endpoint; typed answers, returned snapshot and per-call USD cost | Alpha API; separate account/key and another service hop; verify performance locally |
| OpenAI Decisions API | Documented choice distributions; `gpt-6-luna`; base input price $0.10/M tokens, no output-token charge | Public beta; different wire schema from Jev; account access and local performance still need live verification |
| OpenAI Responses + Structured Outputs | Documented strict enum output; usable comparison baseline | Generates structured text; no equivalent Jev distribution in this sample; not evidence about Decisions API |

Sources: [TypeSafe Choice](https://docs.typesafe.ai/primitives/choice), [gateway contract](https://decisions-api.dev/docs), [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

Three points sharpen the [reference article](https://huggingface.co/blog/a2aprotocol/openai-decisions-api-vs-jev-a-practical-guide):

1. **“Decision API versus prose” is an incomplete comparison.** Strict Structured Outputs already constrains a generative model to an enum. A specialized interface must earn its place through measured routing quality, latency, uncertainty quality, and cost; its name does not establish superiority. Schema correctness also does not establish semantic correctness. [OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs)
2. **Confidence is not a second independent safety signal.** TypeSafe documents it as a statistic derived from the same probability distribution. A peaked distribution can still select the wrong route. This runner records multiclass Brier score where distributions exist, but this small dataset cannot establish calibration or justify a universal automation threshold. [TypeSafe confidence](https://docs.typesafe.ai/confidence)
3. **Measure the whole service and the cost of errors.** Gateway measurements include another service hop. An emergency miss and a general-information error should not be treated as equally acceptable because overall accuracy is high. This runner exposes protected-call automation errors and emergency misses separately; a production decision also needs staffing cost, clarification burden, and failed-call cost. The article's recommendation to measure locally is sound; it does not supply a head-to-head result. [Gateway measurement and billing definitions](https://decisions-api.dev/docs)

Use a small dataset as a smoke test and rubric review. For a single pass over N cases, one mistake changes overall accuracy by 100/N percentage points. Before selection, use a separate, larger pharmacist-labeled holdout with realistic traffic weights, noisy transcripts, multilingual speech, and rare clinical escalations. Measure transcription plus routing plus handoff latency separately from this classifier-only timer. Repeats assess variability; they do not enlarge the independent dataset. Compare safe automation coverage at an agreed error ceiling, then cost per correctly handled call. No winner is claimed from unrun code. OpenAI advertises roughly 10x faster answers than Responses; treat that as a vendor claim until measured here, including latency, routing quality, and errors. Regional and long-context pricing adjustments may apply to the base Decisions rate. [Decisions guide](https://developers.openai.com/api/docs/guides/decisions)

Additional implementation references: [TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart), [TypeSafe OpenAPI](https://api.typesafe.ai/openapi.json), [OpenAI changelog](https://developers.openai.com/api/docs/changelog), [baseline model](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
