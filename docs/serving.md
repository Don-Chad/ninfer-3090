# HTTP serving

`build/apps/ninfer-serve` loads one v3 `.ninfer` artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype int8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
when running alone, and two requests run concurrently when their complete reservations fit.

With `C=2` and two extra Device checkpoint slots, the process owns two active StateImage guarantees
plus a global pool of two Device-resident checkpoints. Eight pinned Host State slots and 8 GiB of
pinned Host KV retain inactive continuations under Device pressure. Active request capacity is two.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`, with
`--lm-head-draft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Add
`--vision` when the server must accept image or video input. Speculative residency is likewise
frozen by `--spec mtp|dflash|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
with `--vision`; each accelerates generated-text decode after multimodal prefill, while Vision encode
and prefill remain outside speculative acceleration. A later request cannot enable a capability
omitted at startup. The artifact need only contain the Text backbone and the optional components
selected for this process.

### Vision residency

`--vision` keeps the Vision tower, its encode workspace and the item handoff resident for the
process lifetime. `--vision-residency overlay` removes that cost on memory-tight cards: the tower
lives in pinned host memory, the sequence plan reserves nothing for Vision, and each image is
encoded inside a bounded window whose device memory is borrowed for the duration of the encode.

A window is funded from free KV pages when they cover it. Those pages hold nothing, so nothing is
copied, the text weights stay mapped, and the encode runs on its own stream while other lanes keep
decoding: the lane that owns the image simply yields its prefill units until the encode completes.
The pages are out of circulation while the loan is open, so the admission capacity shrinks with it
and no request is ever admitted into memory that has been lent away.

When free pages cannot cover the window, it falls back to the evict-ranked tail of the text weights
(lm_head, token embedding, draft head, MTP head), restored from a pinned mirror before the prefill
unit ends. That window is exclusive: the borrowed weights are unmapped, so nothing else runs until
it closes. The fallback is always available, so a vision request never fails for lack of memory,
and a KV cache smaller than one window simply uses it for every image.

Either way the window streams the tower through two layer slots, lands the embeddings in pinned
memory, and the prefill then uploads only each chunk's embedding columns. A follow-up turn over the
same image reuses the prefix and opens no window. Embeddings are produced by the same kernels on the
same bytes, so completions are identical between residencies. `--vision-max-merged N` bounds one
item's merged tokens (media above it is downscaled at preprocessing) and sizes the window. The
request log line reports `overlay=<windows>x<conc|excl|mixed> <ms> (evict <MiB> <ms>, restore <ms>,
staged <MiB>)` and the JSON record carries `vision_overlay`, including `exclusive_windows`.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health and build version (see [Server version](#server-version)) |
| `GET /v1/load` | serving capacity, current load, and monotonic token counters (see [Load](#load)) |
| `GET /slots` | per-slot occupancy of the private context cache (see [Slots](#slots)) |
| `POST /slots/{id}?action=save\|restore\|erase` | save a retained session to a file, restore one, or evict one (see [Slots](#slots)) |
| `GET /metrics` | Prometheus text counters, llama.cpp-compatible names (see [Metrics](#metrics)) |
| `GET /v1/models` | configured OpenAI model alias, effective context limit (`max_model_len`/`context_window`/`context_length`) and input modalities (see [Model discovery](#model-discovery)) |
| `GET /v1/models/{id}` | lookup of the same model object by its alias |
| `GET /props` | read-only llama.cpp-style server properties (see [Model discovery](#model-discovery)) |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. On Linux, accepted
connections also use TCP keepalive and a 15-second `TCP_USER_TIMEOUT`; together with the heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.

The HTTP transport is thread-per-connection, and a worker stays with a connection for its whole
life, including the idle window between keep-alive requests. The pool therefore reserves one base
worker per admissible request (`max_concurrency + max_pending_requests + 1`) and grows on demand up
to 64 further workers for connections that are merely open; the extra workers retire once idle.
Without that headroom a client-side connection pool of otherwise idle sockets occupies every worker
and the server accepts real requests strictly one at a time.

### Model discovery

`/v1/models` lists one model object:

```json
{"id": "qwen3.8-27b", "object": "model", "created": 1790000000, "owned_by": "ninfer",
 "max_model_len": 65536, "context_window": 65536, "context_length": 65536,
 "architecture": {"input_modalities": ["text", "image", "video"], "output_modalities": ["text"]}}
```

The context limit is `--max-context`, mirrored under the vLLM/llama.cpp, Anthropic, and
OpenRouter/Ollama field names. `architecture` uses the OpenRouter shape that llama.cpp's router-mode
`/models` also emits; `input_modalities` lists `image` and `video` only when the server runs with
`--vision`, and is `["text"]` otherwise.

`GET /props` serves clients that discover a llama.cpp server. It is read-only and fills only the
llama.cpp fields NInfer can state truthfully:

```json
{"default_generation_settings": {"n_ctx": 65536,
   "params": {"n_predict": -1, "max_tokens": -1, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
              "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}},
 "total_slots": 1, "model_alias": "qwen3.8-27b", "model_path": "models/qwen3_8_27b.ninfer",
 "modalities": {"vision": false, "audio": false}, "build_info": "ninfer 0.14.3-rtx3090"}
```

`n_ctx` is `--max-context` and `total_slots` is `--max-concurrency`. `n_predict` and its alias
`max_tokens` are the [default output limit](#default-output-limit): `-1`, llama.cpp's "no fixed
cap", when it is derived per request from the prompt and lane share, or the `--default-max-tokens`
cap. The sampler is the loaded model's preset
for the default thinking mode (thinking unless `--no-thinking`) under the process sampling flags and
`--greedy`; request fields still override it per request. `seed` appears only with `--seed`, since
requests otherwise draw a fresh random seed. `model_alias` is the public model id and `model_path`
the artifact path the server was started with. `build_info` is `ninfer <version>` (see
[Server version](#server-version)). There is no `chat_template` or
writable `POST /props`, and `/slots`, `/metrics`, and llama.cpp's non-`/v1` route aliases are not
served.

### Server version

The running build is reported four ways, all from the same string: `ninfer-serve --version` (and
`ninfer --version`) print it and exit without loading a model; `GET /health` returns it as
`{"status": "ok", "version": "0.14.3-rtx3090"}`; every response, including the `503` during model
load and `401` failures, carries an `X-NInfer-Version` header; and the `server_start` log record
and `/props` `build_info` include it.

The string is the repository's `VERSION` file. A build from any other commit appends
`+<9-char commit>` (`0.14.4-rtx3090+1a2b3c4d5`), and uncommitted changes to tracked files append
`-dirty`, so a development binary is never mistaken for the release it descends from. A build
without git metadata, such as a `git archive` snapshot, reports `VERSION` alone, which is why a
release must be built after `VERSION` is set.

### Startup readiness

The server binds its port and starts accepting connections before the Engine has loaded weights
and finished warmup, so a port clash is reported in milliseconds rather than after loading. Every
route -- including `/health`, `OPTIONS`, and requests that would otherwise be unauthenticated --
answers `503` with `Retry-After: 2` until warmup completes, in the target's own error shape:

```json
{"error":{"message":"The model is still loading. Retry shortly.","type":"service_unavailable","code":"model_loading"}}
```

`POST /v1/messages` and `/v1/messages/count_tokens` receive the Anthropic envelope instead, with a
`request_id` field and a `request-id` header, matching every other error response on those
endpoints:

```json
{"type":"error","error":{"type":"api_error","message":"The model is still loading. Retry shortly."},"request_id":"req_..."}
```

A readiness probe should poll `GET /health` (or any endpoint) and expect `503` until the model is
ready rather than treating an accepted TCP connection as a signal of readiness.

A waiting request that cannot be planned for admission fails alone with HTTP 500: the requests
already running are not touched, the context cache is kept, and the failure is logged as
`engine admission failure ... contained` with the exception text and does not count toward the
latch below.

After startup, any other internal host-side failure fails only the requests that were running or
being admitted, clears the context cache, and keeps serving the queue; each such recovery is counted as
a top-level `engine_recoveries` counter on the request log's `throughput` event. `GET /health` turns
`503` for good only when the engine cannot verify a clean recovery, or after three failures in a row
(each less than 30 seconds after the previous one, with no request completing between them), and
then the server needs a restart. Every such failure is logged at error severity with the exception
text, the execution unit (`boundary`, `admission`, `control`, `prefill` or `decode`), the affected
request ids and lanes, and the streak count; a latch is logged `FATAL`, and `ninfer-serve` exits
with status 3 after a five second grace period (so in-flight error responses flush and `/health`
answers 503 meanwhile) for a supervisor to restart. `--no-exit-on-engine-failure` keeps the
process alive instead. The exception text is operator diagnostics and is never sent to clients; the
request log's `request_error` record already carries it for requests that failed this way.

### Load

`GET /v1/load` is a cheap, pollable snapshot for load balancers and gateways that schedule across
several servers. It requires the API key when one is configured, answers `503 model_loading` until
warmup completes like every other route, and reads only the Engine's already-published runtime
counters and the ingress count, so polling it does not wait on or delay the GPU executor.

```bash
curl http://127.0.0.1:8080/v1/load -H 'Authorization: Bearer local-secret'
```

```json
{
  "object": "ninfer.load",
  "model": "qwen3.8-27b",
  "uptime_seconds": 812.4,
  "capacity": {"max_concurrency": 4, "max_pending_requests": 32, "max_admitted_requests": 36,
               "max_context": 65536, "kv_capacity_tokens": 131072, "kv_capacity_pages": 2048,
               "kv_page_tokens": 64, "device_state_slots": 8, "host_state_slots": 24,
               "host_kv_bytes": 17179869184},
  "requests": {"admitted": 6, "running": 4, "prefilling": 1, "decode_ready": 3, "waiting": 2,
               "materializing": 0},
  "occupancy": {"device_main_kv_pages": 1500, "device_main_kv_tokens": 96000,
                "device_state_slots": 5, "host_state_slots": 7, "host_kv_bytes": 2147483648},
  "counters": {"computed_prefill_tokens": 48200113, "committed_decode_tokens": 6120452,
               "reused_prompt_tokens": 30911840, "decode_rounds": 861307,
               "decode_row_rounds": 2448180}
}
```

- `uptime_seconds` counts from the moment the server became ready, not from process start.
- `capacity` is fixed once the Engine is ready. `max_admitted_requests` is
  `max_concurrency + max_pending_requests`, the ingress bound described in
  [Execution behavior](#execution-behavior); `kv_capacity_tokens` is the resolved page-aligned Main
  KV pool and `kv_page_tokens` its page size.
- `requests.admitted` counts requests holding ingress capacity, from preparation until the response
  is released; a new generation request is rejected with `server_overloaded` (HTTP 429, or 529 on
  Anthropic endpoints) when it would exceed `max_admitted_requests`. `running` counts occupied execution lanes (at most
  `max_concurrency`), of which `prefilling` counts the lanes that own a staged prefill (at most
  `--max-prefill-lanes`) and
  `decode_ready` the lanes in the decode batch. `waiting` counts requests submitted to the Engine
  FIFO that have not been admitted to a lane, including those held back until their KV entitlement
  fits.
- `occupancy` reports current Main KV pages (and tokens), Device and Host StateImage slots, and Host
  KV bytes in use. Main KV occupancy includes retained reusable prefixes, which the resource planner
  may evict under pressure, so a full pool does not by itself mean new requests will wait.
- `counters` are monotonic since startup; derive rates by differencing two polls.
  `computed_prefill_tokens` excludes prefix-reused prompt tokens (reported separately in
  `reused_prompt_tokens`). `committed_decode_tokens` counts tokens committed by decode rounds and
  excludes each request's first token, which prefill emits; with speculative decoding a round commits
  several tokens per row. `decode_row_rounds` is the sum of decode batch sizes over `decode_rounds`.
- Gauges and counters come from the snapshot the Engine publishes at execution boundaries, so they
  can trail the instant of the poll by up to one boundary.

### Slots

A slot is one private context-cache cell: the place a finished conversation is retained so its next
turn reuses the cached prefix. There are `--max-private-continuations` slots. `GET /slots` lists them
like llama.cpp's endpoint and reads only published state, so it never waits on the GPU:

```json
[{"id": 0, "is_processing": false, "retained": true, "session_digest": "8c3f1e0a7b2d4c19",
  "checkpoints": [{"frontier": 1812, "session_digest": "51d0..."},
                  {"frontier": 2409, "session_digest": "e27a90c4d15b3f68"}],
  "n_ctx": 131072, "n_prompt_tokens": 2410, "n_prompt_tokens_cache": 2410, "speculative": true,
  "snapshot_file": "chat-7.snap", "last_used_unix_ms": 1791292927534, "reuse_count": 3,
  "reused_tokens": 7120}]
```

A retained slot reports the session depth as both token counts, its session digest (FNV-1a 64 of
the token ids, 16 hex characters) and the checkpoints a later request can resume from, each with the
digest of its prefix. The endpoint checkpoint sits at the executed frontier, which is usually one
token short of the session: the last sampled token is recorded but not yet in the KV or the
recurrent state, so the next turn executes it first. A slot an
active request will publish into reports `is_processing` with that request's prompt and reused
tokens. Chat Completions responses carry the slot and digest a finished session was retained under
as top-level `id_slot` and `session_digest`, on the aggregate response and on the final streamed
chunk that carries timings.

A retained slot also reports what a client needs to decide which sessions are worth saving.
`snapshot_file` is the name of the slot file the session is bound to (the file a save or restore
last named, which an involuntary eviction writes back to), or `null`. A conversation can move to a
different cell from one turn to the next, and its binding moves with it, so `snapshot_file`, not
`id`, says which file holds a conversation: save a continued conversation under the name it
already carries and each conversation keeps one file. `last_used_unix_ms` is the wall-clock time
the session was last published or restored, `reuse_count` the number of turns that continued it
from a retained copy, and `reused_tokens` the prompt tokens those turns reused. These follow a
conversation from cell to cell, and a restored session starts again from zero. A slot with no
retained session reports `null` for all four.

With `--slot-save-path DIR`, `POST /slots/{id}?action=...` persists sessions across restarts and
evictions. Without it the route answers `501 slot_persistence_disabled`.

| Action | Body | Response |
|---|---|---|
| `save` | `{"filename": NAME, "if_digest": DIGEST?}` | `id_slot`, `filename`, `n_saved` tokens, `n_written` bytes, `session_digest`, `timings.save_ms` |
| `restore` | `{"filename": NAME}` | `id_slot`, `filename`, `n_restored` tokens, `n_read` bytes, `session_digest`, `timings.restore_ms` |
| `erase` | `{"if_digest": DIGEST?}` | `id_slot`, `n_erased` tokens (0 for an empty slot) |

`NAME` is 1-128 characters of `[A-Za-z0-9._-]`, may not start or end with a dot, and may not be a
Windows device name; files live directly in `DIR`. Names are case-insensitive: the server stores and
reports them lowercase, so one file never has two names. `if_digest` makes save or erase conditional on the slot
still holding that session, checked atomically with the operation. Restore replaces whatever the
slot held and makes the restored session an ordinary cache entry that any request with a matching
prefix reuses, including from its checkpoints. A snapshot restores only on a server with the same
model artifact, weight formats, KV dtype, speculative backend, draft tokens and draft head; DFlash
servers do not support persistence. Files are written to a temporary name and renamed, and end
with a checksum that restore verifies before it allocates anything.

Errors: `409 slot_busy` while the slot or any context-cache transaction is in use (retry),
`409 slot_session_mismatch` for a failed `if_digest`, `400 invalid_slot`, `invalid_action`,
`invalid_filename`, and `400 slot_save_failed`/`slot_restore_failed` for a missing, corrupt or
incompatible file or a slot with nothing to save. A failed restore leaves the slot empty.

With `--auto-save-evicted`, a session last saved to or restored from a file is written back to that
file, on a background thread, before an involuntary eviction destroys it. Continuing the
conversation keeps the binding, so the file tracks its newest turn. An explicit erase never writes.
A spill never replaces a file with a shallower copy of the session than the last save or restore
recorded, and at most two spills wait for the writer. An explicit save, restore or erase of a file
supersedes every spill of it still waiting (a restore first writes the waiting spills of the file it
reads, so it reads the newest state). The operational log reports each spill, skip or failure. Snapshots are uncompressed. Besides its KV pages, a session stores one recurrent-state
image per checkpoint it retains (endpoint, rewrite checkpoint, long anchors), about 150 MB each on
the 27B, so even a short session is a few hundred MB: a 39-token Qwen3.8-27B session saved as
295 MiB, with save and restore at about 0.3 s each on an RTX 3090.

### Metrics

`GET /metrics` serves Prometheus text format. Like `/v1/load` it requires the API key when one is
configured and reads only already-published counters. The `llamacpp:` series use llama.cpp's
`--metrics` names and meaning, so dashboards built for llama.cpp work unchanged; they come from the
Engine's per-unit totals and advance during a request rather than at its completion.

| Series | Type | Meaning |
|---|---|---|
| `llamacpp:prompt_tokens_total` | counter | prompt tokens computed by prefill; prefix-cache hits excluded |
| `llamacpp:prompt_seconds_total` | counter | prefill execution time |
| `llamacpp:tokens_predicted_total` | counter | tokens committed by decode rounds |
| `llamacpp:tokens_predicted_seconds_total` | counter | decode execution time |
| `llamacpp:requests_processing` | gauge | admitted requests up to `--max-concurrency` |
| `llamacpp:requests_deferred` | gauge | admitted requests waiting beyond `--max-concurrency` |
| `ninfer:requests_total` | counter | requests completed with an outcome |
| `ninfer:requests_failed_total` | counter | accepted requests that ended in an error |
| `ninfer:requests_rejected_total` | counter | generation requests rejected during preparation, one per `request_rejected` request-log event: overload, invalid or oversized prompt or media. Unparseable and oversized (413) HTTP bodies are not counted; failures after acceptance, including a queue timeout after submission, count in `requests_failed_total` |
| `ninfer:prefix_cache_hit_tokens_total` | counter | prompt tokens served from the context cache |
| `ninfer:draft_tokens_total` | counter | speculative draft tokens proposed |
| `ninfer:draft_accepted_tokens_total` | counter | speculative draft tokens accepted |

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history;
- string content and ordered text/refusal parts; adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn;
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension using HTTP(S) or data URIs; image detail is omitted or `auto`;
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; zero performs prompt
  processing without generation, and omitting both applies the
  [default output limit](#default-output-limit);
- `temperature`, `top_p`, presence/frequency penalties, and signed integer `seed`;
- the compatible `top_k` (`0..20`) and `min_p` (`0..1`) sampler extensions;
- up to four non-empty stop strings, applied to both reasoning and answer output;
- the `ignore_eos` benchmarking extension shared with vLLM, SGLang and llama.cpp: a boolean,
  default `false`, that drops the checkpoint's own stop tokens so generation runs to the output
  budget (`finish_reason:"length"`) unless a caller-supplied stop string or context capacity ends
  it first. Caller stop strings still apply; any other value type is rejected with 400. Output past
  the model's natural end is not meaningful text. `/v1/responses` and `/v1/messages` do not
  honor it;
- `n:1` and text-only `modalities`;
- `response_format` of type `text`, `json_object`, or `json_schema` (`name`, optional
  `description`, `schema` and `strict`); JSON formats are enforced token by token, see
  [Structured output](#structured-output);
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- non-strict function tools with `tool_choice` `auto`, `none`, or `allowed_tools` in `auto` mode,
  parallel calls enabled, assistant tool-call history, tool-result messages, and legacy
  function-call history;
- the top-level `reasoning_effort` field;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- the `graft` extension selecting a [prompt graft](#prompt-grafts);
- the `thinking_budget` extension, a positive per-request [thinking cap](#openai-chat-completions);
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes nonzero `logit_bias`, requested log probabilities,
audio/file input or audio output, `strict:true`, required or named tool choice,
`parallel_tool_calls:false` with enabled tools, explicit low/high image detail, web search,
moderation, low/high verbosity, stored Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.
The vLLM/llama.cpp constrained-decoding extensions (`grammar`, `structured_outputs`, `guided_json`,
`guided_regex`, `guided_choice`, and `guided_grammar`) are rejected explicitly instead of being
treated as unknown hints; structured JSON is requested through `response_format`.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

A string `name` on a `tool` message is accepted as an ignored, output-neutral compatibility
extension for clients that mirror the function name onto tool results. It does not participate in
tool identity, prompt rendering, or output. Non-string values are malformed; non-empty names on
other message roles remain unsupported because they carry participant identity that the loaded chat
template cannot represent.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1`, and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors.

The request `model` must equal the public model ID: the artifact `identity.model_id` by default, or
the explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`.

Across Chat Completions, Responses, and Anthropic Messages, an explicit top-level tool-parameter
type controls conversion of Qwen's untyped parameter text. String-admitting values remain strings;
other explicitly typed values are decoded as JSON without coercion. NInfer does not validate
generated arguments against the full JSON Schema.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Clients send a wider effort vocabulary than the maintained Qwen templates accept (`low`, `medium`,
`xhigh`). This fork collapses `minimal` onto `low` and `high`/`max` onto `xhigh` before rendering,
so OpenAI and Claude Code requests that send `high` do not fail inside the template.

Control-token spellings quoted in message content, tool data or ordinary template kwargs are
encoded as text. Media placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. A request overrides this default with a positive
integer: `thinking_budget` on Chat Completions and Responses (a NInfer extension, accepted at top
level), or `thinking:{"type":"enabled","budget_tokens":N}` on Anthropic Messages (a value of at
least 1024 and below `max_tokens`, per that protocol). A request that does not think receives no
cap, whatever it sends.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

At the cap boundary, Engine first honors a natural `</think>`, stop condition, cancellation, or
total output/context limit. If thinking remains open, it commits Qwen's canonical early-close
guidance and close marker to the same model sequence without sampling, streams the guidance as a
reasoning delta, and continues normal content or tool-call generation. Inserted tokens count in
completion usage and the request's `max_tokens`/`max_output_tokens` budget.

Near the end of the output window the early-close guidance may not fit. Let B be the requested
thinking budget, C the output the request can still produce (the smaller of its output limit and
the context left after the prompt, counting the final writable position), and R the number of tokens
that early close needs: the tokenizer-derived guidance and close marker plus one post-close model
token. R depends on the model's tokenizer and is not a fixed number. A request is never rejected
because C falls in B < C < B + R; Engine resolves it as follows:

- C <= B, or C - B >= R: the budget is B and early close is available, as above.
- C > R (and C - B < R): the effective budget is C - R, which is always below B. Early-close
  guidance and at least one post-close model token still fit.
- C <= R (and C > B): the budget cannot be enforced, so thinking runs to the output limit with no
  guidance inserted. The response can end inside the reasoning with empty content.

Logs report the requested budget, and the effective budget when it differs. A request that cannot
enforce its budget reports the requested value only. The server does not promise that the model will
emit nonempty content or a tool call after the marker.

For Chat Completions, `reasoning_effort: "none"` requests disabled thinking. The other standard
values (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`) reach the template on its three rungs:
`minimal` runs as `low`, `high` and `max` as `xhigh`, so clients such as Claude Code that send
`high` work against the bundled Qwen templates.

`--reasoning-effort minimal|low|medium|high|xhigh|max` sets the effort of every thinking-enabled
request that states none on any endpoint, collapsed onto the same rungs. A request effort
(`reasoning_effort`, Responses `reasoning.effort`, Anthropic `output_config.effort`, or
`chat_template_kwargs.reasoning_effort`) overrides it. The default never enables thinking: requests
that run without thinking, through `--no-thinking` or their own options, receive no effort.
Conflicting explicit `enable_thinking` and effort values return `conflicting_template_option`.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unspecified
thinking, effort and preservation options use the template's defaults.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The current Vision runtime also
has a 32,768 merged-token envelope (131,072 raw patches); the effective Vision limit is therefore
`min(--max-context, 32768)`. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with the [default output limit](#default-output-limit) but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; other standard effort values pass to the selected template |
| `reasoning.summary`, `reasoning.generate_summary` | `auto`, `concise` or `detailed` (other values are rejected, and the two must agree when both are sent); accepted as a hint only. NInfer produces no reasoning summaries, so the response reports `reasoning.summary: null` and an empty `summary` on reasoning Items. `reasoning.context` and `reasoning.mode` are rejected with `reasoning_option_not_supported` |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `graft` | NInfer extension: name of a [prompt graft](#prompt-grafts), or `null`; also accepted by input token count |
| `thinking_budget` | NInfer extension: positive per-request [thinking cap](#openai-chat-completions), or `null`; rejected by input token count, which does not generate |
| `text.format` | `{"type":"text"}`, `{"type":"json_object"}`, or `{"type":"json_schema","name",...,"schema",...}` with optional `description` and `strict`; JSON formats are enforced (see [Structured output](#structured-output)) and echoed in the Response object |
| `tools` | direct function definitions or namespace groups containing function definitions; see below |
| `tool_choice` | `auto`, `none`, or function-only `allowed_tools` with mode `auto`; a namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` is accepted only when no effective tool is callable |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | valid `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, `safety_identifier`, and `user` values are accepted without being mapped to Engine session identity |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call`. Multiple message Items append their
content parts, multiple calls retain declaration order, and a reasoning-only turn is retained. A
user, system, developer, or `function_call_output` Item ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Up to four such values become shared stable-prefix
boundaries; they affect reuse opportunities, not prompt identity or output semantics. String
message status/phase metadata is accepted but has no Qwen prompt representation.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning metadata without raw
reasoning text, partial tool Items, and other Item/content types are not supported. HTTP media URLs
stored in a response chain are fetched again when that chain is continued; use data URIs when the
historical media bytes must be immutable.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions'
nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

NInfer gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Only functions in the current effective tool set can become structured calls;
undeclared model output remains ordinary text. `allowed_tools` with mode `auto` filters that set
without changing declaration order, while `tool_choice:"none"` disables structured tool output even
when the history contains earlier calls.

NInfer does not execute functions or enforce JSON Schema through constrained decoding, so
`strict:true`, required or named tool choice, hosted tools, remote MCP tools, and custom free-form
tools are rejected. Deferred loading, output schemas, and caller restrictions that exclude direct
invocation are also rejected because their semantics cannot be honored.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous `<tool_call>` suffix or the structured tool region is held.
Malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function tools and
tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, Structured Outputs/JSON mode, non-empty `include`, background execution, compaction,
files/audio, and OpenAI-hosted/MCP/custom tools. These are compatibility boundaries, not silently
accepted placeholders.


## Structured output

All three generation endpoints can require the answer to be JSON:

| Endpoint | Field | Formats |
|---|---|---|
| `/v1/chat/completions` | `response_format` | `json_object`; `json_schema` with `json_schema.{name, description?, schema?, strict?}` |
| `/v1/responses` | `text.format` | `json_object`; `json_schema` with `{name, schema, description?, strict?}` |
| `/v1/messages` | `output_config.format` | `json_schema` with `schema` (always strict) |

The format is enforced during sampling, not checked afterwards: before every sampled position the
Engine restricts the vocabulary to the tokens that keep the output a valid prefix of the requested
JSON, so the answer cannot leave the grammar. `json_object` admits any JSON object;
`json_schema` admits JSON conforming to the schema (types, `properties`/`required`, `enum`/`const`,
arrays, `anyOf`, string `pattern`/`format`, numeric bounds and the other constructs supported by
XGrammar's JSON Schema converter; `$ref` into `$defs`/`definitions` is resolved). With `strict:true`
an object schema without `additionalProperties` admits only its declared properties, as OpenAI's
and Anthropic's strict mode require; with `strict:false` or omitted, standard JSON Schema defaults
apply. A chat `json_schema` without `schema` admits any JSON value. Properties are emitted in
schema order. A schema the converter cannot represent is rejected with 400
`invalid_output_format` before the request is queued; compilation happens on the request thread
and repeated schemas are served from a cache.

Reasoning is not constrained. When Thinking is on, the model reasons freely and the format applies
to the answer after `</think>`; with Thinking off it applies from the first token. The answer may
start with at most two whitespace characters and, once the JSON value is complete, only an end of
turn may follow, so `finish_reason` is `stop`. The value can still be cut short by `max_tokens`
(`finish_reason:"length"`), a caller stop string, or a Thinking run that never closes; give
structured requests an output budget that covers the reasoning as well as the answer, or disable
Thinking or cap it with a thinking budget.

Structured output cannot be combined with active tools (a tool call is not a JSON value; send
`tool_choice:"none"` or omit the format) or with `ignore_eos` (a completed value can only be followed
by the stop token); both combinations are rejected with 400.

Every speculative backend stays active for structured requests. MTP and context-lookup drafts are
known before a round, so their verification masks are built first; DFlash/DFlash2 propose inside
the round, so the Engine reads the proposal back after the draft pass and builds the masks while the
target verifies it. Each verification column uses the mask for its own position: a draft token the
grammar forbids is rejected there, and the correction or bonus token is sampled from the licensed
set, so constrained output has the same distribution as non-speculative constrained sampling.

Cost, measured 2026-09-28 on one RTX 3090 with Qwen3.8-27B (rk4v4 KV, thinking off, greedy, the
same 823-token JSON answer with and without `json_object`): decode 187.3 → 183.3 tok/s with
DFlash2 K=7 (-2%), 96.6 → 90.8 tok/s with MTP3 plus context lookup (-6%; its masks are built
before the round rather than beside the target), and 46.7 → 45.9 tok/s without speculation (-2%).
Unconstrained requests are unaffected (DFlash2 tg256 63.6 vs 63.5 tok/s, MTP3 79.5 vs 79.6 tok/s
before/after). The first structured request after startup builds the tokenizer index (about
0.1 s); a new schema compiles in milliseconds and a repeated one is cached.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE. Consecutive User or Assistant messages are joined without adding separators.
System messages in `messages` may appear anywhere, including first, between two User messages,
directly after an Assistant message, or last; each renders as its own System turn at that position
after the top-level `system` text, so appending one keeps the earlier rendered prompt reusable. The
one excluded position is between an Assistant `tool_use` and the User message carrying its
`tool_result`, which is rejected as invalid tool history. A final text-only Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media, Thinking, or tool calls and cannot start with Thinking
enabled.

`max_tokens` is optional for local clients and otherwise uses the
[default output limit](#default-output-limit); a positive
value is the complete output budget. `max_tokens:0` is rejected because NInfer does not expose a
completed zero-output cache-prewarm lifecycle. `temperature`, `top_p`, `top_k`, and
`stop_sequences` enter Engine execution. A matched custom stop is returned as
`stop_reason:"stop_sequence"` together with the actual `stop_sequence`; context exhaustion returns
`model_context_window_exceeded`.

Thinking supports `disabled`, `adaptive`, and `enabled`. Enabled Thinking requires
`budget_tokens >= 1024` and less than `max_tokens`, and that budget is passed to Engine. Visible
Thinking is returned with an opaque local signature; SSE emits its `signature_delta` before
closing the block. Assistant Thinking blocks must be passed back unmodified with that signature;
signatures belong to the current serve process and are invalid after it restarts.
`display:"omitted"` is rejected because NInfer cannot provide Anthropic's
encrypted hidden-reasoning restore semantics. `preserve_thinking` remains a NInfer extension for
closed-turn reasoning history, and `graft` selects a [prompt graft](#prompt-grafts). `output_config.effort` passes its protocol-validated value to the
selected template. `output_config.format` accepts `{"type":"json_schema","schema":{...}}`, enforced
strictly (see [Structured output](#structured-output)); Count Tokens ignores it.

User-defined, non-strict tools support `name`, `description`, object `input_schema`, and
`input_examples`. `tool_choice:auto` and `none` are executable. Forced or named choice,
`strict:true`, active single-call enforcement, deferred tools, tools that exclude direct model
calls, Anthropic-provided/server tools, toolsets, MCP, and containers are rejected because their
required constraint or executor is absent. `tool_result` preserves text/image order and marks
`is_error:true` explicitly in the model prompt. For a visible Assistant tool-use turn, the next
User turn must provide exactly one leading result for every declared ID; valid results are matched
by ID and normalized to call order. A history that begins with results remains valid as a truncated
or imported conversation.

Ephemeral `cache_control` on the request, tools, System blocks, and User text/image frontiers is a
best-effort retention hint. NInfer maps representable breakpoints to exact prompt frontiers, keeps
the latest markers allowed by the Engine configuration, and ignores TTL and unrepresentable cache
hints rather than rejecting generation. Reuse still requires exact rendered-token compatibility;
aggregate usage reports verified reused tokens in `cache_read_input_tokens` and leaves cache
creation unknown. Streaming emits `message_start` after Engine admission commits the prefix
selection and before transfer/prefill output, so its uncached/cache-read split is already exact;
terminal cumulative usage matches the aggregate response.

Documents, Search Results, Files, server-tool results, container uploads, and
other execution-dependent blocks are rejected with the missing capability identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages; output-only sampling and
streaming fields do not affect the count:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Prompt grafts

A prompt graft is a hidden conversation prefix, for example a system turn plus an assistant
acknowledgement, which the operator loads at startup and a request selects by name. Grafts are the
[phantom-kv](https://github.com/lordx64/phantom-kv) `prefill_kv` container (format_version 1): a
safetensors file plus a `.json` sidecar beside it.

```bash
ninfer-serve model.ninfer --graft v1=C:/grafts/v1_q38_nf4.bin --graft red=C:/grafts/red.bin
```

A request selects one with the top-level `"graft": "NAME"` field on OpenAI Chat Completions,
Responses (create and input token count) and Anthropic Messages. A name the server did not load
fails with `400 unknown_graft`, and a non-string value fails as a malformed `graft` field.

`--default-graft NAME` makes one loaded graft the default for requests that state none. It must
name a `--graft`, or the server refuses to start.

| request `graft` | without `--default-graft` | with `--default-graft D` |
|---|---|---|
| absent or `null` | none | `D` |
| `""` | none | none (explicit opt-out) |
| `"X"` | `X` | `X` |

A grafted request runs exactly as if the graft's hidden turns preceded its own messages. The
graft's tokens occupy positions `[0, n)` and the request's own rendered prompt starts at `n`. With
greedy decoding, output is identical to sending those turns as literal messages. The rendered
prompt should carry no system turn of its own, because the graft already holds one. Graft tokens
count toward `--max-context` and the reported prompt/input tokens.

The container also stores the cache state that phantom-kv computed for the graft (attention K/V
and Gated DeltaNet conv/recurrent state). NInfer does not inject that state. It replays the
graft's own token ids through its own prefill, which is exact for the loaded weights and KV
storage and covers every layer, including the MTP draft layer. The end of the graft is offered to
the shared-prefix cache, so conversations after the first reuse the whole graft instead of
re-prefilling it. When the shared catalog is already full of other prefixes, the engine's
admission rule for structural candidates admits the graft on its second use.

Startup validates each graft against the loaded model and refuses to start on any mismatch:
- the per-layer attention/linear-attention layout;
- the KV-head, conv and recurrent-state geometry;
- the sidecar's payload sha256;
- the replay ids against the vocabulary.

Trained `softprompt_kv` and `direct_kv` grafts carry no replayable token ids. Their stored K/V and
Gated DeltaNet state are instead written at startup into a pinned shared-prefix slot, and grafted
requests start from it. Because nothing is replayed, these grafts cover the text layers only: the
graft carries no draft-backend state, so under `--spec` the MTP or DFlash cache over the graft's
positions is zero-filled and the draft proposes without graft context there. Output is unchanged,
because the target verifies every proposal against the injected state; only the acceptance rate
can fall. Each one holds a Device StateImage and a shared-prefix slot for the life of the server;
startup adds them on top of `--device-state-slots` and `--max-shared-prefixes`, and a disabled
context cache refuses them. Grafted requests use the context cache like any other: the pinned slot
is the root, and later turns and shared prefixes are captured after it. In the prompt the graft's
positions are held by ids derived from the container's sha256, so a cached prefix is only ever
matched by requests using the same graft. With `--devices`, each layer's K/V and state are written on the device
of the stage that holds that layer.

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated; `GET /v1/load`
and `GET /metrics` require the key.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

The table lists executable defaults. The startup example selects a long-context FP8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `600000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--max-prefill-lanes N` | requests that may prefill at once, at most `--max-concurrency`; each prefill unit goes to the lane with the shortest remaining prompt suffix, so short and prefix-cached prompts are not stuck behind a long one (see below) | `1` |
| `--prefill-max-skip N` | prefill units a lane may be passed over before it is served ahead of shorter lanes | `8` |
| `--decode-rounds-per-prefill N` | decode rounds run after each prefill unit while other requests generate; `0` means `--prefill-chunk` / 64 (see below) | `0` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8, Linux; see the [README](../README.md#several-gpus-pipeline-stages---devices-ab)); overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|nvfp4\|k8v4` | KV-cache storage. `rk8v4` is opt-in RotorQuant and `rk4v4` opt-in Lloyd-Max 4-bit keys; all seven are accepted on this fork's sm_86/sm_89 targets | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2 | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--lookup-ngram N` | context-lookup drafting alongside `--spec`: the last `N` tokens are matched against the sequence so far and what followed is proposed; exact, since verification rejects a wrong guess | `0` (off) |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)) | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--default-max-tokens N` | output limit when omitted by a request; see [default output limit](#default-output-limit) | largest budget that keeps every lane admissible |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay` | `overlay` keeps the Vision tower in pinned host memory and encodes each image inside a window borrowed from the evict-ranked text weight tail, so `--vision` no longer reserves device memory and `--kv-capacity auto` resolves the no-vision capacity; requires `--vision` and CUDA virtual memory management | `resident` |
| `--vision-max-merged N` | merged-token budget of one media item, `[64, 16384]`; larger images and video frame pairs are downscaled at preprocessing instead of being rejected, and the overlay window is sized for it | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--device-state-slots N` | extra Device checkpoint StateImages beyond the active-lane guarantee | `max-concurrency` |
| `--host-state-slots N` | pinned Host StateImage capacity | `8` |
| `--host-kv-mib N` | shared pinned Host Main/Backend KV byte capacity in MiB | `8192` |
| `--max-private-continuations N` | private continuation descriptor capacity | `2 * max-concurrency` |
| `--max-shared-prefixes N` | shared stable-prefix descriptor capacity | `max-concurrency` |
| `--host-cache-reserve-mib N` | with `--auto-host-cache`, host memory left unpinned beneath what is available | `3072` |
| `--host-cache-max-mib N` | with `--auto-host-cache`, the most it may pin (state slots and KV together), applied after the reserve; for machines whose memory other tenants share | no cap |
| `--host-cache-percent N` | with `--auto-host-cache`, the most it may pin as a percentage (1-100) of the machine's total memory (or its container's limit), however much is free at startup; the smallest of this, the cap and the free memory less the reserve applies | no limit |
| `--auto-host-cache` | size `--host-state-slots`, `--host-kv-mib`, `--max-private-continuations` and `--max-shared-prefixes` from the host memory still free once the model has loaded, for a machine that exists to serve; see [automatic host cache](#automatic-host-cache). Replaces those four options and refuses them | off |
| `--max-long-anchors-per-continuation N` | private long-anchor limit per continuation | `2` |
| `--auto-long-anchors N` | propose a private long anchor at each of the last `N` interior message boundaries of every prompt, so a rewrite of recent history restores at the anchor below the edit instead of re-prefilling from token zero; clamped to the anchor limit, `0` disables | the anchor limit |
| `--progress-anchor-tokens N` | propose a private long anchor at every multiple of `N` tokens of a prompt, and keep the anchors a cancelled prefill already holds, so a client that times out or disconnects part way through a very long prompt and retries resumes from the last anchor instead of prefilling from token zero; see the request-lifecycle section on cancelled requests. Shares the long-anchor limit with `--auto-long-anchors`; `0` disables, otherwise at least `256` | `16384` |
| `--max-cache-markers-per-request N` | caller marker input-complexity bound | `4` |
| `--slot-save-path DIR` | enable `POST /slots/{id}` save/restore/erase with files in `DIR` (created at startup) | disabled |
| `--auto-save-evicted` | write an evicted session back to its bound slot file; requires `--slot-save-path` | off |
| `--no-exit-on-engine-failure` | stay alive (answering 503) when the engine latches unavailable, instead of logging FATAL and exiting with status 3 after 5 s | exit |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--graft NAME=PATH` | load a [prompt graft](#prompt-grafts) a request may select by name; repeatable | none |
| `--default-graft NAME` | apply a loaded graft to requests that state none; `"graft": ""` opts out | none |
| `--reasoning-effort minimal\|low\|medium\|high\|xhigh\|max` | effort for thinking-enabled requests that state none | template default |
| `--cors` | permissive browser CORS headers | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..20`; zero selects the top-20 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup; the operational context-cost record and JSONL `server_start` identify the
selected source.

Engine selects sampling defaults from the loaded architecture and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`:
`C` slots guarantee active requests and `H` is a global checkpoint pool. Host State and Host KV are
independent startup-fixed pinned-memory capacities; Host KV is shared by Main and the selected
Backend pool and is consumed in physical page extents. `--no-prefix-reuse` selects root-only Engine
mode and cannot be combined with any of the seven explicit context-cache capacity flags, including
zero-valued flags.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Add `--request-log-jsonl profiles/bench/run/server.requests.jsonl` to the startup command to write
the log at that path.

Every line is one `ninfer_serve_request_log` schema-v22 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | build version, artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, Host State/KV capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, request-owned materialization cost/search diagnostics, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request_done.materialization` is the context-cache decision committed for that request: predicted
immediate, future-loss and total nanoseconds (`predicted_now_ns`, `predicted_future_loss_ns`,
`predicted_total_ns`, and `initial_predicted_total_ns` before search); `targets_evaluated`,
`projection_work`, `planning_elapsed_ns` and `search_elapsed_ns`; `stop_reason`; `budget_exhausted`;
`selected_degradation_units` and `selected_maximal_fallback`; `best_reuse_prompt_tokens`, the most
prompt reuse any admission candidate offered regardless of the plan chosen (beside a `root` plan, 0
means no reusable prefix was found and a large value means the planner priced reuse out); and the
optional-search accounting
`first_improvement_ns` (or `null`), `incumbent_improvements`, `search_work`, `search_granted_ns`,
`search_renewals`, `search_discovery_used`, `search_overshoot_ns`, `search_stop_phase` and
`search_boundary_limited`. Stop reasons are `no_pressure`, `queue_exhausted`, `target_budget`,
`expansion_capacity`, `time_budget`, `insufficient_expected_gain` and `work_budget`; `time_budget`
means the wall or control allowance ran out, while a request too cheap to justify optional search
reports `insufficient_expected_gain`. Search phases are `none`, `setup`, `construction`,
`assessment`, `expansion` and `refinement`. Search is bounded and heuristic; these diagnostics do not
claim model or global optimality, and aborted planning attempts are not published.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, and a stable text-fallback reason. Fallback reasons are `none`,
`malformed_structure`, `duplicate_parameter`, `invalid_tool_name`, `undeclared_tool`, and
`trailing_content`. These counters contain no tool arguments or generated text.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the aggregate prompt and maximum-item token
bounds plus encode peak and handoff layout/usage within that same allocation; these bytes must not
be added to `workspace.capacity_bytes`. The field is `null` when Vision is disabled.

`request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
`device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
`host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
`engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait, and
round count; `units` reports its prefill/control unit counts. In a compact batch every participating
request is delayed by the full round, so these values explain request latency but **must not be
summed across concurrent requests**.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. The existing stderr summaries remain available for operators
but are rounded and are not the aggregation source. Console lines use local
`[YYYY-MM-DD HH:MM:SS.mmm] [level]` timestamps. OpenAI Responses, OpenAI Chat, and Anthropic
generation requests receive a request ID when they enter synchronous preparation. Successful
preparation produces `request_start`; a preparation failure produces `request_rejected` without a
matching start. Later generation failures produce `request_error`. Schema/model validation
rejections before preparation and token-count-only calls are not measurement requests and do not
receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `materializing`, `capture_pending`, and
`terminal_pending` fields are the Engine scheduler snapshot at the end of the interval. The JSONL
`context_cache` object reports selection, capture, transfer, COW, pressure spill, private/shared
owner degradation and eviction, checkpoint drop, pressure search, budget exhaustion, maximal fallback, and historical-fork
counters as interval deltas; `occupancy` and `last_selection` are end-of-interval gauges. Materialization predictions are
request-owned and appear only on the corresponding `request_done` event. The top-level
`engine_recoveries` field is the interval delta of the worker-recovery counter described above; it
is not part of `context_cache` because a recovery is an engine-wide event, not a context-cache
operation.
`pressure.searches` counts plans accepted into Program resource transactions, including a transaction that later ends in
request-local abort; committed victim counters likewise report the resulting stable cache changes.

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose admission, context-transaction, replica, and
stats-publication slow paths; these detail values are already contained in a top-level Host phase
and must not be added to `total`. Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. The stderr interval line shows only total Host milliseconds,
decode Host/device-wait microseconds per round, boundary, and maintenance; use JSONL for analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.

## Execution behavior

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
requests. At each decode boundary, every decode-ready request is compacted into one batch and
processed by one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
There is no admission ETA or unbounded overflow queue. Because there is no preemption, a queued
request waits out the generations ahead of it, so the deadline has to be scaled to the longest
response the deployment allows rather than to a connection timeout: at C1 on an RTX 3090 a single
6,500-token response occupies the engine for about 106 seconds. The 600,000 ms default admits a
queued caller behind roughly ten such responses; lower it only to fail fast on purpose.

A cancelled request (a client timeout, a dropped connection) normally frees its lane and everything
it computed. For a very long prompt that makes a retrying client livelock the server: each attempt
is cancelled at about the same point and the next one starts again from the last cached frontier.
With the context cache on, the Engine therefore cancels a prefilling request by publishing the
long-anchor checkpoints it has already captured, instead of discarding them, and
`--progress-anchor-tokens` makes sure there are some: every request proposes an anchor at each
multiple of that many prompt tokens. The identical retry matches the deepest surviving anchor and
prefills only the tokens after it, so a cancelled attempt loses at most the stride (plus whatever
lies beyond the last captured anchor).

The anchors share the per-continuation long-anchor budget (`--max-long-anchors-per-continuation`,
default `2`), and a full set replaces its shallowest anchor, so a long prompt keeps its deepest
anchors. A prefill that has captured no anchor, whose lane is still holding a just-reused
checkpoint for its first write, or when capacity cannot hold the anchor, is discarded as before. An
anchor occupies one cached state slot, which under pressure the context cache may evict like any
other retained checkpoint.

By default one request owns the staged prefill at a time, so a very long prompt holds the lane for
its whole prefill (340-370 s for a 200k-token prompt on an RTX 3090 with `--kv-dtype rk4v4` and
chunk 512, about 550-590 tok/s against ~1,100 tok/s at shallow context) and a prefix-cached request
behind it, which still needs one prefill unit, waits it out. `--max-prefill-lanes N` lets up to N
requests hold a staged prefill. Each prefill unit then goes to the lane with the shortest remaining
prompt suffix (a lane's first unit always runs first), and a lane passed over `--prefill-max-skip`
units is served before any shorter one, so the long prompt is slowed but never starved. A short
request now waits for at most the unit in flight plus `--decode-rounds-per-prefill` decode rounds
(16 at the default chunk of 1024, one under strict alternation) between each of its prefill units,
plus its own units, and the long prompt finishes later by the prefill work of the requests that passed it. Admission stays
FIFO, and admission reserves each request's full prompt and output KV, so the KV capacity has to
hold the long prompt and the short ones together; a request that does not fit still waits as the
FIFO head however many prefill lanes are configured. The following was measured before
`--decode-rounds-per-prefill` existed, so it is strict alternation (`--decode-rounds-per-prefill 1`)
and has not been re-measured with the default of more rounds, which adds the extra decode rounds to
each wait. On an RTX 3090 with the 27B and
`--kv-dtype rk4v4`, three lanes, two 80k-token prompts and one short prompt: the short request
finished in 3.1 s at chunk 512 and 14.8 s at `--prefill-cublas --prefill-chunk 4096` (each of its
decode steps waits for one chunk of the long prefills), against about 170 s and 105 s behind both
long prompts one at a time. The first long prompt's first token came 9% later than it would alone
(92 s against 85 s at chunk 512, 57 s against 53 s with cuBLAS); the second arrived when the two
prefills would have finished back to back (171 s and 105 s). The prompt that finishes first is the
one with less left to prefill, not the one that arrived first.

The executor runs a single prefill chunk, then
`--decode-rounds-per-prefill` decode rounds, so `--prefill-chunk` sets the worst-case pause every
active stream sees while a new prompt is ingested, and the round count sets how much of the GPU the
streams keep during it. A decode round takes tens of milliseconds and a prefill chunk hundreds, so
strict alternation (`1`) leaves a stream about one token per chunk: on an RTX 3090 with the 27B, a
24.7K-token prompt prefilling at chunk 1024 took a concurrent stream from about 48 tok/s to 1.4 tok/s
(median gap 725 ms), and with `--prefill-cublas --prefill-chunk 4096` to 0.75 tok/s (1.6 s gaps).
The default of `0` runs `--prefill-chunk` / 64 rounds (16 at chunk 1024), which trades some prefill
speed while someone is generating for streams that keep moving; with no request generating, prefill
runs unchanged. On an RTX 3090 ingesting a 4,900-token prompt behind four active
streams with strict alternation, the largest inter-token gap measured 1,043 ms at chunk 1024, 515 ms
at 512, and 312 ms at 256, against an 82 ms median decode interval; the ingesting request's own
prefill rate fell only from 1,135 to 1,130 to 1,110 tok/s. Prefill units of different requests are
never batched together, so a smaller chunk trades almost no ingestion throughput for a
proportionally smaller stall. The shipped concurrent launcher uses 512.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting
media request retains the same cancellation and timeout deadline. Model output is bounded by the
same finite request count and each request's effective output-token limit; output callbacks and
network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` is each sequence's logical ceiling. `--kv-capacity` fixes the shared Main Text KV
pool used by active requests and retained prefixes. `auto` accounts for the complete enabled runtime
and leaves 1 GiB of sizing headroom; omitting the option makes it follow `--max-context`. Capacity
resolves once at startup.

Admission reserves the full prompt-plus-effective-output page entitlement through request
completion. A request remains queued until a legal resource plan can satisfy that entitlement.

### Automatic host cache

The host tier of the context cache (pinned StateImages and KV pages that hold prefixes after they
leave the GPU) defaults to a fixed 8 slots and 8 GiB. `--auto-host-cache` replaces that with sizes
taken from the machine, once, after the model has loaded:

- The available memory is the smaller of the system's available memory (`MemAvailable` on Linux,
  available physical memory on Windows) and what remains under the container's cgroup limit, so a
  rented container is sized by its own limit rather than the host's RAM. Startup fails, naming the
  explicit options, if the platform reports neither.
- The machine is assumed to serve only this process, so everything but a fixed reserve is pinned:
  `--host-cache-reserve-mib` (default 3072) is left for what still grows after sizing: CUDA and
  cuBLAS host state, the tokenizer and server, request buffers and the Responses store (256 MiB by
  default). On the 27B that growth measured about 2 GiB after sizing, and two short requests added
  about 30 MB; the default adds margin for long prompts. Pinned pages cannot be reclaimed, and in a
  container exceeding the limit kills the process, so lower the reserve only after watching the
  process's memory under your own load.
- `--host-cache-max-mib` caps that budget. Without it, a machine with a lot of free memory pins
  nearly all of it, which is only safe if nothing else will want the memory later. On a rented GPU
  box the host's RAM is shared with other tenants: a 64 GB host with 29 GB in use by others left
  `MemAvailable` at about 52 GB, 49 GiB was pinned, and the engine was killed by the kernel's OOM
  killer when the neighbours grew (the container's own limit was never reached, so the reserve and
  the cgroup term could not help). Set the cap to what the workload actually uses of the host tier;
  `host_kv_bytes` in `GET /v1/load` shows how much of it is occupied.
- With vision enabled the reserve also covers the media caches, which grow after sizing and are
  bounded only by their options: `--media-cache-mib` and `--media-live-mib` (1 GiB and 2 GiB by
  default) are added to `--host-cache-reserve-mib`. Without that, a vision workload could outgrow the
  3 GiB margin, which was measured on text requests only.
- `--host-cache-percent N` is the "take half the RAM" setting: `--host-cache-percent 50` pins at most
  half of the machine's total memory (physical memory, or the container's cgroup limit when that is
  lower). It does not depend on what is free at startup, so two servers started together or a
  neighbour that was idle at launch do not change it. It only lowers the budget; the free memory less
  the reserve, and `--host-cache-max-mib`, still apply, and the smallest of the three wins. The pinned
  state slots and KV together never exceed that budget; pinned memory is allocated once at startup and
  the cache evicts its oldest entries to stay inside it rather than growing.
- An eighth of that budget buys StateImage slots (at most 128, each one whole GDN snapshot, so its
  size depends on the model and `--gdn-state-fp16`); the remainder is host KV.
- The private-continuation catalog is the number of resident states (active lanes, device
  checkpoints and host slots), never below `2 * max-concurrency`; the shared-prefix catalog is a
  quarter of that, between `max(max-concurrency, 4)` and 32, plus one per injected graft.

The resolved values are what the `context cache |` startup line and the request log report. The option
combines with `--device-state-slots` and `--max-long-anchors-per-continuation`, but not with
`--host-state-slots`, `--host-kv-mib`, `--max-private-continuations` or `--max-shared-prefixes`, and
not with `--no-prefix-reuse`. On Windows, pinned host memory is charged against the GPU's memory
(see `--host-kv-mib`), so the whole budget is first clamped to half of the device memory left after
the model and KV pool, less 1 GiB, and only then split. A card the model nearly fills therefore gets
a small host cache, down to none, while the same machine on Linux is not limited this way. The
logged figures are the ones pinned. The constants are conservative defaults chosen by reasoning, not
measured against hit rates on a live workload.

### Default output limit

A request that omits its output limit (`max_completion_tokens`/`max_tokens` on Chat Completions,
`max_output_tokens` on Responses, `max_tokens` on Messages) receives the largest budget that still
lets every configured lane be admitted at the same time. Once its prompt is prepared, the Engine
finds the largest output whose admission entitlement -- Main KV pages for the prompt and output,
plus the MTP draft-window or DFlash backend KV pages when speculation is on -- fits one lane's share
(`1/--max-concurrency`) of each KV pool, and clamps it to the remaining context
(`--max-context` minus the prompt). With one lane, or a pool of at least `--max-concurrency` times
`--max-context`, that is the whole remaining context, so long reasoning runs are not cut at an
arbitrary count; with several lanes over a smaller pool, each limitless request stays inside its
share and they all run concurrently. A prompt that alone overruns one lane's share can never run
beside full-share lanes, so it keeps the whole remaining context.

A run that exhausts the budget finishes with `finish_reason:"length"`, Responses `incomplete` with
reason `max_output_tokens`, or Anthropic `stop_reason:"max_tokens"`, or
`stop_reason:"model_context_window_exceeded"` when the budget was the remaining context. An explicit
request limit always wins, and `--default-max-tokens N` replaces the derived default with a fixed cap
(still bounded by the remaining context). The JSONL request record reports the budget actually
submitted as `requested_output_tokens`.

Each reusable checkpoint contains KV and complete continuation state. At admission, capture, and
finish boundaries, resource pressure may keep it on Device, move its StateImage and/or KV replicas
to pinned Host memory, or evict it. The planner compares incoming-request work with the later
recovery cost imposed on retained checkpoints. Active requests retain their state and completion
reservations, and placement choices preserve model semantics. The full policy and invariants are
defined in [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The completion log reports the
reused token count as `cache=`.

The completion log reports one of six reuse paths: `root`, `private_endpoint`,
`private_turn_closure`, `private_response_replay`, `private_long_anchor`, or
`shared_stable_prefix`. Reuse validation covers KV, recurrent state, hidden state, selected-backend
state, and the exact prompt frontier. With stable `preserve_thinking=true`, the auxiliary checkpoint
rolls to the message frontier immediately before the current response's deterministic generation
prologue. A normalized response, compact-summary instruction, or replacement user suffix therefore
replays the small generation prologue and only the changed suffix while retaining the complete
stable conversation prefix. Stable `false` places the turn-closure checkpoint before the first
assistant opener in the open turn, so closing that turn can recompute its opener and omit its
reasoning without discarding the preceding conversation.

`preserve_thinking` selects the capture frontier for newly created checkpoints. Existing exact
checkpoints remain reusable across a mode change. If the desired boundary is behind the selected
reuse frontier and has no snapshot, the Engine keeps the valid hit and defers the new checkpoint. A
later request that diverges before every retained checkpoint starts from root. The JSONL completion
record exposes the restored checkpoint as `prefix_reuse_path`. Reasoning-effort changes participate
in rendered-token identity and exact-prefix selection.

An appended mid-conversation system message is an ordinary prompt suffix, so an unchanged prior
history remains eligible for `private_endpoint`. If the client modifies, removes, or moves a
historical system message, the token prefix genuinely differs and a miss/reset is correct.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools and does not enforce client JSON Schema through constrained
decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
