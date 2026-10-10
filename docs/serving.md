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
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
when running alone. Requests acquire KV pages as execution advances; if concurrent growth exhausts
the pool, the scheduler can pause a request and restore it later.

With `C=2` and two extra Device slots, the process owns four Device StateImages. The default shared
pinned Host budget is 8 GiB plus eight model StateImages. It holds retained state, KV and pause
snapshots, including in-flight destinations; `--host-context-mib` sets an explicit total instead.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Add
`--vision` when the server must accept image or video input. Speculative residency is likewise
frozen by `--spec mtp|dflash|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend.
Any `--spec` also loads the optimized proposal head (`--lm-head-draft` is implied). DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
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
decoding: the window is opened as soon as the image's prefill unit is licensed, after every other
lane's unit for that round, and the lane yields its prefill turn until the encode completes. When
free pages fall short, cached context is reclaimed (demoted or released) to make room; the window
never revokes a paused request's snapshot or pauses another request. The pages are out of
circulation while the loan is open, and any request that then finds itself short of KV pages first
waits for the encode to finish and the pages to return, so a loan never causes an eviction or a
preemption of its own.

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
request log line reports `overlay=<windows>x<conc|excl|mixed>[+ahead] <ms> (evict <MiB> <ms>,
restore <ms>, staged <MiB>)`, where `+ahead` marks windows that encoded beside other lanes' work,
and the JSON record carries `vision_overlay`, including `exclusive_windows` and `ahead_windows`.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health and build version (see [Server version](#server-version)) |
| `GET /v1/load` | serving capacity, current load, and monotonic token counters (see [Load](#load)) |
| `GET /metrics` | Prometheus counters, gauges and latency histograms, plus llama.cpp-compatible series (see [Metrics](#metrics)) |
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
cap", when a request may run to the remaining context, or the `--default-max-tokens` cap. `--max-output-tokens` bounds either, so it is reported instead when it is smaller or when
no default is set: the largest budget any request can get. The sampler is the loaded model's preset
for the default thinking mode (thinking unless `--no-thinking`) under the process sampling flags and
`--greedy`; request fields still override it per request. `seed` appears only with `--seed`, since
requests otherwise draw a fresh random seed. `model_alias` is the public model id and `model_path`
the artifact path the server was started with. `build_info` is `ninfer <version>` (see
[Server version](#server-version)). There is no `chat_template` or
writable `POST /props`, and `/slots` and llama.cpp's non-`/v1` route aliases are not served.

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

After startup, an internal host-side failure fails only the requests that were running or being
admitted, clears the context cache, and keeps serving: queued requests stay queued, and a paused
request keeps its place but gives up its saved snapshot and resumes by recomputing its prompt and
output. Each such recovery is counted in `ninfer_engine_recoveries_total` and as a top-level
`engine_recoveries` counter on the request log's `throughput` event, and `GET /health` stays `200`.
A failure confined to admitting one request (planning it or starting its binding) fails that
request alone and is not a recovery. `GET /health` turns `503` for good only when the engine cannot
prove a clean recovery (cleanup fails, or the GPU memory and host cache in use do not return exactly
to their state after startup), when the GPU itself reports an error, or after three failures in a
row (each less than 30 seconds after the previous one, with no request completing between them);
the server then needs a restart. Every such failure is logged at error severity with the exception
text, the execution unit (`boundary`, `admission`, `control`, `prefill` or `decode`), the affected
request ids and lanes, and the streak count; a latch is logged `FATAL`, and `ninfer-serve` exits
with status 3 after a five second grace period (so in-flight error responses flush and `/health`
answers 503 meanwhile) for a supervisor to restart. `--no-exit-on-engine-failure` keeps the
process alive instead, answering `503`. The exception text is operator diagnostics and is never
sent to clients; the request log's `request_error` record carries it for requests that failed this
way.

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
               "kv_page_tokens": 64, "device_state_slots": 8, "host_context_bytes": 17179869184},
  "requests": {"admitted": 6, "running": 4, "prefilling": 1, "decode_ready": 3, "waiting": 2,
               "paused": 0, "replaying": 0, "materializing": 0},
  "occupancy": {"device_main_kv_pages": 1500, "device_main_kv_tokens": 96000,
                "device_state_slots": 5, "host_context_bytes": 2684354560,
                "host_state_slots": 7, "host_kv_bytes": 2147483648},
  "counters": {"computed_prefill_tokens": 48200113, "committed_decode_tokens": 6120452,
               "reused_prompt_tokens": 30911840, "decode_rounds": 861307,
               "decode_row_rounds": 2448180},
  "host": {"active_seconds": 3120.4, "device_wait_seconds": 41872.9},
  "last_request": {"completion_tokens": 24, "generation_wall_seconds": 0.17,
                   "inter_token_seconds": 0.0074, "decode_host_seconds": 0.012}
}
```

- `uptime_seconds` counts from the moment the server became ready, not from process start.
- `capacity` is fixed once the Engine is ready. `max_admitted_requests` is
  `max_concurrency + max_pending_requests`, the ingress bound described in
  [Execution behavior](#execution-behavior); `kv_capacity_tokens` is the resolved page-aligned Main
  KV pool and `kv_page_tokens` its page size.
- `requests.admitted` counts requests holding ingress capacity, from preparation until the response
  is released; a new generation request is rejected with `server_overloaded` (HTTP 429, or 529 on
  Anthropic endpoints) when it would exceed `max_admitted_requests`. `running` counts occupied
  execution lanes (at most `max_concurrency`), of which `prefilling` counts the lanes with prompt
  left to prefill, `replaying` the lanes rebuilding a paused request's state, and `decode_ready` the
  lanes in the decode batch. `waiting` counts requests submitted to the Engine FIFO that have not
  been admitted to a lane yet. `paused` counts admitted requests that resource pressure paused; they
  hold no lane and resume in their original order.
- `capacity.host_context_bytes` is the one pinned Host budget (see
  [Automatic host cache](#automatic-host-cache)). `occupancy` reports current Main KV pages (and
  tokens), Device StateImage slots, and the Host context bytes in use, including destinations
  reserved for transfers in flight; `host_state_slots` and `host_kv_bytes` break that Host figure
  down and must not be added to it. Main KV occupancy includes retained reusable prefixes, which the
  scheduler reclaims under pressure, so a full pool does not by itself mean new requests will wait.
- `counters` are monotonic since startup; derive rates by differencing two polls.
  `computed_prefill_tokens` excludes prefix-reused prompt tokens (reported separately in
  `reused_prompt_tokens`). `committed_decode_tokens` counts tokens committed by decode rounds and
  excludes each request's first token, which prefill emits; with speculative decoding a round commits
  several tokens per row. `decode_row_rounds` is the sum of decode batch sizes over `decode_rounds`.
- Gauges and counters come from the snapshot the Engine publishes at execution boundaries, so they
  can trail the instant of the poll by up to one boundary.
- `host` is cumulative Engine host-active seconds since startup (device wait excluded) and,
  separately, seconds the host thread waited on the device. Take two polls and divide the host delta
  by the interval to see how host-bound the Engine is.
- `last_request` is the pace of the last finished request that produced at least two tokens, or
  `null` before the first. `inter_token_seconds` is first-to-last token wall time over the token
  gaps. A healthy Engine stays in milliseconds; an Engine that stalls for seconds right after every
  first token reports whole seconds even for a short answer while `/health` stays 200, so a
  supervisor can restart on, say, `inter_token_seconds > 0.25` for several consecutive polls.

### Context store

`--context-store DIR` makes the context cache survive a restart or a crash. It is off by default
and needs a Host context cache (the default 8 GiB, `--host-context-mib` or `--auto-host-cache`),
because stored sessions come back into the Host tier. What is stored is one retained conversation
(continuation) at a time: its recovery points (the endpoint of the last turn, the point where the
next turn's input starts, any long anchors) with their KV pages and recurrent states, byte for byte
what the Host tier would hold for them. Public shared prefixes are not stored. A conversation is
written to `DIR`

- in the background once it has been unused for `--context-store-idle-seconds` (30 s) and has
  changed since it was last written, so a crash loses at most that much of a conversation, and
- at shutdown, for every conversation not already stored, most recently used first, within
  `--context-store-flush-seconds` (60 s).

A conversation evicted from the cache before it was idle that long is not written; with
`--context-store-idle-seconds 0` conversations are written only at shutdown.

On start-up the most recently used conversations are restored into the Host tier, most recent first,
while they fit in its free space and until `--context-store-restore-seconds` (120 s) is spent, before
the server accepts requests. Nothing is evicted for them. A request then reuses a restored
conversation exactly as it would have before the restart, restoring it from the Host tier to the
GPU like any other Host-resident checkpoint. There is nothing for a supervisor or gateway to call.

The store is also read while the server runs. When a request is about to be admitted and the store
holds a checkpoint of that very prompt at least 4,096 tokens deeper than the deepest checkpoint the
cache offers it (the conversation was evicted since), a background reader loads the image while the
request waits, out of the admission order but without holding up other requests, for at most
`--context-store-restore-seconds` or the request's queue deadline. The image is then rebuilt in the
Host tier, giving up the cache's lowest-ranked Host contents if it needs the room (never another
waiting request's chosen source), and the request resumes from it instead of prefilling the
difference. A stored conversation that cannot be read, does not fit or misses its deadline is a miss,
and the request is prefilled as without the store. A read counts as a hydration only if the
admission actually resumes from it. On the 27B on an RTX 3090, a 7.8k-token conversation (an
856 MB image) read back from a warm file cache in about 1.5 s gave a first token after 1.6 s, against
5.1 s for prefilling it and 0.1 s when it had stayed resident.

The store does not support the DFlash speculative backends (their draft-side state has not been
verified through a stored image); the Engine refuses to start with both.

An image does not depend on `--devices`/`--stage-layers`: a conversation stored by a server split
into pipeline stages restores on one GPU and the reverse. A worker failure that clears the context
cache leaves the store as it was, and the next turn of a stored conversation reads it back. Prompt
grafts are installed context, never stored; a conversation that started from a graft is stored like
any other, and changing the configured grafts (their names or files) makes the existing images
misses, as a different model would.

A session image is several GB for a deep context (about 18 KB per token with `--kv-dtype rk4v4`,
plus about 150 MB of recurrent state per recovery point), and consecutive images of one conversation
share almost all of it. The store therefore splits an image into fixed 32 MiB chunks named by a hash
of their content and writes only chunks it does not already hold: keeping a long conversation current
costs the newest pages and the endpoint state, not the whole session. Older images of a conversation
are removed once a newer image of it is stored. Writes are atomic (a temporary name, then a rename),
every chunk is verified when read, and a session that cannot be read back intact is removed and
simply re-prefilled by the next request; a damaged store costs cache hits, never a wrong answer.

The store keeps at most `--context-store-max-gib` (default: half the free space of the volume when
the server starts), removing the least recently used sessions first, and removes sessions unused for
`--context-store-ttl-hours` (default 168). A store written by a different model, quantization, KV
configuration or set of prompt grafts is ignored and ages out. The `ninfer_context_store_*` series (see
[Metrics](#metrics)) report size, writes, bytes reused, what was restored at start-up and how long it
took. A background write is skipped while any request is waiting, binding, prefilling or replaying,
and the write queue holds at most two images, so a slow disk does not hold up requests. Taking the
image itself (copying a deep conversation out of the GPU or the Host tier) is a single step of the
Engine worker, so a request that arrives during it waits for that copy; a conversation that could not
be queued is tried again later.


#### Keeping a copy in a bucket

`--context-store-s3-endpoint URL --context-store-s3-bucket NAME` (off by default, and only with
`--context-store`) keep a copy of the store in an S3-compatible bucket: AWS S3, MinIO, Cloudflare
R2, Backblaze B2 or any server that speaks the S3 API with path-style addressing and Signature V4.
The local directory becomes a cache of the bucket.

- Every session written to the directory is also uploaded in the background, chunks first and its
  manifest last, so another engine never sees a session whose chunks are missing. A chunk the bucket
  already holds is not sent again, so a conversation that grows uploads only what changed. Shutdown
  waits up to two minutes for the uploads still queued.
- At start-up, and every minute after, the engine lists the bucket and registers sessions it does
  not have. They appear in the store as not local; start-up restores the most recently used ones,
  fetching their chunks (every chunk is verified, a damaged one makes the session a miss and it is
  not offered again). A new engine, a replacement box or a second engine therefore starts warm from
  what the others wrote, without sharing a disk.
- A request that would resume from a session only the bucket holds waits for it like any stored
  session (see above): the background reader fetches the missing chunks, within the same deadline,
  without holding up the Engine worker or other requests.
- A session evicted from the directory for space stays in the bucket and comes back the same way.
- Credentials come from the environment, never the command line: `NINFER_S3_ACCESS_KEY_ID` and
  `NINFER_S3_SECRET_ACCESS_KEY` (or `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; an
  `AWS_SESSION_TOKEN` is honoured). `--context-store-s3-region` (default `us-east-1`) names the
  signing region and `--context-store-s3-prefix` namespaces the keys when a bucket is shared. Use
  an `https://` endpoint outside a trusted network: payloads are not part of the request signature.
- The bucket is the long-term tier, so its expiry is the bucket's: add a lifecycle rule that expires
  objects under the prefix after 7 days (the engine assumes at least a day). `--context-store-ttl-hours`
  governs only the files in this directory; a session only the bucket holds ages there, and drops out
  of the index at the next listing once the bucket no longer lists it. A session that is used has the
  age of its objects restarted (a server-side copy onto itself, no data transferred, at most once a day
  per session), and an object the bucket has lost in the meantime is uploaded again from the directory,
  so something in use does not expire under it. A write is uploaded from a snapshot taken when it was
  written, with its chunk files kept on disk until the upload ends (at most 16 uploads wait, each
  newer write of a session replacing its queued upload); a chunk that is not intact on disk stops the
  upload instead of publishing a session whose data is damaged. A chunk found damaged on disk when a
  session is loaded is fetched again from the bucket.
- An unreachable bucket costs the uploads and the fetches, counted in `ninfer_context_store_remote_*`;
  the directory keeps working. Shutdown waits for queued uploads up to the remote flush budget (two
  minutes) and then interrupts the transfer in progress, so it never waits out a stalled connection.


### Metrics

`GET /metrics` serves Prometheus text format 0.0.4 on the same port. Like `/v1/load` it requires
the API key when one is configured, answers `503 model_loading` until warmup completes, and works
independently of `--request-log-jsonl` and `--log-stats-interval-ms`. Engine failure leaves the
endpoint readable with `ninfer_engine_ready 0`. The handler copies published snapshots and formats
them outside the Engine worker; scraping does not reset counters or initiate device work.

```bash
curl http://127.0.0.1:8080/metrics -H 'Authorization: Bearer local-secret'
```

The `llamacpp:` series use llama.cpp's `--metrics` names and meaning, so dashboards and routers
built for llama.cpp work unchanged. They are the Engine's totals since startup and advance per
prefill unit and decode round, during a request rather than at its completion.

| Series | Type | Meaning |
|---|---|---|
| `llamacpp:prompt_tokens_total` | counter | prompt tokens computed by prefill; prefix-cache hits excluded |
| `llamacpp:prompt_seconds_total` | counter | prefill execution time |
| `llamacpp:tokens_predicted_total` | counter | tokens committed by decode rounds |
| `llamacpp:tokens_predicted_seconds_total` | counter | decode execution time |
| `llamacpp:requests_processing` | gauge | admitted requests up to `--max-concurrency` |
| `llamacpp:requests_deferred` | gauge | admitted requests waiting beyond `--max-concurrency` |

Every other series is named `ninfer_*`. Its counters start after startup warmup and reset when the
server restarts.

| Series | Meaning |
|---|---|
| `ninfer_engine_ready`, `ninfer_server_start_time_seconds` | whether the Engine accepts work, and the Unix time the server attached it after warmup |
| `ninfer_model_info`, `ninfer_max_concurrency`, `ninfer_max_context_tokens`, `ninfer_spec_decode_draft_window` | model/backend identity and startup limits |
| `ninfer_requests_running`, `waiting`, `paused`, `prefilling`, `decode_ready`, `replaying`, `materializing` | current Engine gauges; prefill/decode/replay are subsets of resident requests |
| `ninfer_prompt_tokens_total`, `ninfer_prompt_tokens_cached_total` | full input and context-cache-reused tokens, counted once on initial binding |
| `ninfer_prefill_tokens_total`, `ninfer_replayed_tokens_total` | actual initial prefill and separate recovery recomputation |
| `ninfer_generation_tokens_total`, `ninfer_decode_tokens_total` | all committed outputs, or decode/control outputs excluding the first token; include thinking and injected control tokens |
| `ninfer_prefill_seconds_total`, `ninfer_decode_seconds_total` | prefill-unit and decode-round execution time |
| `ninfer_decode_rounds_total`, `ninfer_decode_row_rounds_total` | decode batch executions and the sum of their batch sizes |
| `ninfer_spec_decode_{rounds,draft_tokens,accepted_tokens,fallback_steps}_total` | live native speculative work, including MTP and DFlash/DFlash2 |
| `ninfer_root_selections_total`, `ninfer_checkpoint_selections_total` | admissions by the context-cache source they started from: root is a miss (full prefill), checkpoint a reuse. Hit rate is `checkpoint / (root + checkpoint)` |
| `ninfer_{preemptions,snapshot_restores,replay_restores}_total` | resource-pressure pauses and recovery routes |
| `ninfer_waiting_cancelled_requests_total`, `ninfer_waiting_expired_requests_total`, `ninfer_waiting_abandoned_seconds_total` | requests the client cancelled, or the pending timeout expired, before admission, and the total time they had waited (a client that gives up after 60 s shows up here) |
| `ninfer_cancelled_prefills_total`, `ninfer_cancelled_prefill_computed_tokens_total` | requests cancelled while prefilling, and the prompt tokens they had computed |
| `ninfer_engine_recoveries_total` | host-side worker failures the server survived by failing the in-flight requests and clearing the context cache |
| `ninfer_device_kv_{used,capacity}_pages`, `ninfer_device_backend_kv_used_pages`, `ninfer_device_state_{used,capacity}_slots` | physical Main KV, speculative backend KV and StateImage occupancy; retained history also occupies these pools |
| `ninfer_host_context_{used,reserved,capacity,peak}_bytes`, `ninfer_host_state_images`, `ninfer_host_kv_used_bytes` | unified Host backing; reserved bytes, StateImages and KV bytes are breakdowns already included in used bytes |
| `ninfer_context_transfer_bytes_total{resource,direction}`, `ninfer_context_transfer_seconds_total` | actual `state`/`main_kv`/`backend_kv` payload transfers, `d2h`, `h2d` or `d2d`, and their time |
| `ninfer_context_pressure_spill_pages_total` | KV pages moved from Device to Host to relieve pressure |
| `ninfer_host_work_seconds_total{phase}`, `ninfer_device_wait_seconds_total` | instrumented worker wall time; device wait is not CUDA kernel time |
| `ninfer_engine_host_seconds_total` | the `host_work` phases summed: Engine host-active seconds, device wait excluded. Its rate against token throughput exposes a host-bound Engine |
| `ninfer_token_intervals_total`, `ninfer_token_interval_seconds_total` | gaps between output tokens of finished requests and their first-to-last-token wall time; the ratio of their rates is the mean inter-token time |
| `ninfer_last_request_inter_token_seconds`, `ninfer_last_request_decode_host_seconds` | gauges: mean inter-token time and decode host time of the last finished request with two or more tokens (absent before the first). A healthy Engine stays in milliseconds; one that stalls after every first token reports seconds while `/health` stays 200, which a supervisor can restart on |
| `ninfer_constraint_requests_total{outcome}`, `ninfer_constraint_cache_total{result}` | settled constrained requests by completion state, and compilation-cache access |
| `ninfer_constraint_{prepare,mask,matcher}_seconds_total`, `ninfer_constraint_mask_{positions,upload_bytes}_total` | [constraint](#output-constraints) work aggregated at request settlement, including truncated/cancelled results |
| `ninfer_constraint_draft_wait_seconds_total` | live draft-ready wait counted once per batch; a subset of device wait |
| `ninfer_requests_total{outcome}`, `ninfer_response_failures_total` | generation attempts entering preparation by outcome (`completed`, `cancelled`, `failed`, `rejected`), and response failures after settlement; protocol/model validation failures and token-count requests are excluded |
| `ninfer_time_to_first_token_seconds` | histogram updated once at the first committed token, including preparation, queueing and binding |
| `ninfer_request_duration_seconds`, `ninfer_request_queue_seconds` | histograms for settled generation outcomes, including cancellation; exceptional failures have separate counts |
| `ninfer_context_store_images`, `ninfer_context_store_used_bytes` | gauges: sessions and bytes held by the [context store](#context-store) (zero when it is off) |
| `ninfer_context_store_writes_total`, `_write_failures_total`, `_dropped_total` | sessions written, writes that failed, sessions not queued because the write queue was full |
| `ninfer_context_store_bytes_written_total`, `_bytes_reused_total` | new chunk bytes written, and chunk bytes a write found already stored |
| `ninfer_context_store_evicted_total`, `_corrupt_total` | sessions removed for space, age or supersession, and because they could not be read back intact |
| `ninfer_context_store_restored_sessions`, `_restored_bytes`, `_restore_seconds` | gauges: what start-up restored into the cache, and how long it took |
| `ninfer_context_store_hydrations_total`, `_hydrated_tokens_total`, `_hydration_failures_total`, `_hydration_seconds_total` | stored sessions read back for a request, the prompt tokens that saved, failures (the request was prefilled, or its admission did not use what was read) and the time requests waited for the reads, failed ones included |
| `ninfer_context_store_remote_images` | gauge: with a bucket, sessions it holds that the directory does not hold in full |
| `ninfer_context_store_remote_uploads_total`, `_remote_upload_bytes_total`, `_remote_upload_failures_total` | objects and bytes uploaded to the bucket, and uploads that failed |
| `ninfer_context_store_remote_downloads_total`, `_remote_download_bytes_total`, `_remote_download_failures_total` | chunks and bytes fetched from the bucket, and listings or fetches that failed or returned damaged data |

Histograms expose `_bucket`, `_sum` and `_count`. Metrics have bounded labels; they do not retain
request IDs or request text. Rates are calculated by the consumer, for example:

```promql
rate(ninfer_generation_tokens_total[1m])

rate(ninfer_spec_decode_accepted_tokens_total[1m])
/ rate(ninfer_spec_decode_draft_tokens_total[1m])
```

Detailed per-request records remain available through the
[structured request log](#structured-request-log). Earlier releases of this fork named these series
`ninfer:*`; colons are reserved for Prometheus recording rules, so they are now `ninfer_*`, and the
colon names are no longer emitted.

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
- `response_format` of type `text`, `json_object`, or `json_schema`, and the `structured_outputs`
  extension (grammar, choice or regex); see [Output constraints](#output-constraints);
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- function tools, optional `strict:true` argument schemas, `tool_choice` `auto`/`none`/`required`,
  named selection, `allowed_tools`, and `parallel_tool_calls`; assistant tool-call history,
  tool-result messages, and legacy function-call history;
- the top-level `reasoning_effort` field;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- the `graft` extension selecting a [prompt graft](#prompt-grafts);
- the `thinking_budget` extension, a positive per-request [thinking cap](#openai-chat-completions);
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes nonzero `logit_bias`, requested log probabilities,
audio/file input or audio output, explicit low/high image detail, web search,
moderation, low/high verbosity, stored Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.

### Output constraints

JSON mode and JSON Schema use the standard protocol fields:

| Endpoint | Field |
|---|---|
| Chat Completions | `response_format: {"type":"json_object"}` or `{"type":"json_schema","json_schema":{"name":"answer","schema":{...},"strict":true}}` |
| Responses | `text.format: {"type":"json_object"}` or `{"type":"json_schema","name":"answer","schema":{...},"strict":true}` |
| Anthropic Messages | `output_config.format: {"type":"json_schema","schema":{...}}` |

`json_object` requires an object root. Schema mode follows the supplied root type and enforces the
[supported assertions](maintainer/constrained-decoding.md#42-json-与-schema-的执行合同), including when
`strict` is omitted or false. Unsupported assertions return HTTP 400 before generation. OpenAI
errors distinguish `invalid_json_schema`, `unsupported_json_schema` and `unsatisfiable_json_schema`;
`param` identifies the request field followed by the schema JSON Pointer. Anthropic uses its
`invalid_request_error` envelope with the schema location in the message. Responses echoes the
selected `text.format` in aggregate responses and SSE response objects.

JSON output uses compact separators and declared property order. State the desired content in the
prompt; the schema is not inserted into it. Only one output constraint may be supplied.

Schemas support positional arrays (`prefixItems` plus tail `items`) and inclusive/exclusive
`number` ranges. Bounded numbers use exact int64 integers or finite binary64-compatible decimal
and scientific notation with up to 17 significant digits. Bounds must retain their value when the
schema is parsed; numbers requiring greater precision receive `unsupported_json_schema`.
These capabilities also apply to strict tool parameters.

GBNF, choice and regex are available through the NInfer extension `structured_outputs`
on Chat Completions, Responses and Anthropic Messages. Supply exactly one member:

```json
{"structured_outputs": {"grammar": "root ::= \"yes\" | \"no\""}}
```

```json
{"structured_outputs": {"choice": ["positive", "neutral", "negative"]}}
```

```json
{"structured_outputs": {"regex": "(BUG|TASK)-[0-9]{4}"}}
```

Choice returns one literal string, preserving case and whitespace. The list must be nonempty;
duplicate entries have no extra weight, and an empty-string entry permits empty content.
Regex matches the complete content. It supports character classes, groups, alternatives and
repetition; `.` excludes line terminators, `\d`/`\w` use ASCII ranges, and `\s` includes Unicode
whitespace. Empty regex permits only empty content. Anchors are supported at the ends of top-level
alternatives. Lookaround, backreferences, word boundaries, Unicode properties, flags and unknown
escapes return HTTP 400. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice).
Invalid choices and regexes use `invalid_choice` and `invalid_regex`, with the request field in `param`.

These constraints apply to answer content; thinking is separate. GBNF supports recursive rules,
Unicode and repetition. All modes support streaming and all speculative backends. For assistant
continuation, the grammar covers the existing assistant content plus the generated suffix. Completion uses the model's EOS tokens;
output limits and cancellation can produce an incomplete answer. JSON modes can be combined with
active tools; GBNF, choice and regex require no active tools or `tool_choice:"none"`.
Output constraints reject custom stops. OpenAI errors use
`invalid_grammar` for invalid grammars and `constraint_dead_end` for a reachable prefix without a
legal next token. Anthropic reports these through its `invalid_request_error` envelope.
The `grammar` and `guided_*` aliases are not accepted.

Constrained responses include a NInfer `constraint` observation. `branch` is `undecided`, `content`,
or `tools`; `complete` means the committed language can end, and `terminated` means it accepted EOS.
A complete JSON value can therefore have `complete:true`, `terminated:false` and a length finish
reason. The observation also includes `cache` (`hit`, `built`, `waited`), `mask_positions`,
`mask_upload_bytes`, and `timings_seconds` for preparation, CPU mask work and matcher work.
These times are parts of existing request time and can overlap GPU execution.
Streaming sends the observation once: the Chat finish/usage chunk, the Responses terminal response
object, or Anthropic `message_delta`. Unconstrained responses omit it.

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

The request `model` must equal the public model ID: the artifact `metadata.name` by default
(falling back to its architecture name when absent), or the explicit `--model-id` override.
Reasoning is returned separately as `reasoning_content`; answer text remains in `content`.

For non-strict tools, a direct top-level tool-parameter
`type`, or an `anyOf`/`oneOf` composed entirely of explicit primitive types, guides conversion of
Qwen's untyped parameter text. It does not decide whether structurally complete markup is a tool
call. String-admitting values remain strings, including the empty string. An empty block for a
declared non-string parameter is omitted. Admitted JSON values retain their JSON type;
case-insensitive boolean text is normalized to `true` or `false`. A nonempty schema mismatch remains
a structured call: valid JSON retains its represented type and other text becomes a JSON string so
the tool consumer can report the validation error and continue the agent loop. Schemas without a
supported explicit type retain untyped inference. NInfer does not apply defaults, enforce required
properties, or perform recursive JSON Schema validation on this route.

On the unconstrained route, string parameters preserve function/tool-call markers and balanced nested
`<parameter=...>...</parameter>` text as value bytes. The Qwen wire format has no delimiter escape,
so an unmatched nested parameter opener or a standalone `</parameter>` cannot be represented
unambiguously; either causes the complete tool-call region to fall back to ordinary content.

### Tool constraints

The three protocols share one constrained tool implementation:

| Choice | Generated calls |
|---|---|
| `auto` | Text or calls; zero to many |
| `none` | No tool calls; declarations remain in the prompt |
| OpenAI `required` / Anthropic `any` | One or more calls |
| OpenAI named function | Exactly one call to that function |
| Anthropic named `tool` | One or more calls to that tool |
| OpenAI `parallel_tool_calls:false` / Anthropic `disable_parallel_tool_use:true` | At most one call; exactly one when a call is required |

OpenAI `allowed_tools` supports `auto` and `required`. Selection changes generation permissions,
while all declarations retain their original order in the prompt. Requests with tools enable
**basic structural constraints by default**, including ordinary `auto` calls without `strict`.
The model can answer normally or start a tool call; a call must use the model's tool framing and a
declared function name.

Non-strict parameter names and order remain open. Their schema supplies the existing value
normalization hints; it is not compiled as a strict constraint. Open objects, root unions, and
unsupported schema assertions therefore remain usable. Repeated parameter names use the last
value, retaining the first key position; the published JSON object contains each key once.
`strict:true` additionally enforces the parameter contract below.

Top-level `tool_constraints:"auto"` opts into request-driven constraints: ordinary non-strict
`tool_choice:"auto"` then uses free generation. Strict tools, selection/count restrictions, and
`tool_choice:"none"` still enforce their requirements. `tool_constraints:"basic"` is the default.

With JSON object/schema output, `auto` permits either a JSON answer or a complete tool-call sequence.
Required/named choices permit calls for that turn; after supplying the tool result, use `auto` for
the final JSON answer. `none` permits only JSON. The JSON schema is validated on every turn.
This combination enforces tool framing even with `tool_constraints:"auto"`; `strict` continues to
control argument-value validation. Tool markers inside JSON strings remain ordinary string data.

A function's `strict:true` also constrains its argument values against its schema. Its parameter
root must reduce to a `type:"object"` schema with `additionalProperties:false`, including supported
`allOf` and local-reference combinations.
Properties are emitted in declaration order; optional properties may be omitted. Root
const/enum/unions are not supported. Values use the supported JSON Schema subset described above.
Top-level pure string parameters use raw text and preserve whitespace. Other values use JSON;
a top-level string/non-string union (such as string/null) is rejected because the Qwen parameter
format cannot distinguish those branches. Such unions inside JSON objects or arrays are supported.
Raw values cannot contain the delimiter `\n</parameter>`; unsatisfiable required values are rejected.
Integer arguments use signed 64-bit values; number arguments use finite binary64-compatible
representations. Unsupported schemas fail with HTTP 400 before generation.

For `auto` without JSON output, text can precede the first call. Required/named choices start directly with calls
(after thinking, if enabled). Once a constrained call starts, the suffix consists of complete calls
and model EOS. Active tool constraints require model EOS, so they cannot coexist with custom stop
strings or `ignore_eos`. Basic constraints are only a default: a request that sends stop strings
or `ignore_eos` with ordinary non-strict `auto` tools and no `tool_constraints` field falls back to
free tool-call generation instead of being refused, because many OpenAI-compatible agent clients
send stop strings with every request. A request that asks for constraints explicitly
(`tool_constraints:"basic"`, a required or named `tool_choice`, `parallel_tool_calls:false` or a
strict tool) together with custom stops is still rejected with HTTP 400 on `stop` (or
`ignore_eos`).
Token limits and cancellation can still stop generation: only completed calls are published.
A later call truncated by the token limit keeps `length`/`max_tokens`/Responses `incomplete` as the
terminal status. Streaming publishes each completed call in the terminal event sequence;
arguments are not streamed incrementally.
Assistant continuation may finish a partial call; a prefix containing a completed call is rejected.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.
Generated token segmentation can also differ from re-encoding the same text, limiting prefix reuse.

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

## OpenAI prompt caching

Chat Completions and Responses translate OpenAI cache hints into optional shared-prefix write
candidates:

- omitted `prompt_cache_options` creates a default implicit candidate at the end of the latest
  cacheable content part;
- `mode:"implicit"` requests the same automatic candidate explicitly;
- `mode:"explicit"` disables that implicit write for the request;
- `prompt_cache_breakpoint:{"mode":"explicit"}` on supported content creates an explicit
  candidate.

The automatic candidate precedes the message closing tokens, allowing reuse when the same message
body grows and its previous tokens remain an exact prefix. An explicit marker keeps its requested
location. Responses applies this policy after expanding stored history.

One request carries at most four writes. Explicit markers take precedence: four explicit candidates
leave no extra slot for an automatic candidate. If more explicit markers appear in the history, the
latest four remain write candidates. Exact reads of already-published prefixes do not require the
request to repeat a marker.

These fields are optimization hints. A legal boundary that cannot be represented as an exact
rendered-token frontier is ignored without changing prompt content. `prompt_cache_key` is not an
Engine session key or prefix identity. Valid TTL/retention values are accepted, but NInfer does not
promise their wall-clock residency; physical retention follows the resource scheduler.

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
| `text.format` | `text` (default), `json_object`, or `json_schema`; see [Output constraints](#output-constraints) |
| `tools` | direct function definitions or namespace groups containing function definitions; see below |
| `tool_choice` | `auto`, `none`, `required`, a named function, or function-only `allowed_tools` with mode `auto`/`required`; namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` enforces at most one call |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, and explicit breakpoints follow [OpenAI prompt caching](#openai-prompt-caching); `safety_identifier` and `user` are accepted as client hints |

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
`prompt_cache_breakpoint:{"mode":"explicit"}`. Write selection follows
[OpenAI prompt caching](#openai-prompt-caching); boundaries affect reuse opportunities, not prompt
identity or output semantics. String message status/phase metadata is accepted but has no Qwen
prompt representation.

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
    "required": ["city"],
    "additionalProperties": false
  },
  "strict": true
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
a later request. Selection and strict argument enforcement follow the common tool contract above.

Hosted tools, remote MCP tools, custom free-form tools, deferred loading, output schemas, and
caller restrictions that exclude direct invocation remain unsupported.

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
On the unconstrained route, malformed tool markup is flushed back as ordinary text without losing bytes.

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
moderation, non-empty `include`, background execution, compaction,
files/audio, and OpenAI-hosted/MCP/custom tools. These are compatibility boundaries, not silently
accepted placeholders.

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
selected template. `output_config.format` accepts JSON Schema output as described in
[Output constraints](#output-constraints).

User-defined tools support `name`, `description`, object `input_schema`, `input_examples`, and
`strict`. `tool_choice` accepts `auto`, `none`, `any`, or named `tool`; `disable_parallel_tool_use`
enforces a single-call limit. See the common tool contract above for schema and framing details.
Deferred tools, tools that exclude direct model calls, Anthropic-provided/server tools, toolsets,
MCP, and containers remain unsupported. `tool_result` preserves text/image order and marks
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
the shared-prefix cache as a save opportunity, so conversations after the first can reuse the
whole graft instead of re-prefilling it; whether it is kept follows the
[retention rules](maintainer/resource-scheduling-and-context-cache.md#retention).

Startup validates each graft against the loaded model and refuses to start on any mismatch:
- the per-layer attention/linear-attention layout;
- the KV-head, conv and recurrent-state geometry;
- the sidecar's payload sha256;
- the replay ids against the vocabulary.

Trained `softprompt_kv` and `direct_kv` grafts carry no replayable token ids. Their stored K/V and
Gated DeltaNet state are instead installed at startup as a pinned context-cache checkpoint, and
grafted requests start from it. The K/V are written through the same append Op prefill uses, so
every `--kv-dtype` stores what prefill would store for the same rows. Because nothing is replayed,
these grafts cover the text layers only: the graft carries no draft-backend state, so under `--spec`
the MTP or DFlash context over the graft's positions is zero-filled and the draft proposes without
graft context there. Output is that of the target over the installed state, because the target
verifies every proposal; only the acceptance rate can fall. Each one holds a Device StateImage and
its KV pages for the life of the server; startup adds them on top of `--device-state-slots` and the
planned KV capacity, and a disabled context cache (`--no-prefix-reuse`) refuses them. Grafted
requests use the context cache like any other: the pinned checkpoint is the root, it is never
evicted, and later turns and shared prefixes are captured after it. In the prompt the graft's
positions are held by ids derived from the container's sha256, so the checkpoint is only ever
matched by requests using the same graft.

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
| `--model-id ID` | override the public OpenAI model alias | artifact `metadata.name`, or architecture name |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--max-concurrency N` | resident execution lanes; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `600000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
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
| `--lm-head-draft` | optimized proposal head; implied by `--spec`, accepted for compatibility | on with `--spec` |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)) | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--default-max-tokens N` | output limit when omitted by a request; see [default output limit](#default-output-limit) | the remaining context |
| `--max-output-tokens N` | upper bound on every request's output budget, stated or derived; see [default output limit](#default-output-limit) | none |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay` | `overlay` keeps the Vision tower in pinned host memory and encodes each image inside a window borrowed from free KV pages, or from the evict-ranked text weight tail when those fall short, so `--vision` no longer reserves device memory and `--kv-capacity auto` resolves the no-vision capacity; requires `--vision` and CUDA virtual memory management | `resident` |
| `--vision-max-merged N` | merged-token budget of one media item, `[64, 16384]`; larger images and video frame pairs are downscaled at preprocessing instead of being rejected, and the overlay window is sized for it | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--device-state-slots N` | extra Device StateImages beyond `max-concurrency` | `max-concurrency` |
| `--host-context-mib N` | shared pinned Host budget for StateImages, KV and pause snapshots, including in-flight destinations; refused together with `--auto-host-cache` | `8192 MiB + 8 native StateImages` |
| `--auto-host-cache` | size the Host context budget from the host memory still free once the model has loaded, for a machine that exists to serve; see [automatic host cache](#automatic-host-cache). Refuses an explicit `--host-context-mib` | off |
| `--host-cache-reserve-mib N` | with `--auto-host-cache`, host memory left unpinned beneath what is available | `3072` |
| `--host-cache-max-mib N` | with `--auto-host-cache`, the most it may pin, applied after the reserve; for machines whose memory other tenants share | no cap |
| `--host-cache-percent N` | with `--auto-host-cache`, the most it may pin as a percentage (1-100) of the machine's total memory (or its container's limit), however much is free at startup; the smallest of this, the cap and the free memory less the reserve applies | no limit |
| `--context-store DIR` | keep retained sessions on disk so a restart or crash does not lose the context cache; see [Context store](#context-store). Needs a nonzero Host context budget | off |
| `--context-store-max-gib N` | bound the store; the least recently used sessions are removed beyond it. Requires `--context-store` | half the volume's free space |
| `--context-store-ttl-hours N` | remove sessions unused this long; `0` keeps them until space is needed | `168` |
| `--context-store-idle-seconds N` | write a session unused this long, and changed since it was last written, in the background; `0` writes only at shutdown | `30` |
| `--context-store-restore-seconds N` | time budget for restoring sessions at start-up, and the longest a request waits for its stored session to be read back | `120` |
| `--context-store-flush-seconds N` | time budget for writing sessions that are not yet stored at shutdown | `60` |
| `--context-store-s3-endpoint URL`, `--context-store-s3-bucket NAME` | keep a copy of the store in an S3-compatible bucket (both required together; credentials from `NINFER_S3_ACCESS_KEY_ID` / `NINFER_S3_SECRET_ACCESS_KEY` or the `AWS_` equivalents); see [Keeping a copy in a bucket](#keeping-a-copy-in-a-bucket). Requires `--context-store` | off |
| `--context-store-s3-prefix P`, `--context-store-s3-region R` | key prefix inside the bucket, and the signing region | none, `us-east-1` |
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

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`.
Host state and Main/Backend KV share one startup-fixed byte budget; this is context storage, not a
limit on total process RAM. `--host-context-mib 0` disables Host context backing.
`--no-prefix-reuse` disables cross-request history reads and writes; pause/replay recovery remains
available, and the capacity flags may still be specified.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Every line is one `ninfer_serve_request_log` schema-v26 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | build version, artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, unified Host context capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, preemption/recovery counters, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters |
| `request_scheduling` | request identity, pause/restore/recovery transitions, Snapshot revocation, Engine observation time and cumulative global/request work counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, and a stable text-fallback reason. Fallback reasons are `none`,
`malformed_structure`, `duplicate_parameter`, `invalid_tool_name`, `undeclared_tool`, and
`trailing_content`. These counters contain no tool arguments or generated text.

`request_done.constraint` carries the same constraint observation as the HTTP terminal result,
or `null` for unconstrained requests. Preparation failures and execution errors use the existing
rejection/error records rather than successful constraint outcomes.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.
`generation.scheduling` records preemptions, snapshot/replay restores, replayed tokens, paused time
and request-owned transfer bytes. Replay rebuilds committed state without adding new output usage.

`generation.admission` records the initial `preferred_reused_tokens`, `source_wait_seconds`,
`revoked_checkpoints`, and `fallback_reason`. Source waiting is a subset of initial queue time;
selecting or retaining a checkpoint does not itself count as a cache hit. Revocations count retained
checkpoint references removed under resource pressure. Fallback reasons are `none`, `source_invalid`,
`source_revoked`, `cost_changed`, `capacity_limit`, and `isolated_capacity`.

`request_scheduling` records `pause_started`, `paused`, `restore_started`, `restored`,
`replay_complete`, `recovery_complete`, `snapshot_revoked`, and a `terminal` boundary for preempted
requests. `preemption_index` identifies each pause cycle; `route` is `snapshot`, `replay`, or `null`
while pause preparation has not yet selected the saved representation. `steady_ns` is captured on
the Engine worker, and `elapsed_ns` starts at Engine submission; JSONL writes happen on the request
consumer thread. Compare `steady_ns` rather than delivery order across requests. `restored` ends
binding; `recovery_complete` marks the first fresh committed unit or normal terminal progress,
not merely rebuilding the old frontier. Cancellation can end the cycle without that event.

Each event's `progress` carries global and request-owned prefill, decode/control and replay token
counters. Between two boundaries, subtract the request delta from the global delta to measure
other requests' completed work. In particular, `restored` to `replay_complete` establishes whether
other work advanced during Replay without relying on periodic scheduler gauges. Events are enabled
only with request logging and add no per-token records.

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
`constraint_draft_wait_exposed_seconds` is the request's exposure to the batch's draft-ready wait,
already included in `device_wait_exposed_seconds`. The `throughput.host_work.constraint_draft_wait_seconds`
interval and Prometheus counter count each batch once.

`request_done.first_output_timing` freezes observations immediately before Engine publishes its
first nonempty output delta. It is `null` when no such output exists. This boundary differs from the
first accepted model token and the client's first HTTP output. Engine elapsed time begins at submit:
initial queue ends when the successful binding attempt starts, initial binding ends when the
binding is installed, and paused time includes pause preparation, waiting and restoration. The remaining
interval is resident time. Its `engine` observations describe resident Host/Device-wait exposure;
`prefill` and `replay` describe this request's submitted work. Terminal `engine_timing` still covers
the whole request.

The work `gpu_seconds` measures Text prefill stream intervals, including their MTP/DFlash work;
Vision encode, standalone bridges and exact-hit sampling fall outside that interval.
`context_transfers` reports completed request-owned State/Main KV/backend KV copies by direction,
using transfer-event time and bytes. GPU intervals overlap Host submission and waits, so they are
separate evidence, not additional wall-time stages. Background reclamation remains Engine-wide.

`result.generated_token_ids` records committed output token IDs for exact prefix analysis.
The JSONL file never records an API-key value; `argv` replaces it with `<redacted>`.
Operational stderr summaries are rounded and are not the
aggregation source. OpenAI Responses, OpenAI Chat, and Anthropic generation requests receive a
request ID when they enter synchronous preparation. Successful preparation produces
`request_start`; a preparation failure produces `request_rejected` without a matching start. Each
started generation transaction then has exactly one machine terminal: `request_done` when Engine
returns its outcome, or `request_error` when generation fails before an outcome exists. Later
response rendering, Responses storage, or terminal transport failures are operational response
events only and do not add a second JSONL terminal. Schema/model validation rejections before
preparation and token-count-only calls are not measurement requests and do not receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `paused`, `replaying`, `materializing`,
`capture_pending`, and `terminal_pending` fields are the Engine scheduler snapshot at the end of the
interval. The JSONL `context_cache` object reports selections, captures, StateImage operations,
transfers, tail-page COW and pressure spills as interval deltas; `occupancy` and `last_selection` are
end-of-interval gauges. The separate `scheduling` object reports preemptions, restores and replayed
tokens. Occupancy includes Host reservations while transfers are in flight.
The top-level `engine_recoveries` field is the interval delta of the worker-recovery counter. It is
not part of `context_cache`, because a recovery is an engine-wide event (see
[Startup readiness](#startup-readiness)).

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose stats-publication work; these detail values
are already contained in a top-level Host phase and must not be added to `total`.
Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. Pretty throughput contains nonzero token rates and counts,
the current running/prefill/decode-ready composition, nonzero waiting/materialization/terminal
states, average decode batch, and Host-active time plus its fraction of the interval. Use JSONL for
complete measurement analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.

## Execution behavior

The server owns one resident Engine with `1..8` execution lanes fixed at startup. At each decode
boundary, eligible decode-ready requests form one compact batch, processed by one model traversal
and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
Once admitted, a request may pause for resource pressure without restarting this initial-admission
deadline. Fresh requests enter in FIFO order with a bounded bypass allowance when an earlier request
cannot fit. There is no admission ETA or unbounded overflow queue. A fresh request never preempts a
resident one, so when every lane is busy a queued request waits out the generations ahead of it, and
the deadline has to be scaled to the longest response the deployment allows rather than to a
connection timeout: at C1 on an RTX 3090 a single 6,500-token response occupies the engine for about
106 seconds. The 600,000 ms default admits a queued caller behind roughly ten such responses; lower
it only to fail fast on purpose.

Each worker cycle runs the permitted control and decode work and then one prefill or replay chunk.
Prefill chunks rotate among resident requests, so a short prompt that already holds a lane is not
held behind another lane's long prefill. A request that has no lane yet still waits for a resident
to finish, be cancelled or pause, and a smaller chunk does not admit it sooner. `--prefill-chunk`
therefore sets the worst-case pause every active stream sees while a prompt is ingested. A decode
round takes tens of milliseconds and a prefill chunk hundreds, so a stream gets about one token per
chunk while another request prefills. Measured on an RTX 3090 with the 27B on the previous engine,
which alternated one chunk with one decode round in the same way (not re-measured on this engine):
a 24.7K-token prompt prefilling at chunk 1024 took a concurrent stream from about 48 tok/s to
1.4 tok/s (median gap 725 ms), and with `--prefill-cublas --prefill-chunk 4096` to 0.75 tok/s
(1.6 s gaps). Ingesting a 4,900-token prompt behind four active streams, the largest inter-token gap
measured 1,043 ms at chunk 1024, 515 ms at 512, and 312 ms at 256, against an 82 ms median decode
interval; the ingesting request's own prefill rate fell only from 1,135 to 1,130 to 1,110 tok/s.
Prefill units of different requests are never batched together, so a smaller chunk trades almost no
ingestion throughput for a proportionally smaller stall. The shipped concurrent launcher uses 512.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media preparation uses a shared permit pool sized from `--media-live-mib`
and the maximum supported prepared-payload size per request. Waiting media requests retain the same
cancellation and timeout deadline. Model output is bounded by the same finite request count and
each request's effective output-token limit; output callbacks and network serialization run
outside the GPU executor and do not delay formation of the next batch.

`--max-context` is each sequence's logical ceiling. `--kv-capacity` fixes the shared Main Text KV
pool used by active requests and retained prefixes. `auto` accounts for the complete enabled runtime
and leaves 1 GiB of sizing headroom; omitting the option makes it follow `--max-context`. Capacity
resolves once at startup.

Before each prefill, decode or replay unit, the runtime reserves the additional pages and temporary
storage required by that unit. It first reclaims inactive cache resources when capacity is short.
If resident requests still cannot advance together, it pauses a younger request while preserving
progress for the oldest resident request. A paused request does not block fresh requests that fit
the remaining capacity. Restoration follows original request order and reserves enough space to
rebuild the saved frontier and complete one new execution unit.

A paused request keeps its committed output and protocol state. With sufficient Host backing, it
can restore a snapshot; otherwise it rebuilds model state from retained input and committed tokens.
Replay does not resample or republish those tokens, but it consumes compute and can increase gaps in
the output stream. The policy and ownership rules are defined in
[Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The pretty completion record
shows `cache N (P%, path)` using readable path labels; JSONL retains the exact
`prefix_cache_hit_tokens` and `prefix_reuse_path` fields (`root` or `checkpoint`). Reuse requires
matching KV, recurrent state, hidden state, selected-backend state and exact prefix identity.

Completed conversation endpoints serve direct continuations. A separate input checkpoint preserves
the stable boundary before a response that a later prompt may normalize or replace. With
`preserve_thinking=true`, that boundary precedes the response's generation prologue; with `false`,
it precedes the assistant turn whose closed reasoning may be omitted. The next request can therefore
recompute the changed suffix while retaining the preceding conversation. Capture follows these
semantic boundaries and shared-prefix hints.

Changing `preserve_thinking` or reasoning effort changes the rendered prompt where applicable;
already retained exact prefixes remain usable. An appended mid-conversation system message is an
ordinary suffix. Modifying, removing or moving a historical message changes the prefix and may
require an earlier checkpoint or a root prefill.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools or enforce tool-argument schemas through constrained decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.

### Automatic host cache

The Host context budget (pinned memory for StateImages, KV pages and pause snapshots once they leave
the GPU) defaults to 8 GiB plus eight model StateImages, or the explicit `--host-context-mib`.
`--auto-host-cache` replaces that with one size taken from the machine, once, after the model has
loaded:

- The available memory is the smaller of the system's available memory (`MemAvailable` on Linux,
  available physical memory on Windows) and what remains under the container's cgroup limit, so a
  rented container is sized by its own limit rather than the host's RAM. Startup fails, naming the
  explicit option, if the platform reports neither.
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
  the cgroup term could not help). Set the cap to what the workload actually uses of the Host
  context; `occupancy.host_context_bytes` in `GET /v1/load` shows how much of it is occupied.
- With vision enabled the reserve also covers the media caches, which grow after sizing and are
  bounded only by their options: `--media-cache-mib` and `--media-live-mib` (1 GiB and 2 GiB by
  default) are added to `--host-cache-reserve-mib`. Without that, a vision workload could outgrow the
  3 GiB margin, which was measured on text requests only.
- `--host-cache-percent N` is the "take half the RAM" setting: `--host-cache-percent 50` pins at most
  half of the machine's total memory (physical memory, or the container's cgroup limit when that is
  lower). It does not depend on what is free at startup, so two servers started together or a
  neighbour that was idle at launch do not change it. It only lowers the budget; the free memory less
  the reserve, and `--host-cache-max-mib`, still apply, and the smallest of the three wins.

The result is the same single shared budget an explicit `--host-context-mib` sets: StateImages, KV
and pause snapshots together never exceed it, it is pinned once at startup, and the cache reclaims
retained entries to stay inside it rather than growing. The resolved value is what the
`context cache |` startup line and the request log report. The option combines with
`--device-state-slots` but not with an explicit `--host-context-mib`.

The budget, automatic or explicit, is pinned in full on Windows as on Linux. Earlier builds clamped
it on Windows to half of the device memory left after the model and KV pool, less 1 GiB, because
WDDM once charged pinned host memory against the card; on current drivers pinning no longer tracks
free VRAM (measured on an RTX 3090, driver 616.64: 16 GiB pinned with 804 MiB of the card free), so
the clamp was removed and an 8 GiB budget really pins 8 GiB of RAM. The logged figures are the ones
pinned. The automatic sizing constants are conservative defaults chosen by reasoning, not measured
against hit rates on a live workload.

### Default output limit

A request that omits its output limit (`max_completion_tokens`/`max_tokens` on Chat Completions,
`max_output_tokens` on Responses, `max_tokens` on Messages) may generate up to the remaining context
(`--max-context` minus the prompt). `--default-max-tokens N` replaces that with a fixed cap, still
bounded by the remaining context. An explicit request limit always wins, and `--max-output-tokens N`
bounds every request, including ones that state a larger limit; the bound is not an error, the run
just ends with the length finish reasons below. The JSONL request record reports the budget actually
submitted as `requested_output_tokens`.

A run that exhausts the budget finishes with `finish_reason:"length"`, Responses `incomplete` with
reason `max_output_tokens`, or Anthropic `stop_reason:"max_tokens"`, or
`stop_reason:"model_context_window_exceeded"` when the budget was the remaining context.

An output limit does not reserve KV. A request acquires pages as it generates, so a client that
states a very large limit does not evict other sessions' retained contexts up front; when concurrent
growth exhausts the pool, the scheduler reclaims retained checkpoints first and then pauses a younger
request, as described above.
