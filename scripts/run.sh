#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Serve a model on one RTX 3090.
#
#   run.sh <model> [profile]
#
#   model             profiles
#   qwen38-27b        tuned (default), int8, c8   <- recommended
#   qwen36-35b-a3b    tuned (default)
#
# `tuned` is the recommended profile: rk4v4 KV, speculation plus the draft head, the memory flags,
# vision in overlay residency, and an 8 GiB pinned Host context budget for the cache. `int8` and
# `c8` are the older reference profiles for the 27B -- one user at 64K of INT8 KV (the quality
# default), and eight lanes at 8K -- with every serving flag fixed.
#
# Every measurement behind these defaults, the memory model, and the reasoning for each flag are
# in docs/maintainer/launcher-profiles.md. What follows is what you need to run it.
#
# QWEN3.8-27B, `tuned`: two flag sets, each measured (docs/performance.md, "Recommended
# configurations"), chosen with NINFER_SPEC. The default is the fast one.
#
#   NINFER_SPEC=dflash2 (default): fastest at one stream, the full 262,144-token context headless
#
#     --spec dflash2 --draft-tokens 7 --lm-head-draft \
#     --prefill-cublas --prefill-chunk 4096 \
#     --kv-dtype rk4v4 --gdn-state-fp16 \
#     --vision --vision-residency overlay
#
#   NINFER_SPEC=mtp: the full context, two lanes sharing it, still fast
#
#     --spec mtp --draft-tokens 3 --lm-head-draft \
#     --prefill-cublas --prefill-chunk 2048 \
#     --kv-dtype rk4v4 --gdn-state-fp16 \
#     --vision --vision-residency overlay
#
#   MTP accepts NINFER_DRAFT_TOKENS up to 15. Three suits chat and prose; for coding work that
#   returns edited files, 11-15 decodes up to 1.85x faster (docs/performance.md has the table).
#
# rk4v4 KV (Lloyd-Max 4-bit keys) is 31% smaller than rk8v4 at the same decode speed, for +0.10%
# perplexity over it. Measured beside a desktop on an RTX 3090 (2026-09-24), the DFlash2 set starts
# at up to 180,224 tokens and needs about 1.45 GB more for 262,144 -- roughly what a headless card
# gets back -- so the full context below is the headless extrapolation, and the step-down catches
# a card that falls short. The mtp set starts at 262,144 with two lanes even beside a desktop.
# `none` is the mtp set without speculation.
# The qwen3_8_27b.ninfer that download-model.sh fetches is the DFlash2 bundle and carries the MTP
# weights too, so one file serves both.
#
# IF THE CARD IS BUSY. A desktop (or another job) holding VRAM can leave too little for the default
# context. When the `tuned` profile is refused at startup for lack of GPU memory, this launcher steps
# down on its own -- an eighth of the context at a time, up to five times, from the second step with a 2048 prefill chunk
# and a smaller Host context budget -- and says what it did, so the first run starts instead of ending in an
# error. It
# only does that for the defaults: an explicit NINFER_CONTEXT, NINFER_PREFILL_CHUNK,
# NINFER_HOST_CONTEXT_MIB or NINFER_KV_CAPACITY is honoured as given and fails loudly, and
# NINFER_FALLBACK=off turns the step-down off.
#
# OVERRIDES, from the environment. All profiles: NINFER_MODEL (artifact path), NINFER_MODEL_DIR,
# NINFER_SERVER, NINFER_HOST, NINFER_PORT, NINFER_CHAT_TEMPLATE (path to a local Jinja file, passed
# straight to --chat-template; overrides the artifact's built-in template). `tuned` also:
# NINFER_CONTEXT, NINFER_CONCURRENCY,
# NINFER_KV_CAPACITY, NINFER_KV_DTYPE, NINFER_SPEC, NINFER_DRAFT_TOKENS, NINFER_PREFILL_CHUNK,
# NINFER_VISION (on|off), NINFER_VISION_RESIDENCY, NINFER_HOST_CONTEXT_MIB (8192), NINFER_MIN_P (0.03) and
# NINFER_PRESENCE_PENALTY (0.5), the loop guard ("default" keeps the registered preset).
#
# SERVING KNOBS, `tuned` only. Each is passed to ninfer-serve only when set, so leaving them all
# unset keeps the profile exactly as measured; `ninfer-serve --help` says what each one does.
#   NINFER_AUTO_HOST_CACHE=on           --auto-host-cache              size the Host context budget from free
#                                                                      RAM instead of NINFER_HOST_CONTEXT_MIB
#     NINFER_HOST_CACHE_PERCENT         --host-cache-percent N         share of free RAM to use (1-100)
#     NINFER_HOST_CACHE_RESERVE_MIB     --host-cache-reserve-mib N     RAM to always leave free
#     NINFER_HOST_CACHE_MAX_MIB         --host-cache-max-mib N         hard cap on pinned RAM
#   NINFER_MAX_OUTPUT_TOKENS            --max-output-tokens N          cap every request's output budget
#   NINFER_LOOKUP_NGRAM                 --lookup-ngram N               context-lookup drafting beside --spec
#   NINFER_MLP_A8_DECODE=on             --mlp-a8-decode                INT8-activation MLP decode
#   NINFER_CONTEXT_STORE                --context-store DIR            keep the context cache across restarts
#     NINFER_CONTEXT_STORE_MAX_GIB      --context-store-max-gib N      disk budget for the store
#
# Each spec's defaults (context, lanes, chunk)
# are the ones that fit; the context figures below are extrapolated for a headless card, so treat
# the first start as the confirmation and drop a rung if it refuses:
# 229376 / 212992 / 196608 / 163840 / 131072 / 114688 / 98304 / 65536.
# ------------------------------------------------------------------------------------------------
set -euo pipefail

usage() {
  printf 'usage: %s <model> [profile]\n' "${0##*/}"
  printf '  qwen38-27b       tuned (default), int8, c8   (recommended)\n'
  printf '  qwen36-35b-a3b   tuned (default)\n'
}

model_key="${1:-}"
profile="${2:-tuned}"
case "$model_key" in
  -h|--help) usage; exit 0 ;;
  qwen38-27b)     artifact='qwen3_8_27b.ninfer';     title='Qwen3.8-27B' ;;
  qwen36-35b-a3b) artifact='qwen3_6_35b_a3b.ninfer'; title='Qwen3.6-35B-A3B' ;;
  '') usage >&2; exit 2 ;;
  *) printf 'Unknown model: %s\n' "$model_key" >&2; usage >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/.." && pwd)"
# Two layouts reach this script: a checkout, where the artifacts sit in the repository's own
# models/ directory beside build-linux/, and an unpacked release archive, where the launcher
# sits next to ninfer-serve and models/. Probe for the checkout first, then the archive -- the
# same order the server lookup below uses.
#
# An explicit NINFER_MODEL_DIR is taken verbatim and never probed. Falling back past a
# directory the caller named turns their typo into a 'Missing model' error about a path they
# never mentioned, which is worse than failing on the one they did.
if [[ -n "${NINFER_MODEL_DIR:-}" ]]; then
  model_dir="$NINFER_MODEL_DIR"
else
  # Test for the artifact, not merely for a models/ directory. An archive unpacked below a
  # directory that happens to have its own models/ would otherwise stop at the parent and
  # never look beside the launcher, failing while its own artifact sits right there.
  model_dir="$root/models"
  [[ -f "$model_dir/$artifact" ]] || model_dir="$script_dir/models"
fi
MODEL="${NINFER_MODEL:-$model_dir/$artifact}"
HOST="${NINFER_HOST:-127.0.0.1}"
PORT="${NINFER_PORT:-8080}"

server="${NINFER_SERVER:-$root/build-linux/apps/ninfer-serve}"
# Same two layouts as the artifact lookup above: build tree in a checkout, then the archive
# root, where this launcher sits beside the binary.
[[ -x "$server" ]] || server="$script_dir/ninfer-serve"

# The profile fixes the whole serving shape. `label` is the banner; `profile_args` is everything
# after --host/--port.
label=''
profile_args=()
case "$model_key/$profile" in
  qwen38-27b/tuned)
    # The speculative backend fixes everything that has to move with it: the prefill chunk (the
    # cuBLAS route's workspace scales with it) and the context and lanes that fit. The artifact
    # stores the embedding as Q4 and the head as Q6, so no load-time transcode is passed.
    SPEC="${NINFER_SPEC:-dflash2}"
    case "$SPEC" in
      dflash2)
        spec_args=(--spec dflash2 --draft-tokens "${NINFER_DRAFT_TOKENS:-7}" --lm-head-draft)
        default_context=262144; default_concurrency=1; default_chunk=4096
        spec_label="DFlash2 K=${NINFER_DRAFT_TOKENS:-7} + draft head" ;;
      mtp)
        spec_args=(--spec mtp --draft-tokens "${NINFER_DRAFT_TOKENS:-3}" --lm-head-draft)
        default_context=262144; default_concurrency=2; default_chunk=2048
        spec_label="MTP${NINFER_DRAFT_TOKENS:-3} + draft head, two lanes" ;;
      none)
        spec_args=()
        default_context=262144; default_concurrency=2; default_chunk=2048
        spec_label='no speculation' ;;
      *) printf 'NINFER_SPEC must be dflash2, mtp or none, got %s\n' "$SPEC" >&2; exit 2 ;;
    esac
    CONTEXT="${NINFER_CONTEXT:-$default_context}"
    CONCURRENCY="${NINFER_CONCURRENCY:-$default_concurrency}"
    KV_CAPACITY="${NINFER_KV_CAPACITY:-$CONTEXT}"
    KV_DTYPE="${NINFER_KV_DTYPE:-rk4v4}"
    PREFILL_CHUNK="${NINFER_PREFILL_CHUNK:-$default_chunk}"
    profile_args=(
      --max-concurrency "$CONCURRENCY" --max-context "$CONTEXT" --kv-capacity "$KV_CAPACITY"
      --kv-dtype "$KV_DTYPE"
      ${spec_args[@]+"${spec_args[@]}"}
      --gdn-state-fp16
      --prefill-cublas --prefill-chunk "$PREFILL_CHUNK"
    )
    label="C$CONCURRENCY  |  context $CONTEXT  |  KV pool $KV_CAPACITY  |  $KV_DTYPE  |  $spec_label"
    prefill_note="Prefill: cuBLAS route, chunk $PREFILL_CHUNK"
    if [[ "$SPEC" == 'dflash2' ]]; then
      hint="Need a second lane?  NINFER_SPEC=mtp ${0##*/} $model_key  (slower decode)"
    fi ;;

  qwen36-35b-a3b/tuned)
    SPEC="${NINFER_SPEC:-mtp}"
    case "$SPEC" in
      mtp)  spec_args=(--spec mtp --draft-tokens "${NINFER_DRAFT_TOKENS:-3}" --lm-head-draft --mtp-experts-q4)
            spec_label="MTP${NINFER_DRAFT_TOKENS:-3} + draft head" ;;
      none) spec_args=(); spec_label='no speculation' ;;
      *) printf 'NINFER_SPEC must be mtp or none, got %s\n' "$SPEC" >&2; exit 2 ;;
    esac
    # rk4v4 fits three lanes at the full context even beside a desktop (2026-09-24; four fall 48 MB
    # short there, so a headless card may take NINFER_CONCURRENCY=4). Lanes share the one
    # --kv-capacity pool: any request may use all of it, but not every lane at once.
    CONTEXT="${NINFER_CONTEXT:-262144}"
    CONCURRENCY="${NINFER_CONCURRENCY:-3}"
    KV_CAPACITY="${NINFER_KV_CAPACITY:-$CONTEXT}"
    KV_DTYPE="${NINFER_KV_DTYPE:-rk4v4}"
    # cuBLAS prefill at chunk 4096: 8,848 tok/s on a 4K prompt against 5,470 at the old
    # default-route chunk 512 (2026-10-02), for about 240 MiB more workspace.
    PREFILL_CHUNK="${NINFER_PREFILL_CHUNK:-4096}"
    profile_args=(
      --max-concurrency "$CONCURRENCY" --max-context "$CONTEXT" --kv-capacity "$KV_CAPACITY"
      --kv-dtype "$KV_DTYPE"
      ${spec_args[@]+"${spec_args[@]}"}
      --gdn-state-fp16 --prefill-cublas --prefill-chunk "$PREFILL_CHUNK"
    )
    label="C$CONCURRENCY  |  context $CONTEXT  |  KV pool $KV_CAPACITY  |  $KV_DTYPE  |  $spec_label" ;;

  qwen38-27b/int8)
    profile_args=(
      --max-context 65536 --kv-capacity 65536
      --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000
      --prefill-chunk 1024 --kv-dtype int8
      --spec mtp --draft-tokens 3 --lm-head-draft
    )
    label='one request  |  64K context  |  INT8 KV  |  MTP3, ReplaySSM' ;;

  qwen38-27b/c8)
    # Context cache sized per lane, so several agents rotating through the lanes find their own
    # conversation still cached instead of re-prefilling it: one checkpoint StateImage per lane on
    # the card beyond the active ones, and the same 8 GiB pinned Host context budget as `tuned` for
    # the conversations that leave the card. MEMORY COST: this profile keeps the GDN state in BF16,
    # so a StateImage is 147 MiB. The device slots take 8 x 147 MiB = 1.15 GiB of VRAM, the engine
    # default at eight lanes; the Host budget pins 8 GiB of RAM (StateImages, KV pages and pause
    # snapshots share it). Retention on the card is still bounded by the 16,384-token KV pool.
    c8_lanes=8
    profile_args=(
      --max-context 8192 --kv-capacity 16384
      --max-concurrency "$c8_lanes" --max-pending-requests 32 --pending-timeout-ms 600000
      --prefill-chunk 512 --kv-dtype int8
      --spec mtp --draft-tokens 3 --lm-head-draft
      --device-state-slots "$c8_lanes" --host-context-mib 8192
    )
    label='up to eight requests  |  8K context  |  INT8 KV  |  MTP3, ReplaySSM' ;;

  *)
    printf 'Model %s has no profile %s\n' "$model_key" "$profile" >&2
    usage >&2
    exit 2 ;;
esac

# Only `tuned` carries vision and the sampling guard: the reference profiles are deliberately
# minimal. Vision stays on there -- overlay residency keeps the tower host-pinned and streams each
# image through a borrowed device window, so it costs about 10 MiB of runtime reservation.
if [[ "$profile" == 'tuned' ]]; then
  VISION="${NINFER_VISION:-on}"
  case "$VISION" in
    on)  vision_args=(--vision --vision-residency "${NINFER_VISION_RESIDENCY:-overlay}")
         vision_label="vision (${NINFER_VISION_RESIDENCY:-overlay})" ;;
    off) vision_args=(); vision_label='text only' ;;
    *) printf 'NINFER_VISION must be on or off, got %s\n' "$VISION" >&2; exit 2 ;;
  esac
  label="$label  |  $vision_label"
  # Pinned Host context budget, in MiB: one byte budget that retained StateImages (74.5 MiB each on
  # the 27B with the FP16 state), KV pages and pause snapshots share once they leave the card. It is
  # host RAM, not device memory, pinned in full at startup. Lower it if the box is short on RAM; 0
  # keeps everything on the card.
  HOST_CONTEXT_MIB="${NINFER_HOST_CONTEXT_MIB:-8192}"
  profile_args+=(
    --max-pending-requests 16 --pending-timeout-ms 600000
    ${vision_args[@]+"${vision_args[@]}"}
  )
  # --auto-host-cache sizes that budget from the RAM still free once the model has loaded, and
  # refuses --host-context-mib, so it is either/or: with it on, NINFER_HOST_CONTEXT_MIB (and the
  # step-down ladder's smaller budget) is ignored.
  AUTO_HOST_CACHE="${NINFER_AUTO_HOST_CACHE:-off}"
  case "$AUTO_HOST_CACHE" in
    on)
      profile_args+=(--auto-host-cache)
      [[ -z "${NINFER_HOST_CACHE_PERCENT:-}" ]] || profile_args+=(--host-cache-percent "$NINFER_HOST_CACHE_PERCENT")
      [[ -z "${NINFER_HOST_CACHE_RESERVE_MIB:-}" ]] || profile_args+=(--host-cache-reserve-mib "$NINFER_HOST_CACHE_RESERVE_MIB")
      [[ -z "${NINFER_HOST_CACHE_MAX_MIB:-}" ]] || profile_args+=(--host-cache-max-mib "$NINFER_HOST_CACHE_MAX_MIB")
      cache_note='Context cache: sized automatically from free host RAM  (NINFER_AUTO_HOST_CACHE)' ;;
    off)
      if [[ -n "${NINFER_HOST_CACHE_PERCENT:-}${NINFER_HOST_CACHE_RESERVE_MIB:-}${NINFER_HOST_CACHE_MAX_MIB:-}" ]]; then
        printf 'NINFER_HOST_CACHE_* needs NINFER_AUTO_HOST_CACHE=on\n' >&2; exit 2
      fi
      profile_args+=(--host-context-mib "$HOST_CONTEXT_MIB")
      cache_note="Context cache: $HOST_CONTEXT_MIB MiB pinned host RAM  (NINFER_HOST_CONTEXT_MIB)" ;;
    *) printf 'NINFER_AUTO_HOST_CACHE must be on or off, got %s\n' "$AUTO_HOST_CACHE" >&2; exit 2 ;;
  esac
  # Opt-in serving knobs: each is appended only when set, so the defaults above are untouched.
  [[ -z "${NINFER_MAX_OUTPUT_TOKENS:-}" ]] || profile_args+=(--max-output-tokens "$NINFER_MAX_OUTPUT_TOKENS")
  [[ -z "${NINFER_LOOKUP_NGRAM:-}" ]] || profile_args+=(--lookup-ngram "$NINFER_LOOKUP_NGRAM")
  case "${NINFER_MLP_A8_DECODE:-off}" in
    on) profile_args+=(--mlp-a8-decode) ;;
    off) ;;
    *) printf 'NINFER_MLP_A8_DECODE must be on or off, got %s\n' "$NINFER_MLP_A8_DECODE" >&2; exit 2 ;;
  esac
  if [[ -n "${NINFER_CONTEXT_STORE:-}" ]]; then
    profile_args+=(--context-store "$NINFER_CONTEXT_STORE")
    [[ -z "${NINFER_CONTEXT_STORE_MAX_GIB:-}" ]] || profile_args+=(--context-store-max-gib "$NINFER_CONTEXT_STORE_MAX_GIB")
  elif [[ -n "${NINFER_CONTEXT_STORE_MAX_GIB:-}" ]]; then
    printf 'NINFER_CONTEXT_STORE_MAX_GIB needs NINFER_CONTEXT_STORE\n' >&2; exit 2
  fi
  # Loop guard for the small quant: a mild min-p trims the noisy token tail, and a mild presence
  # penalty breaks repetition loops in the reasoning and the answer. These are process-level
  # overrides, so they replace the registered presets in both thinking and non-thinking mode
  # (thinking presence 0, non-thinking 1.5); a request that sets its own value still wins. Kept
  # low because code legitimately repeats identifiers and syntax -- a penalty near 1.5 or any
  # frequency penalty damages generated code. Temperature, top-p and top-k stay at the registered
  # values. "default" omits the flag and leaves the registered preset in force.
  MIN_P="${NINFER_MIN_P:-0.03}"
  PRESENCE_PENALTY="${NINFER_PRESENCE_PENALTY:-0.5}"
  [[ "$MIN_P" == 'default' ]] || profile_args+=(--min-p "$MIN_P")
  [[ "$PRESENCE_PENALTY" == 'default' ]] || profile_args+=(--presence-penalty "$PRESENCE_PENALTY")
fi

if [[ -n "${NINFER_CHAT_TEMPLATE:-}" ]]; then
  profile_args+=(--chat-template "$NINFER_CHAT_TEMPLATE")
fi

if [[ ! -x "$server" ]]; then
  printf 'Missing ninfer-serve (looked for %s)\n' "$server" >&2
  printf 'Build it first:  ./scripts/build.sh\n' >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  printf 'Missing model: %s\n' "$MODEL" >&2
  printf 'Download it first:  ./download-model.sh %s\n' "$model_key" >&2
  exit 1
fi

printf '%s  |  %s\n' "$title" "$label"
[[ -z "${prefill_note:-}" ]] || printf '%s\n' "$prefill_note"
if [[ "$profile" == 'tuned' ]]; then
  printf '%s\n' "$cache_note"
  printf 'Sampling guard: min-p %s, presence penalty %s  (NINFER_MIN_P, NINFER_PRESENCE_PENALTY; "default" = registered preset)\n' \
    "$MIN_P" "$PRESENCE_PENALTY"
fi
[[ -z "${NINFER_CHAT_TEMPLATE:-}" ]] || printf 'Chat template: %s\n' "$NINFER_CHAT_TEMPLATE"
[[ -z "${hint:-}" ]] || printf '%s\n' "$hint"
printf 'API: http://%s:%s/v1\n\n' "$HOST" "$PORT"

# Step-down eligibility. Only the defaults of the tuned profile may be second-guessed; a value the
# caller chose is theirs. A step-down pass is a re-run of this script with the overrides below set,
# marked by NINFER_FALLBACK_RUNG, so the settings are rebuilt by the same code as a first run.
ladder=0
if [[ "$profile" == 'tuned' && "${NINFER_FALLBACK:-on}" != 'off' ]]; then
  if [[ -n "${NINFER_FALLBACK_RUNG:-}" ]]; then
    ladder=1
  elif [[ -z "${NINFER_CONTEXT:-}${NINFER_PREFILL_CHUNK:-}${NINFER_HOST_CONTEXT_MIB:-}${NINFER_KV_CAPACITY:-}" ]]; then
    ladder=1
  fi
fi

# --host-context-mib 8192 really pins 8 GiB of host RAM, here and in run.bat alike (see
# docs/maintainer/launcher-profiles.md).
if (( ! ladder )); then
  exec "$server" "$MODEL" --host "$HOST" --port "$PORT" "${profile_args[@]}"
fi

# Run the server with its output shown and kept, so a refusal for lack of memory can be told apart
# from any other failure. Only that failure steps down; a crash or a bad artifact does not.
rung="${NINFER_FALLBACK_RUNG:-0}"
base_context="${NINFER_FALLBACK_BASE_CONTEXT:-$CONTEXT}"
base_chunk="${NINFER_FALLBACK_BASE_CHUNK:-$PREFILL_CHUNK}"
base_host_mib="${NINFER_FALLBACK_BASE_HOST_MIB:-$HOST_CONTEXT_MIB}"
server_log="$(mktemp)"
trap 'rm -f -- "$server_log"' EXIT
set +e
"$server" "$MODEL" --host "$HOST" --port "$PORT" "${profile_args[@]}" 2>&1 | tee "$server_log"
status="${PIPESTATUS[0]}"
set -e
if (( status != 0 && rung < 5 )) &&
   grep -q -E 'runtime reservation requires|cudaMallocHost failed' "$server_log"; then
  next=$((rung + 1))
  next_context=$(( base_context * (8 - next) / 8 / 1024 * 1024 ))
  # The first step trims context only: an eighth of it frees more than a card that just misses
  # needs, and prefill speed and cached prefixes are worth keeping. Later steps also give up the
  # wider prefill chunk and halve the Host context budget every other step (floor 0).
  next_chunk=$base_chunk
  (( next < 2 || base_chunk <= 2048 )) || next_chunk=2048
  next_host_mib=$(( base_host_mib >> (next / 2) ))
  (( next_host_mib >= 0 )) || next_host_mib=0
  printf '\nNot enough free memory to start at context %s. Retrying at %s (prefill chunk %s, %s MiB host context).\n' \
    "$CONTEXT" "$next_context" "$next_chunk" "$next_host_mib"
  printf 'Set NINFER_CONTEXT to choose your own, or NINFER_FALLBACK=off to fail instead.\n\n'
  exec env NINFER_CONTEXT="$next_context" NINFER_PREFILL_CHUNK="$next_chunk" \
    NINFER_HOST_CONTEXT_MIB="$next_host_mib" NINFER_FALLBACK_RUNG="$next" \
    NINFER_FALLBACK_BASE_CONTEXT="$base_context" NINFER_FALLBACK_BASE_CHUNK="$base_chunk" \
    NINFER_FALLBACK_BASE_HOST_MIB="$base_host_mib" "$0" "$@"
fi
exit "$status"
