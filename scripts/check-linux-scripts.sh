#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

for script in "$root"/*.sh; do
  bash -n "$script"
done

# CRLF in a shell script is not cosmetic. Inside a `case` block the stray CR becomes part of the
# `in` token and Linux bash refuses the file outright -- which is exactly what happened to the two
# launchers the README recommended as the Linux entry point. `bash -n` above catches that particular shape, but a CRLF
# script without a `case` parses and then misbehaves at runtime instead, so check the bytes.
#
# -U forces binary matching. Without it, Git Bash's grep translates CR away and reports every file
# clean on a Windows checkout, which is how this survived as long as it did.
#
# Scoped to the whole repository rather than scripts/, because .gitattributes states the policy
# repository-wide and eval/ already holds three shell scripts that a scripts/-only check would
# never look at. Driven off `git ls-files` so a new directory is covered the day it appears
# instead of the day someone remembers to add a glob here.
repo_root="$(git -C "$root" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$root/..")"
crlf=''
while IFS= read -r candidate; do
  [[ -f "$repo_root/$candidate" ]] || continue
  if LC_ALL=C grep -qU $'\r' "$repo_root/$candidate"; then
    crlf+="  $candidate"$'\n'
  fi
done < <(git -C "$repo_root" ls-files '*.sh' '*.bash' 2>/dev/null)

if [[ -n "$crlf" ]]; then
  printf 'Shell scripts with CRLF line endings (must be LF; see .gitattributes):\n%s' "$crlf" >&2
  printf 'Linux bash rejects these outright inside a `case` block.\n' >&2
  exit 1
fi
# Every Windows script needs a Bash sibling. There used to be an exemption list here; the
# counterpart was written instead, so the list is gone rather than empty. An exemption list is where the next Windows-only script quietly goes.
#
# Accumulate rather than exiting on the first miss. This loop used to `exit 1` immediately, and
# because it was failing on a cosmetic counterpart rule, none of the downloader coverage below it
# ran at all -- which is how the curl fixture further down went stale unnoticed. A guard that
# stops at its first complaint disables everything after it.
missing=''
for windows_script in "$root"/*.bat "$root"/*.ps1; do
  counterpart="${windows_script%.*}.sh"
  [[ -f "$counterpart" ]] || missing+="  $(basename -- "$counterpart")"$'
'
done
if [[ -n "$missing" ]]; then
  printf 'Missing Bash counterpart for a Windows script:
%s' "$missing" >&2
  exit 1
fi

# The executable bit, read from the index rather than the filesystem. On a Windows checkout every
# file looks executable to bash, and under WSL /mnt/c reports 777 for everything, so `-x` on the
# working tree cannot see a mode-644 script here at all -- while a real Linux checkout gets a file
# that will not run. A release's Linux packager once sat at 644 for exactly that reason. `git ls-files -s` reports what is committed, which is the thing
# that actually reaches a Linux user.
not_executable=''
while IFS= read -r entry; do
  mode="${entry%% *}"
  path="${entry#*$'	'}"
  [[ "$mode" == '100755' ]] || not_executable+="  $path (mode $mode)"$'
'
done < <(git -C "$repo_root" ls-files -s '*.sh' '*.bash' 2>/dev/null)
if [[ -n "$not_executable" ]]; then
  printf 'Shell scripts committed without the executable bit:
%s' "$not_executable" >&2
  printf 'Fix with: git update-index --chmod=+x <path>
' >&2
  exit 1
fi

cat > "$tmp/ninfer-serve" <<'SERVER'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$NINFER_TEST_ARGS"
SERVER
chmod +x "$tmp/ninfer-serve"
touch "$tmp/qwen3_8_27b.ninfer" "$tmp/qwen3_6_35b_a3b.ninfer"

# Hermetic fixtures. A developer's shell may already export NINFER_MODEL, NINFER_HOST and friends
# -- the machine this was written on exports both -- and inheriting them makes every assertion below
# describe that box rather than the layout under test. The first draft of the archive case failed
# exactly that way, against a NINFER_MODEL naming a directory that no longer exists. Clear every
# NINFER_* and pass back only what a case sets deliberately.
ninfer_clear=()
while IFS= read -r name; do
  ninfer_clear+=(-u "$name")
done < <(env | sed -n 's/^\(NINFER_[A-Za-z0-9_]*\)=.*/\1/p')
clear_env() { env ${ninfer_clear[@]+"${ninfer_clear[@]}"} "$@"; }

# scripts/run.sh serves every shipped profile, chosen by `run.sh <model> [profile]`. The README and
# the launcher's own header tell users which flag set to reach for, so pin each set as a whole: a
# flag dropped from one -- or a load-time transcode the 27B artifact no longer needs (it stores the
# embedding as Q4 and the head as Q6) -- fails here instead of at a user's first start.
record() { # record <label> [NAME=value ...] -- <run.sh arguments>; prints the recorded server args
  local label="$1"; shift
  local recorded="$tmp/run.$label.args" assignments=()
  while [[ "$1" != '--' ]]; do assignments+=("$1"); shift; done
  shift
  clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$recorded" NINFER_MODEL_DIR="$tmp" \
    ${assignments[@]+"${assignments[@]}"} "$root/run.sh" "$@" >/dev/null
  printf '%s' "$recorded"
}
expect_flags() { # expect_flags <label> <recorded> <flag sequence>...: each appears, in order, adjacent
  local label="$1" recorded="$2" flat sequence; shift 2
  flat=" $(tr '\n' ' ' < "$recorded")"
  for sequence in "$@"; do
    [[ "$flat" == *" $sequence "* ]] && continue
    printf 'run.sh (%s) is missing "%s"\nargs:%s\n' "$label" "$sequence" "$flat" >&2
    exit 1
  done
}
refuse_flag() { # refuse_flag <label> <recorded> <flag>
  if [[ " $(tr '\n' ' ' < "$2")" == *" $3 "* ]]; then
    printf 'run.sh (%s) must not pass %s\n' "$1" "$3" >&2
    exit 1
  fi
}
expect_exit() { # expect_exit <status> <label> <command...>: the command exits with exactly <status>
  local want="$1" label="$2" got=0; shift 2
  "$@" >/dev/null 2>&1 || got=$?
  [[ "$got" == "$want" ]] && return 0
  printf 'run.sh (%s) exited %s, expected %s\n' "$label" "$got" "$want" >&2
  exit 1
}

# Qwen3.8-27B `tuned`: the two measured flag sets and the speculation-free variant.
recorded="$(record dflash2 -- qwen38-27b)"
expect_flags '27B default' "$recorded" \
  '--spec dflash2 --draft-tokens 7 --lm-head-draft' \
  '--prefill-cublas --prefill-chunk 4096' \
  '--kv-dtype rk4v4' '--gdn-state-fp16' \
  '--vision --vision-residency overlay' '--max-context 262144' '--max-concurrency 1'
refuse_flag '27B default' "$recorded" '--lm-head-q6'
refuse_flag '27B default' "$recorded" '--embedding-q4'

recorded="$(record mtp NINFER_SPEC=mtp -- qwen38-27b tuned)"
expect_flags '27B NINFER_SPEC=mtp' "$recorded" \
  '--spec mtp --draft-tokens 3 --lm-head-draft' \
  '--prefill-cublas --prefill-chunk 2048' \
  '--kv-dtype rk4v4' '--gdn-state-fp16' \
  '--vision --vision-residency overlay' '--max-context 262144' '--max-concurrency 2'
refuse_flag '27B NINFER_SPEC=mtp' "$recorded" '--lm-head-q6'
refuse_flag '27B NINFER_SPEC=mtp' "$recorded" '--embedding-q4'

recorded="$(record none NINFER_SPEC=none -- qwen38-27b)"
expect_flags '27B NINFER_SPEC=none' "$recorded" '--prefill-cublas --prefill-chunk 2048'
refuse_flag '27B NINFER_SPEC=none' "$recorded" '--embedding-q4'
refuse_flag '27B NINFER_SPEC=none' "$recorded" '--spec'
refuse_flag '27B NINFER_SPEC=none' "$recorded" '--lm-head-q6'

# An explicit override beats the profile's default; vision can be turned off.
recorded="$(record override NINFER_CONTEXT=65536 NINFER_PREFILL_CHUNK=1024 NINFER_CONCURRENCY=2 \
  NINFER_VISION=off -- qwen38-27b)"
expect_flags '27B overrides' "$recorded" '--max-context 65536' '--prefill-chunk 1024' '--max-concurrency 2'
refuse_flag '27B overrides' "$recorded" '--vision'

# The Host context budget is the pinned-RAM override a box short on memory needs, and its default
# must not move.
expect_flags '27B default host context' "$(record host8192 -- qwen38-27b)" '--host-context-mib 8192'
expect_flags '27B host context override' "$(record host1024 NINFER_HOST_CONTEXT_MIB=1024 -- qwen38-27b)" '--host-context-mib 1024'

# NINFER_CHAT_TEMPLATE forwards straight to --chat-template, and is absent when unset.
expect_flags '27B chat template override' \
  "$(record chat_template NINFER_CHAT_TEMPLATE="$tmp/custom.jinja" -- qwen38-27b)" \
  "--chat-template $tmp/custom.jinja"
refuse_flag '27B no chat template override' "$(record no_chat_template -- qwen38-27b)" '--chat-template'

# The banner reports what is being served: an explicit vision residency shows up in it, not the
# default. (The stub server prints nothing, so stdout here is the launcher's own.)
banner="$(clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$tmp/unused.args" \
  NINFER_MODEL_DIR="$tmp" NINFER_VISION_RESIDENCY=resident "$root/run.sh" qwen38-27b)"
[[ "$banner" == *'vision (resident)'* && "$banner" != *'vision (overlay)'* ]] || {
  printf 'run.sh banner ignored NINFER_VISION_RESIDENCY: %s\n' "$banner" >&2
  exit 1
}

# The reference profiles are fixed and minimal: no vision, and only c8 sizes its context cache,
# per lane (two retained conversations and two host states per lane, one device state per lane).
recorded="$(record int8 -- qwen38-27b int8)"
expect_flags '27B int8' "$recorded" '--max-context 65536' '--max-concurrency 1' '--kv-dtype int8' \
  '--spec mtp --draft-tokens 3 --lm-head-draft'
refuse_flag '27B int8' "$recorded" '--vision'
recorded="$(record c8 -- qwen38-27b c8)"
expect_flags '27B c8' "$recorded" '--max-concurrency 8' '--max-context 8192' '--kv-capacity 16384' \
  '--kv-dtype int8' '--spec mtp --draft-tokens 3 --lm-head-draft' \
  '--device-state-slots 8' '--host-context-mib 8192'
refuse_flag '27B c8' "$recorded" '--vision'

# Qwen3.6-35B-A3B `tuned`: the cuBLAS prefill route at chunk 4096 (it engages on the 35B's dense
# projections from chunk 2048), no Q4 embedding, and its own draft head flags.
recorded="$(record 35b -- qwen36-35b-a3b)"
expect_flags '35B default' "$recorded" \
  '--spec mtp --draft-tokens 3 --lm-head-draft --mtp-experts-q4' \
  '--kv-dtype rk4v4' '--gdn-state-fp16' '--prefill-cublas --prefill-chunk 4096' \
  '--vision --vision-residency overlay' '--max-context 262144' '--max-concurrency 3' '--host-context-mib 8192'
refuse_flag '35B default' "$recorded" '--embedding-q4'
recorded="$(record 35bnone NINFER_SPEC=none -- qwen36-35b-a3b)"
refuse_flag '35B NINFER_SPEC=none' "$recorded" '--spec'

# Bad input is refused rather than falling through to a profile: no model, an unknown model, a
# profile the model does not have, an unknown speculation setting -- and --help succeeds.
serve_env=(NINFER_SERVER="$tmp/ninfer-serve" NINFER_MODEL_DIR="$tmp" NINFER_TEST_ARGS="$tmp/unused.args")
expect_exit 2 'no model' clear_env "${serve_env[@]}" "$root/run.sh"
expect_exit 2 'unknown model' clear_env "${serve_env[@]}" "$root/run.sh" gpt
expect_exit 2 'profile the model lacks' clear_env "${serve_env[@]}" "$root/run.sh" qwen36-35b-a3b c8
expect_exit 2 'unknown profile' clear_env "${serve_env[@]}" "$root/run.sh" qwen38-27b fastest
expect_exit 2 'unknown NINFER_SPEC' clear_env "${serve_env[@]}" NINFER_SPEC=bogus "$root/run.sh" qwen38-27b
expect_exit 0 '--help' clear_env "${serve_env[@]}" "$root/run.sh" --help

# Step-down ladder. A desktop holding VRAM can leave too little for the default context, and the
# tuned profile is meant to start anyway: refused for lack of memory, it retries with a smaller
# context, a 2048 chunk and fewer host state slots. The stub refuses (with the engine's own message)
# until the requested context fits, and logs every attempt it is given.
cat > "$tmp/ninfer-serve-tight" <<'TIGHT'
#!/usr/bin/env bash
context=''; previous=''
for argument in "$@"; do [[ "$previous" == '--max-context' ]] && context="$argument"; previous="$argument"; done
printf '%s\n' "$*" >> "$NINFER_TEST_LOG"
if (( context > ${NINFER_TEST_FITS:-0} )); then
  echo "${NINFER_TEST_FAILURE:-FATAL server failed during startup | requested Engine runtime reservation requires 5173992960 bytes, but only 1 bytes are available}"
  exit 1
fi
TIGHT
chmod +x "$tmp/ninfer-serve-tight"
attempts() { # attempts <fits> [NAME=value ...] -> log of every attempt; sets $ladder_status
  local fits="$1"; shift
  ladder_log="$tmp/ladder.$RANDOM.log"; : > "$ladder_log"
  ladder_status=0
  clear_env NINFER_SERVER="$tmp/ninfer-serve-tight" NINFER_MODEL_DIR="$tmp" NINFER_TEST_LOG="$ladder_log" \
    NINFER_TEST_FITS="$fits" ${@+"$@"} "$root/run.sh" qwen38-27b >/dev/null 2>&1 || ladder_status=$?
}
field() { # field <flag>: that flag's value on each logged attempt, space separated
  awk -v flag="$1" '{ for (i = 1; i < NF; i++) if ($i == flag) printf "%s ", $(i + 1) } END { print "" }' "$ladder_log"
}
expect_eq() { # expect_eq <label> <got> <want>
  [[ "$2" == "$3" ]] && return 0
  printf 'run.sh ladder (%s): got "%s", expected "%s"\n' "$1" "$2" "$3" >&2
  exit 1
}

attempts 100000
expect_eq 'contexts stepped' "$(field --max-context)" '262144 229376 196608 163840 131072 98304 '
expect_eq 'chunk drops to 2048' "$(field --prefill-chunk)" '4096 4096 2048 2048 2048 2048 '
expect_eq 'host context halves' "$(field --host-context-mib)" '8192 8192 4096 4096 2048 2048 '
expect_eq 'it started, so it exits 0' "$ladder_status" '0'

attempts 1000
expect_eq 'never fits: six attempts, then stops' "$(field --max-context)" '262144 229376 196608 163840 131072 98304 '
expect_eq 'never fits: reports the failure' "$ladder_status" '1'

# Values the caller chose are theirs, and a switch turns the ladder off: one attempt, loud failure.
attempts 1000 NINFER_CONTEXT=131072
expect_eq 'explicit context is not second-guessed' "$(field --max-context)" '131072 '
attempts 1000 NINFER_HOST_CONTEXT_MIB=8192
expect_eq 'explicit host context is not second-guessed' "$(field --max-context)" '262144 '
attempts 1000 NINFER_FALLBACK=off
expect_eq 'NINFER_FALLBACK=off' "$(field --max-context)" '262144 '

# Only a memory refusal steps down. Any other startup failure is not something a smaller context fixes.
attempts 1000 NINFER_TEST_FAILURE='FATAL server failed during startup | artifact is corrupt'
expect_eq 'other failures do not retry' "$(field --max-context)" '262144 '

# NINFER_CHAT_TEMPLATE plays no part in the step-down math, but it must still ride along on every
# retry -- a value set once should not silently drop off a later rung.
attempts 1000 NINFER_CHAT_TEMPLATE="$tmp/custom.jinja"
expect_eq 'chat template rides every rung' "$(field --chat-template)" \
  "$tmp/custom.jinja $tmp/custom.jinja $tmp/custom.jinja $tmp/custom.jinja $tmp/custom.jinja $tmp/custom.jinja "

# The reference profiles are fixed shapes and never step down.
ladder_log="$tmp/ladder.fixed.log"; : > "$ladder_log"
clear_env NINFER_SERVER="$tmp/ninfer-serve-tight" NINFER_MODEL_DIR="$tmp" NINFER_TEST_LOG="$ladder_log" \
  NINFER_TEST_FITS=1000 "$root/run.sh" qwen38-27b int8 >/dev/null 2>&1 || true
expect_eq 'int8 profile has no ladder' "$(field --max-context)" '65536 '

# run.sh ships *inside* the release archive as well as living here, and README calls it the Linux
# entry point. It once resolved both the server and the artifact from `dirname(script)/..` -- the
# archive's parent once packaged -- so the documented quick start exited before serving, with a
# "Missing ninfer-serve" naming a path outside the archive. Nothing tested that, because every case
# above hands the launcher an explicit NINFER_SERVER and model directory.
#
# So drive it the way a user does: unpack and run, flat layout, no environment overrides beyond the
# stub hook. The launcher must find the binary beside itself and the artifact under its own
# models/, not one directory up.
archive="$tmp/archive"
mkdir -- "$archive" "$archive/models"
cp -- "$tmp/ninfer-serve" "$archive/ninfer-serve"
cp -- "$root/run.sh" "$archive/run.sh"
for entry in 'qwen38-27b:qwen3_8_27b.ninfer' 'qwen36-35b-a3b:qwen3_6_35b_a3b.ninfer'; do
  IFS=: read -r key model <<< "$entry"
  : > "$archive/models/$model"
  args="$tmp/run.$key.archive.args"
  ( cd -- "$archive" && clear_env NINFER_TEST_ARGS="$args" ./run.sh "$key" >/dev/null )
  if ! grep -Fx -- "$archive/models/$model" "$args" >/dev/null; then
    printf 'run.sh %s did not resolve its artifact inside the archive: %s\n' "$key" "$(head -1 -- "$args")" >&2
    exit 1
  fi
done

# The probe must select the candidate that holds the artifact, not the first models/ directory that
# happens to exist. An archive unpacked below a directory with its own (empty) models/ would
# otherwise stop at the parent and fail while its own artifact sat beside the launcher.
decoy="$tmp/decoy"
mkdir -p -- "$decoy/models" "$decoy/inner/models"
cp -- "$tmp/ninfer-serve" "$decoy/inner/ninfer-serve"
: > "$decoy/inner/models/qwen3_8_27b.ninfer"
cp -- "$root/run.sh" "$decoy/inner/run.sh"
args="$tmp/run.decoy.args"
( cd -- "$decoy/inner" && clear_env NINFER_TEST_ARGS="$args" ./run.sh qwen38-27b >/dev/null )
if ! grep -Fx -- "$decoy/inner/models/qwen3_8_27b.ninfer" "$args" >/dev/null; then
  printf 'run.sh stopped at an empty parent models/: %s\n' "$(head -1 -- "$args")" >&2
  exit 1
fi

# An explicit NINFER_MODEL_DIR must be honoured verbatim, never probed past. The two-candidate
# fallback above is for when the caller said nothing; applying it to a directory the caller *named*
# reports a missing artifact under a path they never mentioned, and hides their typo. Assert the
# error names the directory that was actually asked for.
err="$tmp/explicit-dir.err"
if ( cd -- "$archive" && clear_env NINFER_MODEL_DIR="$tmp/nope" NINFER_TEST_ARGS="$tmp/unused.args" \
       ./run.sh qwen38-27b >/dev/null 2>"$err" ); then
  printf 'run.sh accepted a nonexistent NINFER_MODEL_DIR\n' >&2
  exit 1
fi
if ! grep -Fq -- "$tmp/nope/qwen3_8_27b.ninfer" "$err"; then
  printf 'run.sh ignored an explicit NINFER_MODEL_DIR: %s\n' "$(head -1 -- "$err")" >&2
  exit 1
fi

# The other half of the two-candidate lookup: a checkout must still prefer its own build tree and
# models/ directory. This is the case that regresses if someone "simplifies" the fallback away.
checkout="$tmp/checkout"
mkdir -p -- "$checkout/scripts" "$checkout/build-linux/apps" "$checkout/models"
cp -- "$tmp/ninfer-serve" "$checkout/build-linux/apps/ninfer-serve"
: > "$checkout/models/qwen3_8_27b.ninfer"
cp -- "$root/run.sh" "$checkout/scripts/run.sh"
args="$tmp/run.checkout.args"
( cd -- "$checkout" && clear_env NINFER_TEST_ARGS="$args" ./scripts/run.sh qwen38-27b >/dev/null )
if ! grep -Fx -- "$checkout/models/qwen3_8_27b.ninfer" "$args" >/dev/null; then
  printf 'run.sh did not resolve the checkout artifact: %s\n' "$(head -1 -- "$args")" >&2
  exit 1
fi


mkdir -- "$tmp/bin" "$tmp/models"
# Everything below drives download-model.sh. Run it as a user with no Hugging Face login, so the
# optional graft fetch after a model is in place has no token to use whatever this machine holds.
unset HF_TOKEN HF_TOKEN_PATH HF_HOME XDG_CACHE_HOME
mkdir -- "$tmp/no-login-home"
export HOME="$tmp/no-login-home"
# NINFER_TEST_FAKE_SIZE makes the fixture produce a file of exactly the pinned downloaders'
# expected_size (via truncate, so this stays instant regardless of how large the real artifact
# is) instead of an empty one: download-model.sh verifies size (and, unless
# NINFER_SKIP_SHA256=1, sha256) before promoting the file, so an empty fixture output fails
# verification and never reaches the file-existence assertions below.
cat > "$tmp/bin/curl" <<'CURL'
#!/usr/bin/env bash
while (( $# )); do
  if [[ "$1" == '--output' ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
: > "$output"
if [[ -n "${NINFER_TEST_FAKE_SIZE:-}" ]]; then
  truncate -s "$NINFER_TEST_FAKE_SIZE" "$output"
fi
CURL
chmod +x "$tmp/bin/curl"
# download-model.sh prefers aria2c over curl when aria2c is on PATH, and GitHub's hosted runner
# image ships aria2c -- so without a stub here, every case below that reaches the fetch branch
# would download the real multi-GB artifact from Hugging Face instead of exercising the fixture,
# which is what made this suite take minutes instead of seconds. -d/-o mirror aria2c's own flags
# (see download-model.sh's invocation), not curl's --output.
cat > "$tmp/bin/aria2c" <<'ARIA2C'
#!/usr/bin/env bash
dir='.' out=''
while (( $# )); do
  case "$1" in
    -d) dir="$2"; shift 2 ;;
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
output="$dir/$out"
: > "$output"
if [[ -n "${NINFER_TEST_FAKE_SIZE:-}" ]]; then
  truncate -s "$NINFER_TEST_FAKE_SIZE" "$output"
fi
ARIA2C
chmod +x "$tmp/bin/aria2c"
# Stubs sha256sum for the checksum-rejection fixture below. verify() hashes whatever it is given,
# and a real sha256sum reads every logical byte even of a sparse file -- tens of GB per downloader,
# which is instant to allocate but not free to read, and turned this fixture into a multi-minute
# hash of empty space. The stub returns a fixed value that cannot match any pinned expected_sha256
# without reading the file at all, which is enough to exercise verify()'s mismatch-rejection branch
# -- the property under test is the script's response to a failed comparison, not whether
# sha256sum itself hashes correctly.
cat > "$tmp/bin/sha256sum" <<'HASH'
#!/usr/bin/env bash
printf -v hash '%064d' 0
printf '%s  %s\n' "$hash" "${*: -1}"
HASH
chmod +x "$tmp/bin/sha256sum"
# One downloader serves every model and pins each one's revision, size and sha256 in a case block,
# verifying size and sha256 before promoting. Every model is driven the same way: hand the fixture the
# model's own expected_size and skip the hash, which proves the promotion path without needing a real
# 19 GB payload.
downloader="$root/download-model.sh"
models=(qwen38-27b qwen36-27b qwen36-35b-a3b)
artifact_of() {
  case "$1" in
    qwen38-27b) printf 'qwen3_8_27b.ninfer' ;;
    qwen36-27b) printf 'qwen3_6_27b.ninfer' ;;
    qwen36-35b-a3b) printf 'qwen3_6_35b_a3b.ninfer' ;;
  esac
}
# The pin lives in download-model.sh's `  <model>)` block, so read it from there rather than repeating it.
expected_size_of() {
  sed -n "/^  $1)\$/,/^    ;;\$/ s/^ *expected_size=\\([0-9]\\+\\)\$/\\1/p" "$downloader"
}
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"
  size="$(expected_size_of "$key")"
  if [[ -z "$size" ]]; then
    printf 'download-model.sh has no expected_size to verify %s against\n' "$key" >&2
    exit 1
  fi
  PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 \
    NINFER_TEST_FAKE_SIZE="$size" "$downloader" "$key" >/dev/null
  if [[ ! -f "$tmp/models/$model" ]]; then
    printf 'download-model.sh %s did not promote a payload matching its pin to %s\n' "$key" "$model" >&2
    exit 1
  fi
done

# A model name is required and must be one of the pinned ones. Neither case may touch the network or
# the models directory: exit 2 with a usage message that names every model, before any curl or
# aria2c runs.
# The models directory it is pointed at does not exist, so creating it would be caught below.
for bad in '' 'qwen38-27' 'qwen3_8_27b'; do
  if [[ -z "$bad" ]]; then bad_args=(); else bad_args=("$bad"); fi
  status=0
  usage_output="$(PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/never-created" "$downloader" \
    ${bad_args[@]+"${bad_args[@]}"} 2>&1 >/dev/null)" || status=$?
  if [[ "$status" != 2 ]]; then
    printf 'download-model.sh %s exited %s, expected 2\n' "${bad:-<no model>}" "$status" >&2
    exit 1
  fi
  for key in "${models[@]}"; do
    if [[ "$usage_output" != *"$key"* ]]; then
      printf 'download-model.sh usage for %s omits %s\n' "${bad:-<no model>}" "$key" >&2
      exit 1
    fi
  done
  if [[ -e "$tmp/never-created" ]]; then
    printf 'download-model.sh created the model directory before validating its argument\n' >&2
    exit 1
  fi
done

# An artifact that already verifies is left alone: no download, not even an attempted one. Both
# stubs in this directory fail every call, so a script that fetched again -- via either path --
# would exit non-zero instead of quietly reaching the real network.
mkdir -- "$tmp/failing-curl"
printf '#!/usr/bin/env bash\nexit 22\n' > "$tmp/failing-curl/curl"
chmod +x "$tmp/failing-curl/curl"
cp -- "$tmp/failing-curl/curl" "$tmp/failing-curl/aria2c"
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"
  size="$(expected_size_of "$key")"
  # Sparse, like the fixture's own payloads: instant to allocate whatever the pinned size.
  truncate -s "$size" "$tmp/models/$model"
  # Capture rather than pipe into grep -q: the script prints more after this line, and a reader that
  # exits at the first match would hand it SIGPIPE, failing the pipeline under pipefail.
  present_output="$(PATH="$tmp/failing-curl:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 \
    "$downloader" "$key" 2>&1)" || present_output=''
  if [[ "$present_output" != *'Model already present'* ]]; then
    printf 'download-model.sh %s did not accept an artifact that already verifies\n' "$key" >&2
    exit 1
  fi
done

# The opposite: a file at the final path that does not verify is replaced, not trusted. An empty
# file is the wrong size, so the script must fetch (the fixture supplies a payload of the pinned
# size) and end with the pinned-size artifact rather than the corrupt one.
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"
  size="$(expected_size_of "$key")"
  : > "$tmp/models/$model"
  PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 \
    NINFER_TEST_FAKE_SIZE="$size" "$downloader" "$key" >/dev/null 2>&1
  if [[ "$(wc -c < "$tmp/models/$model" | tr -d '[:space:]')" != "$size" ]]; then
    printf 'download-model.sh %s kept a corrupt artifact instead of re-fetching it\n' "$key" >&2
    exit 1
  fi
done

# The other half of the contract. Above proves a payload matching the pin is promoted; this proves
# one that does not is refused, which is the property the staging and checksum work exists for and
# the one a regression would silently remove. Without NINFER_TEST_FAKE_SIZE the fixture writes an
# empty file, so every pinned model should be rejected, leave the real artifact path alone, and
# keep the revision-scoped .part it was told to delete.
rm -f -- "$tmp/models"/*.ninfer
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"

  if PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" \
       "$downloader" "$key" >/dev/null 2>&1; then
    printf 'download-model.sh %s accepted a payload that does not match its pin\n' "$key" >&2
    exit 1
  fi
  if [[ -e "$tmp/models/$model" ]]; then
    printf 'download-model.sh %s promoted an unverified payload to %s\n' "$key" "$model" >&2
    exit 1
  fi
  if ! compgen -G "$tmp/models/$model."*".part" >/dev/null; then
    printf 'download-model.sh %s did not stage its download under a revision-scoped name\n' "$key" >&2
    exit 1
  fi
  rm -f -- "$tmp/models/$model."*".part"
done

# The size check alone is not the checksum contract: a payload of the right size but wrong content
# must be rejected too, and every case above either skips the hash (NINFER_SKIP_SHA256=1) or never
# reaches it (wrong size fails first). Drive the fixture with the correct size and the hash check
# left on -- the stubbed sha256sum above always returns a value that cannot match a real pinned
# hash, so this exercises verify()'s mismatch-rejection branch without hashing tens of GB of
# logical (sparse) data to get there.
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"
  size="$(expected_size_of "$key")"

  if PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_TEST_FAKE_SIZE="$size" \
       "$downloader" "$key" >/dev/null 2>&1; then
    printf 'download-model.sh %s accepted a size-correct payload with the wrong checksum\n' "$key" >&2
    exit 1
  fi
  if [[ -e "$tmp/models/$model" ]]; then
    printf 'download-model.sh %s promoted a payload that failed checksum verification to %s\n' "$key" "$model" >&2
    exit 1
  fi
  rm -f -- "$tmp/models/$model."*".part"
done

# The qwen38-27b godmode graft is fetched best-effort after the model is in place. It is a pair the
# launcher trusts by the .bin alone and the loader needs whole with its .json, so the contract is
# all-or-nothing: whatever happens, a failed fetch must leave a whole verified pair or neither file,
# and must never fail the model download. These cases run the real fetch_graft against a stub curl
# that serves tiny payloads (the model itself is a sparse file that already verifies, so nothing
# large is fetched) and a stub sha256sum that recognises those payloads; no network is touched.
# The graft's two pins sit in fetch_graft; the model pins earlier in the file are different hashes.
graft_shas="$(sed -n '/^fetch_graft() {$/,/^}$/p' "$downloader" | grep -oE '[0-9a-f]{64}')"
graft_bin_sha="$(sed -n 1p <<<"$graft_shas")"
graft_json_sha="$(sed -n 2p <<<"$graft_shas")"
if [[ -z "$graft_bin_sha" || -z "$graft_json_sha" ]]; then
  printf 'download-model.sh has no graft pins to verify against\n' >&2
  exit 1
fi
mkdir -- "$tmp/graft-bin" "$tmp/graft-home" "$tmp/graft-models"
cat > "$tmp/graft-bin/curl" <<'CURL'
#!/usr/bin/env bash
# NINFER_TEST_GRAFT_MODE: ok (default), deny_json (the sidecar request is refused) or bad_json
# (the sidecar arrives with the wrong content). Authorization headers are logged to GRAFT_LOG.
header='' output='' url=''
while (( $# )); do
  case "$1" in
    -H) header="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
printf '%s %s\n' "$header" "$url" >> "$GRAFT_LOG"
if [[ "${NINFER_TEST_GRAFT_MODE:-ok}" == deny_json && "$url" == *.json ]]; then exit 22; fi
if [[ "${NINFER_TEST_GRAFT_MODE:-ok}" == bad_json && "$url" == *.json ]]; then printf 'bad' > "$output"; exit 0; fi
printf 'good' > "$output"
CURL
cat > "$tmp/graft-bin/sha256sum" <<HASH
#!/usr/bin/env bash
file="\${*: -1}"
if [[ "\$(cat "\$file")" == good ]]; then
  case "\${file%.part}" in
    *.bin) printf '%s  %s\n' '$graft_bin_sha' "\$file" ;;
    *) printf '%s  %s\n' '$graft_json_sha' "\$file" ;;
  esac
else
  printf '%064d  %s\n' 0 "\$file"
fi
HASH
chmod +x "$tmp/graft-bin/curl" "$tmp/graft-bin/sha256sum"
graft_size="$(expected_size_of qwen38-27b)"
truncate -s "$graft_size" "$tmp/graft-models/$(artifact_of qwen38-27b)"
gdir="$tmp/graft-dir"
export GRAFT_LOG="$tmp/graft.log"

# Runs the downloader for qwen38-27b with no token configured anywhere unless the caller adds one.
run_graft() {
  env -u HF_TOKEN -u HF_TOKEN_PATH -u HF_HOME -u XDG_CACHE_HOME -u NINFER_GRAFTS \
    HOME="$tmp/graft-home" PATH="$tmp/graft-bin:$PATH" NINFER_MODEL_DIR="$tmp/graft-models" \
    NINFER_SKIP_SHA256=1 NINFER_GRAFT_DIR="$gdir" "$@" "$downloader" qwen38-27b 2>&1
}
graft_files() { (cd "$gdir" 2>/dev/null && ls -A | tr '\n' ' '); }
graft_fail() { printf 'download-model.sh graft: %s\n' "$1" >&2; exit 1; }
graft_reset() { rm -rf -- "$gdir"; : > "$GRAFT_LOG"; }
pair='godmode_q38_trained.bin godmode_q38_trained.json '

# No token anywhere: nothing is fetched, a lone half or a staged .part is dropped, and the model
# download still succeeds.
graft_reset; mkdir -- "$gdir"; printf 'good' > "$gdir/godmode_q38_trained.bin"; printf 'x' > "$gdir/godmode_q38_trained.json.part"
out="$(run_graft)" || graft_fail 'a missing token failed the model download'
[[ "$out" == *'no Hugging Face token'* ]] || graft_fail 'a missing token was not reported'
[[ -s "$GRAFT_LOG" ]] && graft_fail 'a missing token still reached curl'
[[ -z "$(graft_files)" ]] || graft_fail "a missing token left a broken pair behind: $(graft_files)"
# A whole pair that does not match the pinned hashes may be the user's own; without a token it stays.
printf 'mine' > "$gdir/godmode_q38_trained.bin"; printf 'mine' > "$gdir/godmode_q38_trained.json"
run_graft >/dev/null || graft_fail 'a missing token failed the model download'
[[ "$(graft_files)" == "$pair" ]] || graft_fail 'a whole unverified pair was deleted without a token'

# Token discovery follows huggingface_hub: HF_TOKEN, HF_TOKEN_PATH, HF_HOME/token, then
# XDG_CACHE_HOME/huggingface/token. Each must be honoured, and the pair installs whole.
mkdir -p -- "$tmp/graft-tokens/home" "$tmp/graft-tokens/xdg/huggingface"
printf 'tok-path\n' > "$tmp/graft-tokens/path-token"
printf 'tok-home\n' > "$tmp/graft-tokens/home/token"
printf 'tok-xdg\n' > "$tmp/graft-tokens/xdg/huggingface/token"
check_token() { # <expected token> <env assignment>...
  local expected="$1"; shift
  graft_reset
  out="$(run_graft "$@")" || graft_fail "the download failed with $*"
  grep -Fq "Bearer $expected " "$GRAFT_LOG" || graft_fail "token $expected was not used with $*"
  [[ "$(graft_files)" == "$pair" ]] || graft_fail "no whole pair after $*: $(graft_files)"
  [[ "$out" == *'Graft ready'* ]] || graft_fail "graft not reported ready with $*"
}
check_token env-token HF_TOKEN=env-token HF_TOKEN_PATH="$tmp/graft-tokens/path-token"
check_token tok-path HF_TOKEN_PATH="$tmp/graft-tokens/path-token" HF_HOME="$tmp/graft-tokens/home"
check_token tok-home HF_HOME="$tmp/graft-tokens/home" XDG_CACHE_HOME="$tmp/graft-tokens/xdg"
check_token tok-xdg XDG_CACHE_HOME="$tmp/graft-tokens/xdg"

# A verified pair is accepted without any fetch, and a second run is idempotent.
: > "$GRAFT_LOG"
out="$(run_graft HF_TOKEN=t)" || graft_fail 'a verified pair failed the model download'
[[ "$out" == *'Graft ready'* && ! -s "$GRAFT_LOG" ]] || graft_fail 'a verified pair was fetched again'

# Access denied for the second file, with a stale lone .bin on disk: neither file remains.
graft_reset; mkdir -- "$gdir"; printf 'stale' > "$gdir/godmode_q38_trained.bin"
out="$(run_graft HF_TOKEN=t NINFER_TEST_GRAFT_MODE=deny_json)" || graft_fail 'a denied sidecar failed the model download'
[[ "$out" == *'no access'* ]] || graft_fail 'a denied sidecar was not reported'
[[ -z "$(graft_files)" ]] || graft_fail "a denied sidecar left files behind: $(graft_files)"

# A sidecar with the wrong hash is not installed, and the binary that arrived first goes with it.
graft_reset
out="$(run_graft HF_TOKEN=t NINFER_TEST_GRAFT_MODE=bad_json)" || graft_fail 'a bad sidecar hash failed the model download'
[[ "$out" == *'failed sha256 verification'* ]] || graft_fail 'a bad sidecar hash was not reported'
[[ -z "$(graft_files)" ]] || graft_fail "a bad sidecar hash left files behind: $(graft_files)"

# A mismatched pair already on disk is replaced by the verified one.
graft_reset; mkdir -- "$gdir"; printf 'old' > "$gdir/godmode_q38_trained.bin"; printf 'old' > "$gdir/godmode_q38_trained.json"
run_graft HF_TOKEN=t >/dev/null || graft_fail 'replacing a mismatched pair failed the model download'
[[ "$(graft_files)" == "$pair" && "$(cat "$gdir/godmode_q38_trained.bin")" == good ]] || graft_fail 'a mismatched pair was not replaced'

# A directory that cannot be created skips the graft; the model download result is unchanged.
: > "$tmp/graft-afile"
out="$(run_graft HF_TOKEN=t NINFER_GRAFT_DIR="$tmp/graft-afile/sub")" || graft_fail 'an uncreatable graft directory failed the model download'
[[ "$out" == *'cannot create'* ]] || graft_fail 'an uncreatable graft directory was not reported'

# NINFER_GRAFTS=off skips the graft entirely.
graft_reset
run_graft HF_TOKEN=t NINFER_GRAFTS=off >/dev/null || graft_fail 'NINFER_GRAFTS=off failed the model download'
[[ ! -s "$GRAFT_LOG" && ! -e "$gdir" ]] || graft_fail 'NINFER_GRAFTS=off still fetched'

printf '%s\n' 'Linux script checks passed.'
