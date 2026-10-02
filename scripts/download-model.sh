#!/usr/bin/env bash
set -euo pipefail

# Downloads one pinned model artifact, verifies it, and only then promotes it to its final name.
#
#   download-model.sh <model>
#
#   qwen38-27b       Qwen3.8-27B, 17.7 GiB. The default 27B for every benchmark in this repository.
#                    This fork's own conversion (WarlaxZ/Qwen3.8-27B-NInfer-3090, recipe
#                    qwen3_8_27b): the imatrix-weighted grouped_search encoder, a 4-bit embedding, a
#                    6-bit head and 4-bit mixer outputs and MLP down in layers 36-63. It carries the
#                    vision, MTP and DFlash2 weights, so one file serves --vision, --spec mtp and
#                    --spec dflash2. Executables older than the 5120x17408 Q4 linear_add route refuse
#                    it at startup. Figures published before 2026-10 were measured against the
#                    upstream neroued pin 1cbd84e7 (v2 18dfc887), 19.0 GiB.
#   qwen36-27b       Qwen3.6-27B groupwise-int, 16.3 GiB, target_key qwen3_6_27b. A different model
#                    family from qwen3_8_27b, which is why having the latter does not satisfy the
#                    former: four real-model tests -- ninfer_qwen3_6_27b_prefix_real_test,
#                    _score_real_test, _load_plan_test and the Qwen3.6 27B half of the engine suite --
#                    skip without it.
#   qwen36-35b-a3b   Qwen3.6-35B-A3B, 21.2 GiB, the RTX 3090-compatible vision model. The official v3
#                    artifact revision: its artifact-manifest.json lists the text, vision, mtp and
#                    dflash components, and the DFlash bundle (from z-lab/Qwen3.6-35B-A3B-DFlash) is
#                    what --spec dflash needs.
#
# To repin a model, edit its block below: take the size and sha256 from what HuggingFace reports for
# the new revision (X-Linked-Size and X-Linked-ETag on the resolve URL, or the revision's
# artifact-manifest.json), and for qwen36-35b-a3b check the manifest still lists a dflash component.
# Changing a pin makes the staging hazard below live rather than hypothetical.
#
# Environment: NINFER_MODEL_DIR (default: models/ beside this script), NINFER_SKIP_SHA256=1 to accept
# a file on size alone.
#
# Uses aria2c (16 parallel ranges) when it is on PATH, since single-stream curl against these
# artifacts has been observed to throttle to ~1 MB/s or stall outright; falls back to curl
# otherwise. `apt-get install aria2` / `brew install aria2` to opt in.

usage() {
  printf 'usage: %s <model>\n\n  qwen38-27b\n  qwen36-27b\n  qwen36-35b-a3b\n' "${0##*/}" >&2
}

case "${1:-}" in
  qwen38-27b)
    artifact='qwen3_8_27b.ninfer'
    owner='WarlaxZ'
    repo='Qwen3.8-27B-NInfer-3090'
    revision='0b0baabf51748811d29872ba5e4d92740ace041d'
    expected_size=18982458624
    expected_sha256='7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706'
    label='the Qwen3.8-27B model (17.7 GiB)'
    tests_variable='NINFER_QWEN3_8_27B_WEIGHTS'
    ;;
  qwen36-27b)
    artifact='qwen3_6_27b.ninfer'
    owner='neroued'
    repo='Qwen3.6-27B-NInfer'
    revision='3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829'
    expected_size=17495538688
    expected_sha256='9b610a7d051e7c4dbf89adb604bd269d248b643c8f6c7ef75bdb871af92c6f6b'
    label='the Qwen3.6-27B model (16.3 GiB)'
    tests_variable='NINFER_QWEN3_6_27B_WEIGHTS'
    ;;
  qwen36-35b-a3b)
    artifact='qwen3_6_35b_a3b.ninfer'
    owner='neroued'
    repo='Qwen3.6-35B-A3B-NInfer'
    revision='ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6'
    expected_size=22790484480
    expected_sha256='3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84'
    label='the RTX 3090-compatible Qwen3.6-35B-A3B vision model (21.2 GiB)'
    tests_variable='NINFER_QWEN3_6_35B_A3B_WEIGHTS'
    ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/$artifact"

# Staged under a revision-scoped name so that curl -C - can only ever resume the same artifact.
# curl -C - resumes by appending at the current file length, without checking what wrote those
# bytes, so a leftover partial from a different revision -- for instance the previous pin's, on any
# machine that ran an older script -- would be spliced onto the head of this one: a file of entirely
# plausible size that is corrupt throughout. A name that carries the revision means a resume can only
# ever continue the same artifact, and the checks below are what promote it to the final name.
part="$model.$revision.part"

file_size() { wc -c < "$1" | tr -d '[:space:]'; }

# Verifies a file against expected_size and, unless NINFER_SKIP_SHA256=1, expected_sha256.
# Used both for an existing "$model" (so a same-sized-but-corrupt file is not accepted forever
# just because it happened to pass once, or was replaced out from under this script) and for a
# freshly downloaded "$part" -- one check that cannot drift out of sync with itself. A missing
# sha256sum/shasum fails closed rather than silently promoting an unverified file: the whole
# reason these artifacts are checksummed is to catch a same-sized-but-corrupt file, and silently
# skipping that would defeat it, not just once, but for every future run on that host.
verify() {
  [ "$(file_size "$1")" = "$expected_size" ] || return 1
  [ "${NINFER_SKIP_SHA256:-0}" = '1' ] && return 0
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum -- "$1" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 -- "$1" | cut -d' ' -f1)"
  else
    printf 'No sha256sum or shasum found; cannot verify %s. Set NINFER_SKIP_SHA256=1 to accept it unverified.\n' "$1" >&2
    return 1
  fi
  [ "$actual" = "$expected_sha256" ]
}

mkdir -p -- "$model_dir"

if [ -f "$model" ]; then
  if verify "$model"; then
    printf 'Model already present: %s\n' "$model"
    exit 0
  fi
  printf '%s\n' "Existing $model did not verify against revision $revision; fetching the pinned one." >&2
  # A v2 file from an earlier release does not load any more, but it does not need to be fetched
  # again either: tools/upgrade_ninfer_v2_to_v3.py rewrites it locally in a couple of minutes.
  printf '%s\n' "If it is the v2 file from an earlier release, you can upgrade it instead of downloading:" \
    "  python tools/upgrade_ninfer_v2_to_v3.py $model $model.v3 && mv $model.v3 $model" >&2
fi

printf '%s\n' "Downloading $label..."
url="https://huggingface.co/$owner/$repo/resolve/$revision/$artifact"
if command -v aria2c >/dev/null 2>&1; then
  # 16 parallel ranges over one file, not single-stream curl. On the Vast rental boxes this repo
  # downloads at ~98 MB/s over aria2c against the same throttled-to-~1MB/s (and once outright
  # stalled at 11 of 21 GiB) single-stream curl -- see scripts/multi-gpu-testing/README.md. -c
  # resumes into the same .part name curl would use, and HuggingFace serves ranges fine.
  aria2c -x16 -s16 -c --file-allocation=none -d "$model_dir" -o "$(basename -- "$part")" "$url"
else
  curl -L -C - --fail --output "$part" "$url"
fi
if [ $? -ne 0 ]; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi

if ! verify "$part"; then
  printf 'Downloaded file at %s failed verification against revision %s. Delete it and run this script again.\n' \
    "$part" "$revision" >&2
  exit 1
fi

mv -f -- "$part" "$model"
printf 'Model ready: %s\n' "$model"
printf 'Point the tests at it with:  export %s=%s\n' "$tests_variable" "$model"
