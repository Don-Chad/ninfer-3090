"""Frequency-shortlist selection for the optional v4 draft `lm_head`.

The converter embeds an optional `[N, 5120]` Q4G64 draft head plus a paired
`[N]` int32 id-map (see ../../docs/q5090_packed_file_format_v4.md sections 8, 9.4,
14, 18). This module owns the deterministic selection of the N shortlist tokens
from a frequency ranking so that both the converter (`convert.py`) and the
verifier (`verify.py`) derive an identical id-map.

Ranking file layout (`ranking.train.counts.i64`): a little-endian int64 array
reshaped to `[rows, vocab]` where row 0 is the total emit histogram and rows 1..
are per-domain strata. Only row 0 (totals) drives selection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

VOCAB_SIZE = 248320
DRAFT_HEAD_N = 131072
DEFAULT_RANKING = "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
DEFAULT_TOKENIZER = "/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16"


def load_counts(path: str, vocab: int) -> np.ndarray:
    """Return the [rows, vocab] int64 histogram (row 0 = total, 1.. = strata)."""
    raw = np.fromfile(path, dtype="<i8")
    if raw.size == 0:
        raise SystemExit(f"{path}: empty ranking file")
    if raw.size % vocab != 0:
        raise SystemExit(f"{path}: size {raw.size} not a multiple of vocab {vocab}")
    return raw.reshape(-1, vocab)


def read_special_ids(tokenizer_dir: str) -> list:
    """Special/control token ids from tokenizer_config.json added_tokens_decoder."""
    cfg = os.path.join(tokenizer_dir, "tokenizer_config.json")
    if not os.path.isfile(cfg):
        return []
    with open(cfg) as f:
        data = json.load(f)
    out = []
    for id_str, meta in data.get("added_tokens_decoder", {}).items():
        if isinstance(meta, dict) and meta.get("special", False):
            out.append(int(id_str))
    return sorted(set(out))


def select_shortlist(total: np.ndarray, n: int, force_include: list) -> np.ndarray:
    """Top-n tokens by frequency (desc), guaranteeing force_include membership.

    Ties and the zero-count tail are broken by ascending token id for
    determinism. Returns an int64 array of exactly n unique vocab ids in
    frequency-rank order. The result depends only on the ranking totals and the
    force-include set, so any consumer reproduces the same id-map.
    """
    vocab = int(total.size)
    if n > vocab:
        raise SystemExit(f"N={n} exceeds vocab={vocab}")
    order = np.argsort(-total.astype(np.int64), kind="stable")  # freq desc, id asc on ties
    forced = np.array(sorted(set(i for i in force_include if 0 <= i < vocab)), dtype=np.int64)
    if forced.size >= n:
        forced_by_freq = forced[np.argsort(-total[forced].astype(np.int64), kind="stable")]
        return np.sort(forced_by_freq[:n])
    forced_set = set(forced.tolist())
    picked = []
    for tok in order.tolist():
        if len(picked) >= n - forced.size:
            break
        if tok not in forced_set:
            picked.append(tok)
    selected = np.array(picked, dtype=np.int64)
    selected = np.concatenate([selected, forced])
    selected = selected[np.argsort(-total[selected].astype(np.int64), kind="stable")]
    if selected.size != n or np.unique(selected).size != n:
        raise SystemExit(f"shortlist build failed: size={selected.size} unique={np.unique(selected).size}")
    return selected


@dataclass(frozen=True)
class DraftHeadContext:
    """Materialization context for the two synthetic draft-head blocks."""

    n: int
    selected: np.ndarray  # int64 [n], real vocab ids in frequency-rank order
    ranking: str
    tokenizer: str
    force_include: tuple


def compute_shortlist(
    ranking_path: str,
    tokenizer_dir: str,
    n: int,
    vocab: int = VOCAB_SIZE,
) -> DraftHeadContext:
    counts = load_counts(ranking_path, vocab)
    total = counts[0]
    force_include = read_special_ids(tokenizer_dir)
    selected = select_shortlist(total, n, force_include)
    return DraftHeadContext(
        n=int(n),
        selected=selected,
        ranking=ranking_path,
        tokenizer=tokenizer_dir,
        force_include=tuple(force_include),
    )
