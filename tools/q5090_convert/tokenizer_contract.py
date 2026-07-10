"""Authoritative v4.1 embedded-tokenizer semantic validation."""

from __future__ import annotations

import json
import hashlib
from typing import Mapping

from . import format as fmt


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _reject_surrogates(value: object, label: str) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError(f"{label} contains a lone surrogate")
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item, label)
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key, label)
            _reject_surrogates(item, label)


def _decode_json(data: bytes, label: str) -> dict:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8: {exc}") from exc
    try:
        root = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed {label}: {exc}") from exc
    if not isinstance(root, dict):
        raise ValueError(f"{label} root must be an object")
    _reject_surrogates(root, label)
    return root


def _token_id(value: object, vocab_size: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or value >= vocab_size:
        raise ValueError(f"{label}={value} is outside [0,{vocab_size})")
    return value


def validate_tokenizer_assets(
    assets: Mapping[int, bytes], vocab_size: int, *, require_canonical: bool = False
) -> None:
    """Reject any bundle that the C++ tokenizer cannot safely consume for this model."""

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if set(assets) != set(fmt.TOKENIZER_KINDS):
        raise ValueError("tokenizer bundle must contain exactly the three v4.1 asset kinds")

    tokenizer = _decode_json(assets[fmt.TOKENIZER_JSON], "tokenizer.json")
    model = tokenizer.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError("tokenizer.json model.type must be BPE")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer.json model.vocab must be a non-empty object")

    occupied: set[int] = set()
    for token, raw_id in vocab.items():
        if not isinstance(token, str):
            raise ValueError("tokenizer.json vocab token must be a string")
        token_id = _token_id(raw_id, vocab_size, "tokenizer vocab id")
        if token_id in occupied:
            raise ValueError(f"tokenizer.json vocab contains duplicate id {token_id}")
        occupied.add(token_id)

    added = tokenizer.get("added_tokens")
    if not isinstance(added, list):
        raise ValueError("tokenizer.json added_tokens must be an array")
    added_ids: set[int] = set()
    for item in added:
        if not isinstance(item, dict):
            raise ValueError("added_tokens entry must be an object")
        token_id = _token_id(item.get("id"), vocab_size, "added token id")
        if token_id in occupied or token_id in added_ids:
            raise ValueError(f"added token id {token_id} overlaps or is duplicated")
        if not isinstance(item.get("content"), str):
            raise ValueError("added token content must be a string")
        for field in ("single_word", "lstrip", "rstrip", "normalized", "special"):
            if not isinstance(item.get(field), bool):
                raise ValueError(f"added token {field} must be boolean")
        if any(item[field] for field in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("unsupported added-token matching flags must all be false")
        added_ids.add(token_id)
        occupied.add(token_id)

    generation = _decode_json(
        assets[fmt.TOKENIZER_GENERATION_CONFIG], "generation_config.json"
    )
    eos = generation.get("eos_token_id")
    eos_values = eos if isinstance(eos, list) else [eos]
    if not eos_values:
        raise ValueError("generation_config.json eos_token_id array must not be empty")
    for raw_id in eos_values:
        token_id = _token_id(raw_id, vocab_size, "eos_token_id")
        if token_id not in occupied:
            raise ValueError(f"eos_token_id {token_id} is absent from tokenizer vocab")

    if require_canonical:
        if vocab_size != 248320:
            raise ValueError(f"canonical tokenizer requires vocab_size 248320, got {vocab_size}")
        expected_ids = set(range(fmt.TOKENIZER_OCCUPIED_ID_COUNT))
        if occupied != expected_ids:
            raise ValueError(
                "canonical tokenizer occupied ids must be exactly "
                f"[0,{fmt.TOKENIZER_OCCUPIED_ID_COUNT})"
            )
        for kind, expected_digest in fmt.TOKENIZER_CANONICAL_SHA256.items():
            if hashlib.sha256(assets[kind]).digest() != expected_digest:
                raise ValueError(
                    f"{fmt.TOKENIZER_SOURCE_NAME[kind]} does not match canonical Qwen3.6-27B identity"
                )

    merges_data = assets[fmt.TOKENIZER_MERGES]
    try:
        merges = merges_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"merges.txt is not UTF-8: {exc}") from exc
    if "\x00" in merges:
        raise ValueError("merges.txt contains NUL")
    pairs: set[tuple[str, str]] = set()
    saw_content = False
    for raw_line in merges.split("\n"):
        line = raw_line.removesuffix("\r")
        if not line:
            continue
        if not saw_content and line.startswith("#version"):
            saw_content = True
            continue
        saw_content = True
        if line.count(" ") != 1:
            raise ValueError(f"malformed merges.txt line: {line}")
        left, right = line.split(" ")
        if not left or not right:
            raise ValueError(f"malformed merges.txt line: {line}")
        pair = (left, right)
        if pair in pairs:
            raise ValueError(f"duplicate merges.txt pair: {line}")
        pairs.add(pair)
