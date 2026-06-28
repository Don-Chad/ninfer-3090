#!/usr/bin/env python3
"""Regenerate or verify M2.8 prompt `.ids` files and fixture manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench import tokenizer_common as common


MANIFEST_NAME = f"{common.FIXTURE_SET}.manifest.json"


GENERATION = {
    "stop_token_ids": common.STOP_TOKEN_IDS,
    "stop_token_names": common.STOP_TOKEN_NAMES,
    "sampling_policy": "Fixture ids are prompt-only chat-template inputs; decode sampling is configured by benchmark callers.",
}


def _render_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=common.ADD_GENERATION_PROMPT,
        **common.CHAT_TEMPLATE_KWARGS,
        return_dict=False,
    )
    if not ids:
        raise RuntimeError("tokenizer produced no ids")
    if not all(isinstance(value, int) and value >= 0 for value in ids):
        raise RuntimeError("tokenizer produced a non-integer or negative id")
    return list(ids)


def _render_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=common.ADD_GENERATION_PROMPT,
        **common.CHAT_TEMPLATE_KWARGS,
        return_dict=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("tokenizer produced empty rendered prompt text")
    return rendered


def _expected_manifest(
    fixture_dir: Path,
    tokenizer: Any,
    tokenizer_metadata: dict[str, str],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for name in common.REQUIRED_CASES:
        messages_filename = f"{name}{common.MESSAGE_FILE_SUFFIX}"
        ids_filename = f"{name}.ids"
        messages_path = fixture_dir / messages_filename
        ids = fixture_dir / f"{name}.ids"
        if not messages_path.exists():
            raise RuntimeError(f"missing required prompt messages: {messages_path}")
        if not ids.exists():
            raise RuntimeError(f"missing required prompt ids: {ids}")
        messages = common.read_messages(messages_path)
        token_ids = common.read_ids(ids)
        rendered = _render_text(tokenizer, messages)
        cases.append(
            {
                "name": name,
                "messages": messages_filename,
                "ids": ids_filename,
                "prompt_tokens": len(token_ids),
                "messages_sha256": common.sha256_file(messages_path),
                "rendered_prompt_sha256": common.sha256_text(rendered),
                "ids_sha256": common.sha256_file(ids),
                "prompt_format": common.PROMPT_FORMAT,
                "add_generation_prompt": common.ADD_GENERATION_PROMPT,
                "add_special_tokens": common.ADD_SPECIAL_TOKENS,
                "chat_template_kwargs": common.CHAT_TEMPLATE_KWARGS,
            }
        )
    return {
        "fixture_set": common.FIXTURE_SET,
        "tokenizer": tokenizer_metadata,
        "generation": GENERATION,
        "cases": cases,
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    names = [case["name"] for case in manifest["cases"]]
    if names != list(common.REQUIRED_CASES):
        raise RuntimeError(f"manifest case order mismatch: {names}")
    token_counts = {case["name"]: int(case["prompt_tokens"]) for case in manifest["cases"]}
    if token_counts["long_2k"] < 2048:
        raise RuntimeError(f"long_2k must have at least 2048 tokens, got {token_counts['long_2k']}")


def write_fixtures(
    fixture_dir: Path,
    tokenizer: Any,
    tokenizer_metadata: dict[str, str],
    check: bool,
) -> dict[str, Any]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, list[int]] = {}
    for name in common.REQUIRED_CASES:
        messages_path = fixture_dir / f"{name}{common.MESSAGE_FILE_SUFFIX}"
        if not messages_path.exists():
            raise RuntimeError(f"missing required prompt messages: {messages_path}")
        generated[name] = _render_ids(tokenizer, common.read_messages(messages_path))

    if check:
        for name, ids in generated.items():
            ids_path = fixture_dir / f"{name}.ids"
            if not ids_path.exists():
                raise RuntimeError(f"missing ids file in check mode: {ids_path}")
            if common.read_ids(ids_path) != ids:
                raise RuntimeError(f"stale ids file: {ids_path}")
    else:
        for name, ids in generated.items():
            common.write_ids(fixture_dir / f"{name}.ids", ids)

    manifest = _expected_manifest(fixture_dir, tokenizer, tokenizer_metadata)
    _validate_manifest(manifest)
    manifest_path = fixture_dir / MANIFEST_NAME
    if check:
        if not manifest_path.exists():
            raise RuntimeError(f"missing manifest in check mode: {manifest_path}")
        existing = common.read_json(manifest_path)
        if existing != manifest:
            raise RuntimeError(f"stale fixture manifest: {manifest_path}")
    else:
        common.write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=common.repo_root() / "bench/fixtures/prompts")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    tokenizer_path = common.resolve_tokenizer_path(args.tokenizer_path)
    tokenizer = common.load_tokenizer(tokenizer_path)
    metadata = common.tokenizer_metadata(tokenizer_path, redact_path=True)
    manifest = write_fixtures(args.fixture_dir, tokenizer, metadata, check=args.check)
    action = "checked" if args.check else "wrote"
    print(f"{action} {args.fixture_dir / MANIFEST_NAME}")
    for case in manifest["cases"]:
        print(f"{case['name']}: {case['prompt_tokens']} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
