#!/usr/bin/env python3
"""Bake the qus_bench meaningful-token corpus.

qus_bench slices the first P token ids of this corpus to build a prefill of an exact
length P, so the corpus must be real, in-distribution text (not random/dummy tokens) and
long enough to cover the largest prefill length you want to benchmark.

The corpus is curated mixed-domain prose (Chinese / English / code / math). It is encoded
with a local Hugging Face Qwen3.6 tokenizer WITHOUT the chat template and WITHOUT special
tokens, then repeated to reach a target token count. The output is deterministic for a fixed
text + tokenizer, so `--check` can verify the committed artifact.

Outputs:
  bench/fixtures/bench_corpus.ids           whitespace-separated decimal token ids
  bench/fixtures/bench_corpus.manifest.json provenance (tokenizer id, token count, sha256)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

TOKENIZER_MODEL_ID = "Qwen/Qwen3.6-27B"

# Curated, in-distribution prose. Order and content are part of the corpus contract: changing
# them changes the committed ids. Keep paragraphs meaningful and self-contained.
PARAGRAPHS: tuple[str, ...] = (
    "周末旅行规划资料：目的地是一个适合亲子散步的湖边小城，市中心到湖区有直达公交，车程大约"
    "四十分钟。周六下午两点以后人流会增加，建议提前购买返程车票，并把儿童水杯、薄外套、充电宝"
    "和少量现金放在随身包里。",
    "天气预报显示周六白天多云，傍晚可能有小雨，湖边风会比城区更明显。行程中可以安排室内书店、"
    "湖边步道和家庭餐厅三个停留点，这样下雨时也能快速调整，不需要临时寻找避雨地点。",
    "餐饮方面，湖区东门附近有一家面馆和一家简餐店，面馆上菜快，适合孩子饿的时候优先选择；简餐店"
    "座位宽敞，但晚餐高峰可能排队二十分钟。预算按交通、饮料、晚餐和临时雨具计算，一家三口控制在"
    "三百元以内比较稳妥。",
    "A single-stream language model runtime spends almost all of its decode time moving weights "
    "from memory, so throughput at batch size one is bounded by memory bandwidth rather than raw "
    "arithmetic. Prefill, by contrast, processes the whole prompt at once and is usually compute "
    "bound, which is why the two phases are measured separately.",
    "When you profile an inference engine, keep the prompt content fixed and vary only the lengths "
    "you care about. Report prefill tokens per second and decode tokens per second as independent "
    "numbers, because a change that speeds up one phase can easily slow down the other.",
    "def moving_average(values, window):\n"
    "    if window <= 0:\n"
    "        raise ValueError(\"window must be positive\")\n"
    "    total = 0.0\n"
    "    result = []\n"
    "    for index, value in enumerate(values):\n"
    "        total += value\n"
    "        if index >= window:\n"
    "            total -= values[index - window]\n"
    "        if index >= window - 1:\n"
    "            result.append(total / window)\n"
    "    return result",
    "Consider a sequence defined by a_1 = 1 and a_{n+1} = a_n + 1 / a_n for n >= 1. Each step adds a "
    "positive amount, so the sequence is strictly increasing. Squaring the recurrence gives "
    "a_{n+1}^2 = a_n^2 + 2 + 1 / a_n^2, hence a_n^2 grows at least linearly and a_n is on the order "
    "of the square root of 2n for large n.",
    "会议纪要：本周完成了基准测试工具的重构评审，确认新的吞吐量工具只负责测速，正确性与逐层对齐仍由"
    "独立的校验工具负责。下一步先固化预填充与解码两个阶段的计时边界，再补充机器可读的输出格式，方便"
    "把历史结果归档和比较。",
)


def build_text(paragraphs: Sequence[str]) -> str:
    return "\n\n".join(paragraphs) + "\n\n"


def resolve_tokenizer_path(cli_path: str | None) -> Path:
    raw = cli_path or os.environ.get("QUS_TOKENIZER_PATH")
    if not raw:
        raise SystemExit(
            "tokenizer path required; pass --tokenizer-path or set QUS_TOKENIZER_PATH"
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"tokenizer path is not a directory: {path}")
    return path


def load_tokenizer(tokenizer_path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required; install tools/bench/requirements.txt"
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )


def encode_corpus(tokenizer: Any, min_tokens: int) -> list[int]:
    base_text = build_text(PARAGRAPHS)
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_ids:
        raise SystemExit("tokenizer produced no ids for the base corpus text")
    repeats = (min_tokens + len(base_ids) - 1) // len(base_ids) + 1
    full_text = base_text * repeats
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(ids) < min_tokens:
        raise SystemExit(
            f"encoded corpus has {len(ids)} tokens, below --min-tokens {min_tokens}"
        )
    for value in ids:
        if not isinstance(value, int) or value < 0:
            raise SystemExit(f"tokenizer produced an invalid id: {value!r}")
    return ids


def format_ids(ids: Sequence[int], per_line: int = 32) -> str:
    lines = [
        " ".join(str(v) for v in ids[i : i + per_line])
        for i in range(0, len(ids), per_line)
    ]
    return "\n".join(lines) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(ids: Sequence[int], ids_text: str, min_tokens: int) -> dict[str, Any]:
    return {
        "artifact_type": "qus_bench_corpus",
        "schema_version": 1,
        "tokenizer_source": "local_hf",
        "tokenizer_model_id": TOKENIZER_MODEL_ID,
        "add_special_tokens": False,
        "chat_template": False,
        "min_tokens": min_tokens,
        "token_count": len(ids),
        "ids_sha256": sha256_text(ids_text),
        "source": "curated mixed-domain prose (zh/en/code/math), repeated to reach min_tokens",
    }


def write_outputs(out_path: Path, manifest_path: Path, ids: Sequence[int], min_tokens: int) -> None:
    ids_text = format_ids(ids)
    manifest = build_manifest(ids, ids_text, min_tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ids_text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def check_outputs(out_path: Path, manifest_path: Path, ids: Sequence[int], min_tokens: int) -> int:
    if not out_path.exists() or not manifest_path.exists():
        print(f"missing corpus artifact: {out_path} or {manifest_path}", file=sys.stderr)
        return 1
    ids_text = format_ids(ids)
    expected = build_manifest(ids, ids_text, min_tokens)
    actual_ids_text = out_path.read_text(encoding="utf-8")
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = 0
    if actual_ids_text != ids_text:
        print("bench_corpus.ids differs from freshly baked corpus", file=sys.stderr)
        failures += 1
    for key in ("token_count", "ids_sha256"):
        if actual_manifest.get(key) != expected[key]:
            print(
                f"manifest.{key} mismatch: {actual_manifest.get(key)!r} != {expected[key]!r}",
                file=sys.stderr,
            )
            failures += 1
    if failures == 0:
        print(f"corpus OK: {expected['token_count']} tokens, sha256 {expected['ids_sha256']}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", default=None)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--out", type=Path, default=repo_root / "bench/fixtures/bench_corpus.ids")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--min-tokens", type=int, default=9216)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed artifacts match a fresh bake instead of writing",
    )
    args = parser.parse_args(argv)
    if args.min_tokens < 1:
        raise SystemExit("--min-tokens must be positive")
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")

    tokenizer = load_tokenizer(resolve_tokenizer_path(args.tokenizer_path))
    ids = encode_corpus(tokenizer, args.min_tokens)

    if args.check:
        return check_outputs(args.out, manifest_path, ids, args.min_tokens)

    write_outputs(args.out, manifest_path, ids, args.min_tokens)
    print(f"wrote {args.out} ({len(ids)} tokens) and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
