#!/usr/bin/env python3
"""Decode generated token ids from an e2e report into sidecar text artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench import tokenizer_common as common


def _default_output_dir(report_path: Path) -> Path:
    return report_path.with_suffix(".decoded")


def _require_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise RuntimeError("expected schema_version 1")
    if report.get("artifact_type") != "qus_e2e_benchmark_report":
        raise RuntimeError("expected qus_e2e_benchmark_report")
    if report.get("status") != "ok":
        raise RuntimeError("decode sidecars require an ok e2e report")
    if not isinstance(report.get("cases"), list):
        raise RuntimeError("report cases must be a list")


def decode_report(
    report_path: Path,
    tokenizer: Any,
    tokenizer_metadata: dict[str, str],
    output_dir: Path | None,
) -> dict[str, Any]:
    source_report = str(report_path)
    report = common.read_json(report_path)
    _require_report(report)
    out_dir = output_dir if output_dir is not None else _default_output_dir(report_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    for case_index, case in enumerate(report["cases"]):
        case_name = str(case.get("name", f"case{case_index}"))
        repeats = case.get("repeats")
        if not isinstance(repeats, list):
            raise RuntimeError(f"case {case_name} repeats must be a list")
        for repeat in repeats:
            repeat_index = int(repeat.get("repeat_index", len(artifacts)))
            ids = repeat.get("generated_token_ids")
            if not isinstance(ids, list) or not all(isinstance(v, int) and v >= 0 for v in ids):
                raise RuntimeError(f"case {case_name} repeat {repeat_index} has invalid token ids")
            decoded = tokenizer.decode(list(ids), skip_special_tokens=False)
            filename = f"case{case_index}_{common.safe_name(case_name)}_repeat{repeat_index}.txt"
            decoded_path = out_dir / filename
            decoded_path.write_text(decoded, encoding="utf-8")
            artifacts.append(
                {
                    "case_index": case_index,
                    "case_name": case_name,
                    "repeat_index": repeat_index,
                    "decoded_text_path": str(decoded_path),
                    "generated_tokens_total": len(ids),
                }
            )

    if not artifacts:
        raise RuntimeError("report contains no generated token ids to decode")

    manifest = {
        "artifact_type": "qus_decoded_text_artifacts",
        "source_report": source_report,
        "readability_gate": "human_smoke_only",
        "tokenizer": tokenizer_metadata,
        "artifacts": artifacts,
    }
    common.write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    tokenizer_path = common.resolve_tokenizer_path(args.tokenizer_path)
    tokenizer = common.load_tokenizer(tokenizer_path)
    metadata = common.tokenizer_metadata(tokenizer_path, redact_path=False)
    manifest = decode_report(args.report, tokenizer, metadata, args.output_dir)
    manifest_path = (
        args.output_dir if args.output_dir is not None else args.report.with_suffix(".decoded")
    ) / "manifest.json"
    print(f"wrote {manifest_path}")
    print(f"decoded {len(manifest['artifacts'])} repeats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
