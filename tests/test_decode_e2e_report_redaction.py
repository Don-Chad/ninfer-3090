#!/usr/bin/env python3
"""Tests for decoded e2e manifest tokenizer path redaction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            raise AssertionError("decode sidecars must preserve generated ids")
        return "".join(chr(i) for i in ids)


class DecodeE2EReportRedactionTests(unittest.TestCase):
    def test_cli_redacts_tokenizer_path_in_manifest_by_default(self) -> None:
        from tools.bench import decode_e2e_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_path = root / "local-tokenizer"
            tokenizer_path.mkdir()
            report_path = root / "report.json"
            output_dir = root / "decoded"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "qus_e2e_benchmark_report",
                        "status": "ok",
                        "cases": [
                            {
                                "name": "cn_short",
                                "repeats": [
                                    {"repeat_index": 0, "generated_token_ids": [65, 66, 67]},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "decode_e2e_report.py",
                "--report",
                str(report_path),
                "--tokenizer-path",
                str(tokenizer_path),
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                decode_e2e_report.common, "load_tokenizer", return_value=FakeTokenizer()
            ):
                self.assertEqual(decode_e2e_report.main(), 0)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tokenizer"]["tokenizer_path"], "")


if __name__ == "__main__":
    unittest.main()
