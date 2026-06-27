#!/usr/bin/env python3
"""Unit tests for M2.8 bench tokenizer/decode helpers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.bench import tokenizer_common as common  # noqa: E402


class FakeTokenizer:
    def __init__(self) -> None:
        self.encoded: dict[str, list[int]] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("fixtures must not add special tokens implicitly")
        if text in self.encoded:
            return self.encoded[text]
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            raise AssertionError("decode sidecars must preserve generated ids")
        return "".join(chr(i) for i in ids)


class TokenizerCommonTests(unittest.TestCase):
    def test_ids_roundtrip_and_rejects_metadata(self) -> None:
        ids = [1, 2, 3, 4096, 151643]
        text = common.format_ids(ids)
        self.assertEqual(common.parse_ids_text(text), ids)
        self.assertEqual(common.parse_ids_text("1 2\n3\t4\n"), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            common.parse_ids_text("")
        with self.assertRaises(ValueError):
            common.parse_ids_text("1 -2 3")
        with self.assertRaises(ValueError):
            common.parse_ids_text("1 2 # 3")
        with self.assertRaises(ValueError):
            common.parse_ids_text("1 two 3")

    def test_sha256_and_tokenizer_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = root / "tokenizer.json"
            config = root / "tokenizer_config.json"
            special = root / "special_tokens_map.json"
            tokenizer.write_text("tokenizer\n", encoding="utf-8")
            config.write_text("config\n", encoding="utf-8")
            special.write_text("special\n", encoding="utf-8")

            metadata = common.tokenizer_metadata(root)
            self.assertEqual(metadata["tokenizer_source"], "local_hf")
            self.assertEqual(metadata["tokenizer_model_id"], "Qwen/Qwen3.6-27B")
            self.assertEqual(metadata["tokenizer_path"], str(root))
            self.assertEqual(metadata["tokenizer_json_sha256"], common.sha256_file(tokenizer))
            self.assertEqual(metadata["tokenizer_config_sha256"], common.sha256_file(config))
            self.assertEqual(metadata["special_tokens_map_sha256"], common.sha256_file(special))

    def test_resolve_tokenizer_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(common.resolve_tokenizer_path(str(root)), root)
            old = os.environ.get("QUS_TOKENIZER_PATH")
            try:
                os.environ["QUS_TOKENIZER_PATH"] = str(root)
                self.assertEqual(common.resolve_tokenizer_path(None), root)
            finally:
                if old is None:
                    os.environ.pop("QUS_TOKENIZER_PATH", None)
                else:
                    os.environ["QUS_TOKENIZER_PATH"] = old
            with self.assertRaises(RuntimeError):
                old = os.environ.pop("QUS_TOKENIZER_PATH", None)
                try:
                    common.resolve_tokenizer_path(None)
                finally:
                    if old is not None:
                        os.environ["QUS_TOKENIZER_PATH"] = old


class FixtureManifestTests(unittest.TestCase):
    def test_build_manifest_records_cases_in_required_order(self) -> None:
        from tools.bench import tokenize_prompts

        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "prompts"
            fixture_dir.mkdir()
            fake = FakeTokenizer()
            for index, name in enumerate(common.REQUIRED_CASES):
                text = f"{name} fixture text"
                fake.encoded[text] = (
                    list(range(2050)) if name == "long_2k" else [index + 1, index + 2, index + 3]
                )
                (fixture_dir / f"{name}.txt").write_text(text, encoding="utf-8")

            metadata = {
                "tokenizer_source": "local_hf",
                "tokenizer_model_id": common.TOKENIZER_MODEL_ID,
                "tokenizer_path": "/tmp/tokenizer",
                "tokenizer_json_sha256": "tok",
                "tokenizer_config_sha256": "cfg",
                "special_tokens_map_sha256": "special",
            }
            manifest = tokenize_prompts.write_fixtures(
                fixture_dir=fixture_dir,
                tokenizer=fake,
                tokenizer_metadata=metadata,
                check=False,
            )
            self.assertEqual(manifest["fixture_set"], common.FIXTURE_SET)
            self.assertEqual([case["name"] for case in manifest["cases"]], list(common.REQUIRED_CASES))
            for case in manifest["cases"]:
                ids = common.read_ids(fixture_dir / case["ids"])
                self.assertEqual(case["prompt_tokens"], len(ids))
                self.assertEqual(case["txt_sha256"], common.sha256_file(fixture_dir / case["txt"]))
                self.assertEqual(case["ids_sha256"], common.sha256_file(fixture_dir / case["ids"]))

            checked = tokenize_prompts.write_fixtures(
                fixture_dir=fixture_dir,
                tokenizer=fake,
                tokenizer_metadata=metadata,
                check=True,
            )
            self.assertEqual(checked, manifest)

    def test_check_mode_rejects_stale_ids(self) -> None:
        from tools.bench import tokenize_prompts

        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "prompts"
            fixture_dir.mkdir()
            fake = FakeTokenizer()
            for index, name in enumerate(common.REQUIRED_CASES):
                text = f"{name} fixture text"
                fake.encoded[text] = (
                    list(range(2050)) if name == "long_2k" else [index + 10, index + 11]
                )
                (fixture_dir / f"{name}.txt").write_text(text, encoding="utf-8")
            metadata = common.tokenizer_metadata(Path(tmp), redact_path=True)
            tokenize_prompts.write_fixtures(fixture_dir, fake, metadata, check=False)
            (fixture_dir / "cn_short.ids").write_text("1 2 3\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tokenize_prompts.write_fixtures(fixture_dir, fake, metadata, check=True)


class DecodeReportTests(unittest.TestCase):
    def test_decode_report_writes_sidecar_manifest_and_text(self) -> None:
        from tools.bench import decode_e2e_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            report = {
                "schema_version": 1,
                "artifact_type": "qus_e2e_benchmark_report",
                "status": "ok",
                "cases": [
                    {
                        "name": "cn_short",
                        "repeats": [
                            {"repeat_index": 0, "generated_token_ids": [65, 66, 67]},
                            {"repeat_index": 1, "generated_token_ids": [68, 69]},
                        ],
                    }
                ],
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            metadata = {
                "tokenizer_source": "local_hf",
                "tokenizer_model_id": common.TOKENIZER_MODEL_ID,
                "tokenizer_path": "/tmp/tokenizer",
                "tokenizer_json_sha256": "tok",
                "tokenizer_config_sha256": "cfg",
                "special_tokens_map_sha256": "special",
            }
            manifest = decode_e2e_report.decode_report(
                report_path=report_path,
                tokenizer=FakeTokenizer(),
                tokenizer_metadata=metadata,
                output_dir=None,
            )
            self.assertEqual(manifest["artifact_type"], "qus_decoded_text_artifacts")
            self.assertEqual(manifest["readability_gate"], "human_smoke_only")
            self.assertEqual(manifest["source_report"], str(report_path))
            self.assertEqual(len(manifest["artifacts"]), 2)
            manifest_path = root / "report.decoded" / "manifest.json"
            self.assertTrue(manifest_path.exists())
            first = Path(manifest["artifacts"][0]["decoded_text_path"])
            self.assertEqual(first.read_text(encoding="utf-8"), "ABC")

    def test_decode_report_rejects_non_ok_report(self) -> None:
        from tools.bench import decode_e2e_report

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "qus_e2e_benchmark_report",
                        "status": "error",
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                decode_e2e_report.decode_report(
                    report_path=report_path,
                    tokenizer=FakeTokenizer(),
                    tokenizer_metadata=common.tokenizer_metadata(Path(tmp), redact_path=True),
                    output_dir=None,
                )


if __name__ == "__main__":
    unittest.main()
