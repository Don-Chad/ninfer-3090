from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.convert.qwen3_8_27b import convert, inventory


def test_huihui_is_an_explicit_registered_profile():
    profile = convert.resolve_model_profile(inventory.HUIHUI_ABLITERATED_MODEL_ID)

    assert profile.model_id == "huihui-qwen3.8-27b-abliterated"
    assert profile.target_key == "huihui_qwen3_8_27b_abliterated"
    assert profile.source_repository == convert.HUIHUI_SOURCE_REPOSITORY
    assert profile.source_revision == convert.HUIHUI_SOURCE_REVISION


def _write_pinned_source(root: Path) -> tuple[convert.ModelProfile, Path]:
    shard = "model-00001-of-00001.safetensors"
    index = {"weight_map": {"test.weight": shard}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))

    filenames = [
        "config.json",
        "model.safetensors.index.json",
        shard,
        *(name.removeprefix("frontend/") for name in convert.OFFICIAL_RESOURCE_SHA256),
    ]
    metadata = root / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    etags = {}
    for filename in filenames:
        path = root / filename
        if not path.exists():
            path.write_bytes(b"x")
        # Exercise both digest formats without requiring a real checkpoint.
        expected = "0" * (64 if filename == shard else 40)
        etags[filename] = convert._content_etag(path, expected)
    profile = convert.ModelProfile(
        model_id=inventory.HUIHUI_ABLITERATED_MODEL_ID,
        target_key="huihui_qwen3_8_27b_abliterated",
        source_repository=convert.HUIHUI_SOURCE_REPOSITORY,
        source_revision=convert.HUIHUI_SOURCE_REVISION,
        source_etags=etags,
    )
    for filename in filenames:
        (metadata / f"{filename}.metadata").write_text(
            f"{profile.source_revision}\n{etags[filename]}\n0\n"
        )
    return profile, metadata


def test_huihui_profile_requires_the_pinned_hub_revision(tmp_path: Path):
    profile, metadata = _write_pinned_source(tmp_path)

    convert.validate_source_profile(tmp_path, profile)

    (metadata / "config.json.metadata").write_text(
        f"wrong-revision\n{profile.source_etags['config.json']}\n0\n"
    )
    with pytest.raises(ValueError, match="source revision mismatch for config.json"):
        convert.validate_source_profile(tmp_path, profile)


def test_huihui_profile_rejects_changed_checkpoint_bytes(tmp_path: Path):
    profile, _ = _write_pinned_source(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="source content mismatch"):
        convert.validate_source_profile(tmp_path, profile)


def test_unregistered_qwen38_identity_is_rejected():
    with pytest.raises(ValueError, match="unregistered Qwen3.8 model id"):
        convert.resolve_model_profile("qwen3.8-27b-similar-but-unregistered")
