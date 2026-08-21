"""Convert a registered Qwen3.8-27B checkpoint into one complete artifact.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert \
      --model /path/to/Qwen3.8-27B \
      --out out/qwen3_8_27b.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import convert as qwen3_6_convert
from tools.convert.qwen3_6_27b import draft_head, recipe

from . import inventory


RECIPE_ID = "qwen3_8_27b-v1"
OFFICIAL_SOURCE_REPOSITORY = "Qwen/Qwen3.8-27B"
OFFICIAL_SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HUIHUI_SOURCE_REPOSITORY = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
HUIHUI_SOURCE_REVISION = "d42ca8978c5a66e92c3446d46e8adfe03ef692ff"

# Hugging Face uses a Git blob SHA-1 for ordinary files and a content SHA-256
# for LFS objects. These manifests bind each artifact identity to source bytes.
COMMON_SOURCE_ETAGS = {
    "chat_template.jinja": "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3",
    "config.json": "706cebd746c4b6f2b1d1f892630867acfdfd3df8",
    "generation_config.json": "023756cfadf88e5bf69eefeee3e172f38c448d64",
    "model.safetensors.index.json": "da35e3c564457dface7d138f0b6cac284ff8958c",
    "preprocessor_config.json": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
    "tokenizer_config.json": "5de744b3fca2129d7186979ae47c06be33903243",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "video_preprocessor_config.json": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
}
OFFICIAL_SOURCE_ETAGS = {
    **COMMON_SOURCE_ETAGS,
    "model-00001-of-00018.safetensors": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
    "model-00002-of-00018.safetensors": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
    "model-00003-of-00018.safetensors": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
    "model-00004-of-00018.safetensors": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
    "model-00005-of-00018.safetensors": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
    "model-00006-of-00018.safetensors": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
    "model-00007-of-00018.safetensors": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
    "model-00008-of-00018.safetensors": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
    "model-00009-of-00018.safetensors": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
    "model-00010-of-00018.safetensors": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
    "model-00011-of-00018.safetensors": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
    "model-00012-of-00018.safetensors": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
    "model-00013-of-00018.safetensors": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
    "model-00014-of-00018.safetensors": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
    "model-00015-of-00018.safetensors": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
    "model-00016-of-00018.safetensors": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
    "model-00017-of-00018.safetensors": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
    "model-00018-of-00018.safetensors": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
}
HUIHUI_SOURCE_ETAGS = {
    **COMMON_SOURCE_ETAGS,
    "model-00001-of-00018.safetensors": "746142a26219375775e31233a25b9d06bb50cc36a45b833dfad58d45bbeb68ac",
    "model-00002-of-00018.safetensors": "32e1fcb17fea75306e94eb0adfacf7e4069226f938582920df5962dc11c3becb",
    "model-00003-of-00018.safetensors": "1e22bff91476ac31fd41fa6c5b6772286cacb458fcb0b13cd0d401d15504cf42",
    "model-00004-of-00018.safetensors": "955da8a0ee55a8864617fd26e845a340bdbb4467087a223af0ef36caeb772397",
    "model-00005-of-00018.safetensors": "b1abd9cf4146923b9bdc2a943a3a3cf6600c0d622ff902626185377fc1b0e18a",
    "model-00006-of-00018.safetensors": "f69e98460c36b5c2b53b52de139bd1473c9ac872c7fe5575a0aac7ab6a823aef",
    "model-00007-of-00018.safetensors": "fa7adc386e8f88b5c68b26303bbe65c7a578797c66219f204699290427e91277",
    "model-00008-of-00018.safetensors": "4aafde533aa91b53df84b58bc6a7b61cd6b547878ae975f0250fde88203e288c",
    "model-00009-of-00018.safetensors": "75eedafc64705770e1a88132095cb848616c73a38689f33ec3802f3fada44a03",
    "model-00010-of-00018.safetensors": "79e1151f7868618544645443228956ca3c47734ce7f5ff57e701399d593d37cf",
    "model-00011-of-00018.safetensors": "545a750b0b0ef3e4a9fdec1f14177472c0d5c3c8028b52de3419098a1b90a0fa",
    "model-00012-of-00018.safetensors": "71fce00e3a7a171c0938685a63405d8266ea7697ec243553a38e94c6f69add60",
    "model-00013-of-00018.safetensors": "74cb96b32e845358f1db196134fb51fa60b288bfbf050e40a33abbe14ebce5d6",
    "model-00014-of-00018.safetensors": "0aec080a2c075370573a87a5f3baabe7969d0dd3b304c77feb5d24349bc91d7e",
    "model-00015-of-00018.safetensors": "fd49765c104b7db242f1e0e58866fc378e89fc435022bba5004d81cb94f4944f",
    "model-00016-of-00018.safetensors": "08dbb33dd635e396782a1f604b0de40f16721ee0f7ad0b19fccffc4888c28531",
    "model-00017-of-00018.safetensors": "fa2aec9367b5878d5694f819fc3d192cfaa941a18ac344330785a9f37d4e9e11",
    "model-00018-of-00018.safetensors": "fe72369eebdd81b26cb9c22ea63aec33f003ddcbb5cf08e4e7165f7cbcc5628b",
}

OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
    ),
    "frontend/tokenizer_config.json": (
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
    ),
    "frontend/chat_template.jinja": (
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    ),
    "frontend/generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_id: str
    target_key: str
    source_repository: str
    source_revision: str
    source_etags: Mapping[str, str]


MODEL_PROFILES = {
    inventory.MODEL_ID: ModelProfile(
        model_id=inventory.MODEL_ID,
        target_key=inventory.TARGET_KEY,
        source_repository=OFFICIAL_SOURCE_REPOSITORY,
        source_revision=OFFICIAL_SOURCE_REVISION,
        source_etags=OFFICIAL_SOURCE_ETAGS,
    ),
    inventory.HUIHUI_ABLITERATED_MODEL_ID: ModelProfile(
        model_id=inventory.HUIHUI_ABLITERATED_MODEL_ID,
        target_key="huihui_qwen3_8_27b_abliterated",
        source_repository=HUIHUI_SOURCE_REPOSITORY,
        source_revision=HUIHUI_SOURCE_REVISION,
        source_etags=HUIHUI_SOURCE_ETAGS,
    ),
}


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    profile: ModelProfile
    config_summary: dict[str, object]
    source: recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_model_profile(model_id: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[model_id]
    except KeyError as error:
        raise ValueError(
            f"unregistered Qwen3.8 model id {model_id!r}; expected one of "
            f"{tuple(MODEL_PROFILES)!r}"
        ) from error


def _content_etag(path: Path, expected: str) -> str:
    if len(expected) == 64:
        digest = hashlib.sha256()
    elif len(expected) == 40:
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode())
    else:
        raise ValueError(f"unsupported pinned ETag for {path.name}: {expected!r}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_profile(model_dir: str | Path, profile: ModelProfile) -> None:
    """Bind an artifact identity to every byte in its pinned source manifest."""

    model = Path(model_dir)
    index = family_conversion.load_json(model / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    shard_names = tuple(sorted(set(weight_map.values())))
    if not all(isinstance(name, str) and name for name in shard_names):
        raise ValueError("model.safetensors.index.json has an invalid shard name")

    resource_filenames = tuple(
        name.removeprefix("frontend/") for name in OFFICIAL_RESOURCE_SHA256
    )
    required_files = (
        "config.json",
        "model.safetensors.index.json",
        *resource_filenames,
        *shard_names,
    )
    if set(required_files) != set(profile.source_etags):
        raise ValueError("pinned source manifest does not match required checkpoint files")
    metadata_root = model / ".cache" / "huggingface" / "download"
    for filename in required_files:
        if not (model / filename).is_file():
            raise ValueError(f"pinned source file is missing: {filename}")
        metadata = metadata_root / f"{filename}.metadata"
        if not metadata.is_file():
            raise ValueError(
                f"pinned source metadata is missing for {filename}; download "
                f"{profile.source_repository}@{profile.source_revision} with `hf download "
                "--revision ... --local-dir ...`"
            )
        lines = metadata.read_text(encoding="utf-8").splitlines()
        actual_revision = lines[0] if lines else ""
        if actual_revision != profile.source_revision:
            raise ValueError(
                f"source revision mismatch for {filename}: expected "
                f"{profile.source_revision}, got {actual_revision or '<empty>'}"
            )
        recorded_etag = lines[1] if len(lines) > 1 else ""
        expected_etag = profile.source_etags[filename]
        if recorded_etag != expected_etag:
            raise ValueError(
                f"source ETag mismatch for {filename}: expected {expected_etag}, "
                f"got {recorded_etag or '<empty>'}"
            )
        actual_etag = _content_etag(model / filename, expected_etag)
        if actual_etag != expected_etag:
            raise ValueError(
                f"source content mismatch for {filename}: expected {expected_etag}, "
                f"got {actual_etag}"
            )


def preflight_inventory() -> None:
    if (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    ) != (6, 771, 2, 12, 333, 1118, 1124):
        raise ValueError("registered inventory is incomplete")
    recipe.validate_recipe_coverage()


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    spec_names = tuple(spec.name for spec in inventory.RESOURCE_SPECS)
    if spec_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the official Qwen3.8 profile: "
            f"expected {expected_names!r}, got {spec_names!r}"
        )
    resources = family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)
    actual_names = tuple(resource.name for resource in resources)
    if actual_names != expected_names:
        raise ValueError(
            "Qwen3.8 frontend resource set mismatch: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    for resource in resources:
        actual = hashlib.sha256(resource.data).hexdigest()
        expected = OFFICIAL_RESOURCE_SHA256[resource.name]
        if actual != expected:
            filename = resource.name.removeprefix("frontend/")
            raise ValueError(
                f"official Qwen3.8 resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )
    return resources


def preflight_conversion(
    model_dir: str | Path,
    *,
    model_id: str,
) -> ConversionPreflight:
    model = Path(model_dir)
    profile = resolve_model_profile(model_id)
    validate_source_profile(model, profile)
    config = family_conversion.load_json(model / "config.json")
    config_summary = qwen3_6_convert.validate_config(config)
    preflight_inventory()
    source = recipe.preflight_sources(model)
    resources = load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        profile=profile,
        config_summary=config_summary,
        source=source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    return qwen3_6_convert.materialize_tensor(spec, reader, draft)


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: recipe.SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
    model_id: str,
) -> dict[str, object]:
    profile = resolve_model_profile(model_id)
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(profile.model_id, inventory.WEIGHTS_ID),
        target_key=profile.target_key,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary=config_summary,
        source_preflight=source_preflight,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
    )
    report["source"]["repository"] = profile.source_repository
    report["source"]["revision"] = profile.source_revision
    report["source"]["manifest"] = dict(profile.source_etags)
    return report


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    model_id: str,
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model, model_id=model_id)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    with ShardReader(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(preflight.profile.model_id, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = materialize_tensor(spec, reader, preflight.draft)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": requested_device,
        "model_id": preflight.profile.model_id,
    }
    report = build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
        model_id=preflight.profile.model_id,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-id",
        required=True,
        choices=inventory.MODEL_IDS,
        help="registered artifact identity; the Huihui identity requires its pinned Hub revision",
    )
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, model_id=args.model_id)


if __name__ == "__main__":
    main()
