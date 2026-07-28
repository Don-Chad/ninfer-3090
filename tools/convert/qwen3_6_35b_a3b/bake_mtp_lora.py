"""Bake the trained MTP LoRA into the native W8 input projection.

This intentionally has no command-line parser. Edit the constants below or set
NINFER_MTP_LORA_ACTION to ``apply``, ``verify``, or ``restore``. The original
encoded tensor payload is backed up before the artifact is changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from tools.artifact.container import Artifact
from tools.convert.common.quantize import quantize_and_encode


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = REPO_ROOT / "models/qwen3_6_35b_a3b.ninfer"
SOURCE_MODEL_DIR = Path(r"D:\mtp-k3-training\source")
ADAPTER_PATH = Path(
    r"D:\mtp-k3-training\checkpoints\python_mtp\best_python_mtp_k3.pt"
)
BACKUP_PATH = ARTIFACT_PATH.with_suffix(".mtp_input_projection.original.bin")
REPORT_PATH = ARTIFACT_PATH.with_suffix(".mtp_lora_bake.json")
ACTION = os.environ.get("NINFER_MTP_LORA_ACTION", "apply").strip().lower()
DEVICE = "cuda"

OBJECT_NAME = "mtp/input_projection"
SOURCE_NAME = "mtp.fc.weight"
EXPECTED_FORMAT = "W8G32_F16S"
EXPECTED_SHAPE = (2048, 4096)
EXPECTED_ADAPTER_FORMAT = "ninfer_mtp_lora_v1"
EXPECTED_ADAPTER_MODULE = "mtp.fc"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_payload() -> tuple[int, bytes]:
    with Artifact.open(ARTIFACT_PATH) as artifact:
        obj = artifact.find(OBJECT_NAME)
        if obj.kind != "tensor":
            raise ValueError(f"{OBJECT_NAME} is not a tensor")
        if obj.format != EXPECTED_FORMAT or obj.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"unexpected tensor contract: {obj.format} {obj.shape}"
            )
        absolute_offset = artifact.payload_offset + obj.offset
        payload = bytes(artifact.payload(obj))
    return absolute_offset, payload


def _source_shard() -> Path:
    index_path = SOURCE_MODEL_DIR / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index["weight_map"][SOURCE_NAME]
    return SOURCE_MODEL_DIR / shard_name


def _load_source_weight() -> torch.Tensor:
    shard = _source_shard()
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(SOURCE_NAME)
    if tuple(weight.shape) != EXPECTED_SHAPE:
        raise ValueError(f"unexpected {SOURCE_NAME} shape: {tuple(weight.shape)}")
    return weight


def _load_delta() -> tuple[torch.Tensor, dict[str, object]]:
    checkpoint = torch.load(
        ADAPTER_PATH, map_location="cpu", weights_only=True
    )
    if checkpoint.get("format") != EXPECTED_ADAPTER_FORMAT:
        raise ValueError(f"unexpected adapter format: {checkpoint.get('format')!r}")
    if checkpoint.get("module") != EXPECTED_ADAPTER_MODULE:
        raise ValueError(f"unexpected adapter module: {checkpoint.get('module')!r}")

    lora_a = checkpoint["lora_a"].float()
    lora_b = checkpoint["lora_b"].float()
    rank = int(checkpoint["rank"])
    alpha = float(checkpoint["alpha"])
    if tuple(lora_a.shape) != (4096, rank):
        raise ValueError(f"unexpected lora_a shape: {tuple(lora_a.shape)}")
    if tuple(lora_b.shape) != (rank, 2048):
        raise ValueError(f"unexpected lora_b shape: {tuple(lora_b.shape)}")

    scale = alpha / rank
    # PyTorch Linear stores [out, in], while training evaluated x @ A @ B.
    delta = (lora_a @ lora_b).T.contiguous().mul_(scale)
    metadata = {
        "adapter": str(ADAPTER_PATH),
        "adapter_format": checkpoint["format"],
        "module": checkpoint["module"],
        "rank": rank,
        "alpha": alpha,
        "scale": scale,
        "steps": int(checkpoint.get("steps", 0)),
    }
    return delta, metadata


def _encode(weight: torch.Tensor) -> bytes:
    return quantize_and_encode(weight, EXPECTED_FORMAT, device=DEVICE)


def _replace_payload(absolute_offset: int, payload: bytes) -> None:
    with ARTIFACT_PATH.open("r+b", buffering=0) as handle:
        handle.seek(absolute_offset)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def apply() -> None:
    absolute_offset, stored_original = _artifact_payload()
    source = _load_source_weight()
    encoded_original = _encode(source)
    if encoded_original != stored_original:
        raise ValueError(
            "artifact payload is not the canonical quantization of the supplied "
            "BF16 source; refusing to patch"
        )

    if BACKUP_PATH.exists():
        existing_backup = BACKUP_PATH.read_bytes()
        if existing_backup != stored_original:
            raise ValueError("existing backup does not match the artifact payload")
    else:
        BACKUP_PATH.write_bytes(stored_original)

    delta, metadata = _load_delta()
    merged = source.float().add_(delta)
    encoded_merged = _encode(merged)
    if len(encoded_merged) != len(stored_original):
        raise ValueError("encoded payload size changed")

    changed_bytes = sum(a != b for a, b in zip(stored_original, encoded_merged))
    _replace_payload(absolute_offset, encoded_merged)
    _, written = _artifact_payload()
    if written != encoded_merged:
        raise OSError("artifact payload verification failed after write")

    report = {
        "action": "apply",
        "artifact": str(ARTIFACT_PATH),
        "object": OBJECT_NAME,
        "format": EXPECTED_FORMAT,
        "shape": list(EXPECTED_SHAPE),
        "source_shard": str(_source_shard()),
        "backup": str(BACKUP_PATH),
        "original_payload_sha256": _sha256(stored_original),
        "merged_payload_sha256": _sha256(encoded_merged),
        "payload_bytes": len(encoded_merged),
        "changed_payload_bytes": changed_bytes,
        "delta_l2": float(torch.linalg.vector_norm(delta)),
        "source_l2": float(torch.linalg.vector_norm(source.float())),
        **metadata,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def restore() -> None:
    if not BACKUP_PATH.exists():
        raise FileNotFoundError(f"missing backup: {BACKUP_PATH}")
    absolute_offset, current = _artifact_payload()
    original = BACKUP_PATH.read_bytes()
    if len(original) != len(current):
        raise ValueError("backup payload size does not match artifact")
    _replace_payload(absolute_offset, original)
    _, restored = _artifact_payload()
    if restored != original:
        raise OSError("artifact payload verification failed after restore")
    print(
        json.dumps(
            {
                "action": "restore",
                "artifact": str(ARTIFACT_PATH),
                "object": OBJECT_NAME,
                "restored_payload_sha256": _sha256(restored),
            },
            indent=2,
        )
    )


def verify() -> None:
    _, current = _artifact_payload()
    result = {
        "action": "verify",
        "artifact": str(ARTIFACT_PATH),
        "object": OBJECT_NAME,
        "current_payload_sha256": _sha256(current),
        "backup_exists": BACKUP_PATH.exists(),
    }
    if BACKUP_PATH.exists():
        original = BACKUP_PATH.read_bytes()
        result["original_payload_sha256"] = _sha256(original)
        result["is_original"] = current == original
    print(json.dumps(result, indent=2))


def main() -> None:
    actions = {"apply": apply, "restore": restore, "verify": verify}
    try:
        action = actions[ACTION]
    except KeyError:
        raise ValueError(f"unknown action {ACTION!r}; expected one of {tuple(actions)}")
    action()


if __name__ == "__main__":
    main()
