"""Activation importance for grouped quantization, imported from llama.cpp imatrix files.

An importance matrix holds, for every projection, the mean squared value of each input channel
over calibration text. ``grouped_search`` weights each weight's rounding error by its channel's
importance. llama.cpp's ``llama-imatrix`` writes these as GGUF tensors ``<tensor>.in_sum2`` and
``<tensor>.counts`` keyed by GGUF tensor names; this module maps them onto NInfer logical
parameter names and NInfer's input-channel order, and stores them as one safetensors file:

    python -m tools.convert.imatrix --gguf imatrix.gguf --config <Qwen3.8 dir>/config.json \
        --out qwen3_8_27b.imatrix.safetensors
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# GGUF tensor whose input activation each NInfer text-layer role reads.
_GGUF_TENSOR = {
    "attention/query": "attn_q",
    "attention/gate": "attn_q",
    "attention/key": "attn_k",
    "attention/value": "attn_v",
    "attention/output": "attn_output",
    "gdn/query": "attn_qkv",
    "gdn/key": "attn_qkv",
    "gdn/value": "attn_qkv",
    "gdn/z": "attn_gate",
    "gdn/output": "ssm_out",
    "mlp/gate": "ffn_gate",
    "mlp/up": "ffn_up",
    "mlp/down": "ffn_down",
}


def v_head_order(k_heads: int, v_heads: int, v_dim: int) -> torch.Tensor:
    """llama.cpp's grouped-to-tiled GDN value-head order: GGUF index i holds HF index order[i]."""
    n = v_heads * v_dim
    order = torch.arange(n).reshape(k_heads, v_heads // k_heads, v_dim)
    return order.permute(1, 0, 2).reshape(n)


def from_gguf(path: Path, config: dict) -> dict[str, torch.Tensor]:
    """Map a llama.cpp imatrix onto ``text/layers/N/<role>`` importance vectors."""
    import gguf

    text = config.get("text_config", config)
    reader = gguf.GGUFReader(str(path))
    sums, counts = {}, {}
    for tensor in reader.tensors:
        values = torch.from_numpy(tensor.data.copy()).to(torch.float32).reshape(-1)
        if tensor.name.endswith(".in_sum2"):
            sums[tensor.name[: -len(".in_sum2")]] = values
        elif tensor.name.endswith(".counts"):
            counts[tensor.name[: -len(".counts")]] = float(values[0])
    order = v_head_order(text["linear_num_key_heads"], text["linear_num_value_heads"],
                         text["linear_value_head_dim"])
    result = {}
    for layer, kind in enumerate(text["layer_types"]):
        prefix = "attention/" if kind == "full_attention" else "gdn/"
        for role, gguf_name in _GGUF_TENSOR.items():
            if not role.startswith((prefix, "mlp/")):
                continue
            name = f"blk.{layer}.{gguf_name}.weight"
            if name not in sums:
                continue
            importance = sums[name] / max(counts.get(name, 1.0), 1.0)
            if role == "gdn/output":
                hf = torch.empty_like(importance)
                hf[order] = importance
                importance = hf
            result[f"text/layers/{layer}/{role}"] = importance.clone()
    return result


@lru_cache(maxsize=2)
def load(path: str) -> dict[str, torch.Tensor]:
    """Importance vectors by logical parameter name, from a file written by this module."""
    return load_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    vectors = from_gguf(args.gguf, json.loads(args.config.read_text()))
    save_file(vectors, str(args.out), metadata={"source": args.gguf.name})
    print(f"{len(vectors)} parameters -> {args.out}")


if __name__ == "__main__":
    main()
