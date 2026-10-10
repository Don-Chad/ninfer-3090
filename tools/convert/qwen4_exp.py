"""Qwen3.8-Flash-Next (GGUF architecture ``qwen4exp``) from a llama.cpp-style GGUF.

Maps GGUF metadata to the text config, every GGUF tensor to exactly one logical parameter (or,
for the routed experts, one per expert), and synthesizes the frontend resources from the GGUF
vocabulary. Values are never re-derived: block tensors keep their GGML blocks and direct tensors
their words. Logical shapes are the reversed GGUF dimensions, which keeps the bytes identical.

Export conventions of the GGUF, bound as stored (evidence in docs/maintainer/flash-next-plan.md):

- Every norm vector except ``gdn/norm`` (GGUF ``ssm_norm``) is stored as ``1 + gamma``; the
  model multiplies by the stored vector. ``gdn/norm`` is a plain multiplier. The converter does
  not subtract 1, which would round.
- GDN value heads are in llama.cpp's tiled order (value head h pairs with key head h % 16) in
  ``gdn/qkv`` (value rows), ``gdn/z``, ``gdn/a_projection``, ``gdn/b_projection``, ``gdn/a``,
  ``gdn/dt_bias``, ``gdn/convolution`` and the K axis of ``gdn/output``. Block formats cannot
  permute K exactly, so the model adopts the tiled order rather than the converter undoing it.
- ``gdn/a`` holds ``-exp(A_log)``.
- ``attention/query_gate`` interleaves each head's 256 query rows with its 256 gate rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .model import Model, Parameter
from .resources import token_domain
from .sources.gguf import GgufError, GgufModel, tensor_source

ARCHITECTURE = "qwen4exp"
_KEY = ARCHITECTURE + "."
# tokenizer.ggml.pre "qwen35": the Qwen3.5 split expression (kQwenSplitPattern in the runtime).
QWEN35_SPLIT_PATTERN = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)
_NORMAL, _CONTROL, _USER_DEFINED, _UNUSED = 1, 3, 4, 5


def _int(metadata: Mapping, key: str) -> int:
    value = metadata.get(key)
    if type(value) is not int or value <= 0:
        raise GgufError(f"{key}: expected a positive integer, got {value!r}")
    return value


def _ints(metadata: Mapping, key: str, length: int | None = None) -> list[int]:
    value = metadata.get(key)
    if (
        not isinstance(value, list)
        or any(type(item) is not int or item < 0 for item in value)
        or (length is not None and len(value) != length)
    ):
        raise GgufError(f"{key}: expected {length or 'a list of'} nonnegative integers")
    return list(value)


def _finite(metadata: Mapping, key: str, default: float | None = None) -> float:
    """A finite float metadata value (NaN and +-inf are never valid, and JSON cannot hold them)."""
    value = metadata.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise GgufError(f"{key}: expected a finite number, got {value!r}")
    return float(value)


def _real(metadata: Mapping, key: str) -> float:
    value = _finite(metadata, key)
    if not value > 0:
        raise GgufError(f"{key}: expected a positive number, got {value!r}")
    return value


def _f32_text(metadata: Mapping, key: str, default: float) -> float:
    """The shortest decimal that rounds to the same FP32 (GGUF stores sampling values as F32)."""
    value = _finite(metadata, key, default)
    with np.errstate(over="ignore"):
        single = np.float32(value)
    if not math.isfinite(single):
        raise GgufError(f"{key}: {value!r} overflows FP32")
    return float(str(single))


def text_config(metadata: Mapping) -> dict:
    if metadata.get("general.architecture") != ARCHITECTURE:
        raise GgufError(
            f"general.architecture is {metadata.get('general.architecture')!r}, "
            f"expected {ARCHITECTURE!r}"
        )
    m = metadata
    layers = _int(m, _KEY + "block_count")
    interval = _int(m, _KEY + "full_attention_interval")
    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(layers)
    ]
    ratios = _ints(m, _KEY + "attention.compress_ratios", layers)
    if any((ratio != 0) != (kind == "full_attention") for ratio, kind in zip(ratios, layer_types)):
        raise GgufError("attention.compress_ratios disagree with full_attention_interval")
    compress = sorted({ratio for ratio in ratios if ratio})
    if len(compress) != 1:
        raise GgufError(f"expected one QSA compress ratio, got {compress}")
    head_dim = _int(m, _KEY + "attention.key_length")
    if _int(m, _KEY + "attention.value_length") != head_dim:
        raise GgufError("attention key and value lengths differ")
    rotary = _int(m, _KEY + "rope.dimension_count")
    sections = _ints(m, _KEY + "rope.dimension_sections", 4)
    if sections[3] != 0 or sum(sections) * 2 != rotary or rotary > head_dim:
        raise GgufError(f"rope sections {sections} disagree with rotary width {rotary}")
    value_heads = _int(m, _KEY + "ssm.time_step_rank")
    inner = _int(m, _KEY + "ssm.inner_size")
    key_heads = _int(m, _KEY + "ssm.group_count")
    if inner % value_heads or value_heads % key_heads:
        raise GgufError("GDN head counts do not divide the inner size")
    ple_layers = _ints(m, _KEY + "ple.layers")
    offsets = _ints(m, _KEY + "ple.head_offsets")
    sizes = _ints(m, _KEY + "ple.head_vocab_sizes", len(offsets))
    ngram = _int(m, _KEY + "ple.ngram_size")
    if ngram < 2:
        raise GgufError(f"{_KEY}ple.ngram_size: expected at least 2, got {ngram}")
    heads_per_ngram = _int(m, _KEY + "ple.heads_per_ngram")
    if len(offsets) != (ngram - 1) * heads_per_ngram:
        raise GgufError("PLE head tables do not match ngram_size and heads_per_ngram")
    running = 0
    for offset, size in zip(offsets, sizes):
        if size <= 0 or offset != running:
            raise GgufError(
                "PLE head offsets are not the running sum of the head sizes from zero"
            )
        running += size
    if any(not 0 <= layer < layers for layer in ple_layers):
        raise GgufError(f"PLE layers {ple_layers} are outside the model")
    experts, used = _int(m, _KEY + "expert_count"), _int(m, _KEY + "expert_used_count")
    if used > experts:
        raise GgufError("expert_used_count exceeds expert_count")
    return {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp_text",
        "hidden_size": _int(m, _KEY + "embedding_length"),
        "num_hidden_layers": layers,
        "max_position_embeddings": _int(m, _KEY + "context_length"),
        "tie_word_embeddings": False,
        "rms_norm_eps": _real(m, _KEY + "attention.layer_norm_rms_epsilon"),
        "layer_types": layer_types,
        "num_attention_heads": _int(m, _KEY + "attention.head_count"),
        "num_key_value_heads": _int(m, _KEY + "attention.head_count_kv"),
        "head_dim": head_dim,
        "rope_parameters": {
            "rope_theta": _real(m, _KEY + "rope.freq_base"),
            "partial_rotary_factor": rotary / head_dim,
            "mrope_section": sections[:3],
            "mrope_interleaved": True,
        },
        "linear_num_key_heads": key_heads,
        "linear_key_head_dim": _int(m, _KEY + "ssm.state_size"),
        "linear_num_value_heads": value_heads,
        "linear_value_head_dim": inner // value_heads,
        "linear_conv_kernel_dim": _int(m, _KEY + "ssm.conv_kernel"),
        "num_experts": experts,
        "num_experts_per_tok": used,
        "moe_intermediate_size": _int(m, _KEY + "expert_feed_forward_length"),
        "shared_expert_intermediate_size": _int(
            m, _KEY + "expert_shared_feed_forward_length"
        ),
        "hc_count": _int(m, _KEY + "hyper_connection.count"),
        "hc_low_rank": _int(m, _KEY + "hyper_connection.low_rank"),
        "indexer_num_heads": _int(m, _KEY + "attention.indexer.head_count"),
        "indexer_head_dim": _int(m, _KEY + "attention.indexer.key_length"),
        "indexer_top_k": _int(m, _KEY + "attention.indexer.top_k"),
        "indexer_compress_ratio": compress[0],
        "ple_layers": ple_layers,
        "ple_ngram_size": ngram,
        "ple_heads_per_ngram": heads_per_ngram,
        "ple_conv_kernel": _int(m, _KEY + "ple.conv_kernel"),
        "ple_embedding_dim": _int(m, _KEY + "embedding_length_per_layer_input"),
        "ple_eos_token_id": _int(m, _KEY + "ple.eos_token_id"),
        "ple_image_token_id": _int(m, _KEY + "ple.image_token_id"),
        "ple_layer_multipliers": _ints(m, _KEY + "ple.layer_multipliers", ngram),
        "ple_head_offsets": offsets,
        "ple_head_vocab_sizes": sizes,
    }


# GGUF per-layer suffix -> (logical role, mathematical inputs relative to the layer prefix).
_LAYER_ROLES = {
    "hc_attn_down.weight": ("hc_mixer/down", ("hc_mixer/stream",)),
    "hc_attn_up.weight": ("hc_mixer/up", ("hc_mixer/low_rank",)),
    "hc_attn_inject.weight": ("hc_mixer/inject", ("hc_mixer/stream",)),
    "hc_attn_norm.weight": ("hc_mixer/norm", ()),
    "hc_ffn_down.weight": ("hc_ffn/down", ("hc_ffn/stream",)),
    "hc_ffn_up.weight": ("hc_ffn/up", ("hc_ffn/low_rank",)),
    "hc_ffn_inject.weight": ("hc_ffn/inject", ("hc_ffn/stream",)),
    "hc_ffn_norm.weight": ("hc_ffn/norm", ()),
    "attn_qkv.weight": ("gdn/qkv", ("mixer_input",)),
    "attn_gate.weight": ("gdn/z", ("mixer_input",)),
    "ssm_alpha.weight": ("gdn/a_projection", ("mixer_input",)),
    "ssm_beta.weight": ("gdn/b_projection", ("mixer_input",)),
    "ssm_a": ("gdn/a", ()),
    "ssm_dt.bias": ("gdn/dt_bias", ()),
    "ssm_conv1d.weight": ("gdn/convolution", ()),
    "ssm_norm.weight": ("gdn/norm", ()),
    "ssm_out.weight": ("gdn/output", ("gdn/gated_output",)),
    "attn_q.weight": ("attention/query_gate", ("mixer_input",)),
    "attn_k.weight": ("attention/key", ("mixer_input",)),
    "attn_v.weight": ("attention/value", ("mixer_input",)),
    "attn_output.weight": ("attention/output", ("attention/gated_output",)),
    "attn_q_norm.weight": ("attention/query_norm", ()),
    "attn_k_norm.weight": ("attention/key_norm", ()),
    "indexer.q_proj.weight": ("indexer/query", ("mixer_input",)),
    "indexer.k_proj.weight": ("indexer/key", ("mixer_input",)),
    "indexer.q_norm.weight": ("indexer/query_norm", ()),
    "indexer.k_norm.weight": ("indexer/key_norm", ()),
    "ffn_gate_inp.weight": ("moe/router", ("ffn_input",)),
    "ffn_gate_inp_shexp.weight": ("moe/shared_score", ("ffn_input",)),
    "ffn_gate_shexp.weight": ("moe/shared/gate", ("ffn_input",)),
    "ffn_up_shexp.weight": ("moe/shared/up", ("ffn_input",)),
    "ffn_down_shexp.weight": ("moe/shared/down", ("moe/shared/product",)),
    "ple_key.weight": ("ple/key", ("ple/ngram_embedding",)),
    "ple_value.weight": ("ple/value", ("ple/ngram_embedding",)),
    "ple_norm_key.weight": ("ple/key_norm", ()),
    "ple_norm_query.weight": ("ple/query_norm", ()),
    "ple_norm_conv.weight": ("ple/conv_norm", ()),
    "ple_conv1d.weight": ("ple/convolution", ()),
}
_EXPERT_ROLES = {
    "ffn_gate_exps.weight": "gate",
    "ffn_up_exps.weight": "up",
    "ffn_down_exps.weight": "down",
}
_COMMON = ("hc_attn_", "hc_ffn_", "ffn_")
_MIXER_SUFFIXES = {
    "linear_attention": {
        "attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight",
        "ssm_a", "ssm_dt.bias", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight",
    },
    "full_attention": {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_proj.weight",
        "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight",
    },
}
_PLE_SUFFIXES = {key for key in _LAYER_ROLES if key.startswith("ple_")}
_LAYER_COMMON = {key for key in _LAYER_ROLES if key.startswith(_COMMON)} | set(_EXPERT_ROLES)
_GLOBAL_ROLES = {
    "token_embd.weight": ("text/token_embedding", ()),
    "output.weight": ("text/output_head", ("text/final_hidden",)),
    "output_hc_down.weight": ("text/output_hc/down", ("text/output_hc/stream",)),
    "output_hc_up.weight": ("text/output_hc/up", ("text/output_hc/low_rank",)),
    "output_hc_norm.weight": ("text/output_hc/norm", ()),
    "per_layer_token_embd.weight": ("text/ple/table", ()),
}
PLE_TABLE = "per_layer_token_embd.weight"


@dataclass(frozen=True, slots=True)
class Subset:
    """A development subset: chosen tensors, whole expert banks, and the first PLE rows.

    Such an artifact is not a loadable model; later PRs use it as a real-weight fixture.
    """

    tensors: frozenset[str]
    direct_layers: frozenset[int]
    expert_layers: frozenset[int]
    ple_rows: int

    def keeps(self, gguf: GgufModel, name: str) -> bool:
        if name in self.tensors or name == PLE_TABLE:
            return True
        layer, suffix = _split_layer(name)
        if layer is None:
            return name.startswith("output_hc_")
        if suffix in _EXPERT_ROLES:
            return layer in self.expert_layers
        return layer in self.direct_layers and gguf.tensor(name).type.dtype is not None


SUBSETS = {
    # One real tensor of every stored GGML type, the direct tensors of a GDN layer, the PLE
    # layer and a QSA layer, three expert banks (IQ2_S, IQ2_XXS and IQ1_M with Q2_0 down) and
    # the first 100 PLE pages.
    "dev": Subset(
        tensors=frozenset(
            {
                "token_embd.weight",
                "output.weight",
                "blk.0.attn_qkv.weight",
                "blk.0.attn_gate.weight",
                "blk.1.ffn_gate_shexp.weight",
                "blk.0.ffn_down_shexp.weight",
                "blk.3.ffn_down_shexp.weight",
                "blk.6.ffn_down_shexp.weight",
            }
        ),
        direct_layers=frozenset({0, 1, 3}),
        expert_layers=frozenset({0, 1, 8}),
        ple_rows=4500,
    ),
}


def _split_layer(name: str) -> tuple[int | None, str]:
    if not name.startswith("blk."):
        return None, name
    _, layer, suffix = name.split(".", 2)
    if not layer.isdigit():
        raise GgufError(f"{name}: malformed layer index")
    return int(layer), suffix


@dataclass(frozen=True, slots=True)
class TensorMapping:
    """One logical parameter of a GGUF tensor: its row range, logical shape and inputs."""

    gguf: str
    parameter: str
    rows: tuple[int, int] | None
    shape: tuple[int, ...]
    inputs: tuple[str, ...]


def name_map(gguf: GgufModel, config: dict) -> list[TensorMapping]:
    """Map every GGUF tensor; reject unknown, missing or misplaced tensors."""
    layers = config["num_hidden_layers"]
    expected: dict[int, set[str]] = {}
    for layer, kind in enumerate(config["layer_types"]):
        expected[layer] = _LAYER_COMMON | _MIXER_SUFFIXES[kind]
        if layer in config["ple_layers"]:
            expected[layer] |= _PLE_SUFFIXES
    found: dict[int, set[str]] = {layer: set() for layer in range(layers)}
    result = []
    experts = config["num_experts"]
    for name, tensor in gguf.tensors.items():
        layer, suffix = _split_layer(name)
        if layer is None:
            if name not in _GLOBAL_ROLES:
                raise GgufError(f"unknown GGUF tensor {name!r}")
            parameter, inputs = _GLOBAL_ROLES[name]
            result.append(TensorMapping(name, parameter, None, tensor.shape, inputs))
            continue
        if layer >= layers or suffix not in expected[layer]:
            raise GgufError(f"unexpected GGUF tensor {name!r} for this layer")
        found[layer].add(suffix)
        prefix = f"text/layers/{layer}/"
        if suffix in _EXPERT_ROLES:
            if len(tensor.shape) != 3 or tensor.shape[0] != experts:
                raise GgufError(f"{name}: expected [{experts}, N, K], got {tensor.shape}")
            role = _EXPERT_ROLES[suffix]
            per = tensor.shape[1]
            for expert in range(experts):
                expert_prefix = f"{prefix}moe/experts/{expert}/"
                use = expert_prefix + "product" if role == "down" else prefix + "ffn_input"
                result.append(
                    TensorMapping(
                        name,
                        expert_prefix + role,
                        (expert * per, (expert + 1) * per),
                        tensor.shape[1:],
                        (use,),
                    )
                )
            continue
        role, inputs = _LAYER_ROLES[suffix]
        shape = tensor.shape
        if suffix == "ffn_gate_inp_shexp.weight":
            shape = (1, *shape)
        result.append(
            TensorMapping(name, prefix + role, None, shape, tuple(prefix + i for i in inputs))
        )
    for name in _GLOBAL_ROLES:
        if name not in gguf.tensors:
            raise GgufError(f"GGUF tensor {name!r} is missing")
    for layer in range(layers):
        missing = expected[layer] - found[layer]
        if missing:
            raise GgufError(f"layer {layer} is missing {sorted(missing)}")
    _check_shapes(gguf, config)
    return result


def _check_shapes(gguf: GgufModel, config: dict) -> None:
    h, vocab = config["hidden_size"], gguf.tensor("token_embd.weight").shape[0]
    qkv = 2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
    vg = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    hc, low = config["hc_count"], config["hc_low_rank"]
    experts, moe = config["num_experts"], config["moe_intermediate_size"]
    shared = config["shared_expert_intermediate_size"]
    value_heads = config["linear_num_value_heads"]
    checks = {
        "token_embd.weight": (vocab, h),
        "output_hc_down.weight": (low, hc * h),
        "output_hc_up.weight": (hc * h, low),
        "output_hc_norm.weight": (hc * h,),
        "output.weight": (vocab, h),
        PLE_TABLE: (
            gguf.tensor(PLE_TABLE).shape[0],
            config["ple_embedding_dim"],
        ),
    }
    for layer, kind in enumerate(config["layer_types"]):
        p = f"blk.{layer}."
        for part in ("attn", "ffn"):
            checks[p + f"hc_{part}_down.weight"] = (low, hc * h)
            checks[p + f"hc_{part}_up.weight"] = (hc * h, low)
            checks[p + f"hc_{part}_inject.weight"] = (hc, hc * h)
            checks[p + f"hc_{part}_norm.weight"] = (hc * h,)
        checks[p + "ffn_gate_inp.weight"] = (experts, h)
        checks[p + "ffn_gate_inp_shexp.weight"] = (h,)
        checks[p + "ffn_gate_shexp.weight"] = (shared, h)
        checks[p + "ffn_up_shexp.weight"] = (shared, h)
        checks[p + "ffn_down_shexp.weight"] = (h, shared)
        checks[p + "ffn_gate_exps.weight"] = (experts, moe, h)
        checks[p + "ffn_up_exps.weight"] = (experts, moe, h)
        checks[p + "ffn_down_exps.weight"] = (experts, h, moe)
        if kind == "linear_attention":
            checks[p + "attn_qkv.weight"] = (qkv + vg, h)
            checks[p + "attn_gate.weight"] = (vg, h)
            checks[p + "ssm_alpha.weight"] = (value_heads, h)
            checks[p + "ssm_beta.weight"] = (value_heads, h)
            checks[p + "ssm_a"] = (value_heads,)
            checks[p + "ssm_dt.bias"] = (value_heads,)
            checks[p + "ssm_conv1d.weight"] = (
                qkv + vg, config["linear_conv_kernel_dim"]
            )
            checks[p + "ssm_out.weight"] = (h, vg)
        else:
            heads, d = config["num_attention_heads"], config["head_dim"]
            checks[p + "attn_q.weight"] = (2 * heads * d, h)
            kv = config["num_key_value_heads"] * d
            checks[p + "attn_k.weight"] = (kv, h)
            checks[p + "attn_v.weight"] = (kv, h)
            checks[p + "attn_q_norm.weight"] = (d,)
            checks[p + "attn_k_norm.weight"] = (d,)
            checks[p + "attn_output.weight"] = (h, heads * d)
            # The indexer reads the same mixer input as attention. Its query has one
            # indexer_head_dim vector per indexer head and its key a single such vector.
            idx_heads, idx_dim = config["indexer_num_heads"], config["indexer_head_dim"]
            checks[p + "indexer.q_proj.weight"] = (idx_heads * idx_dim, h)
            checks[p + "indexer.k_proj.weight"] = (idx_dim, h)
            checks[p + "indexer.q_norm.weight"] = (idx_dim,)
            checks[p + "indexer.k_norm.weight"] = (idx_dim,)
    # Not checked, because nothing in the config or the repository pins their dimensions:
    # ple_key/ple_value/ple_norm_*/ple_conv1d. Their input width (2 * ple_embedding_dim in the
    # fixture) is ngram_size - 1 or heads_per_ngram times it, and the hyper-connection width
    # they share is only evidenced by that fixture.
    for name, shape in checks.items():
        if gguf.tensor(name).shape != tuple(shape):
            raise GgufError(f"{name}: shape {gguf.tensor(name).shape}, expected {shape}")
    rows = gguf.tensor(PLE_TABLE).shape[0]
    if config["ple_head_offsets"][-1] + config["ple_head_vocab_sizes"][-1] > rows:
        raise GgufError("PLE head ranges exceed the table rows")


def tokenizer_resources(metadata: Mapping, config: dict) -> dict[str, bytes]:
    """Synthesize Hugging Face tokenizer resources from the GGUF vocabulary and merges."""
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get(
        "tokenizer.ggml.pre"
    ) != "qwen35":
        raise GgufError("only the gpt2 BPE tokenizer with the qwen35 pre-tokenizer is supported")
    tokens = metadata.get("tokenizer.ggml.tokens")
    types = metadata.get("tokenizer.ggml.token_type")
    merges = metadata.get("tokenizer.ggml.merges")
    if (
        not isinstance(tokens, list)
        or not isinstance(types, list)
        or len(tokens) != len(types)
        or not isinstance(merges, list)
    ):
        raise GgufError("tokenizer.ggml tokens, token types and merges are required")
    vocab, added = {}, []
    unused_from = None
    for index, (token, kind) in enumerate(zip(tokens, types)):
        if kind == _UNUSED:
            unused_from = index if unused_from is None else unused_from
            continue
        if unused_from is not None:
            raise GgufError(f"token {index} follows unused padding tokens")
        if kind == _NORMAL:
            if added:
                raise GgufError(f"normal token {index} follows added tokens")
            if token in vocab:
                raise GgufError(f"duplicate vocabulary token {token!r}")
            vocab[token] = index
        elif kind in (_CONTROL, _USER_DEFINED):
            added.append(
                {
                    "id": index,
                    "content": token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": kind == _CONTROL,
                }
            )
        else:
            raise GgufError(f"token {index} has unsupported token type {kind}")
    byte_level = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": QWEN35_SPLIT_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                dict(byte_level),
            ],
        },
        "post_processor": dict(byte_level),
        "decoder": dict(byte_level),
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": list(merges),
        },
    }

    def token(key: str, index=None) -> str:
        index = metadata.get(key) if index is None else index
        if type(index) is not int or not 0 <= index < len(tokens) or types[index] == _UNUSED:
            raise GgufError(f"{key} does not name a used token")
        return tokens[index]

    if metadata.get("tokenizer.ggml.add_bos_token", False) is not False:
        raise GgufError("this tokenizer does not prepend BOS")
    tokenizer_config = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": _added_tokens_decoder(added),
        "bos_token": None,
        "clean_up_tokenization_spaces": False,
        "eos_token": token("tokenizer.ggml.eos_token_id"),
        "errors": "replace",
        "model_max_length": config["max_position_embeddings"],
        "pad_token": token("tokenizer.ggml.padding_token_id"),
        "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer",
        "unk_token": None,
    }
    token(_KEY + "ple.eos_token_id", config["ple_eos_token_id"])
    if metadata.get("tokenizer.ggml.bos_token_id") is not None:
        token("tokenizer.ggml.bos_token_id")
    eos = [metadata["tokenizer.ggml.eos_token_id"], config["ple_eos_token_id"]]
    generation = {
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "do_sample": True,
        "eos_token_id": list(dict.fromkeys(eos)),
        "pad_token_id": metadata["tokenizer.ggml.padding_token_id"],
        "temperature": _f32_text(metadata, "general.sampling.temp", 1.0),
        "top_k": metadata.get("general.sampling.top_k", 20),
        "top_p": _f32_text(metadata, "general.sampling.top_p", 1.0),
    }
    template = metadata.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template:
        raise GgufError("tokenizer.chat_template is required")

    return {
        "tokenizer.json": _json_bytes(tokenizer),
        "tokenizer_config.json": _json_bytes(tokenizer_config),
        "chat_template.jinja": template.encode("utf-8"),
        "generation_config.json": _json_bytes(generation),
    }


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _added_tokens_decoder(added: list[dict]) -> dict:
    return {
        str(item["id"]): {key: item[key] for key in sorted(item) if key != "id"}
        for item in added
    }


def looks_special(content: str) -> bool:
    """llama.cpp's converter stores ``<|...|>`` added tokens as CONTROL even when not special."""
    return content.startswith("<|") and content.endswith("|>")


def check_tokenizer(candidate: dict, synthesized: dict) -> list[int]:
    """Require a supplied tokenizer.json to encode exactly like the GGUF vocabulary.

    Vocabulary, merges, pre-tokenizer pipeline and added-token ids and contents must be equal.
    The ``special`` flag may differ only where the GGUF cannot record it: a ``<|...|>`` token
    that llama.cpp marked CONTROL. Returns the ids whose flag the candidate changes.
    """
    for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder"):
        if candidate.get(key) != synthesized[key]:
            raise ValueError(f"tokenizer.json {key} differs from the GGUF tokenizer")
    model, expected = candidate.get("model", {}), synthesized["model"]
    for key in expected:
        if model.get(key) != expected[key]:
            raise ValueError(f"tokenizer.json model.{key} differs from the GGUF tokenizer")
    added = candidate.get("added_tokens", [])
    if [(t.get("id"), t.get("content")) for t in added] != [
        (t["id"], t["content"]) for t in synthesized["added_tokens"]
    ]:
        raise ValueError("tokenizer.json added tokens differ from the GGUF tokenizer")
    changed = []
    for ours, theirs in zip(synthesized["added_tokens"], added):
        if {k: v for k, v in theirs.items() if k != "special"} != {
            k: v for k, v in ours.items() if k != "special"
        }:
            raise ValueError(f"tokenizer.json token {ours['id']} attributes differ")
        if theirs.get("special") != ours["special"]:
            if not (ours["special"] and looks_special(ours["content"])):
                raise ValueError(f"tokenizer.json token {ours['id']} special flag conflicts")
            changed.append(ours["id"])
    return changed


def build_model(
    gguf: GgufModel,
    *,
    subset: Subset | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    config = text_config(gguf.metadata)
    mappings = name_map(gguf, config)
    resources = tokenizer_resources(gguf.metadata, config)
    for role, path in (resource_overrides or {}).items():
        if role not in resources:
            raise ValueError(f"resource override {role!r} has no consumer")
        data = Path(path).read_bytes()
        if role == "tokenizer.json":
            candidate = json.loads(data)
            check_tokenizer(candidate, json.loads(resources[role]))
            if "tokenizer_config.json" not in resource_overrides:
                # Keep the synthesized decoder's special flags equal to the chosen tokenizer.
                settings = json.loads(resources["tokenizer_config.json"])
                settings["added_tokens_decoder"] = _added_tokens_decoder(
                    candidate["added_tokens"]
                )
                resources["tokenizer_config.json"] = _json_bytes(settings)
        resources[role] = data
    vocab = gguf.tensor("token_embd.weight").shape[0]
    config["vocab_size"] = vocab
    count, special = token_domain(
        json.loads(resources["tokenizer.json"]),
        json.loads(resources["tokenizer_config.json"]),
        vocab,
    )
    references = {role: f"resource/text/{role}" for role in resources}
    model = Model(
        {"text": {"config": config, "resources": references}},
        resources={references[role]: data for role, data in resources.items()},
        token_count=count,
        special_token_ids=special,
    )
    groups: dict[tuple[int, str], list[str]] = {}
    for item in mappings:
        if subset is not None and not subset.keeps(gguf, item.gguf):
            continue
        tensor = gguf.tensor(item.gguf)
        rows, shape = item.rows, item.shape
        if item.gguf == PLE_TABLE and subset is not None:
            rows, shape = (0, subset.ple_rows), (subset.ple_rows, shape[1])
        source = tensor_source(gguf, item.gguf, rows=rows, shape=shape)
        model.add(
            Parameter(
                item.parameter,
                shape,
                source,
                inputs=item.inputs,
                # Unassigned block tensors fall back to their exact FP32 decode, never a rounding.
                direct_format=tensor.type.format if tensor.type.dtype is not None else "fp32",
            )
        )
        if item.rows is not None:
            layer = int(item.parameter.split("/")[2])
            bank = "down" if item.parameter.endswith("/down") else "gate_up"
            groups.setdefault((layer, bank), []).append(item.parameter)
    for (layer, bank), names in sorted(groups.items()):
        if bank == "gate_up":
            # Expert e is gate_e's rows then up_e's rows: two contiguous byte ranges per expert.
            names = sorted(names, key=lambda name: (int(name.split("/")[5]), name.endswith("/up")))
        else:
            names = sorted(names, key=lambda name: int(name.split("/")[5]))
        model.packing_groups.append(tuple(names))
    return model


def encoded_format(parameter: Parameter) -> str | None:
    """The GGML block format of a parameter's source rows, or None for direct words."""
    if parameter.source.read_encoded is None:
        return None
    return parameter.source.read_encoded(0, 1).format
