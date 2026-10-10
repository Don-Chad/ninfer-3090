"""A tiny synthetic qwen4exp GGUF with every tensor family, for converter tests."""

from __future__ import annotations

from pathlib import Path

from .gguf_writer import tensor_bytes, write_gguf

H, F, E, HC, LOW = 64, 128, 4, 2, 8
F32, F16, Q8_0, IQ4_NL, BF16, Q2_0 = 0, 1, 8, 20, 30, 42
PLE_ROWS, PLE_DIM = 300, 32
TOKENS = [chr(ord("a") + i) for i in range(10)] + [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "[PAD14]",
    "[PAD15]",
]
TOKEN_TYPES = [1] * 10 + [3, 3, 3, 4, 5, 5]


def metadata(**changes) -> list[tuple[str, int, object]]:
    a = "qwen4exp."
    items = {
        "general.architecture": (8, "qwen4exp"),
        a + "block_count": (4, 4),
        a + "context_length": (4, 4096),
        a + "embedding_length": (4, H),
        a + "attention.head_count": (4, 2),
        a + "attention.head_count_kv": (4, 1),
        a + "rope.dimension_sections": (9, (5, [2, 1, 1, 0])),
        a + "rope.freq_base": (6, 10000000.0),
        a + "attention.layer_norm_rms_epsilon": (6, 1e-6),
        a + "expert_count": (4, E),
        a + "expert_used_count": (4, 2),
        a + "attention.key_length": (4, 32),
        a + "attention.value_length": (4, 32),
        a + "expert_feed_forward_length": (4, F),
        a + "expert_shared_feed_forward_length": (4, F),
        a + "ssm.conv_kernel": (4, 4),
        a + "ssm.state_size": (4, 16),
        a + "ssm.group_count": (4, 2),
        a + "ssm.time_step_rank": (4, 4),
        a + "ssm.inner_size": (4, 64),
        a + "full_attention_interval": (4, 4),
        a + "rope.dimension_count": (4, 8),
        a + "hyper_connection.count": (4, HC),
        a + "hyper_connection.low_rank": (4, LOW),
        a + "attention.indexer.head_count": (4, 2),
        a + "attention.indexer.key_length": (4, 16),
        a + "attention.indexer.top_k": (4, 8),
        a + "attention.compress_ratios": (9, (5, [0, 0, 0, 4])),
        a + "ple.layers": (9, (5, [1])),
        a + "ple.ngram_size": (4, 3),
        a + "ple.heads_per_ngram": (4, 2),
        a + "ple.conv_kernel": (4, 4),
        a + "ple.eos_token_id": (4, 10),
        a + "ple.image_token_id": (4, 11),
        a + "embedding_length_per_layer_input": (4, PLE_DIM),
        a + "ple.layer_multipliers": (9, (11, [11, 13, 17])),
        a + "ple.head_offsets": (9, (11, [0, 5, 12, 23])),
        a + "ple.head_vocab_sizes": (9, (11, [5, 7, 11, 13])),
        "tokenizer.ggml.model": (8, "gpt2"),
        "tokenizer.ggml.pre": (8, "qwen35"),
        "tokenizer.ggml.tokens": (9, (8, TOKENS)),
        "tokenizer.ggml.token_type": (9, (5, TOKEN_TYPES)),
        "tokenizer.ggml.merges": (9, (8, ["a b", "ab c"])),
        "tokenizer.ggml.eos_token_id": (4, 12),
        "tokenizer.ggml.padding_token_id": (4, 10),
        "tokenizer.ggml.bos_token_id": (4, 10),
        "tokenizer.ggml.add_bos_token": (7, False),
        "tokenizer.chat_template": (8, "{{ messages }}"),
        "general.sampling.top_k": (5, 20),
        "general.sampling.top_p": (6, 0.95),
        "general.sampling.temp": (6, 1.0),
    }
    items.update(changes)
    return [(key, kind, value) for key, (kind, value) in items.items() if kind is not None]


def tensors(drop=(), extra=()) -> list[tuple[str, int, tuple[int, ...], bytes]]:
    specs = [
        ("token_embd.weight", Q8_0, (H, 16)),
        ("output.weight", Q8_0, (H, 16)),
        ("output_hc_down.weight", BF16, (HC * H, LOW)),
        ("output_hc_up.weight", BF16, (LOW, HC * H)),
        ("output_hc_norm.weight", F32, (HC * H,)),
        ("per_layer_token_embd.weight", IQ4_NL, (PLE_DIM, PLE_ROWS)),
    ]
    for layer in range(4):
        p = f"blk.{layer}."
        for part in ("attn", "ffn"):
            specs += [
                (p + f"hc_{part}_down.weight", BF16, (HC * H, LOW)),
                (p + f"hc_{part}_up.weight", BF16, (LOW, HC * H)),
                (p + f"hc_{part}_inject.weight", BF16, (HC * H, HC)),
                (p + f"hc_{part}_norm.weight", F32, (HC * H,)),
            ]
        specs += [
            (p + "ffn_gate_inp.weight", BF16, (H, E)),
            (p + "ffn_gate_inp_shexp.weight", BF16, (H,)),
            (p + "ffn_gate_shexp.weight", Q8_0, (H, F)),
            (p + "ffn_up_shexp.weight", IQ4_NL, (H, F)),
            (p + "ffn_down_shexp.weight", Q2_0, (F, H)),
            (p + "ffn_gate_exps.weight", Q8_0, (H, F, E)),
            (p + "ffn_up_exps.weight", Q8_0, (H, F, E)),
            (p + "ffn_down_exps.weight", Q2_0, (F, H, E)),
        ]
        if layer == 3:
            specs += [
                (p + "attn_q.weight", Q8_0, (H, 128)),
                (p + "attn_k.weight", Q8_0, (H, 32)),
                (p + "attn_v.weight", IQ4_NL, (H, 32)),
                (p + "attn_output.weight", Q8_0, (H, H)),
                (p + "attn_q_norm.weight", F32, (32,)),
                (p + "attn_k_norm.weight", F32, (32,)),
                (p + "indexer.q_proj.weight", BF16, (H, 32)),
                (p + "indexer.k_proj.weight", BF16, (H, 16)),
                (p + "indexer.q_norm.weight", F32, (16,)),
                (p + "indexer.k_norm.weight", F32, (16,)),
            ]
        else:
            specs += [
                (p + "attn_qkv.weight", Q8_0, (H, 128)),
                (p + "attn_gate.weight", Q8_0, (H, 64)),
                (p + "ssm_alpha.weight", BF16, (H, 4)),
                (p + "ssm_beta.weight", BF16, (H, 4)),
                (p + "ssm_a", F32, (4,)),
                (p + "ssm_dt.bias", F32, (4,)),
                (p + "ssm_conv1d.weight", F32, (4, 128)),
                (p + "ssm_norm.weight", F32, (16,)),
                (p + "ssm_out.weight", Q8_0, (64, H)),
            ]
        if layer == 1:
            specs += [
                (p + "ple_key.weight", BF16, (PLE_DIM * 2, HC * H)),
                (p + "ple_value.weight", BF16, (PLE_DIM * 2, H)),
                (p + "ple_norm_key.weight", F32, (HC * H,)),
                (p + "ple_norm_query.weight", F32, (HC * H,)),
                (p + "ple_norm_conv.weight", F32, (HC * H,)),
                (p + "ple_conv1d.weight", F16, (4, HC * H)),
            ]
    specs = [spec for spec in specs if spec[0] not in drop] + list(extra)
    return [
        (name, kind, ne, tensor_bytes(kind, ne, seed=index))
        for index, (name, kind, ne) in enumerate(specs)
    ]


def write(path: Path, *, metadata_changes=None, drop=(), extra=()) -> Path:
    return write_gguf(path, metadata(**(metadata_changes or {})), tensors(drop, extra))
