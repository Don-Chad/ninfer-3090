---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - multimodal
  - conversational
  - cuda
  - rtx-5090
---

# Qwen3.6-35B-A3B for NInfer

This model card is the version-controlled source for
[neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer).

The repository contains
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) converted to the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. The artifact is intended
only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_6_35b_a3b.ninfer` |
| Size | 22,373,184,256 bytes (20.84 GiB) |
| SHA-256 | `9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163` |
| Container version | 1 |
| NInfer model ID | `qwen3.6-35b-a3b` |
| NInfer target key | `qwen3_6_35b_a3b` |

The file contains the registered Text, Vision, MTP, proposal-head, tokenizer, chat-template,
generation, and media-processor objects required by NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163' \
  'qwen3_6_35b_a3b.ninfer' | sha256sum --check
```

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#build) for source-build dependencies.

## Download and run

```bash
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

For images, videos, structured chat history, and HTTP serving, see the
[NInfer documentation](https://github.com/Neroued/ninfer/tree/master/docs).

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- the NInfer CLI;
- OpenAI Chat Completions and Anthropic Messages serving.

## Performance

On an RTX 5090, the published NInfer MTP=3 measurement for a 7,678-token prompt and 512 generated
tokens reports 13,411.71 prefill tok/s and 436.32 decode tok/s, averaged over five measured runs
after one warm-up. See the
[full methodology and comparison](https://github.com/Neroued/ninfer/blob/master/docs/performance.md).

## Limits

- The artifact is accepted only by the matching NInfer target.
- NInfer currently executes on one RTX 5090, one CUDA device, and one active request per Engine.
- It does not provide continuous batching, multi-GPU execution, CPU/GPU offload, or distributed
  serving.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | `Qwen/Qwen3.6-35B-A3B` |
| Source revision | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Conversion recipe | `qwen3_6_35b_a3b-v1` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter revision | `19d17f0dfe655e4a2e495f0ed992c8c168f31862` |

The complete object inventory and conversion metadata are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/blob/main/artifact-manifest.json).

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.
