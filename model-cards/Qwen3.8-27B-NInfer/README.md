---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - multimodal
  - conversational
  - cuda
  - rtx-5090
---

# Qwen3.8-27B for NInfer

This model card is the version-controlled source for
[neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer).

The repository contains
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted to the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. The artifact is intended
only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b.ninfer` |
| Size | 18,210,531,328 bytes (16.96 GiB) |
| SHA-256 | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `groupwise-int` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,124 (1,118 tensors and 6 resources) |

The Text body uses the registered Q4/Q5/Q6 groupwise allocation, while the token embedding and
full output head use `W8G32_F16S`. The file also contains the registered Vision, MTP,
proposal-head, tokenizer, chat-template, generation, and media-processor objects required by
NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e' \
  'qwen3_8_27b.ninfer' | sha256sum --check
```

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) revision
  [`5232055`](https://github.com/Neroued/ninfer/commit/52320554b5e71a9da96bff809ddf67ac5773ed63)
  or later, built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#build) for source-build dependencies.

## Download and run

```bash
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
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
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI- and Anthropic-compatible serving.

## Limits

- The artifact is accepted only by NInfer revision `5232055` or later and the matching registered
  target.
- NInfer executes on one RTX 5090 and one CUDA device, with a startup-fixed capacity of 1–8 active
  requests per Engine.
- It does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| Conversion recipe | `qwen3_8_27b-v1` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter revision | `52320554b5e71a9da96bff809ddf67ac5773ed63` |
| Minimum runtime revision | `52320554b5e71a9da96bff809ddf67ac5773ed63` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The local source configuration, tensor index, frontend resources, and published CRC32 inventory
were checked against the canonical Hugging Face source revision above before conversion
publication.

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the
[Qwen3.8-27B artifact reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.8-27b-artifact.md).

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.
