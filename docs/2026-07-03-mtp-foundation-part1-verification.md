# MTP Foundation Part 1 Verification

This report records the local verification artifacts for
`docs/2026-07-02-mtp-foundation-part1-design.md` Part 1.

## q5090 v3 MTP_DRAFT Layout

Generated artifact:

```text
out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus
out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus.manifest.json
out/conv_dump.v3_mtp_w8g32.json
```

These files are under `out/` and are ignored by git.

Generation command:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090_convert.convert \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus
```

Converter summary:

```text
blocks=1164 segments=1312 fusion_groups=130 modules=3
TEXT_CORE      blocks=819 payload=16378329088
MTP_DRAFT      blocks=12  payload=451267584
VISION_ENCODER blocks=333 payload=293396992
file_bytes=17123284480
```

Verifier command:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090_convert.verify \
  out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --dump out/conv_dump.v3_mtp_w8g32.json
```

Verifier result:

```text
L0 structural checks done; problems=0
L1 value checks done; problems=0
OK: L0 + L1 passed in 198s

Q4G64_F16S   blocks=182 scale_bad=0 code_bad=0 control_bad=0
Q5G64_F16S   blocks=295 scale_bad=0 code_bad=0 control_bad=0
Q6G64_F16S   blocks=2   scale_bad=0 code_bad=0 control_bad=0
W8G128_F16S  blocks=2   scale_bad=0 code_bad=0 control_bad=0
BF16_CTRL    blocks=582 scale_bad=0 code_bad=0 control_bad=0
FP32_CTRL    blocks=96  scale_bad=0 code_bad=0 control_bad=0
W8G32_F16S   blocks=5   scale_bad=0 code_bad=0 control_bad=0
```

The new MTP_DRAFT layout contributes 12 blocks, 16 segments, and 2 fusion groups:
`ATTN_IN` and `MLP_GATEUP`. The five MTP dense/fused blocks use `W8G32_F16S`; the seven
MTP control tensors use `BF16_CTRL`. `TEXT_CORE` has no W8 tensors. The only `W8G128_F16S`
tensors are the existing VISION merger FC weights.

## Python Ref Model Verified MTP

Verification command:

```bash
/home/neroued/miniconda3/envs/py311/bin/python - <<'PY'
from tools.parity.ref_model import (
    DEFAULT_MODEL,
    DEFAULT_STOP_TOKEN_IDS,
    RefModel,
    chat_prompt_ids,
    load_tokenizer,
    parse_stop_token_ids,
)

weights = 'out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus'
prompt = '''请编写一个 Python 函数 `merge_intervals(intervals)`，要求：
1. 输入为 List[List[int]]，每个区间包含开始和结束两个整数；
2. 合并所有重叠或相邻的区间；
3. 返回按起点升序排列的区间列表；
4. 给出简短解释，并写出两个 assert 测试用例。'''
decode = 64
model = RefModel(weights, device='cuda', resident='auto')
tokenizer = load_tokenizer(DEFAULT_MODEL)
prompt_ids, _ = chat_prompt_ids(tokenizer, [{'role': 'user', 'content': prompt}])
stop = parse_stop_token_ids(DEFAULT_STOP_TOKEN_IDS)

base = model.forward(prompt_ids, decode, stop_token_ids=stop)
base_text = tokenizer.decode(base, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(f'PROMPT_TOKENS {len(prompt_ids)}')
print(f'DECODE {decode}')
print(f'BASE_TEXT {base_text!r}')
for draft_count in range(1, 6):
    tokens, stats = model.forward_mtp_verified(prompt_ids, decode, draft_count=draft_count, stop_token_ids=stop)
    print(
        f'MTP draft_count={draft_count} match={tokens == base} '
        f'draft_tokens={stats.draft_tokens} accepted_tokens={stats.accepted_tokens} '
        f'acceptance_rate={stats.acceptance_rate:.6f} '
        f'acceptance_length={stats.acceptance_length:.6f} '
        f'drafted_per_pos={stats.draft_tokens_per_pos} '
        f'accepted_per_pos={stats.accepted_tokens_per_pos}'
    )
PY
```

Result:

```text
PROMPT_TOKENS 89
DECODE 64
BASE_TEXT '```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    合并所有重叠或相邻的区间。

    算法思路：
    1. 如果输入为空，返回空列表。
    2. 按区间的起点'
```

| draft tokens requested | output matched baseline | verified draft tokens | accepted tokens | acceptance rate | acceptance length | accepted / drafted by draft position |
|---:|---|---:|---:|---:|---:|---|
| 1 | yes | 33 | 30 | 0.909091 | 1.909091 | `[30] / [33]` |
| 2 | yes | 43 | 41 | 0.953488 | 2.863636 | `[22, 19] / [22, 21]` |
| 3 | yes | 52 | 45 | 0.865385 | 3.500000 | `[16, 15, 14] / [18, 17, 17]` |
| 4 | yes | 51 | 49 | 0.960784 | 4.500000 | `[13, 12, 12, 12] / [14, 13, 12, 12]` |
| 5 | yes | 60 | 50 | 0.833333 | 4.846154 | `[13, 12, 10, 8, 7] / [13, 12, 12, 12, 11]` |

The prompt naturally ran to the requested 64 decoded tokens with the default stop-token set. During
validation, the old batched target verification produced a real `draft_count=5` mismatch at token
index 37 by accepting a token from the batched target path that differed from greedy one-token target
decode. The ref model now verifies draft candidates with one-token target decode steps, so verified
MTP output matches non-MTP greedy for draft counts 1..5.

## C++ Scope

This W8G32 MTP precision update is verified in the q5090 Python converter/verifier and Python ref
model. C++ runtime/model-card support for MTP remains outside Part 1.
