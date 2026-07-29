"""Create a 1,500-token code-refactor fixture with one explicit 2x duplicate."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODEL = Path(r"D:\mtp-k3-training\source")
OUTPUT_ROOT = ROOT / "benchmark-results/prompt-lookup-code-2x"
PREFIX_TOKENS = 1_500

DUPLICATED_HELPER = """\
def normalize_retry_policy(config):
    attempts = max(1, min(int(config.get("attempts", 3)), 10))
    delay_ms = max(0, int(config.get("delay_ms", 250)))
    jitter = bool(config.get("jitter", True))
    return {"attempts": attempts, "delay_ms": delay_ms, "jitter": jitter}
"""


def main() -> None:
    matcher = (ROOT / "src/runtime/speculative/prompt_lookup.cpp").read_text(encoding="utf-8")
    options = (ROOT / "apps/cli/options.cpp").read_text(encoding="utf-8")
    prompt = f"""Refactor this mixed Python/C++ patch. Move the duplicated retry-policy helper into
a shared Python module, update both callers, and tighten the C++ validation without changing
observable behavior. Return a concise unified diff.

```cpp
// prompt_lookup.cpp
{matcher}
```

```cpp
// options.cpp excerpt
{options[-9_000:]}
```

```python
# worker.py
{DUPLICATED_HELPER}

def schedule_worker(config, queue):
    policy = normalize_retry_policy(config)
    return queue.schedule(policy["attempts"], policy["delay_ms"], policy["jitter"])
```

```python
# uploader.py
{DUPLICATED_HELPER}

def schedule_upload(config, uploader):
    policy = normalize_retry_policy(config)
    return uploader.schedule(policy["attempts"], policy["delay_ms"], policy["jitter"])
```
"""
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(token_ids) < PREFIX_TOKENS:
        raise ValueError(f"fixture has only {len(token_ids)} tokens")
    prefix = token_ids[-PREFIX_TOKENS:]
    helper_ids = tokenizer.encode(DUPLICATED_HELPER, add_special_tokens=False)
    decoded_prefix = tokenizer.decode(prefix)
    occurrences = decoded_prefix.count("def normalize_retry_policy(config):")
    if occurrences != 2:
        raise ValueError(f"expected exactly two helper copies, found {occurrences}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    corpus = OUTPUT_ROOT / "code_refactor_2x_1500.ids"
    corpus.write_text(" ".join(map(str, prefix)) + "\n", encoding="ascii")
    report = {
        "prompt_kind": "mixed code refactor with one explicit 2x duplicate",
        "rendered_tokens": len(token_ids),
        "benchmark_prefix_tokens": len(prefix),
        "duplicated_helper_tokens": len(helper_ids),
        "duplicated_helper_occurrences": occurrences,
        "corpus": str(corpus),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
