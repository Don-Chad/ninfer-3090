"""Create a compact, realistically repetitive Python code-edit benchmark prefix."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


SOURCE_MODEL = Path(r"D:\mtp-k3-training\source")
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "benchmark-results/prompt-lookup-code"
PREFIX_TOKENS = 128

PROMPT = """Change this Python repository code by implementing clone_user in the same style.
Return only valid Python.

```python
def create_user(store, name, email):
    if not name:
        raise ValueError("name is required")
    if not email:
        raise ValueError("email is required")
    if store.contains(email):
        raise ValueError("email already exists")
    user = User(name=name, email=email)
    store.save(user)
    return user

def update_user(store, user_id, name, email):
    if not name:
        raise ValueError("name is required")
    if not email:
        raise ValueError("email is required")
    if store.contains(email):
        raise ValueError("email already exists")
    user = store.get(user_id)
    user.name = name
    user.email = email
    store.save(user)
    return user

def clone_user(store, user_id, name, email):
```
"""


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(token_ids) < PREFIX_TOKENS:
        raise ValueError(f"fixture has only {len(token_ids)} tokens")
    prefix = token_ids[-PREFIX_TOKENS:]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    corpus = OUTPUT_ROOT / "python_code_edit_repetition.ids"
    corpus.write_text(" ".join(str(token_id) for token_id in prefix) + "\n", encoding="ascii")
    report = {
        "prompt": PROMPT,
        "rendered_tokens": len(token_ids),
        "benchmark_prefix_tokens": len(prefix),
        "corpus": str(corpus),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
