"""Create a 1,500-token repetitive Python repository-edit benchmark prefix."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


SOURCE_MODEL = Path(r"D:\mtp-k3-training\source")
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "benchmark-results/prompt-lookup-code-long"
PREFIX_TOKENS = 1_500
ENTITY_NAMES = (
    "user",
    "project",
    "team",
    "invoice",
    "subscription",
    "workspace",
    "document",
    "comment",
    "notification",
    "integration",
    "deployment",
    "environment",
)


def repository_module() -> str:
    sections = []
    for entity in ENTITY_NAMES:
        title = entity.title().replace("_", "")
        sections.append(
            f'''def create_{entity}(store, name, email):
    if not name:
        raise ValueError("name is required")
    if not email:
        raise ValueError("email is required")
    if store.contains(email):
        raise ValueError("email already exists")
    item = {title}(name=name, email=email)
    store.save(item)
    return item

def update_{entity}(store, item_id, name, email):
    if not name:
        raise ValueError("name is required")
    if not email:
        raise ValueError("email is required")
    if store.contains(email):
        raise ValueError("email already exists")
    item = store.get(item_id)
    item.name = name
    item.email = email
    store.save(item)
    return item
'''
        )
    return "\n".join(sections)


def main() -> None:
    target = ENTITY_NAMES[-1]
    prompt = f"""Change this Python repository code by implementing clone_{target} in the same style.
Return only valid Python.

```python
{repository_module()}
def clone_{target}(store, item_id, name, email):
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

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    corpus = OUTPUT_ROOT / "python_repository_edit_1500.ids"
    corpus.write_text(" ".join(str(token_id) for token_id in prefix) + "\n", encoding="ascii")
    report = {
        "prompt_kind": "repetitive Python repository edit",
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
