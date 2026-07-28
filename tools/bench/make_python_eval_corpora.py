"""Export repository-disjoint Python evaluation prompts as benchmark corpora."""

from __future__ import annotations

import json
from pathlib import Path


TRACE_ROOT = Path(r"D:\mtp-k3-training\traces\eval")
OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "benchmark-results/python-eval-corpora"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest_path in sorted(TRACE_ROOT.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt_ids = manifest["prompt_ids"]
        if not prompt_ids or not all(type(token_id) is int for token_id in prompt_ids):
            raise ValueError(f"invalid prompt_ids in {manifest_path}")
        case_id = manifest_path.parent.name
        output_path = OUTPUT_ROOT / f"{case_id}.ids"
        output_path.write_text(
            " ".join(str(token_id) for token_id in prompt_ids) + "\n",
            encoding="ascii",
        )
        rows.append(
            {
                "case": case_id,
                "tokens": len(prompt_ids),
                "corpus": str(output_path),
                "prompt_source": manifest["prompt_source"],
            }
        )
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
