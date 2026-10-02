"""Score weight encodings of Qwen3.8-27B in llama.cpp against a reference, on NInfer corpora.

NInfer's grouped formats map exactly onto llama.cpp's Q8_0 blocks: a q4/q5/q6 group of 64 is two
Q8_0 blocks of 32 sharing the group's FP16 scale, a q8 group of 32 is one block, and a Q8_0 block
stores signed 8-bit codes with a signed FP16 scale. ``export`` therefore writes any NInfer encoding
of the BF16 source, including per-tensor format plans the engine cannot run yet, as a GGUF whose
weights dequantize to exactly the values the engine would use. It copies a Q8_0 template GGUF
(Unsloth's Qwen3.8-27B Q8_0, which has the same tensor names, types and byte offsets) and
overwrites each projection's blocks in place. The GDN ``a``/``b`` projections and the MTP layer
keep the template's Q8_0 values in every export.

``score`` runs ``llama-perplexity`` per corpus stream against saved reference logits
(``--kl-divergence-base``) and records PPL, mean KLD and top-token agreement.

    python tools/eval/gguf_eval.py export --source <Qwen3.8-27B dir> --template <Q8_0.gguf> \
        --encoder search --imatrix <imatrix.gguf> --out <variant.gguf>
    python tools/eval/gguf_eval.py score --model <variant.gguf> --name <label> \
        --base-dir <dir with STREAM.kld> --out <results dir>
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.artifact.formats import get_format  # noqa: E402
from tools.convert.imatrix import v_head_order as imatrix_order  # noqa: E402
from tools.convert.quantization.groupwise import (  # noqa: E402
    quantize_matrix,
    search_quantize_matrix,
)

Q4, Q5, Q6, Q8 = "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16"
HEAD_DIM = 256          # full-attention head dimension; q_proj interleaves query and gate per head
GDN_K_HEADS, GDN_V_HEADS, GDN_V_DIM = 16, 48, 128
LAYERS = 64
STREAMS = ("wikipedia_en", "wikipedia_zh", "arxiv", "github_code", "own_code", "synthetic_chat")

# Role formats of the qwen3_8_27b layout before 2026-10 (Q4/Q5 by role, Q8 embedding and head);
# --plan entries override them.
RECIPE = {
    "attention/query": Q4, "attention/gate": Q5, "attention/key": Q4, "attention/value": Q5,
    "attention/output": Q5, "gdn/query": Q4, "gdn/key": Q4, "gdn/value": Q5, "gdn/z": Q5,
    "gdn/output": Q5, "mlp/gate": Q4, "mlp/up": Q4, "mlp/down": Q5,
    "text/token_embedding": Q8, "text/output_head": Q8,
}


def v_head_order() -> torch.Tensor:
    """llama.cpp's grouped-to-tiled V-head order for this model: GGUF index i holds HF order[i]."""
    return imatrix_order(GDN_K_HEADS, GDN_V_HEADS, GDN_V_DIM)


@dataclass(frozen=True)
class Segment:
    """Rows ``rows`` of an HF matrix encoded as logical role ``role``."""

    role: str
    rows: torch.Tensor


@dataclass(frozen=True)
class Target:
    gguf: str
    hf: str
    segments: tuple[Segment, ...]
    row_order: torch.Tensor | None = None     # GGUF row i = HF row row_order[i]
    column_order: torch.Tensor | None = None  # GGUF column i = HF column column_order[i]


def targets(layer_types: list[str]) -> list[Target]:
    every = lambda n: torch.arange(n)  # noqa: E731
    out = [
        Target("token_embd.weight", "model.language_model.embed_tokens.weight",
               (Segment("text/token_embedding", every(248320)),)),
        Target("output.weight", "lm_head.weight", (Segment("text/output_head", every(248320)),)),
    ]
    v_order = v_head_order()
    for layer, kind in enumerate(layer_types):
        hf = f"model.language_model.layers.{layer}."
        blk = f"blk.{layer}."
        if kind == "full_attention":
            heads = torch.arange(12288).reshape(-1, 2, HEAD_DIM)
            out += [
                Target(blk + "attn_q.weight", hf + "self_attn.q_proj.weight",
                       (Segment("attention/query", heads[:, 0].reshape(-1)),
                        Segment("attention/gate", heads[:, 1].reshape(-1)))),
                Target(blk + "attn_k.weight", hf + "self_attn.k_proj.weight",
                       (Segment("attention/key", every(1024)),)),
                Target(blk + "attn_v.weight", hf + "self_attn.v_proj.weight",
                       (Segment("attention/value", every(1024)),)),
                Target(blk + "attn_output.weight", hf + "self_attn.o_proj.weight",
                       (Segment("attention/output", every(5120)),)),
            ]
        else:
            qkv_order = torch.cat([torch.arange(4096), 4096 + v_order])
            out += [
                Target(blk + "attn_qkv.weight", hf + "linear_attn.in_proj_qkv.weight",
                       (Segment("gdn/query", torch.arange(0, 2048)),
                        Segment("gdn/key", torch.arange(2048, 4096)),
                        Segment("gdn/value", torch.arange(4096, 10240))),
                       row_order=qkv_order),
                Target(blk + "attn_gate.weight", hf + "linear_attn.in_proj_z.weight",
                       (Segment("gdn/z", every(6144)),), row_order=v_order),
                Target(blk + "ssm_out.weight", hf + "linear_attn.out_proj.weight",
                       (Segment("gdn/output", every(5120)),), column_order=v_order),
            ]
        out += [
            Target(blk + "ffn_gate.weight", hf + "mlp.gate_proj.weight",
                   (Segment("mlp/gate", every(17408)),)),
            Target(blk + "ffn_up.weight", hf + "mlp.up_proj.weight",
                   (Segment("mlp/up", every(17408)),)),
            Target(blk + "ffn_down.weight", hf + "mlp.down_proj.weight",
                   (Segment("mlp/down", every(5120)),)),
        ]
    return out


class Source:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = json.loads((root / "config.json").read_text())

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(str(self.root / self.index[name]), framework="pt") as handle:
            return handle.get_tensor(name)

    def layer_types(self) -> list[str]:
        text = self.config.get("text_config", self.config)
        return list(text["layer_types"])


class GptqSource:
    """Symmetric GPTQ int4 checkpoint (auto_gptq packing, desc_act off), e.g. an AutoRound export.

    ``qweight`` is int32 [K/8, N] with eight 4-bit codes per word along K, ``qzeros`` int32
    [K/G, N/8], ``scales`` FP16 [K/G, N]; a weight is ``scale * (code - zero)``. Checkpoint format
    ``gptq`` (v1) stores each zero minus one. Only symmetric groups (zero 8) are accepted, so
    ``code - 8`` is a signed code in [-8, 7] under the checkpoint's own (possibly negative) scale.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        index = root / "model.safetensors.index.json"
        self.index = json.loads(index.read_text())["weight_map"]
        config = json.loads((root / "config.json").read_text())["quantization_config"]
        if config.get("bits") != 4 or not config.get("sym") or config.get("desc_act"):
            raise ValueError("expected symmetric 4-bit GPTQ without act-order")
        self.group = int(config["group_size"])
        self.zero_offset = 1 if config.get("checkpoint_format", "gptq") == "gptq" else 0

    def _tensor(self, name: str) -> torch.Tensor:
        with safe_open(str(self.root / self.index[name]), framework="pt") as handle:
            return handle.get_tensor(name)

    def decode(self, module: str) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        if module + ".qweight" not in self.index:
            return None
        shifts = torch.arange(0, 32, 4, dtype=torch.int32)
        qweight = self._tensor(module + ".qweight")
        codes = ((qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(-1, qweight.shape[1])
        qzeros = self._tensor(module + ".qzeros")
        zeros = ((qzeros.unsqueeze(2) >> shifts.view(1, 1, 8)) & 0xF).reshape(qzeros.shape[0], -1)
        zeros = zeros + self.zero_offset
        if not bool((zeros == 8).all()):
            raise ValueError(f"{module}: GPTQ zeros are not symmetric")
        scales = self._tensor(module + ".scales").to(torch.float16)
        return (codes.T.contiguous() - 8).to(torch.int8), scales.T.contiguous(), self.group, 4


def load_imatrix(path: Path) -> dict[str, torch.Tensor]:
    import gguf

    reader = gguf.GGUFReader(str(path))
    sums, counts = {}, {}
    for tensor in reader.tensors:
        if tensor.name.endswith(".in_sum2"):
            sums[tensor.name[: -len(".in_sum2")]] = torch.from_numpy(np.array(tensor.data))
        elif tensor.name.endswith(".counts"):
            counts[tensor.name[: -len(".counts")]] = float(np.array(tensor.data).reshape(-1)[0])
    return {name: values / max(counts.get(name, 1.0), 1.0) for name, values in sums.items()}


def parse_plan(text: str | None) -> dict[tuple[int | None, str], str]:
    """``role=fmt`` or ``LAYER:role=fmt`` entries, comma separated, overriding ``RECIPE``."""
    plan: dict[tuple[int | None, str], str] = {}
    for entry in filter(None, (text or "").split(",")):
        key, fmt = entry.split("=")
        layer, _, role = key.rpartition(":")
        fmt = {"q4": Q4, "q5": Q5, "q6": Q6, "q8": Q8}.get(fmt, fmt)
        get_format(fmt)
        plan[(int(layer) if layer else None, role)] = fmt
    return plan


def role_format(plan, layer: int | None, role: str) -> str:
    return plan.get((layer, role)) or plan.get((None, role)) or RECIPE[role]


def encode(weight: torch.Tensor, fmt: str, encoder: str, importance: torch.Tensor | None,
           device: str):
    if encoder == "rtn":
        result = quantize_matrix(weight, fmt, device=device)
    else:
        result = search_quantize_matrix(
            weight, fmt, importance=importance, negative_scales=encoder == "search-neg",
            device=device,
        )
    return result.codes.cpu(), result.scales.cpu()


def q8_0_blocks(codes: torch.Tensor, scales: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand grouped codes/scales to per-32 Q8_0 block codes [rows, K] and scales [rows, K/32]."""
    rows, groups, group = codes.shape
    flat = codes.reshape(rows, groups * group)[:, :k]
    block_scales = scales.repeat_interleave(group // 32, dim=1)[:, : k // 32]
    return flat, block_scales


def q8_0_bytes(target: Target, codes: torch.Tensor, scales: torch.Tensor) -> bytes:
    """Apply llama.cpp's row/column order to HF-layout block codes and pack Q8_0 blocks."""
    n, k = codes.shape
    if target.row_order is not None:
        codes, scales = codes[target.row_order], scales[target.row_order]
    if target.column_order is not None:
        block_order = target.column_order[::32] // 32
        codes, scales = codes[:, target.column_order], scales[:, block_order]
    blocks = torch.empty((n, k // 32, 34), dtype=torch.uint8)
    blocks[:, :, :2] = scales.contiguous().view(torch.uint8).reshape(n, k // 32, 2)
    blocks[:, :, 2:] = codes.contiguous().view(torch.uint8).reshape(n, k // 32, 32)
    return blocks.numpy().tobytes()


def hf_importance(imatrix: dict[str, torch.Tensor], target: Target) -> torch.Tensor | None:
    """The imatrix vector for ``target`` in HF column order."""
    importance = imatrix.get(target.gguf)
    if importance is not None and target.column_order is not None:
        hf_order = torch.empty_like(importance)
        hf_order[target.column_order] = importance
        importance = hf_order
    return importance


def export(args) -> None:
    import gguf

    source = Source(args.source)
    plan = parse_plan(args.plan)
    imatrix = load_imatrix(args.imatrix) if args.imatrix else {}
    gptq = GptqSource(args.gptq) if args.gptq else None
    reader = gguf.GGUFReader(str(args.template))
    info = {tensor.name: tensor for tensor in reader.tensors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(".partial")
    shutil.copyfile(args.template, partial)
    report = {"encoder": args.encoder, "imatrix": str(args.imatrix) if args.imatrix else None,
              "plan": args.plan, "tensors": {}}
    layer_pattern = re.compile(r"^blk\.(\d+)\.")
    with partial.open("r+b") as handle:
        for target in targets(source.layer_types()):
            tensor = info[target.gguf]
            if tensor.tensor_type != gguf.GGMLQuantizationType.Q8_0:
                raise ValueError(f"{target.gguf}: template tensor is not Q8_0")
            weight = source.tensor(target.hf).to(torch.float32)
            n, k = weight.shape
            if [int(x) for x in tensor.shape] != [k, n]:
                raise ValueError(f"{target.gguf}: template shape {tensor.shape} != {(k, n)}")
            importance = hf_importance(imatrix, target)
            match = layer_pattern.match(target.gguf)
            layer = int(match.group(1)) if match else None
            codes = torch.empty((n, k), dtype=torch.int8)
            scales = torch.empty((n, k // 32), dtype=torch.float16)
            module = target.hf[: -len(".weight")]
            imported = None
            if gptq:
                imported = gptq.decode(module)
            pending = list(target.segments)
            if imported is not None:
                icodes, iscales, group, bits = imported
                if icodes.shape != (n, k):
                    raise ValueError(f"{target.gguf}: imported shape {tuple(icodes.shape)} != {(n, k)}")
                block_scales = iscales.repeat_interleave(group // 32, dim=1)
                pending = []
                for segment in target.segments:
                    fmt = role_format(plan, layer, segment.role)
                    spec = get_format(fmt)
                    # GPTQ codes fill roles of the same width; --widen also places them in wider
                    # formats instead of re-encoding those roles with --encoder.
                    usable = spec.bits == bits or (args.widen and spec.bits > bits)
                    if not usable or spec.group_size % group:
                        pending.append(segment)
                        continue
                    codes[segment.rows] = icodes[segment.rows]
                    scales[segment.rows] = block_scales[segment.rows]
                    report["tensors"].setdefault(target.gguf, []).append(
                        {"role": segment.role, "format": fmt, "codes": "imported"})
            for segment in pending:
                fmt = role_format(plan, layer, segment.role)
                rows = weight.index_select(0, segment.rows)
                for begin in range(0, rows.shape[0], 1024):
                    chunk = rows[begin : begin + 1024]
                    c, s = q8_0_blocks(*encode(chunk, fmt, args.encoder, importance,
                                               args.device), k)
                    codes[segment.rows[begin : begin + 1024]] = c
                    scales[segment.rows[begin : begin + 1024]] = s
                report["tensors"].setdefault(target.gguf, []).append(
                    {"role": segment.role, "format": fmt})
            handle.seek(int(tensor.data_offset))
            handle.write(q8_0_bytes(target, codes, scales))
            print(f"{target.gguf:<28} {[s['format'] for s in report['tensors'][target.gguf]]}",
                  flush=True)
    partial.replace(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")


_METRICS = {
    "ppl": r"Mean PPL\(Q\)\s*:\s*([\d.]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([\d.]+)",
    "ln_ppl_ratio": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([-\d.]+)",
    "kld": r"Mean\s+KLD:\s*([-\d.]+)",
    "kld_99": r"99\.0%\s+KLD:\s*([-\d.]+)",
    "same_top": r"Same top p:\s*([\d.]+)",
}


def _llama(args, extra: list[str]) -> str:
    command = [str(args.llama_perplexity), "-m", str(args.model), "-c", "4096", "-b", "4096",
               "-ub", "1024", "-t", str(args.threads), "-ngl", str(args.ngl),
               "--load-mode", getattr(args, "load_mode", "none"), *extra]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed:\n{completed.stderr[-4000:]}")
    return completed.stdout + completed.stderr


def reference(args) -> None:
    """Save reference logits for the first ``--chunks`` 4096-token chunks of each stream."""
    args.base_dir.mkdir(parents=True, exist_ok=True)
    sources = ({args.text.stem: args.text} if args.text else
               {stream: args.corpus / "data" / stream / "00.txt" for stream in args.streams})
    for stream, path in sources.items():
        text = _llama(args, ["-f", str(path),
                             "--chunks", str(args.chunks),
                             "--kl-divergence-base", str(args.base_dir / f"{stream}.kld")])
        (args.base_dir / f"{stream}.log").write_text(text, encoding="utf-8")
        match = re.search(r"Final estimate: PPL = ([\d.]+)", text)
        print(stream, match.group(1) if match else "?", flush=True)


def score(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    results = {"name": args.name, "model": str(args.model), "streams": {}}
    for stream in args.streams:
        base = args.base_dir / f"{stream}.kld"
        log = args.out / f"{args.name}-{stream}.log"
        text = _llama(args, ["--kl-divergence-base", str(base), "--kl-divergence"])
        log.write_text(text, encoding="utf-8")
        metrics = {}
        for key, pattern in _METRICS.items():
            match = re.search(pattern, text)
            metrics[key] = float(match.group(1)) if match else None
        results["streams"][stream] = metrics
        print(stream, metrics, flush=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                encoding="utf-8", newline="\n")


def probe_units(layer_types: list[str], band: int) -> list[tuple[str, list[str]]]:
    """Tensor groups whose sensitivity is probed: four roles per band of layers, plus the
    embedding and the output head."""
    roles = {
        "mixer-in": ("attn_qkv", "attn_gate", "attn_q", "attn_k", "attn_v"),
        "mixer-out": ("ssm_out", "attn_output"),
        "mlp-in": ("ffn_gate", "ffn_up"),
        "mlp-down": ("ffn_down",),
    }
    names = {t.gguf for t in targets(layer_types)}
    units = [("embedding", ["token_embd.weight"]), ("head", ["output.weight"])]
    for start in range(0, len(layer_types), band):
        layers = range(start, min(start + band, len(layer_types)))
        for role, suffixes in roles.items():
            members = [f"blk.{layer}.{suffix}.weight" for layer in layers for suffix in suffixes]
            units.append((f"L{start:02d}-{layers[-1]:02d}-{role}",
                          [name for name in members if name in names]))
    return units


def probe(args) -> None:
    """Measure the KLD each tensor group adds when encoded at ``--format`` inside the
    near-lossless template, one group at a time, restoring the template bytes after each."""
    import gguf

    source = Source(args.source)
    imatrix = load_imatrix(args.imatrix) if args.imatrix else {}
    fmt = parse_plan(f"x={args.format}")[(None, "x")]
    if not args.model.exists():
        shutil.copyfile(args.template, args.model)
    info = {t.name: t for t in gguf.GGUFReader(str(args.model)).tensors}
    by_name = {t.gguf: t for t in targets(source.layer_types())}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    backup = out / "restore.bin"
    restore_index = out / "restore.json"

    def restore() -> None:
        if not restore_index.exists():
            return
        spans = json.loads(restore_index.read_text())
        with backup.open("rb") as saved, args.model.open("r+b") as handle:
            for offset, size in spans:
                handle.seek(offset)
                handle.write(saved.read(size))
        restore_index.unlink()
        backup.unlink()

    restore()
    for unit, names in probe_units(source.layer_types(), args.band):
        result_path = out / f"{unit}.json"
        if result_path.exists() or (args.only and unit not in args.only):
            continue
        payloads = []
        for name in names:
            target = by_name[name]
            weight = source.tensor(target.hf).to(torch.float32)
            n, k = weight.shape
            codes, scales = encode(weight, fmt, args.encoder, hf_importance(imatrix, target),
                                   args.device)
            payloads.append((int(info[name].data_offset),
                             q8_0_bytes(target, *q8_0_blocks(codes, scales, k))))
        with args.model.open("r+b") as handle, backup.open("wb") as saved:
            spans = []
            for offset, payload in payloads:
                handle.seek(offset)
                saved.write(handle.read(len(payload)))
                spans.append((offset, len(payload)))
        restore_index.write_text(json.dumps(spans))
        with args.model.open("r+b") as handle:
            for offset, payload in payloads:
                handle.seek(offset)
                handle.write(payload)
        del payloads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        streams = {}
        for stream in args.streams:
            text = _llama(args, ["--kl-divergence-base", str(args.base_dir / f"{stream}.kld"),
                                 "--kl-divergence"])
            streams[stream] = {key: (float(m.group(1)) if (m := re.search(pattern, text)) else None)
                               for key, pattern in _METRICS.items()}
        restore()
        result_path.write_text(json.dumps({"unit": unit, "format": fmt, "tensors": names,
                                           "streams": streams}, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
        mean = sum(v["kld"] for v in streams.values()) / len(streams)
        print(f"{unit:<26} mean KLD {mean:.6f}", flush=True)


def ninfer_bytes(report: dict, layer_types: list[str], *, decode_only: bool = False) -> int:
    """Stored bytes of the exported projections in NInfer grouped formats (codes plus scales).

    ``decode_only`` leaves out the token embedding, of which decode reads one row per token."""
    rows = {t.gguf: {seg.role: len(seg.rows) for seg in t.segments} for t in targets(layer_types)}
    k_of = {"token_embd.weight": 5120, "output.weight": 5120}
    total = 0
    for name, segments in report["tensors"].items():
        if decode_only and name == "token_embd.weight":
            continue
        k = k_of.get(name) or (17408 if "ffn_down" in name else
                               6144 if ("attn_output" in name or "ssm_out" in name) else 5120)
        for segment in segments:
            spec = get_format(segment["format"])
            total += rows[name][segment["role"]] * k * (spec.bits / 8 + 2 / spec.group_size)
    return int(total)


def table(args) -> None:
    layer_types = Source(args.source).layer_types()
    lines = ["| Model | Size (GB) | Read/token (GB) | Mean KLD | Top-1 same | PPL vs ref | "
             + " | ".join(f"KLD {s}" for s in STREAMS) + " |",
             "|---|---:|---:|---:|---:|---:|" + "---:|" * len(STREAMS)]
    for path in sorted(args.results.glob("*.json")):
        result = json.loads(path.read_text())
        streams = result["streams"]
        if set(streams) != set(STREAMS):
            continue
        model = Path(result["model"])
        report = model.with_suffix(".json")
        if report.exists():
            exported = json.loads(report.read_text())
            size = ninfer_bytes(exported, layer_types) / 1e9
            read = f"{ninfer_bytes(exported, layer_types, decode_only=True) / 1e9:.2f}"
        else:
            size, read = model.stat().st_size / 1e9, "n/a"
        kld = sum(streams[s]["kld"] for s in STREAMS) / len(STREAMS)
        same = sum(streams[s]["same_top"] for s in STREAMS) / len(STREAMS)
        ratio = sum(streams[s]["ln_ppl_ratio"] for s in STREAMS) / len(STREAMS)
        lines.append(f"| {result['name']} | {size:.2f} | {read} | {kld:.4f} | {same:.2f}% | "
                     f"{100 * (np.exp(ratio) - 1):+.2f}% | "
                     + " | ".join(f"{streams[s]['kld']:.4f}" for s in STREAMS) + " |")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--source", type=Path, required=True)
    exporter.add_argument("--template", type=Path, required=True)
    exporter.add_argument("--encoder", choices=("rtn", "search", "search-neg"), default="rtn")
    exporter.add_argument("--imatrix", type=Path)
    exporter.add_argument("--plan", help="format overrides, e.g. 'attention/output=q4,0:mlp/down=q8'")
    exporter.add_argument("--out", type=Path, required=True)
    exporter.add_argument("--device", default="cpu")
    exporter.add_argument("--gptq", type=Path,
                          help="take projection codes from a symmetric GPTQ int4 checkpoint; "
                               "tensors it lacks use --encoder")
    exporter.add_argument("--widen", action="store_true",
                          help="store imported 4-bit codes in wider role formats as they are")
    referencer = commands.add_parser("reference")
    referencer.add_argument("--corpus", type=Path)
    referencer.add_argument("--text", type=Path, help="one text file instead of corpus streams")
    referencer.add_argument("--chunks", type=int, default=8)
    scorer = commands.add_parser("score")
    scorer.add_argument("--name", required=True)
    scorer.add_argument("--out", type=Path, required=True)
    prober = commands.add_parser("probe")
    prober.add_argument("--source", type=Path, required=True)
    prober.add_argument("--template", type=Path, required=True)
    prober.add_argument("--imatrix", type=Path)
    prober.add_argument("--encoder", choices=("rtn", "search", "search-neg"), default="search-neg")
    prober.add_argument("--format", default="q4")
    prober.add_argument("--band", type=int, default=4)
    prober.add_argument("--device", default="cuda")
    prober.add_argument("--only", nargs="*", help="probe only these units")
    prober.add_argument("--load-mode", default="mmap",
                        help="llama.cpp load mode; mmap avoids a pinned host copy next to the "
                             "probe's own tensors")
    prober.add_argument("--out", type=Path, required=True)
    for command in (referencer, scorer, prober):
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--base-dir", type=Path, required=True)
        command.add_argument("--streams", nargs="+", default=list(STREAMS))
        command.add_argument("--threads", type=int, default=16)
        command.add_argument("--ngl", type=int, default=0, help="layers offloaded to the GPU")
        command.add_argument("--llama-perplexity", type=Path, required=True)
    tabler = commands.add_parser("table")
    tabler.add_argument("--results", type=Path, required=True)
    tabler.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    {"export": export, "reference": reference, "score": score, "probe": probe,
     "table": table}[args.command](args)


if __name__ == "__main__":
    main()
