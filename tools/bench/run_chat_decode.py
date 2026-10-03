"""Decode rate on a chat prompt set, through ninfer-serve's OpenAI Chat Completions endpoint.

Mirrors `vllm bench serve --dataset-name custom --custom-output-len 1024 --num-prompts N
--max-concurrency C`, so results can be read against stacks that publish that metric:

    decode rate = concurrency * 1000 / mean TPOT

TPOT comes from the server's own request log (`--request-log-jsonl`), so it is the decode phase
only; `output throughput` (generated tokens / wall clock) is reported alongside it and includes
prefill and the tail where early finishers leave the batch.

Arms are interleaved -- in order on even passes, reversed on odd -- because between-process spread
on one RTX 3090 is 3-5% (TODO.md, "This card is power-capped"), which is larger than most effects
worth measuring. Always A/B inside one sitting.

usage:
  python tools/bench/run_chat_decode.py --model MODEL.ninfer --prompts PROMPTS.jsonl \
      --out DIR --arm name=/path/to/ninfer-serve [--arm name=/path/to/other[;model=OTHER.ninfer][;drop=--flag,...][;add=--flag value ...][;env=K=V,K2=V2]]

  An arm may also name its own model, so two artifacts can be A/B-ed on one binary, in one sitting,
  with the same prompts: --arm old=SERVE;model=old.ninfer --arm new=SERVE;model=new.ninfer (--model
  is then only the default for arms that do not name one). --spec selects the speculative backend
  (mtp, dflash2 or none) and --draft-tokens its draft count. The RESULT line reports the decode
  rate, the draft acceptance and the tokens emitted per round, so it covers an acceptance comparison
  between artifacts as well as a speed one. --model-id is the id the server publishes (an artifact's
  public name), which differs for the 35B-A3B.

  Arms may share one binary and differ only by environment (e.g. env=NINFER_LM_HEAD_Q4=1) --
  useful for env-gated quality trades, where the interleaving above is what makes the A/B valid.

  PROMPTS.jsonl holds one object per line with a "prompt" string. Any chat prompt set works; the
  comparisons in docs/performance.md use the eight prompts of
  syv-ai/qwen38-27b-rtx3090's bench/prompts_real.jsonl (thinking off, 1024 output tokens).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# Dropping a flag has to take its value with it. Filtering tokens one at a time leaves the value
# behind -- dropping --spec would strip the flag and leave a bare "mtp" in argv, which either fails
# to parse or, worse, parses as something else and the arm silently measures a configuration nobody
# asked for. The server's options are either a bare switch or exactly one value, so a flag is
# removed together with a following token that does not itself begin with "--".
def drop_flags(command, drop):
    kept, index = [], 0
    while index < len(command):
        token = command[index]
        if token in drop:
            index += 1
            if index < len(command) and not command[index].startswith("--"):
                index += 1
            continue
        kept.append(token)
        index += 1
    return kept


def parse_arm(raw):
    parts = raw.split(";")
    name, server = parts[0].split("=", 1)
    drop, add, env, model = [], [], {}, None
    for part in parts[1:]:
        key, value = part.split("=", 1)
        if key == "model":
            model = value
        elif key == "drop":
            drop = value.split(",")
        elif key == "add":
            add = value.split()
        elif key == "env":
            for pair in value.split(","):
                k, v = pair.split("=", 1)
                env[k] = v
        else:
            raise SystemExit(f"unknown arm field: {key}")
    # CreateProcess does not resolve a relative path written with forward slashes, so resolve it here.
    return name, shutil.which(server) or os.path.abspath(server), drop, add, env, model


def wait_ready(proc, port, timeout=180.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except Exception:
            time.sleep(0.3)
    raise TimeoutError("server health timeout")


def chat(port, prompt, seed, args):
    body = {
        "model": args.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "seed": seed,
    }
    if args.temperature == 0:
        body["temperature"] = 0
    else:
        body.update({"temperature": args.temperature, "top_p": 0.8, "top_k": 20})
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def run_arm(name, server, drop, add, env, model, prompts, args, rep):
    out = Path(args.out) / f"rep{rep}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "requests.jsonl"
    log.unlink(missing_ok=True)
    command = [server, model, "--host", "127.0.0.1", "--port", str(args.port),
               "--max-context", str(args.max_context), "--kv-capacity", str(args.kv_capacity),
               "--max-concurrency", str(args.concurrency), "--prefill-chunk", "1024",
               "--kv-dtype", args.kv_dtype, "--no-prefix-reuse", "--request-log-jsonl", str(log)]
    if args.spec != "none":
        command += ["--spec", args.spec, "--draft-tokens", str(args.draft_tokens), "--lm-head-draft"]
    if args.temperature == 0:
        command.append("--greedy")
    command = drop_flags(command, drop) + add

    proc_env = {**os.environ, **env}
    with open(out / "stdout.log", "w") as so, open(out / "stderr.log", "w") as se:
        proc = subprocess.Popen(command, stdout=so, stderr=se, env=proc_env)
        try:
            wait_ready(proc, args.port)
            chat(args.port, "Say hi.", 1, args)  # warm the first-request paths outside the measurement
            started = time.monotonic()
            if args.concurrency == 1:
                responses = [chat(args.port, p, 1000 + i, args) for i, p in enumerate(prompts)]
            else:
                with ThreadPoolExecutor(args.concurrency) as pool:
                    responses = list(pool.map(
                        lambda item: chat(args.port, item[1], 1000 + item[0], args),
                        enumerate(prompts)))
            wall = time.monotonic() - started
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    (out / "responses.json").write_text(json.dumps(responses, indent=1))

    done = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    done = [d for d in done if d.get("event") == "request_done"][1:]  # drop the warm-up
    tpot, generated, drafted, accepted, rounds = [], 0, 0, 0, 0
    for record in done:
        tokens = record["result"]["completion_tokens"]
        generated += tokens
        if tokens > 1:
            tpot.append(record["timings_seconds"]["decode"] / (tokens - 1))
        spec = record.get("speculative") or {}
        drafted += spec.get("drafted_tokens", 0)
        accepted += spec.get("accepted_tokens", 0)
        rounds += spec.get("rounds", 0)
    mean_tpot = sum(tpot) / max(len(tpot), 1)
    print(f"RESULT rep{rep} {name} C{args.concurrency} "
          f"decode={args.concurrency / mean_tpot:.2f} tok/s "
          f"output_throughput={generated / wall:.1f} tok/s "
          f"accept={100.0 * accepted / max(drafted, 1):.2f}% "
          f"tok/round={generated / max(rounds, 1):.3f} gen={generated} reqs={len(done)}",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="default artifact for arms without a model= field")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", default="chat-decode-out")
    parser.add_argument("--arm", action="append", required=True,
                        help="name=/path/to/ninfer-serve[;model=PATH][;drop=--flag,...][;add=--flag value ...]"
                             "[;env=K=V,...]")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--spec", choices=("mtp", "dflash2", "none"), default="mtp")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--model-id", default="qwen3.8-27b",
                        help="model id the server publishes (the artifact's public name)")
    parser.add_argument("--kv-dtype", default="int8")
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--kv-capacity", type=int, default=0,
                        help="0 selects max-context for C1 and twice that above")
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    if args.kv_capacity == 0:
        args.kv_capacity = args.max_context if args.concurrency == 1 else 2 * args.max_context

    prompts = [json.loads(line)["prompt"]
               for line in open(args.prompts, encoding="utf-8") if line.strip()]
    arms = [parse_arm(raw) for raw in args.arm]
    arms = [(name, server, drop, add, env, model or args.model) for name, server, drop, add, env, model in arms]
    missing = [name for name, *_, model in arms if model is None]
    if missing:
        raise SystemExit(f"no model for arm(s) {', '.join(missing)}: pass --model or ;model= on the arm")
    for rep in range(args.reps):
        ordered = arms if rep % 2 == 0 else list(reversed(arms))
        for name, server, drop, add, env, model in ordered:
            run_arm(name, server, drop, add, env, model, prompts, args, rep)


if __name__ == "__main__":
    main()
