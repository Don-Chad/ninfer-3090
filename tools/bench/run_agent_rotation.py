#!/usr/bin/env python3
"""Rotate several agent conversations through one running ninfer-serve and measure cache reuse.

The workload reproduces the multi-agent shape the context cache has to survive: N independent
conversations, each growing by one long user turn per visit, visited round-robin so that at most
`--max-concurrency` of them can stay device resident and the rest must come back from the Host tier
(or be re-prefilled). Every `--edit-every`th visit of an agent rewrites an earlier user turn instead
of appending one, which only a long anchor below the edit can serve without a cold prefill.

The client is black-box: it reads TTFT from the stream and `prompt_tokens`/`cached_tokens` from the
final usage chunk. Pair it with `--request-log-jsonl` on the server for the reuse path and transfer
detail of each request.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ninfer_serve.client import NInferServeClient
from tools.ninfer_serve.openai_chat import chat_request

WORDS = (
    "anchor bearing cable damper engine flange gasket housing impeller journal keeper lever "
    "manifold nozzle orifice piston quill rotor spindle tappet union valve washer yoke bracket "
    "coupling detent eccentric fulcrum governor hinge idler jib knuckle linkage mandrel nut "
    "outrigger pawl ratchet sprocket trunnion"
).split()


@dataclass
class Agent:
    index: int
    rng: random.Random
    messages: list[dict[str, Any]] = field(default_factory=list)
    visits: int = 0
    prompt_tokens: int = 0
    done: bool = False


@dataclass
class Sample:
    agent: int
    visit: int
    kind: str
    prompt_tokens: int
    cached_tokens: int
    ttft_seconds: float | None
    total_seconds: float
    error: str | None = None


def filler(rng: random.Random, words: int) -> str:
    lines = []
    for start in range(0, words, 16):
        lines.append(" ".join(rng.choice(WORDS) for _ in range(min(16, words - start))) + ".")
    return "\n".join(lines)


def user_turn(agent: Agent, words: int) -> dict[str, Any]:
    body = filler(agent.rng, words)
    return {
        "role": "user",
        "content": f"Agent {agent.index} visit {agent.visits}. Review this log and summarise the "
        f"faults in one line.\n{body}",
    }


def send(
    client: NInferServeClient, model: str, messages: list[dict[str, Any]], reply_tokens: int
) -> tuple[Sample, str]:
    request = chat_request(model, messages, reply_tokens)
    exchange = client.prepare(request)
    sent_ns: list[int] = []
    result = exchange.execute(on_sent=sent_ns.append)
    first_output = next((e.received_ns for e in result.events if e.kind == "model_output"), None)
    text = "".join(e.output for e in result.events if e.kind == "model_output")
    usage: dict[str, Any] = {}
    for event in result.events:
        if event.payload and isinstance(event.payload.get("usage"), dict):
            usage = event.payload["usage"]
    details = usage.get("prompt_tokens_details") or {}
    start = sent_ns[0] if sent_ns else result.http.sent_ns
    end = result.http.ended_ns or time.perf_counter_ns()
    error = result.protocol_error or result.error_code or result.http.error
    if result.http.status not in (None, 200) and error is None:
        error = f"http {result.http.status}"
    sample = Sample(
        agent=-1,
        visit=-1,
        kind="",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
        ttft_seconds=(first_output - start) / 1e9 if first_output and start else None,
        total_seconds=(end - start) / 1e9 if start else 0.0,
        error=error,
    )
    return sample, text


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summarize(samples: list[Sample]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("first", "append", "edit", "all"):
        selected = [s for s in samples if (kind == "all" or s.kind == kind) and s.error is None]
        if not selected:
            continue
        prompt = sum(s.prompt_tokens for s in selected)
        cached = sum(s.cached_tokens for s in selected)
        ttfts = [s.ttft_seconds for s in selected if s.ttft_seconds is not None]
        out[kind] = {
            "requests": len(selected),
            "weighted_hit_rate": cached / prompt if prompt else 0.0,
            "cold_requests": sum(1 for s in selected if s.cached_tokens == 0),
            "ttft_p50": percentile(ttfts, 0.5),
            "ttft_p95": percentile(ttfts, 0.95),
            "ttft_max": max(ttfts) if ttfts else None,
            "ttft_mean": statistics.fmean(ttfts) if ttfts else None,
        }
    out["errors"] = sum(1 for s in samples if s.error is not None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--target-tokens", type=int, default=200_000,
                        help="stop an agent once its prompt reaches this many tokens")
    parser.add_argument("--turn-words", type=int, default=3000,
                        help="filler words per appended user turn (about 1.3 tokens per word)")
    parser.add_argument("--reply-tokens", type=int, default=32)
    parser.add_argument("--edit-every", type=int, default=6,
                        help="every Nth visit rewrites the user turn two turns back; 0 disables")
    parser.add_argument("--max-requests", type=int, default=0, help="0 = until all agents finish")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--out", type=Path, required=True, help="JSONL of per-request samples")
    args = parser.parse_args()

    client = NInferServeClient(args.base_url, args.timeout_seconds, args.api_key)
    model = client.discover_model()
    agents = [Agent(index, random.Random(args.seed * 1000 + index)) for index in range(args.agents)]
    for agent in agents:
        agent.messages.append({
            "role": "system",
            "content": f"You are maintenance agent {agent.index}. Answer tersely.",
        })

    samples: list[Sample] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as log:
        while not all(agent.done for agent in agents):
            for agent in agents:
                if agent.done:
                    continue
                if args.max_requests and len(samples) >= args.max_requests:
                    agent.done = True
                    continue
                user_positions = [i for i, m in enumerate(agent.messages) if m["role"] == "user"]
                edit = (args.edit_every > 0 and agent.visits > 0 and
                        agent.visits % args.edit_every == 0 and len(user_positions) >= 3)
                if edit:
                    # Rewrite the user turn two back and drop everything after it: the prompt
                    # diverges below the endpoint and below the rewrite checkpoint.
                    position = user_positions[-2]
                    del agent.messages[position:]
                    kind = "edit"
                else:
                    kind = "first" if agent.visits == 0 else "append"
                agent.messages.append(user_turn(agent, args.turn_words))
                sample, reply = send(client, model, agent.messages, args.reply_tokens)
                sample.agent, sample.visit, sample.kind = agent.index, agent.visits, kind
                samples.append(sample)
                log.write(json.dumps(asdict(sample)) + "\n")
                log.flush()
                print(f"agent={agent.index} visit={agent.visits} kind={kind} "
                      f"prompt={sample.prompt_tokens} cached={sample.cached_tokens} "
                      f"ttft={sample.ttft_seconds} error={sample.error}", file=sys.stderr)
                agent.visits += 1
                if sample.error is not None:
                    agent.messages.pop()
                    agent.done = True
                    continue
                # Replay the reply verbatim so the next visit is an exact extension.
                agent.messages.append({"role": "assistant", "content": reply})
                agent.prompt_tokens = sample.prompt_tokens
                if agent.prompt_tokens >= args.target_tokens:
                    agent.done = True

    print(json.dumps(summarize(samples), indent=2))
    return 0 if all(s.error is None for s in samples) else 1


if __name__ == "__main__":
    sys.exit(main())
