#!/usr/bin/env python3
"""Load generator for the large-request / worker-failure reproduction.

Builds multi-turn chat conversations with tool definitions whose prompt token count is calibrated
against the server's own tokenizer (usage.prompt_tokens of a max_tokens=1 probe), then replays the
production traffic mix: a warm streaming agent loop, a very large non-stream request (cancelled at a
client timeout and resent), mid-size streams and small requests.

  loadgen.py probe  --prompt 107600 --tools 278 --overshoot 21
  loadgen.py mix    --window 131072 [--overshoot 21] [--lanes 3] [--rounds 6]
"""
import argparse, http.client, json, random, sys, threading, time

import os
HOST, PORT, KEY = "127.0.0.1", int(os.environ.get("NI_PORT", "8011")), "KEY"
MODEL = "qwen3.8-27b"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.2f}] {msg}", flush=True)


def post(path, body, timeout=None, stream=False, cancel_after=None):
    """Returns (status, parsed-or-text, seconds). With cancel_after, closes the socket at that time."""
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    start = time.time()
    try:
        conn.request("POST", path, json.dumps(body),
                     {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}",
                      "x-api-key": KEY, "anthropic-version": "2023-06-01"})
        if cancel_after is not None:
            conn.sock.settimeout(cancel_after)
        resp = conn.getresponse()
        if stream:
            n = 0
            while True:
                line = resp.readline()
                if not line:
                    break
                n += 1
            return resp.status, f"{n} stream lines", time.time() - start
        data = resp.read().decode("utf-8", "replace")
        try:
            data = json.loads(data)
        except ValueError:
            pass
        return resp.status, data, time.time() - start
    except (TimeoutError, OSError) as e:
        return 499, f"client cancel/timeout: {e!r}", time.time() - start
    finally:
        conn.close()


VOCAB = None


def vocab(seed):
    global VOCAB
    if VOCAB is None:
        r = random.Random(1234)
        syl = ["ka", "to", "mi", "ren", "sol", "vex", "dar", "lu", "pho", "nim", "qua", "zed", "bri", "tan"]
        VOCAB = ["".join(r.choice(syl) for _ in range(r.randint(1, 3))) for _ in range(4000)]
    return VOCAB


def filler(words, seed):
    r = random.Random(seed)
    v = vocab(seed)
    out, line = [], []
    for i in range(words):
        line.append(r.choice(v))
        if len(line) >= 14:
            out.append(" ".join(line) + "."); line = []
    out.append(" ".join(line))
    return " ".join(out)


def make_tools(n, seed):
    r = random.Random(seed)
    tools = []
    for i in range(n):
        props = {f"arg_{j}": {"type": r.choice(["string", "integer", "boolean"]),
                              "description": filler(12, seed * 1000 + i * 10 + j)} for j in range(r.randint(2, 5))}
        tools.append({"type": "function", "function": {
            "name": f"tool_{seed}_{i}", "description": filler(30, seed * 7000 + i),
            "parameters": {"type": "object", "properties": props, "required": list(props)[:1]}}})
    return tools


def make_messages(n_messages, total_words, seed):
    """n_messages alternating user/assistant turns (ending on user), words spread evenly."""
    per = max(1, total_words // n_messages)
    msgs = [{"role": "system", "content": "You are a careful coding agent. " + filler(40, seed)}]
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": filler(per, seed * 100003 + i)})
    if msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": "continue"})
    return msgs


def chat_body(msgs, tools, max_tokens, stream=False):
    b = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "stream": stream}
    if tools:
        b["tools"] = tools
    if stream:
        b["stream_options"] = {"include_usage": True}
    return b


def prompt_tokens(msgs, tools):
    st, d, _ = post("/v1/chat/completions", chat_body(msgs, tools, 1), timeout=1800)
    if st == 400 and "context_length_exceeded" in str(d):
        return 10**9  # larger than the window: caller shrinks
    if st != 200:
        raise RuntimeError(f"probe failed {st}: {str(d)[:300]}")
    return d["usage"]["prompt_tokens"]


CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".calib_cache.json")


def _cache_get(key):
    try:
        return json.load(open(CACHE_FILE)).get(key)
    except (OSError, ValueError):
        return None


def _cache_put(key, val):
    try:
        d = json.load(open(CACHE_FILE))
    except (OSError, ValueError):
        d = {}
    d[key] = val
    json.dump(d, open(CACHE_FILE, "w"))


def build(n_messages, words, tail_words, seed):
    msgs = make_messages(n_messages, words, seed)
    if tail_words:
        msgs[-1]["content"] += " " + filler(tail_words, seed * 31 + 7)
    return msgs


def calibrate(target, n_messages, n_tools, seed, tol=1500):
    """Prompt token count within [target-tol, target+tol].  The sum prompt+max_tokens is what the
    reproduction needs, and the caller sets max_tokens from the measured count, so exactness is not
    needed.  Affine fit from two small probes, then only the last message is tuned so the earlier
    prefix stays in the server's prefix cache."""
    tools = make_tools(n_tools, seed) if n_tools else None
    ckey = f"{target}/{n_messages}/{n_tools}/{seed}"
    hit = _cache_get(ckey)

    def measure(words, tail):
        k = f"m/{n_messages}/{n_tools}/{seed}/{words}/{tail}"
        v = _cache_get(k)
        if v is None:
            v = prompt_tokens(build(n_messages, words, tail, seed), tools)
            _cache_put(k, v)
        log(f"calibrate seed={seed} words={words} tail={tail} -> {v}")
        return v

    if hit:
        words, tail = hit
        return build(n_messages, words, tail, seed), tools, measure(words, tail)
    w1, w2 = 2000, 6000
    p1, p2 = measure(w1, 0), measure(w2, 0)
    slope = (p2 - p1) / (w2 - w1)
    base_w = int(w1 + (target - 0.12 * target - p1) / slope)  # leave ~12% for the tail
    base_w -= base_w % n_messages
    pb = measure(base_w, 0)
    tail = 0
    for _ in range(6):
        if abs(pb - target) <= tol:
            break
        tail = max(0, tail + int((target - pb) / slope))
        pb = measure(base_w, tail)
    if abs(pb - target) > tol:
        raise RuntimeError(f"could not reach {target}: got {pb}")
    _cache_put(ckey, [base_w, tail])
    return build(n_messages, base_w, tail, seed), tools, pb


def health():
    c = http.client.HTTPConnection(HOST, PORT, timeout=10)
    try:
        c.request("GET", "/health"); r = c.getresponse(); return r.status, r.read().decode()[:80]
    except OSError as e:
        return 0, repr(e)
    finally:
        c.close()


def cmd_probe(a):
    msgs, tools, pt = calibrate(a.prompt, a.messages, a.tools, a.seed)
    window = a.window
    max_tokens = window - pt + a.overshoot
    log(f"prompt={pt} max_tokens={max_tokens} sum={pt + max_tokens} window={window} (over by {a.overshoot})")
    st, d, sec = post("/v1/chat/completions", chat_body(msgs, tools, min(max_tokens, a.cap)), timeout=a.timeout)
    log(f"-> HTTP {st} in {sec:.1f}s: {str(d)[:300]}")
    log(f"health {health()}")


def cmd_mix(a):
    W = a.window
    f = W / 262144.0
    sizes = dict(big=int(215210 * f), loop=int(65000 * f), mid=int(169683 * f), small=int(43500 * f))
    log(f"sizes {sizes}")
    big_m, big_t, big_pt = calibrate(sizes["big"], 172, a.big_tools, 11)
    big_max = W - big_pt + a.overshoot
    loop_m, loop_t, loop_pt = calibrate(sizes["loop"], 80, 7, 22)
    mid_m, mid_t, mid_pt = calibrate(sizes["mid"], 165, 19, 33)
    small_m, small_t, small_pt = calibrate(sizes["small"], 2, 0, 44)
    log(f"big={big_pt}+{big_max}={big_pt + big_max} loop={loop_pt} mid={mid_pt} small={small_pt}")
    stop = threading.Event()
    results = []

    def worker(name, body, stream, cancel_after, repeat, gap):
        while not stop.is_set() and repeat != 0:
            log(f"{name}: start")
            st, d, sec = post("/v1/chat/completions", body, timeout=1800, stream=stream, cancel_after=cancel_after)
            log(f"{name}: HTTP {st} in {sec:.1f}s {str(d)[:160]}")
            results.append((name, st))
            repeat -= 1
            stop.wait(gap)

    loop_body = chat_body(loop_m, loop_t, min(32000, a.cap), stream=True)
    mid_body = chat_body(mid_m, mid_t, min(int(91414 * f), a.cap), stream=True)
    small_body = chat_body(small_m, small_t, 2112, stream=False)
    big_body = chat_body(big_m, big_t, min(big_max, a.cap), stream=False)
    ths = [threading.Thread(target=worker, args=("loop", loop_body, True, None, a.rounds, 1))]
    if a.lanes >= 2: ths.append(threading.Thread(target=worker, args=("mid", mid_body, True, None, a.rounds, 1)))
    if a.lanes >= 3: ths.append(threading.Thread(target=worker, args=("small", small_body, False, None, a.rounds * 3, 5)))
    for t in ths: t.start()
    time.sleep(a.warm)  # let the loop's prefix warm
    bigs = threading.Thread(target=worker, args=("BIG", big_body, False, a.big_cancel, a.rounds, 0))
    bigs.start(); ths.append(bigs)
    for t in ths: t.join()
    log(f"health {health()}")
    log(f"done: {len(results)} responses, statuses {sorted(set(s for _, s in results))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("probe")
    pp.add_argument("--prompt", type=int, required=True)
    pp.add_argument("--tools", type=int, default=0)
    pp.add_argument("--messages", type=int, default=172)
    pp.add_argument("--overshoot", type=int, default=0)
    pp.add_argument("--window", type=int, default=131072)
    pp.add_argument("--cap", type=int, default=10**9, help="clamp max_tokens actually sent")
    pp.add_argument("--timeout", type=int, default=1800)
    pp.add_argument("--seed", type=int, default=11)
    pm = sub.add_parser("mix")
    pm.add_argument("--window", type=int, default=131072)
    pm.add_argument("--overshoot", type=int, default=21)
    pm.add_argument("--lanes", type=int, default=3)
    pm.add_argument("--rounds", type=int, default=4)
    pm.add_argument("--warm", type=int, default=20)
    pm.add_argument("--big-tools", type=int, default=278)
    pm.add_argument("--big-cancel", type=float, default=60.0)
    pm.add_argument("--cap", type=int, default=10**9)
    a = p.parse_args()
    {"probe": cmd_probe, "mix": cmd_mix}[a.cmd](a)
