#!/usr/bin/env python3
"""NInfer Studio - optional web UI for a local ninfer server.

Standard library only: no venv, no pip, no build step. It serves the static
frontend and proxies the engine's OpenAI-compatible API, so the browser needs
no CORS exception and no knowledge of the engine port.

The engine is treated strictly as an external service with an optional
capability set. `/health`, `/v1/models` and `/v1/chat/completions` are the
minimum this UI needs; `/v1/load`, `/metrics` and `/v1/models/{id}/residency`
are probed at runtime and reported in `/api/state` under `caps`, so the
frontend can say "not available on this server" instead of showing a number it
does not have.

Configuration (first match wins, later layers override earlier ones):
  1. built-in defaults below
  2. webui/studio.config.json
  3. --config PATH on the command line
  4. environment: NINFER_URL, NINFER_API_KEY, NINFER_HOST, NINFER_PORT,
     NINFER_UI_HOST, NINFER_UI_PORT, STUDIO_LOGS_DIR, STUDIO_MODELS_DIR

No secret belongs in the JSON file: the bearer token comes from the
environment only (NINFER_API_KEY, the same variable ninfer-serve reads).
"""
import json
import os
import ipaddress
import ownership
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(WEBUI_DIR)

DEFAULTS = {
    "engine": {
        "host": "127.0.0.1",
        "port": 8080,               # the port scripts/run.bat serves on
        "url": "",                  # derived from host/port when empty
        "start_command": [],        # e.g. ["scripts\\run.bat", "qwen38-27b", "tuned"]
        "startup_timeout_s": 900,
    },
    "ui": {"host": "127.0.0.1", "port": 8090},
    "paths": {"logs": "logs", "models": "models"},
    "poll_ms": 2000,
}


def _deep_merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _config_path():
    """--config PATH, else STUDIO_CONFIG, else webui/studio.config.json."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if arg.startswith("--config="):
            return os.path.abspath(arg.split("=", 1)[1])
    env = os.environ.get("STUDIO_CONFIG")
    if env:
        return os.path.abspath(env)
    return os.path.join(WEBUI_DIR, "studio.config.json")


def load_config():
    cfg = json.loads(json.dumps(DEFAULTS))
    path = _config_path()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = _deep_merge(cfg, json.load(f))
            cfg["_config_file"] = path
        except (OSError, ValueError) as e:
            sys.stderr.write("[studio] ignoring %s: %s\n" % (path, e))
    else:
        cfg["_config_file"] = None

    eng, ui = cfg["engine"], cfg["ui"]
    # The environment mirrors the names ninfer's own launchers use, so one
    # variable can drive both the engine and this UI.
    if os.environ.get("NINFER_HOST"):
        eng["host"] = os.environ["NINFER_HOST"]
    if os.environ.get("NINFER_PORT"):
        eng["port"] = int(os.environ["NINFER_PORT"])
    if os.environ.get("NINFER_URL"):
        eng["url"] = os.environ["NINFER_URL"]
    if os.environ.get("NINFER_UI_HOST"):
        ui["host"] = os.environ["NINFER_UI_HOST"]
    if os.environ.get("NINFER_UI_PORT"):
        ui["port"] = int(os.environ["NINFER_UI_PORT"])
    if os.environ.get("STUDIO_LOGS_DIR"):
        cfg["paths"]["logs"] = os.environ["STUDIO_LOGS_DIR"]
    if os.environ.get("STUDIO_MODELS_DIR"):
        cfg["paths"]["models"] = os.environ["STUDIO_MODELS_DIR"]

    if not eng.get("url"):
        eng["url"] = "http://%s:%d" % (eng["host"], eng["port"])
    return cfg


CFG = load_config()
ENGINE = CFG["engine"]
UI = CFG["ui"]
UPSTREAM = str(ENGINE["url"]).rstrip("/")
UPSTREAM_HOST = ENGINE.get("host") or "127.0.0.1"
UPSTREAM_PORT = int(ENGINE.get("port") or 8080)
API_KEY = os.environ.get("NINFER_API_KEY", "")
UI_PORT = int(UI.get("port") or 8090)
UI_HOST = UI.get("host") or "127.0.0.1"
if UI_HOST != "localhost" and not ipaddress.ip_address(UI_HOST).is_loopback:
    raise ValueError("Studio must bind a loopback address; use a local browser")

LOGS_DIR = os.path.join(PROJECT, CFG["paths"]["logs"])
MODELS_DIR = os.path.join(PROJECT, CFG["paths"]["models"])
ENGINE_LOG = os.path.join(LOGS_DIR, "engine.log")
UI_LOG = os.path.join(LOGS_DIR, "ui.log")


def _engine_exe():
    """Where ninfer-serve lives, following scripts/run.bat's own rule."""
    for env in (os.environ.get("NINFER_SERVER"), os.environ.get("NINFER_EXE")):
        if env and os.path.isfile(env):
            return os.path.abspath(env)
    candidates = [
        os.path.join(PROJECT, "build-ninja", "apps", "ninfer-serve.exe"),
        os.path.join(PROJECT, "build", "apps", "ninfer-serve.exe"),
        os.path.join(PROJECT, "ninfer-serve.exe"),
        os.path.join(os.path.dirname(WEBUI_DIR), "ninfer-serve.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


ENGINE_EXE = _engine_exe()

_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/tokens.css": ("tokens.css", "text/css; charset=utf-8"),
    "/icons.svg": ("icons.svg", "image/svg+xml"),
}

_MIME = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
    ".html": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}

# Prometheus series this UI uses when /metrics is served. A series that the
# running build does not export simply stays absent - the frontend shows the
# metric as unavailable rather than inventing a value.
SCALARS = {
    "ninfer_engine_ready": "ready",
    "ninfer_decode_tokens_total": "decode_tokens",
    "ninfer_decode_seconds_total": "decode_seconds",
    "ninfer_decode_rounds_total": "rounds",
    "ninfer_generation_tokens_total": "gen_tokens",
    "ninfer_prefill_tokens_total": "prefill_tokens",
    "ninfer_prefill_seconds_total": "prefill_seconds",
    "ninfer_gpu_memory_used_bytes": "vram_used",
    "ninfer_gpu_memory_total_bytes": "vram_total",
    "ninfer_gpu_power_watts": "power",
    "ninfer_gpu_temperature_celsius": "temp",
    "ninfer_gpu_utilization_percent": "gpu_util",
    "ninfer_host_kv_used_bytes": "kv_used",
    "ninfer_host_kv_capacity_bytes": "kv_total",
    "ninfer_device_kv_used_pages": "kv_pages",
    "ninfer_device_kv_capacity_pages": "kv_pages_total",
    "ninfer_host_context_used_bytes": "ctx_used",
    "ninfer_host_context_capacity_bytes": "ctx_total",
    "ninfer_spec_decode_draft_window": "draft_window",
}

LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$")


def _headers(extra=None):
    h = {"Accept": "application/json", "User-Agent": "ninfer-studio/2.0"}
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    if extra:
        h.update(extra)
    return h


def fetch(path, timeout=4.0):
    """GET an upstream path, return (status, body_bytes). 0 = unreachable."""
    req = urllib.request.Request(UPSTREAM + path, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""
    except Exception as e:  # connection refused, timeout, ...
        return 0, str(e).encode()


def _post_upstream(path, payload=None, timeout=60.0):
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        UPSTREAM + path, data=data,
        headers=_headers({"Content-Type": "application/json"}), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""
    except Exception as e:
        return 0, str(e).encode()


_MODEL_CACHE = {"id": None, "modalities": None}


def _input_modalities(model):
    """Input modalities, from either shape a build may use for this field."""
    if not isinstance(model, dict):
        return []
    arch = model.get("architecture") or {}
    mods = arch.get("input_modalities") or model.get("input_modalities") or []
    return [str(m) for m in mods]


def default_model():
    """ninfer requires an explicit model alias; discover the served one once."""
    if _MODEL_CACHE["id"]:
        return _MODEL_CACHE["id"]
    st, body = fetch("/v1/models", timeout=4.0)
    if st == 200:
        try:
            data = json.loads(body).get("data") or []
            if data and data[0].get("id"):
                _MODEL_CACHE["id"] = data[0]["id"]
                _MODEL_CACHE["modalities"] = _input_modalities(data[0])
                return _MODEL_CACHE["id"]
        except Exception:
            pass
    return "default"


# ---------------------------------------------------------------- GPU thermal
# Temperatures come from NVML (nvml.dll, shipped with the driver): no extra
# software, no subprocess per poll. What a card exposes differs by SKU, so
# every sensor carries an availability flag - nothing is estimated or copied
# from another sensor.
_THERMAL_LOCK = threading.Lock()
_THERMAL_CACHE = {"at": 0.0, "data": None}
_NVML = None
_NVML_HANDLE = None
_NVML_ERROR = None

NVML_TEMP_GPU = 0
NVML_TEMP_MEMORY = 1
NVML_THRESHOLDS = ((1, "slowdown"), (0, "shutdown"), (3, "max_operating"), (4, "acquire"))
NVML_RC = {0: "ok", 1: "uninitialized", 2: "not supported", 3: "no permission",
           4: "not found", 6: "invalid argument"}


def _nvml():
    """Load and initialise NVML once; remember why if it is unavailable."""
    global _NVML, _NVML_HANDLE, _NVML_ERROR
    if _NVML is not None or _NVML_ERROR:
        return _NVML
    try:
        import ctypes
        lib = ctypes.WinDLL("nvml.dll") if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
        lib.nvmlInit_v2()
        handle = ctypes.c_void_p()
        rc = lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
        if rc != 0:
            raise OSError("nvmlDeviceGetHandleByIndex_v2 rc=%d" % rc)
        _NVML, _NVML_HANDLE = lib, handle
    except Exception as e:
        _NVML_ERROR = "%s: %s" % (type(e).__name__, e)
    return _NVML


def gpu_thermal():
    """Temperatures, fans and throttle limits, with per-sensor availability."""
    now = time.time()
    with _THERMAL_LOCK:
        if _THERMAL_CACHE["data"] and now - _THERMAL_CACHE["at"] < 2.0:
            return _THERMAL_CACHE["data"]

    import ctypes

    out = {
        "core_c": None, "memory_c": None, "hotspot_c": None, "board_c": None,
        "fan_pct": None, "fan_count": None, "limits": {},
        "available": {}, "reason": {}, "source": "nvml",
    }
    lib = _nvml()
    if not lib:
        out["source"] = "unavailable"
        for key in ("core_c", "memory_c", "hotspot_c", "board_c", "fan_pct"):
            out["available"][key] = False
            out["reason"][key] = "NVML not available (%s)" % (_NVML_ERROR or "unknown")
    else:
        def temp(sensor):
            v = ctypes.c_uint()
            rc = lib.nvmlDeviceGetTemperature(_NVML_HANDLE, sensor, ctypes.byref(v))
            return (v.value, "ok") if rc == 0 else (None, NVML_RC.get(rc, "rc=%d" % rc))

        for key, sensor in (("core_c", NVML_TEMP_GPU), ("memory_c", NVML_TEMP_MEMORY)):
            value, why = temp(sensor)
            out[key], out["available"][key], out["reason"][key] = value, value is not None, why

        # NVML has no hot-spot or board sensor: there is no id to ask for it,
        # so report it as unsupported instead of borrowing the core reading.
        for key in ("hotspot_c", "board_c"):
            out["available"][key] = False
            out["reason"][key] = "not exposed by NVML"

        fan = ctypes.c_uint()
        rc = lib.nvmlDeviceGetFanSpeed(_NVML_HANDLE, ctypes.byref(fan))
        out["fan_pct"] = fan.value if rc == 0 else None
        out["available"]["fan_pct"] = rc == 0
        out["reason"]["fan_pct"] = "ok" if rc == 0 else NVML_RC.get(rc, "rc=%d" % rc)
        nfan = ctypes.c_uint()
        if lib.nvmlDeviceGetNumFans(_NVML_HANDLE, ctypes.byref(nfan)) == 0:
            out["fan_count"] = nfan.value

        for threshold, name in NVML_THRESHOLDS:
            v = ctypes.c_uint()
            if lib.nvmlDeviceGetTemperatureThreshold(_NVML_HANDLE, threshold, ctypes.byref(v)) == 0:
                out["limits"][name] = v.value

    with _THERMAL_LOCK:
        _THERMAL_CACHE["at"], _THERMAL_CACHE["data"] = now, out
    return out


def parse_metrics(text):
    out, hist = {}, {}
    backend = ""
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name, val = m.group(1), m.group(3)
        if name == "ninfer_model_info":
            m2 = re.search(r'speculative_backend="([^"]*)"', m.group(2) or "")
            if m2:
                backend = m2.group(1)
            continue
        if name.endswith("_bucket") or name.endswith("_sum") or name.endswith("_count"):
            hist.setdefault(name, 0.0)
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
    flat = {key: out[name] for name, key in SCALARS.items() if name in out}
    flat["_series"] = len(out)
    flat["_histograms"] = len(hist)
    if backend:
        flat["spec_backend"] = backend
    return flat


def pids_on_port(port):
    """PIDs listening on a TCP port. Windows: netstat, POSIX: lsof."""
    found = []
    try:
        if os.name == "nt":
            res = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                                 text=True, timeout=10,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
                    if parts[1].endswith(":%d" % port):
                        pid = parts[4]
                        if pid.isdigit() and int(pid) > 0 and pid not in found:
                            found.append(pid)
        else:
            res = subprocess.run(["lsof", "-nP", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t"],
                                 capture_output=True, text=True, timeout=10)
            found = [p for p in res.stdout.split() if p.isdigit()]
    except Exception:
        pass
    return found


def engine_pid():
    return pids_on_port(UPSTREAM_PORT)


def tail(path, n=60):
    """Last n lines of a log file. A missing file is not an error."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [ln.rstrip() for ln in data.splitlines() if ln.strip()]
    return lines[-n:]


def _start_engine():
    """Start the engine exactly the way the repository's own launcher does.

    The Studio deliberately does not own serving flags: it runs the configured
    start command (scripts/run.bat by default, which carries the measured
    profile) with NINFER_HOST/NINFER_PORT set, so the official defaults stay
    intact and this UI only chooses the port it talks to.
    """
    live = engine_pid()
    if live:
        return {"ok": True, "already": True, "pids": live}
    pending = ownership.claim("engine", [])
    if pending:
        return {"ok": True, "already": True, "pids": pending}
    cmd = ENGINE.get("start_command") or []
    if not cmd:
        return {"ok": False, "error": "no engine.start_command configured "
                                      "(see webui/studio.config.json)"}
    os.makedirs(LOGS_DIR, exist_ok=True)
    env = dict(os.environ)
    env["NINFER_HOST"] = str(UPSTREAM_HOST)
    env["NINFER_PORT"] = str(UPSTREAM_PORT)
    if ENGINE_EXE:
        env.setdefault("NINFER_SERVER", ENGINE_EXE)
    args = list(cmd)
    if args[0] == "scripts/run":
        args[0] = os.path.join("scripts", "run.bat" if os.name == "nt" else "run.sh")
    if os.name == "nt":
        # run.bat is a batch file: it needs cmd.exe, and it must not inherit a
        # console window from the UI process.
        shell_args = ["cmd", "/c"] + args
    else:
        shell_args = args
    log = open(ENGINE_LOG, "ab", buffering=0)
    log.write(("\n==== studio start %s ====\n" % time.strftime("%Y-%m-%d %H:%M:%S")).encode())
    flags = 0
    if os.name == "nt":
        # CREATE_NO_WINDOW, not DETACHED_PROCESS: a batch file needs a console
        # to run, and without one run.bat dies part-way through the model load.
        flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    proc = subprocess.Popen(shell_args, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env, creationflags=flags if os.name == "nt" else 0)
    log.close()
    ownership.record("engine", proc.pid)
    return {"ok": True, "pid": proc.pid, "command": shell_args, "log": ENGINE_LOG}


def start_engine():
    with ownership.LOCK:
        return _start_engine()


def stop_engine():
    """Ownership is determined by the server, never by a caller's PID list."""
    return ownership.stop("engine", engine_pid())


def model_action(action):
    """suspend / resume the served model (needs a server built with suspend)."""
    mid = default_model()
    return _post_upstream("/v1/models/%s/%s" % (urllib.parse.quote(mid), action), {})


def artifacts():
    rows, total = [], 0
    if os.path.isdir(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, name)
            if os.path.isfile(path) and name.endswith(".ninfer"):
                st = os.stat(path)
                rows.append({"name": name, "bytes": st.st_size, "mtime": int(st.st_mtime)})
                total += st.st_size
    rows.sort(key=lambda r: -r["bytes"])
    try:
        du = shutil.disk_usage(PROJECT)
    except OSError:
        du = None
    return {
        "files": rows, "total": total,
        "disk_free": du.free if du else None, "disk_total": du.total if du else None,
        "models_dir": MODELS_DIR, "engine_exe": ENGINE_EXE,
    }


_CAPS = {"at": 0.0, "data": None}
_CAPS_LOCK = threading.Lock()


def capabilities(refresh=False):
    """Probe which optional upstream endpoints this server really serves.

    Nothing here is assumed from the version: each capability is the observed
    status of one request, and a capability that answers 401/404/0 is reported
    with its reason so the UI can explain itself.
    """
    now = time.time()
    with _CAPS_LOCK:
        if _CAPS["data"] and not refresh and now - _CAPS["at"] < 5.0:
            return _CAPS["data"]
    caps = {}
    st, body = fetch("/health", timeout=2.0)
    caps["health"] = {"ok": st == 200, "status": st,
                      "loading": st == 503,
                      "version": None}
    if st in (200, 503):
        try:
            caps["health"]["version"] = json.loads(body or b"{}").get("version")
        except Exception:
            pass
    st_m, body_m = fetch("/v1/models", timeout=3.0)
    caps["models"] = {"ok": st_m == 200, "status": st_m}
    model = None
    if st_m == 200:
        try:
            data = json.loads(body_m).get("data") or []
            model = data[0] if data else None
            caps["models"]["count"] = len(data)
        except Exception:
            caps["models"]["ok"] = False
    caps["vision"] = any("image" in m for m in _input_modalities(model or {}))
    st_l, _ = fetch("/v1/load", timeout=3.0)
    caps["load"] = {"ok": st_l == 200, "status": st_l}
    st_x, _ = fetch("/metrics", timeout=3.0)
    caps["metrics"] = {"ok": st_x == 200, "status": st_x}
    if st_x == 401:
        caps["metrics"]["reason"] = "auth"
    elif st_x == 404:
        caps["metrics"]["reason"] = "unsupported"
    resid = {"ok": False, "status": 0}
    suspend_off = False
    if model and model.get("id"):
        st_r, body_r = fetch("/v1/models/%s/residency" % urllib.parse.quote(model["id"]), timeout=3.0)
        resid = {"ok": st_r == 200, "status": st_r}
        try:
            jr = json.loads(body_r or b"{}")
        except Exception:
            jr = {}
        # the endpoint answering is not the same as the feature being on: a
        # profile started without --model-suspend reports enabled=false.
        if isinstance(jr, dict) and jr.get("enabled") is False:
            suspend_off = True
            resid["enabled"] = False
        elif isinstance(jr, dict) and jr.get("enabled") is True:
            resid["enabled"] = True
    caps["residency"] = resid
    caps["suspend"] = {"ok": resid["ok"] and not suspend_off, "status": resid["status"]}
    if suspend_off:
        caps["suspend"]["reason"] = "disabled"
    caps["chat"] = {"ok": st_m == 200 or caps["health"]["ok"]}
    with _CAPS_LOCK:
        _CAPS["at"], _CAPS["data"] = now, caps
    return caps


def snapshot():
    """One poll of everything the dashboard needs; each call is independent."""
    result = {"ts": time.time(), "upstream": UPSTREAM, "up": False, "errors": {}}
    got = {}
    jobs = {"health": "/health", "models": "/v1/models", "load": "/v1/load", "metrics": "/metrics"}

    def run(k, p):
        got[k] = fetch(p)

    threads = [threading.Thread(target=run, args=(k, p), daemon=True) for k, p in jobs.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5.0)

    st, body = got.get("health", (0, b""))
    result["up"] = st == 200
    props = {}
    if st in (200, 503):
        try:
            version = json.loads(body or b"{}").get("version")
            if version:
                props["build_info"] = "ninfer-serve " + str(version)
        except Exception:
            pass
    if st != 200:
        result["errors"]["health"] = {"status": st, "body": body[:200].decode("utf-8", "replace")}

    st, body = got.get("models", (0, b""))
    if st == 200:
        try:
            data = json.loads(body).get("data", [])
            result["model"] = data[0] if data else None
        except Exception as e:
            result["errors"]["models"] = str(e)
    else:
        result["errors"]["models"] = st

    st, body = got.get("load", (0, b""))
    if st == 200:
        try:
            result["load"] = json.loads(body)
        except Exception as e:
            result["errors"]["load"] = str(e)
    else:
        result["errors"]["load"] = st

    st, body = got.get("metrics", (0, b""))
    if st == 200:
        result["metrics"] = parse_metrics(body.decode("utf-8", "replace"))
    else:
        result["errors"]["metrics"] = {"status": st, "body": body[:120].decode("utf-8", "replace")}

    mid = (result.get("model") or {}).get("id")
    if result["up"] and mid:
        st, body = fetch("/v1/models/%s/residency" % urllib.parse.quote(mid), timeout=3.0)
        if st == 200:
            try:
                result["residency"] = json.loads(body)
            except Exception:
                pass
    try:
        result["gpu"] = gpu_thermal()
    except Exception as e:
        result["gpu"] = {"source": "error", "error": "%s: %s" % (type(e).__name__, e)}
    # ninfer has no /props; derive the few fields the UI reads from the
    # endpoints that do exist, and leave the rest absent rather than invented.
    model0 = result.get("model") or {}
    if model0.get("context_window") or model0.get("max_model_len"):
        props["n_ctx"] = model0.get("context_window") or model0.get("max_model_len")
    if isinstance(result.get("residency"), dict):
        props["is_sleeping"] = result["residency"].get("state") == "suspended"
    if props:
        result["props"] = props
    caps = capabilities()
    result["caps"] = caps
    result["pids"] = engine_pid() if result["up"] else []
    owned = ownership.claim("engine", result["pids"])
    result["engine_owned"] = bool(owned)
    result["engine_exe"] = ENGINE_EXE
    result["config"] = {"config_file": CFG.get("_config_file"), "ui_port": UI_PORT,
                        "engine_url": UPSTREAM, "start_command": ENGINE.get("start_command"),
                        "poll_ms": CFG.get("poll_ms", 2000)}
    return result


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NinferStudio/2.0"

    def log_message(self, fmt, *args):
        line = fmt % args
        if " 200 " not in line:
            sys.stderr.write("%s - %s\n" % (self.address_string(), line))

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _safe_static(self, name):
        # Only bundled public assets are served. Config, source and runtime
        # state must never become downloadable, even inside WEBUI_DIR.
        if name not in ("/fonts/outfit-latin-wght.woff2", "/fonts/outfit-latin-ext-wght.woff2"):
            return None
        root = os.path.realpath(WEBUI_DIR)
        path = os.path.realpath(os.path.join(root, name.lstrip("/")))
        if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
            return None
        return path

    def _trusted_request(self, changing=False):
        port = self.server.server_port
        allowed = {"127.0.0.1:%d" % port, "localhost:%d" % port,
                   "[::1]:%d" % port, "%s:%d" % (UI_HOST, port)}
        host = self.headers.get("Host", "").lower()
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin != "http://" + host):
            self.close_connection = True
            self._json({"error": "Untrusted Host or Origin"}, 403)
            return False
        if changing and self.headers.get("X-Ninfer-Studio") != "1":
            self.close_connection = True
            self._json({"error": "Missing Studio request header"}, 403)
            return False
        return True

    def do_GET(self):
        if not self._trusted_request():
            return
        path = self.path.split("?", 1)[0]
        if path in _STATIC:
            fname, ctype = _STATIC[path]
            try:
                with open(os.path.join(WEBUI_DIR, fname), "rb") as f:
                    self._send(200, f.read(), ctype)
            except OSError as e:
                self._send(500, "missing %s: %s" % (fname, e), "text/plain; charset=utf-8")
            return
        if path == "/api/state":
            self._json(snapshot())
            return
        if path == "/api/ping":
            self._json({"ok": True, "ts": time.time(), "ui_port": UI_PORT, "engine": UPSTREAM})
            return
        if path == "/api/caps":
            self._json(capabilities(refresh=True))
            return
        if path == "/api/files":
            self._json(artifacts())
            return
        if path == "/api/log":
            want = "ui" if "ui=1" in self.path else "engine"
            path_log = UI_LOG if want == "ui" else ENGINE_LOG
            self._json({"path": path_log, "lines": tail(path_log, 60)})
            return
        full = self._safe_static(path)
        if full:
            ctype = _MIME.get(os.path.splitext(full)[1].lower(), "application/octet-stream")
            with open(full, "rb") as f:
                self._send(200, f.read(), ctype)
            return
        self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if not self._trusted_request(changing=True):
            return
        path = self.path.split("?", 1)[0]

        if path == "/api/engine/start":
            self._json(start_engine())
            return
        if path == "/api/engine/stop":
            # Close after responding rather than interpreting an unused body as
            # another HTTP request on this connection.
            self.close_connection = True
            result = stop_engine()
            self._json(result, 200 if result["ok"] else 409)
            return
        if path in ("/api/model/suspend", "/api/model/resume"):
            action = path.rsplit("/", 1)[-1]
            st, body = model_action(action)
            text = body.decode("utf-8", "replace")
            try:
                obj = json.loads(text) if text.strip() else {}
            except Exception:
                obj = {"raw": text[:400]}
            if st in (200, 202, 409):
                self._json({"ok": st in (200, 202), "status": st, "engine": obj})
            else:
                self._json({"ok": False, "status": st or 0, "error": obj}, 502 if st == 0 else st)
            return

        if path != "/api/chat":
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._json({"error": {"message": "bad request: %s" % e}}, 400)
            return

        payload["stream"] = bool(payload.get("stream", True))
        if not payload.get("model"):
            payload["model"] = default_model()
        req = urllib.request.Request(
            UPSTREAM + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=_headers({"Content-Type": "application/json"}), method="POST")
        try:
            up = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read() or b"", "application/json; charset=utf-8")
            return
        except Exception as e:
            self._json({"error": {"message": "engine unreachable: %s" % e}}, 502)
            return

        ctype = up.headers.get("Content-Type", "text/event-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(b):
            self.wfile.write(b"%X\r\n" % len(b) + b + b"\r\n")
            self.wfile.flush()

        try:
            while True:
                buf = up.read1(2048)
                if not buf:
                    break
                chunk(buf)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            up.close()


def wait_upstream(seconds=None):
    seconds = seconds if seconds is not None else int(ENGINE.get("startup_timeout_s") or 900)
    deadline = time.time() + seconds
    while time.time() < deadline:
        if fetch("/health", timeout=2.0)[0] == 200:
            return True
        time.sleep(1.0)
    return False


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    # Windows SO_REUSEADDR can share a live listener. POSIX needs it to restart
    # after recently closed HTTP connections leave the address in TIME_WAIT.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return  # the browser closed a metric poll / stream: nothing to report
        super().handle_error(request, client_address)


def main():
    if "--wait" in sys.argv:
        ok = wait_upstream()
        sys.stderr.write("[studio] engine %s at %s\n" % ("ready" if ok else "NOT ready", UPSTREAM))
        return 0 if ok else 1
    try:
        srv = StudioServer((UI_HOST, UI_PORT), Handler)
    except OSError as e:
        sys.stderr.write("[studio] cannot bind %s:%d (%s) - already running?\n" % (UI_HOST, UI_PORT, e))
        return 1
    sys.stderr.write("[studio] NInfer Studio on http://%s:%d/ -> %s\n" % (UI_HOST, UI_PORT, UPSTREAM))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
