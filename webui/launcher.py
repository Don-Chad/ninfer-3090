#!/usr/bin/env python3
"""Start, stop and inspect NInfer Studio (web UI) and the ninfer engine.

    python webui/launcher.py start [--no-browser] [--timeout SECONDS]
    python webui/launcher.py stop
    python webui/launcher.py status

Design rules, because this touches a running inference server:

* The engine is started through the repository's own launcher command
  (`engine.start_command`, scripts/run.bat by default). This tool does not
  carry serving flags of its own, so the measured profile stays authoritative.
* Nothing already running is started twice: an engine answering /health is
  reused, and a UI answering /api/ping is reused.
* A port held by a foreign process is reported and left alone; no process is
  ever killed unless this launcher started it and recorded the pid in
  webui/.state/launcher.json.
* The UI is only opened after the engine answers /health with 200, which is
  how ninfer reports "model loaded and serving".
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server as S  # noqa: E402  single source of truth for config and ports

STATE_PATH = os.path.join(S.WEBUI_DIR, ".state", "launcher.json")


def _load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def http_status(url, path, timeout=2.0):
    """HTTP status of one request, 0 when nothing answers."""
    try:
        with urllib.request.urlopen(url + path, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def port_busy(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def engine_ready():
    return http_status(S.UPSTREAM, "/health") == 200


def ui_ready():
    return http_status("http://%s:%d" % (S.UI_HOST, S.UI_PORT), "/api/ping") == 200


def say(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def wait_ready(timeout, on_poll=None):
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        st = http_status(S.UPSTREAM, "/health")
        if on_poll:
            on_poll()
        if st == 200:
            return True
        if st != last:
            say("[.] engine /health -> %s%s" % (st or "no answer",
                                                " (loading the model)" if st == 503 else ""))
            last = st
        time.sleep(2.0)
    return False


def _pid_image(pid):
    """Image name of a pid, so a recycled pid is never killed by mistake."""
    try:
        if os.name == "nt":
            res = subprocess.run(["tasklist", "/FI", "PID eq %s" % pid, "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            first = (res.stdout or "").splitlines()
            if first and first[0].startswith('"'):
                return first[0].split('","')[0].strip('"').lower()
        else:
            res = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                                 capture_output=True, text=True, timeout=10)
            return (res.stdout or "").strip().lower()
    except Exception:
        pass
    return ""


ALLOWED_IMAGES = {"ninfer-serve.exe", "cmd.exe", "python.exe", "python3", "py.exe", "py"}


def claim(pids):
    """Add pids that belong to this launcher's own tree and still look like it."""
    state = _load_state()
    owned = {str(p) for p in (state.get("engine", {}).get("owned") or [])}
    for pid in pids:
        img = _pid_image(pid)
        if img in ALLOWED_IMAGES:
            owned.add(str(pid))
    eng = state.get("engine") or {}
    eng["owned"] = sorted(owned)
    state["engine"] = eng
    _save_state(state)


def cmd_start(argv):
    no_browser = "--no-browser" in argv
    timeout = int(S.ENGINE.get("startup_timeout_s") or 900)
    for i, arg in enumerate(argv):
        if arg == "--timeout" and i + 1 < len(argv):
            timeout = int(argv[i + 1])
        elif arg.startswith("--timeout="):
            timeout = int(arg.split("=", 1)[1])

    state = _load_state()
    state.setdefault("engine", {"owned": []})
    state.setdefault("ui", {"owned": []})

    # ---- engine -------------------------------------------------------
    if engine_ready():
        say("[i] engine already serving on %s" % S.UPSTREAM)
    else:
        busy = S.pids_on_port(S.UPSTREAM_PORT)
        if busy:
            say("[x] port %d is taken by pid %s but %s/health does not answer 200."
                % (S.UPSTREAM_PORT, ",".join(busy), S.UPSTREAM))
            say("    Stop that process yourself, or set another port in webui/studio.config.json")
            say("    (engine.port / NINFER_PORT). Nothing was started.")
            return 1
        started = S.start_engine()
        if not started.get("ok"):
            say("[x] cannot start the engine: %s" % started.get("error"))
            return 1
        say("[i] engine starting: %s (log: %s)" % (" ".join(started.get("command") or []),
                                                   started.get("log")))
        state["engine"] = {"owned": [str(started["pid"])], "started_at": time.time(),
                           "command": started.get("command")}
        _save_state(state)
        if not wait_ready(timeout, on_poll=lambda: claim(S.pids_on_port(S.UPSTREAM_PORT))):
            say("[x] engine did not become ready within %ds. See %s" % (timeout, S.ENGINE_LOG))
            say("    The launcher keeps what it started; run 'launcher.py stop' to clean up.")
            return 1
        # the listening process is a child of the launcher shell: remember it too
        owned = {str(p) for p in (state["engine"].get("owned") or [])} | set(S.pids_on_port(S.UPSTREAM_PORT))
        state["engine"] = {"owned": sorted(owned), "started_at": state["engine"].get("started_at"),
                           "command": state["engine"].get("command")}
        _save_state(state)
        say("[ok] engine ready on %s" % S.UPSTREAM)

    # ---- web UI -------------------------------------------------------
    if ui_ready():
        say("[i] web UI already running on http://%s:%d" % (S.UI_HOST, S.UI_PORT))
    else:
        busy = S.pids_on_port(S.UI_PORT)
        if busy:
            say("[x] port %d is taken by pid %s and it is not this Studio."
                % (S.UI_PORT, ",".join(busy)))
            say("    Change ui.port in webui/studio.config.json (or NINFER_UI_PORT). Nothing was started.")
            return 1
        os.makedirs(S.LOGS_DIR, exist_ok=True)
        log = open(S.UI_LOG, "ab", buffering=0)
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen([sys.executable, os.path.join(S.WEBUI_DIR, "server.py")],
                                cwd=S.PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=flags if os.name == "nt" else 0)
        log.close()
        state["ui"] = {"owned": [proc.pid], "started_at": time.time()}
        _save_state(state)
        deadline = time.time() + 30
        while time.time() < deadline and not ui_ready():
            time.sleep(0.5)
        if not ui_ready():
            say("[x] web UI did not answer /api/ping on port %d. See %s" % (S.UI_PORT, S.UI_LOG))
            return 1
        say("[ok] web UI ready on http://%s:%d" % (S.UI_HOST, S.UI_PORT))

    url = "http://%s:%d/" % (S.UI_HOST, S.UI_PORT)
    say("")
    say("     Studio : %s" % url)
    say("     API    : %s/v1" % S.UPSTREAM)
    say("     logs   : %s" % S.LOGS_DIR)
    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return 0


def cmd_stop(argv):
    state = _load_state()
    rc = 0

    owned_ui = [str(p) for p in (state.get("ui", {}).get("owned") or [])]
    killed, skipped = [], []
    for pid in S.pids_on_port(S.UI_PORT):
        (killed if pid in owned_ui and _pid_image(pid) in ALLOWED_IMAGES else skipped).append(pid)
    if killed:
        _kill(killed)
        say("[ok] stopped web UI (pid %s)" % ", ".join(killed))
    elif skipped:
        say("[!] port %d is held by pid %s, which this launcher did not start - left alone."
            % (S.UI_PORT, ",".join(skipped)))
        rc = 1
    else:
        say("[i] web UI was not running")

    owned_engine = [str(p) for p in (state.get("engine", {}).get("owned") or [])]
    killed, skipped = [], []
    for pid in S.pids_on_port(S.UPSTREAM_PORT):
        (killed if pid in owned_engine and _pid_image(pid) in ALLOWED_IMAGES else skipped).append(pid)
    if killed:
        _kill(killed)
        say("[ok] stopped engine (pid %s)" % ", ".join(killed))
    elif skipped:
        say("[!] engine on port %d (pid %s) was not started by this launcher - left running."
            % (S.UPSTREAM_PORT, ",".join(skipped)))
        rc = 1
    else:
        say("[i] engine was not running")

    _save_state({})
    say("[ok] done. ports %d and %d are %s." % (
        S.UI_PORT, S.UPSTREAM_PORT,
        "free" if not (port_busy(S.UI_PORT) or port_busy(S.UPSTREAM_PORT)) else "still in use"))
    return rc


def _kill(pids):
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"] if os.name == "nt"
                       else ["kill", "-TERM", str(pid)],
                       capture_output=True, timeout=20,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)


def cmd_status(argv):
    state = _load_state()
    eng_ok = engine_ready()
    ui_ok = ui_ready()
    say("engine   : %s  %s" % (S.UPSTREAM, "ready" if eng_ok else
                               "loading" if http_status(S.UPSTREAM, "/health") == 503 else "down"))
    say("           port %d pids: %s" % (S.UPSTREAM_PORT, ",".join(S.pids_on_port(S.UPSTREAM_PORT)) or "-"))
    say("web UI   : http://%s:%d  %s" % (S.UI_HOST, S.UI_PORT, "ready" if ui_ok else "down"))
    say("           port %d pids: %s" % (S.UI_PORT, ",".join(S.pids_on_port(S.UI_PORT)) or "-"))
    say("launcher : engine owned by launcher: %s" % (",".join(map(str, state.get("engine", {}).get("owned") or [])) or "-"))
    say("           engine command: %s" % " ".join(S.ENGINE.get("start_command") or []) or "-")
    say("           config file: %s" % (S.CFG.get("_config_file") or "(built-in defaults)"))
    return 0 if (eng_ok or ui_ok) else 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "start":
        return cmd_start(rest)
    if cmd == "stop":
        return cmd_stop(rest)
    if cmd == "status":
        return cmd_status(rest)
    sys.stderr.write("unknown command %r\n\n%s" % (cmd, __doc__))
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
