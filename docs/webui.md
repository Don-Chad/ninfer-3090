# Web UI (optional)

NInfer Studio is an optional browser front end for a `ninfer-serve` instance:
a streaming chat client, live telemetry, model suspend/resume, an engine log
viewer and a launcher for both processes. It is plain static files plus one
Python standard-library server - no packages, no build step, no node.

It does not replace `scripts/run.bat`. The official launcher stays the way to
serve a model; the Studio only talks to whatever is already listening, and when
it starts an engine itself it does so **by running your configured start
command**, which defaults to `scripts\run.bat qwen38-27b tuned`.

```
scripts\start-ui.bat          # engine + UI + browser
scripts\stop-ui.bat           # stop what start-ui.bat started
python webui\launcher.py status
```

The two `scripts\*_ui.bat` wrappers are Windows-only; they only call
`webui/launcher.py`, which is plain Python.

## Requirements

| | |
|---|---|
| Python | 3.9+ (`py -3` on Windows). Standard library only. |
| Engine | any `ninfer-serve` build: `/health`, `/v1/models`, `/v1/chat/completions` |
| Temperatures | NVIDIA driver (NVML). Without it the panel says why instead of guessing. |

## Policy: what this UI is allowed to touch

* Engine flags, profiles, artifacts and sampling defaults belong to
  `scripts/run.bat` and the documentation around it. The Studio passes none of
  its own; it sets `NINFER_HOST`, `NINFER_PORT` (and `NINFER_SERVER` when it can
  find the binary) and runs the configured command.
* Optional endpoints are probed, never assumed. The dashboard shows a control or
  a number only when the running build answers for it:

| Capability | Endpoint | When missing |
|---|---|---|
| chat | `POST /v1/chat/completions`, `GET /v1/models` | UI reports the engine unreachable |
| load / context | `GET /v1/load` | load, lane and context tiles stay empty |
| metrics | `GET /metrics` | throughput, VRAM, power tiles stay empty, with a note saying why |
| suspend / resume | `GET|POST /v1/models/{id}/residency|suspend|resume`, plus `enabled: true` in the residency body | the buttons are disabled and explain that the server was not started with suspend. A profile started without `--model-suspend` answers the endpoint but reports `enabled: false`, which counts as unavailable. |
| image input | `input_modalities` in `GET /v1/models` | the attach button is disabled |
| build / context limits | `GET /props` | those fields show `—` (that endpoint is llama.cpp heritage, not ninfer) |

`/health` answering `503` is treated as "loading the model", not "offline".

## Start

```
scripts\start-ui.bat                 # opens the browser when ready
scripts\start-ui.bat --no-browser
scripts\start-ui.bat --timeout 1800  # wait longer for a large artifact to load
```

The launcher:

1. reuses an engine already answering `/health` (never a second copy);
2. otherwise runs `engine.start_command` detached, logging to `logs/engine.log`;
3. waits for `/health` to answer `200` and prints the status transitions it saw;
4. starts the UI, waits for `/api/ping`, then opens `http://127.0.0.1:8090/`.

If the engine or UI port is held by a process this launcher did not start, it
prints the port and pid and exits non-zero without touching it.

## Stop

```
scripts\stop-ui.bat
```

Both the dashboard Stop button and the command-line launcher use the same
ownership records in `webui/.state/launcher.json`. They stop only processes
Studio started and verified descendants of those processes. Records include
process creation times so a reused PID cannot claim an unrelated process.
An engine you started yourself keeps running; its dashboard Stop button stays
disabled. Failed starts can also be stopped before they open a listening port.

## Configuration

`webui/studio.config.json` (all keys optional; the committed file holds the
defaults), or a path of your own with `--config`, or `STUDIO_CONFIG`.

```json
{
  "engine": {
    "host": "127.0.0.1",
    "port": 8080,
    "url": "",
    "start_command": ["scripts\\run.bat", "qwen38-27b", "tuned"],
    "startup_timeout_s": 900
  },
  "ui": { "host": "127.0.0.1", "port": 8090 },
  "paths": { "logs": "logs", "models": "models" },
  "poll_ms": 2000
}
```

| Key | Meaning |
|---|---|
| `engine.host` / `engine.port` | where the engine listens; the default `8080` is the port `scripts/run.bat` uses. |
| `engine.url` | full base URL; when set it wins over host/port (e.g. a remote box). |
| `engine.start_command` | argv of the start command, relative to the repository root. Leave empty to use the Studio as a dashboard only. |
| `engine.startup_timeout_s` | how long to wait for `/health` `200` before giving up. |
| `ui.host` / `ui.port` | where the Studio listens; the host must be a loopback address or `localhost`. |
| `paths.logs` / `paths.models` | relative to the repository root; `logs/` is where the launcher writes `engine.log` and `ui.log`. |
| `poll_ms` | dashboard refresh interval (the UI currently polls every 2 s). |

Environment overrides, applied after the file: `NINFER_HOST`, `NINFER_PORT`,
`NINFER_URL`, `NINFER_UI_HOST`, `NINFER_UI_PORT`, `STUDIO_LOGS_DIR`,
`STUDIO_MODELS_DIR`, `STUDIO_CONFIG`.

Serving on another port without editing anything:

```
set NINFER_PORT=8095
set NINFER_UI_PORT=8096
scripts\start-ui.bat
```

`NINFER_API_KEY` is read from the environment only (never from the JSON) and is
forwarded as a bearer token to the engine. `/metrics` and `/v1/load` need it
when the server was started with an API key.

Studio is a local browser application. It validates the request Host and
Origin, and its browser client includes `X-Ninfer-Studio: 1` on control and chat
requests. Cross-origin control requests are rejected. Only bundled public UI
assets are served; source, configuration and launcher state are private.
The resolved `--config` file is forwarded to the UI subprocess as well as used
by the launcher.

## Regression tests

Run `uv run --no-project --python 3.11 webui/test_studio.py -v` from the repository
root. These standard-library tests need no model or GPU. They cover file-access
boundaries, browser request validation, ownership and recycled PIDs, immediate
stream delivery, and a real launcher start/stop/restart cycle with custom ports
and a separate engine that must remain running.

## Temperatures

The Live tab reads temperatures from NVML through `ctypes` (`nvml.dll` on
Windows, `libnvidia-ml.so.1` elsewhere): no subprocess per poll and no extra
software. Each sensor is reported with its own availability:

* **GPU core** - from NVML; falls back to the engine's `ninfer_gpu_temperature_celsius`.
* **fans** - NVML fan speed, with the fan count when the driver reports it.
* **VRAM junction, hot spot, board/VRM** - shown only if the driver exposes them.
  NVML has no hot-spot or board sensor, and consumer GeForce cards do not report
  memory-junction temperature, so those rows read `n/a` with the reason in a
  tooltip. Nothing is estimated from the core temperature.

Throttle thresholds (`slowdown`, `max`, `shutdown`, `acquire`) come from
`nvmlDeviceGetTemperatureThreshold` and are shown as context, not as sensors.

## Troubleshooting

| Symptom | Meaning |
|---|---|
| `port 8080 is taken by pid N but /health does not answer 200` | something else owns the port; stop it yourself or configure another port. Nothing was started. |
| status pill says `loading model` | `/health` returns `503`; the model is still being loaded. Large artifacts take minutes. |
| metric tiles empty + note | the build does not serve `/metrics` (or needs the API key). Chat is unaffected. |
| suspend/resume disabled | the engine was not started with model suspend, or the build has no residency endpoint. |
| `cannot bind 127.0.0.1:8090` | another Studio instance is already running on that port. |
