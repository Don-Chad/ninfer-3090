@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Serve a model on one RTX 3090.
rem
rem   run.bat [model] [profile]     (double-click it and it asks for the model)
rem
rem   model             profiles
rem   qwen38-27b        tuned (default), int8, c8
rem   qwen36-35b-a3b    tuned (default)
rem
rem `tuned` is the recommended profile: rk4v4 KV, speculation plus the draft head, the memory
rem flags, vision in overlay residency, and the tuned context cache with automatic prefix grid.
rem `int8` and `c8` are the older reference profiles for the 27B -- one user at 64K of INT8 KV
rem (the quality default), and eight lanes at 8K -- with every serving flag fixed.
rem
rem Every measurement behind these defaults, the memory model, and the reasoning for each flag are
rem in docs\maintainer\launcher-profiles.md. What follows is what you need to run it.
rem
rem QWEN3.8-27B, `tuned`: two flag sets, each measured (docs\performance.md, "Recommended
rem configurations"), chosen with NINFER_SPEC. The default is the fast one.
rem
rem   NINFER_SPEC=dflash2 (default): fastest at one stream, 172,032 tokens of context
rem
rem     --spec dflash2 --draft-tokens 7 --lm-head-draft
rem     --prefill-cublas --prefill-chunk 4096
rem     --kv-dtype rk4v4 --embedding-q4 --gdn-state-fp16
rem     --vision --vision-residency overlay
rem
rem   NINFER_SPEC=mtp: the full 262,144-token native context, two lanes sharing it, still fast
rem
rem     --spec mtp --draft-tokens 3 --lm-head-draft
rem     --prefill-cublas --prefill-chunk 2048
rem     --kv-dtype rk4v4 --embedding-q4 --lm-head-q6 --gdn-state-fp16
rem     --vision --vision-residency overlay
rem
rem   set NINFER_SPEC=mtp && run.bat qwen38-27b
rem
rem   MTP accepts NINFER_DRAFT_TOKENS up to 15. Three suits chat and prose; for coding work that
rem   returns edited files, 11-15 decodes up to 1.85x faster (docs\cli.md has the table).
rem
rem rk4v4 KV (Lloyd-Max 4-bit keys) is 31%% smaller than rk8v4 at the same decode speed, for +0.10%%
rem perplexity over it. Measured on a desktop RTX 3090 (2026-09-24), the DFlash2 set starts at up to
rem 180,224 tokens (rk8v4: 131,072) and the default keeps a rung of margin; its draft weights and its
rem refusal of --lm-head-q6 are why it stops short of 262,144. The mtp set starts at 262,144 with two
rem lanes and about 1.2 GiB to spare. `none` is the mtp set without speculation.
rem The qwen3_8_27b.ninfer that download-model.bat fetches is the DFlash2 bundle and carries the
rem MTP weights too, so one file serves both.
rem
rem OVERRIDES, from the environment. All profiles: NINFER_MODEL (artifact path), NINFER_MODEL_DIR,
rem NINFER_SERVER, NINFER_HOST, NINFER_PORT, NINFER_GRAFT_DIR (phantom-kv graft directory),
rem NINFER_GRAFTS (set to "off" to disable graft loading). `tuned` also: NINFER_CONTEXT,
rem NINFER_CONCURRENCY, NINFER_KV_DTYPE, NINFER_SPEC, NINFER_DRAFT_TOKENS, NINFER_PREFILL_CHUNK,
rem NINFER_VISION (on^|off), NINFER_VISION_RESIDENCY, NINFER_HOST_STATE_SLOTS. Each spec's defaults
rem (context, lanes, chunk) are the ones measured to fit beside a desktop, which holds roughly 1.5 GiB
rem of the card; if startup refuses, drop a rung of NINFER_CONTEXT: 229376 / 196608 / 163840 / 131072 /
rem 98304 / 65536.
rem
rem IF THE CARD IS BUSY. A desktop (or another job) holding VRAM can leave too little for the default
rem context. When the `tuned` profile is refused at startup for lack of GPU memory, this launcher
rem steps down on its own -- an eighth of the context at a time, up to five times, and from the
rem second step also a 2048 prefill chunk and fewer host state slots -- and says what it did, so the
rem first run starts instead
rem of ending in an error. It only does that for the defaults: an explicit NINFER_CONTEXT, NINFER_PREFILL_CHUNK or
rem NINFER_HOST_STATE_SLOTS is honoured as given and fails loudly, and NINFER_FALLBACK=off turns the
rem step-down off.
rem
rem Loopback by default. 0.0.0.0 publishes an unauthenticated OpenAI-compatible endpoint to every
rem network this machine is on, so it is opt-in per run rather than the shipped default:
rem
rem   set NINFER_HOST=0.0.0.0 && run.bat qwen38-27b
rem ---------------------------------------------------------------------------------------------

set "MODEL_KEY=%~1"
set "PROFILE=%~2"
rem Step-down eligibility, decided once on the first pass: only the defaults of the tuned profile
rem may be second-guessed. A step-down pass sets the overrides itself and jumps back to :model_known.
if not defined RUNG (
  set "RUNG=0"
  set "LADDER=0"
  if "%NINFER_CONTEXT%%NINFER_PREFILL_CHUNK%%NINFER_HOST_STATE_SLOTS%"=="" if /i not "%NINFER_FALLBACK%"=="off" set "LADDER=1"
)
if "%PROFILE%"=="" set "PROFILE=tuned"
if "%MODEL_KEY%"=="" goto :choose_model
:model_resolve
if /i "%MODEL_KEY%"=="-h" goto :help
if /i "%MODEL_KEY%"=="--help" goto :help
if /i "%MODEL_KEY%"=="qwen38-27b" (
  set "ARTIFACT=qwen3_8_27b.ninfer"
  set "TITLE=Qwen3.8-27B"
  set "GRAFT_FILE=v1_q38_nf4_trained.bin"
  goto :model_known
)
if /i "%MODEL_KEY%"=="qwen36-35b-a3b" (
  set "ARTIFACT=qwen3_6_35b_a3b.ninfer"
  set "TITLE=Qwen3.6-35B-A3B"
  set "GRAFT_FILE=v1_q36_35b_nf4_trained.bin"
  goto :model_known
)
echo Unknown model: %MODEL_KEY% 1>&2
call :usage 1>&2
exit /b 2

:choose_model
rem Double-clicked from Explorer there is no argument to give, so ask. choice exits 255 when it has
rem no console to read from; that must not silently pick a model.
echo Which model?
echo   1  Qwen3.6-35B-A3B  (recommended)
echo   2  Qwen3.8-27B
choice /c 12 /n /m "Choose 1 or 2: "
if errorlevel 255 exit /b 2
if errorlevel 2 (
  set "MODEL_KEY=qwen38-27b"
) else (
  set "MODEL_KEY=qwen36-35b-a3b"
)
goto :model_resolve

:help
call :usage
exit /b 0

:model_known
rem Two layouts share this launcher. In the release archive it sits at the archive root beside
rem models\, which is what download-model.bat writes to there. In a checkout it sits in scripts\,
rem one level under the repo root, and models\ (see .gitignore) is beside the repo root, not beside
rem this script -- a directory named scripts with a CMakeLists.txt above it tells the two apart (a
rem bare CMakeLists.txt probe would misfire on an archive unpacked beneath any source tree). This
rem must agree with download-model.bat's own default, which uses the same rule.
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
for %%I in ("%~dp0.") do set "SCRIPT_DIRNAME=%%~nxI"
set "MODEL_DIR=%~dp0models"
if /i "%SCRIPT_DIRNAME%"=="scripts" if exist "%ROOT%\CMakeLists.txt" set "MODEL_DIR=%ROOT%\models"
set "MODEL=%MODEL_DIR%\%ARTIFACT%"
rem An explicit NINFER_MODEL_DIR is taken verbatim and never probed, as in run.sh, so a model
rem downloaded there with download-model.bat is found here.
if not "%NINFER_MODEL_DIR%"=="" set "MODEL=%NINFER_MODEL_DIR%\%ARTIFACT%"
set "HOST=127.0.0.1"
set "PORT=8080"
set "KV_DTYPE=rk4v4"
if not "%NINFER_MODEL%"=="" set "MODEL=%NINFER_MODEL%"
if not "%NINFER_HOST%"=="" set "HOST=%NINFER_HOST%"
if not "%NINFER_PORT%"=="" set "PORT=%NINFER_PORT%"

set "SERVER=%ROOT%\build-ninja\apps\ninfer-serve.exe"
if not exist "%SERVER%" set "SERVER=%~dp0ninfer-serve.exe"
if not "%NINFER_SERVER%"=="" set "SERVER=%NINFER_SERVER%"

rem Phantom-KV graft: default directory is artifacts\grafts in this repo. The graft
rem file name is set per model key above. NINFER_GRAFTS=off disables graft loading entirely;
rem NINFER_GRAFT_DIR overrides where to look.
set "GRAFT_DIR=%ROOT%\artifacts\grafts"
if not "%NINFER_GRAFT_DIR%"=="" set "GRAFT_DIR=%NINFER_GRAFT_DIR%"

rem The profile fixes the whole serving shape. LABEL is the banner; PROFILE_ARGS is everything
rem after --host/--port. Values below use ^| for the separator: a bare pipe inside an expanded
rem variable would be parsed as a pipe when echoed.
set "LABEL="
set "PROFILE_ARGS="
set "PREFILL_NOTE="
set "HINT="
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/tuned" goto :profile_27b_tuned
if /i "%MODEL_KEY%/%PROFILE%"=="qwen36-35b-a3b/tuned" goto :profile_35b_tuned
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/int8" goto :profile_27b_int8
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/c8" goto :profile_27b_c8
echo Model %MODEL_KEY% has no profile %PROFILE% 1>&2
call :usage 1>&2
exit /b 2

:profile_27b_tuned
rem The speculative backend fixes everything that has to move with it: the prefill chunk (the
rem cuBLAS route's workspace scales with it), the context that fits, and --lm-head-q6, which
rem DFlash and DFlash2 refuse. The overrides are applied after these defaults, so an explicit
rem NINFER_CONTEXT or NINFER_PREFILL_CHUNK always wins.
set "SPEC=dflash2"
if not "%NINFER_SPEC%"=="" set "SPEC=%NINFER_SPEC%"
if /i "%SPEC%"=="dflash2" goto :spec_dflash2
if /i "%SPEC%"=="mtp" goto :spec_mtp
if /i "%SPEC%"=="none" goto :spec_none
echo NINFER_SPEC must be dflash2, mtp or none, got %SPEC% 1>&2
exit /b 2

:spec_dflash2
set "SPEC=dflash2"
set "CONTEXT=172032"
set "CONCURRENCY=1"
set "PREFILL_CHUNK=4096"
set "DRAFT_TOKENS=7"
set "MEMORY_ARGS="
goto :spec_done

:spec_mtp
set "SPEC=mtp"
set "CONTEXT=262144"
set "CONCURRENCY=2"
set "PREFILL_CHUNK=2048"
set "DRAFT_TOKENS=3"
set "MEMORY_ARGS=--lm-head-q6"
goto :spec_done

:spec_none
set "SPEC=none"
set "CONTEXT=262144"
set "CONCURRENCY=2"
set "PREFILL_CHUNK=2048"
set "DRAFT_TOKENS="
set "MEMORY_ARGS="

:spec_done
if not "%NINFER_DRAFT_TOKENS%"=="" set "DRAFT_TOKENS=%NINFER_DRAFT_TOKENS%"
if not "%NINFER_CONTEXT%"=="" set "CONTEXT=%NINFER_CONTEXT%"
if not "%NINFER_CONCURRENCY%"=="" set "CONCURRENCY=%NINFER_CONCURRENCY%"
if not "%NINFER_KV_DTYPE%"=="" set "KV_DTYPE=%NINFER_KV_DTYPE%"
if not "%NINFER_PREFILL_CHUNK%"=="" set "PREFILL_CHUNK=%NINFER_PREFILL_CHUNK%"
set "SPEC_ARGS="
if /i not "%SPEC%"=="none" set "SPEC_ARGS=--spec %SPEC% --draft-tokens %DRAFT_TOKENS% --lm-head-draft"
if /i "%SPEC%"=="dflash2" set "SPEC_LABEL=DFlash2 K=%DRAFT_TOKENS% + draft head"
if /i "%SPEC%"=="mtp" set "SPEC_LABEL=MTP%DRAFT_TOKENS% + draft head, Q6 head, full context"
if /i "%SPEC%"=="none" set "SPEC_LABEL=no speculation"
set "PROFILE_ARGS=--max-concurrency %CONCURRENCY% --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-dtype %KV_DTYPE% %SPEC_ARGS% --embedding-q4 %MEMORY_ARGS% --gdn-state-fp16 --prefill-cublas --prefill-chunk %PREFILL_CHUNK%"
set "LABEL=C%CONCURRENCY%  ^|  context %CONTEXT%  ^|  %KV_DTYPE% KV  ^|  %SPEC_LABEL%"
set "PREFILL_NOTE=Prefill: cuBLAS route, chunk %PREFILL_CHUNK%"
if /i "%SPEC%"=="dflash2" set "HINT=Need the full 262K context or a second lane? Set NINFER_SPEC=mtp: slower decode."
goto :profile_tuned_common

:profile_35b_tuned
rem rk4v4 KV fits the full native context with two lanes beside a desktop (three measured to start,
rem 2026-09-24; rk8v4 managed 147,456 with one). DFlash2 is a 27B-only backend. Lanes share the
rem one --kv-capacity pool: any request may use all 262,144 tokens, but the lanes' requests together
rem hold at most that many at a time.
set "SPEC=mtp"
if not "%NINFER_SPEC%"=="" set "SPEC=%NINFER_SPEC%"
set "CONTEXT=262144"
set "CONCURRENCY=2"
set "PREFILL_CHUNK=512"
set "DRAFT_TOKENS=3"
if /i "%SPEC%"=="mtp" goto :spec35_mtp
if /i "%SPEC%"=="none" goto :spec35_none
echo NINFER_SPEC must be mtp or none, got %SPEC% 1>&2
exit /b 2
:spec35_mtp
set "SPEC=mtp"
set "SPEC_LABEL=MTP3 + draft head"
goto :spec35_done
:spec35_none
set "SPEC=none"
set "SPEC_LABEL=no speculation"
:spec35_done
if not "%NINFER_DRAFT_TOKENS%"=="" set "DRAFT_TOKENS=%NINFER_DRAFT_TOKENS%"
if not "%NINFER_CONTEXT%"=="" set "CONTEXT=%NINFER_CONTEXT%"
if not "%NINFER_CONCURRENCY%"=="" set "CONCURRENCY=%NINFER_CONCURRENCY%"
if not "%NINFER_KV_DTYPE%"=="" set "KV_DTYPE=%NINFER_KV_DTYPE%"
if not "%NINFER_PREFILL_CHUNK%"=="" set "PREFILL_CHUNK=%NINFER_PREFILL_CHUNK%"
set "SPEC_ARGS="
if /i "%SPEC%"=="mtp" set "SPEC_ARGS=--spec mtp --draft-tokens %DRAFT_TOKENS% --lm-head-draft --mtp-experts-q4"
if /i "%SPEC%"=="mtp" set "SPEC_LABEL=MTP%DRAFT_TOKENS% + draft head"
set "PROFILE_ARGS=--max-concurrency %CONCURRENCY% --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-dtype %KV_DTYPE% %SPEC_ARGS% --gdn-state-fp16 --prefill-chunk %PREFILL_CHUNK%"
set "LABEL=C%CONCURRENCY%  ^|  context %CONTEXT%  ^|  %KV_DTYPE% KV  ^|  %SPEC_LABEL%"
goto :profile_tuned_common

:profile_27b_int8
set "PROFILE_ARGS=--max-context 65536 --kv-capacity 65536 --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 --prefill-chunk 1024 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft"
set "LABEL=one request  ^|  64K context  ^|  INT8 KV  ^|  MTP3, ReplaySSM"
goto :launch

:profile_27b_c8
rem Context cache sized per lane, so several agents rotating through the lanes find their own
rem conversation still cached instead of re-prefilling it: two retained conversations per lane,
rem one checkpoint StateImage per lane on the card beyond the active ones, and two per lane in
rem pinned host memory. Measured at one lane on the default of two retained conversations, four
rem rotating agents reused 12%% of their prompts (TTFT 14 s); with room for all of them, 76%% (2.9 s).
rem MEMORY COST: this profile keeps the GDN state in BF16, so a StateImage is 147 MiB. The device
rem slots take 8 x 147 MiB = 1.15 GiB of VRAM, the engine default at eight lanes, so that is
rem unchanged; the host slots pin 16 x 147 MiB = 2.3 GiB of RAM, which WDDM also charges against the
rem card. Lower C8_HOST_STATES_PER_LANE first if startup runs short. Retention is still bounded by
rem the 16,384-token KV pool below.
set "C8_LANES=8"
set /a C8_PRIVATE=C8_LANES*2
set /a C8_DEVICE_STATES=C8_LANES
set "C8_HOST_STATES_PER_LANE=2"
set /a C8_HOST_STATES=C8_LANES*C8_HOST_STATES_PER_LANE
set "PROFILE_ARGS=--max-context 8192 --kv-capacity 16384 --max-concurrency %C8_LANES% --max-pending-requests 32 --pending-timeout-ms 600000 --prefill-chunk 512 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --max-private-continuations %C8_PRIVATE% --device-state-slots %C8_DEVICE_STATES% --host-state-slots %C8_HOST_STATES%"
set "LABEL=up to eight requests  ^|  8K context  ^|  INT8 KV  ^|  MTP3, ReplaySSM"
goto :launch

:profile_tuned_common
rem Only `tuned` carries the context cache and vision: the reference profiles are deliberately
rem minimal. Vision stays on -- overlay residency keeps the tower host-pinned and streams each
rem image through a borrowed device window, so it costs about 10 MiB of runtime reservation.
set "VISION=on"
if not "%NINFER_VISION%"=="" set "VISION=%NINFER_VISION%"
set "VISION_RESIDENCY=overlay"
if not "%NINFER_VISION_RESIDENCY%"=="" set "VISION_RESIDENCY=%NINFER_VISION_RESIDENCY%"
set "VISION_ARGS="
rem Pinned host memory for the context cache: 74.5 MiB per slot on the 27B. WDDM charges it against
rem the card, so a busy desktop needs fewer (see the README on startup).
rem --max-private-continuations 8 below is what keeps several rotating conversations cached: the
rem engine default is two per lane, and four agents on one lane then evict each other on every turn
rem (12%% prompt reuse against 76%% with room for all four, measured 2026-09-28). A retained
rem conversation costs no memory by itself; its KV pages and StateImages come from the pools above.
set "HOST_STATE_SLOTS=32"
if not "%NINFER_HOST_STATE_SLOTS%"=="" set "HOST_STATE_SLOTS=%NINFER_HOST_STATE_SLOTS%"
if /i "%VISION%"=="on" (
  set "VISION_ARGS=--vision --vision-residency %VISION_RESIDENCY%"
  set "LABEL=%LABEL%  ^|  vision (%VISION_RESIDENCY%)"
  goto :vision_done
)
if /i "%VISION%"=="off" (
  set "LABEL=%LABEL%  ^|  text only"
  goto :vision_done
)
echo NINFER_VISION must be on or off, got %VISION% 1>&2
exit /b 2
:vision_done
set "PROFILE_ARGS=%PROFILE_ARGS% --max-pending-requests 16 --pending-timeout-ms 600000 %VISION_ARGS% --max-private-continuations 8 --max-shared-prefixes 8 --host-state-slots %HOST_STATE_SLOTS% --host-kv-mib 8192 --auto-prefix-grid"

:launch
set "GRAFT_ARGS="
if /i "%NINFER_GRAFTS%"=="off" goto :graft_done
if "%GRAFT_FILE%"=="" goto :graft_done
if not exist "%GRAFT_DIR%\%GRAFT_FILE%" goto :graft_done
set "GRAFT_ARGS=--graft "v1=%GRAFT_DIR%\%GRAFT_FILE%""
:graft_done
if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Build it first:  .\scripts\build.ps1
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Download it first:  download-model.bat %MODEL_KEY%
  exit /b 1
)

echo %TITLE%  ^|  %LABEL%
if not "%PREFILL_NOTE%"=="" echo %PREFILL_NOTE%
if /i "%PROFILE%"=="tuned" echo Cache: 8 shared / 8 private / %HOST_STATE_SLOTS% host states  ^|  automatic prefix grid on
if not "%GRAFT_ARGS%"=="" echo Graft: v1 = %GRAFT_FILE%
if not "%HINT%"=="" echo %HINT%
echo API: http://%HOST%:%PORT%/v1
echo.

rem WHAT --host-kv-mib 8192 ACTUALLY GETS ON WINDOWS, which is not 8 GiB. WDDM maps a pinned host
rem allocation into the GPU's address space and charges it against the card, so the runtime clamps
rem the request to (free VRAM - 1 GiB) / 2 before the first cudaMallocHost. The flag is kept
rem rather than corrected because it is right on Linux, where the full 8 GiB of host RAM is pinned,
rem and because it is harmless here: the clamp takes what is actually free after the KV cache is
rem allocated, so it costs no context, and prefix reuse falls back to device pages when the pin is
rem zero. Do not read "8192" as a description of this machine. See
rem docs\maintainer\launcher-profiles.md.
if /i not "%PROFILE%"=="tuned" set "LADDER=0"
if "%LADDER%"=="0" (
  "%SERVER%" "%MODEL%" --host %HOST% --port %PORT% %PROFILE_ARGS% %GRAFT_ARGS%
  endlocal
  exit /b %ERRORLEVEL%
)

rem Run the server with its output shown and kept, so a refusal for lack of memory can be told apart
rem from any other failure. Only that failure steps down; a crash or a bad artifact does not. The log
rem is written as ASCII on purpose: Tee-Object writes UTF-16, which findstr cannot search.
if "%RUNG%"=="0" (
  set "BASE_CONTEXT=%CONTEXT%"
  set "BASE_CHUNK=%PREFILL_CHUNK%"
  set "BASE_SLOTS=%HOST_STATE_SLOTS%"
)
set "SERVER_LOG=%TEMP%\ninfer-run-%RANDOM%%RANDOM%.log"
"%SERVER%" "%MODEL%" --host %HOST% --port %PORT% %PROFILE_ARGS% %GRAFT_ARGS% 2>&1 | powershell -NoProfile -Command "$input | ForEach-Object { $_; Add-Content -LiteralPath '%SERVER_LOG%' -Value $_ -Encoding Ascii }"
findstr /c:"runtime reservation requires" /c:"cudaMallocHost failed" "%SERVER_LOG%" >nul 2>&1
if errorlevel 1 goto :server_done
if %RUNG% GEQ 5 goto :server_done
set /a NEXT=RUNG+1
set /a NEXT_CONTEXT=BASE_CONTEXT*(8-NEXT)/8/1024*1024
rem The first step trims context only: an eighth of it frees more than a card that just misses
rem needs, and prefill speed and cached prefixes are worth keeping. Later steps also give up the
rem wider prefill chunk and halve the host state slots every other step.
set "NEXT_CHUNK=%BASE_CHUNK%"
if %NEXT% GEQ 2 if %BASE_CHUNK% GTR 2048 set "NEXT_CHUNK=2048"
set /a "NEXT_SLOTS=BASE_SLOTS>>(NEXT/2)"
echo.
echo Not enough free GPU memory to start at context %CONTEXT%. Retrying at %NEXT_CONTEXT% (prefill chunk %NEXT_CHUNK%, %NEXT_SLOTS% host state slots).
echo Set NINFER_CONTEXT to choose your own, or NINFER_FALLBACK=off to fail instead.
echo.
del "%SERVER_LOG%" >nul 2>&1
set "NINFER_CONTEXT=%NEXT_CONTEXT%"
set "NINFER_PREFILL_CHUNK=%NEXT_CHUNK%"
set "NINFER_HOST_STATE_SLOTS=%NEXT_SLOTS%"
set "RUNG=%NEXT%"
goto :model_known

:server_done
rem The pipe hides the server's own exit status, so report failure from what it logged.
findstr /c:"FATAL" "%SERVER_LOG%" >nul 2>&1
set "SERVER_STATUS=0"
if not errorlevel 1 set "SERVER_STATUS=1"
del "%SERVER_LOG%" >nul 2>&1
endlocal & exit /b %SERVER_STATUS%

:usage
echo usage: run.bat ^<model^> [profile]
echo   qwen38-27b       tuned (default), int8, c8
echo   qwen36-35b-a3b   tuned (default)
exit /b 0
