@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Qwen3.6-35B-A3B on one RTX 3090 -- single user, full native 262,144-token context.
rem
rem Runs the Ninja build directly out of build-ninja\apps, not a packaged release.
rem
rem VRAM budget, measured on this machine with Windows holding ~2.7 GB:
rem
rem   core text weights        19.59 GiB   always resident
rem   KV cache @ 262144 tok     1.99 GiB   rk8v4, 7,969 bytes/token
rem   sequence + workspace      0.14 GiB
rem   ------------------------------------
rem   planned device total     21.77 GiB   leaving ~309 MiB free after startup
rem
rem That headroom is thin on purpose -- you asked for the largest context that fits. If the server
rem fails to start, or dies after you open something GPU-hungry, drop CONTEXT to the next value
rem down. Each 32,768 tokens is worth about 261 MiB:
rem
rem   CONTEXT=262144  ~309 MiB free   full native context (default)
rem   CONTEXT=229376  ~570 MiB free
rem   CONTEXT=196608  ~831 MiB free   comfortable alongside a browser
rem
rem Why no speculative decoding: the MTP head costs 856 MiB of resident weights, which is about
rem 110,000 tokens of context at this KV rate. It buys decode throughput, and this profile spends
rem that budget on context instead. Decode still measures ~181 tok/s here, because only ~3B of the
rem 35B parameters are active per token. If you would rather have the speed, add
rem `--spec mtp --draft-tokens 3` and drop CONTEXT to about 131072.
rem
rem Why rk8v4 KV: 7,969 bytes/token against int8's ~10,560, so it buys roughly 32% more context for
rem +0.082% perplexity. int8 at this context would need ~2.6 GiB of KV and would not fit.
rem
rem Vision is off: the vision tower is another 261 MiB and there is no room for it here.
rem ---------------------------------------------------------------------------------------------

set "CONTEXT=262144"
set "MODEL=C:\Ninefer-3090\models\qwen3_6_35b_a3b.ninfer"

rem Bind address. 0.0.0.0 exposes an unauthenticated OpenAI-compatible endpoint to your whole LAN,
rem which is what the qwen38 launcher does; use 127.0.0.1 to keep it on this machine only.
set "HOST=0.0.0.0"
set "PORT=8080"

rem scripts\ -> repo root -> the Ninja build output.
set "ROOT=%~dp0.."
set "SERVER=%ROOT%\build-ninja\apps\ninfer-serve.exe"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Build it first, from a VS 2022 BuildTools environment:
  echo   cmake --build "%ROOT%\build-ninja"
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  exit /b 1
)

echo Qwen3.6-35B-A3B  ^|  C1  ^|  context %CONTEXT%  ^|  rk8v4 KV  ^|  no speculation
echo API: http://%HOST%:%PORT%/v1
echo.

"%SERVER%" "%MODEL%" ^
  --host %HOST% --port %PORT% ^
  --max-concurrency 1 ^
  --max-context %CONTEXT% ^
  --kv-capacity %CONTEXT% ^
  --kv-dtype rk8v4 ^
  --prefill-chunk 512 ^
  --max-pending-requests 16

endlocal
