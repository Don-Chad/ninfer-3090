@echo off
setlocal

rem ======================== EDITABLE SETTINGS ========================
set "MAX_CONTEXT=131072"
set "OUTPUT_TOKENS=1024"
set "PREFILL_PROMPT_CHARACTERS=28000"
set "COHORTS=1,2,4,8"
set "KV_DTYPE=rk8v4"
set "START_DELAY_SECONDS=10"
set "MODEL=%~dp0..\..\qwen3_8_27b.ninfer"
set "SERVER=%~dp0..\build-sm86-replayssm\apps\Release\ninfer-serve.exe"
rem ==================================================================

for %%I in ("%~dp0..") do set "REPO=%%~fI"
if not exist "%SERVER%" (
  echo ERROR: Server not found: %SERVER%
  exit /b 1
)
if not exist "%MODEL%" (
  echo ERROR: Model not found: %MODEL%
  exit /b 1
)
where uv >nul 2>nul || (
  echo ERROR: uv is not available in PATH.
  exit /b 1
)

echo.
echo RTX 3090 Qwen3.8 benchmark
echo   Shared context : %MAX_CONTEXT% tokens ^(C1 full; C8 capped at 8K per request^)
echo   Decode output  : %OUTPUT_TOKENS% tokens
echo   Cohorts        : C1, C2, C4, C8
echo   KV cache       : %KV_DTYPE%
echo   Results        : %REPO%\benchmark_results\windows_3090_*
if /I "%KV_DTYPE%"=="int8" if %MAX_CONTEXT% GTR 65536 echo WARNING: This high-context INT8 profile is not the recommended 3090 benchmark setting.
echo.
echo Starting in %START_DELAY_SECONDS% seconds. Press Ctrl+C to cancel.
timeout /t %START_DELAY_SECONDS% /nobreak >nul

set "NINFER_BENCH_SERVER=%SERVER%"
set "NINFER_BENCH_MODEL=%MODEL%"
set "NINFER_BENCH_MAX_CONTEXT=%MAX_CONTEXT%"
set "NINFER_BENCH_OUTPUT_TOKENS=%OUTPUT_TOKENS%"
set "NINFER_BENCH_PREFILL_CHARS=%PREFILL_PROMPT_CHARACTERS%"
set "NINFER_BENCH_COHORTS=%COHORTS%"
set "NINFER_BENCH_KV_DTYPE=%KV_DTYPE%"

pushd "%REPO%"
uv run tools\bench\run_qwen38_windows_3090_benchmarks.py
set "RESULT=%ERRORLEVEL%"
popd

if not "%RESULT%"=="0" (
  echo.
  echo BENCHMARK FAILED. Logs and partial results were preserved.
  exit /b %RESULT%
)
echo.
echo BENCHMARK COMPLETE. Open the results directory printed above.
exit /b 0
