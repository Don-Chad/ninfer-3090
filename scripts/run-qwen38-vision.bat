@echo off
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%models\qwen3_8_27b.ninfer"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Put this launcher beside the release files.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Run download-qwen38.bat first, or drag the model onto this launcher.
  exit /b 1
)

echo Starting Qwen3.8-27B Vision at http://127.0.0.1:8080/v1
echo Tested RTX 3090 profile: one request, 32K context, ReplaySSM and MTP3
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port 8080 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-pending-requests 8 --prefill-chunk 512 --kv-dtype int8 --default-max-tokens 1024 --vision --spec mtp --draft-tokens 3 --lm-head-draft
