@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_6_35b_a3b.ninfer"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
echo Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model...
curl.exe -L -C - --fail --output "%MODEL%" "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c/qwen3_6_35b_a3b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)
echo Model ready: %MODEL%
