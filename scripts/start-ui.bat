@echo off
rem Optional web UI for this repository: start the engine (through the official
rem launcher) and the Studio, then open it. Nothing here replaces scripts\run.bat.
setlocal
set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
where py >nul 2>&1
if errorlevel 1 (
  set "PY=python"
) else (
  set "PY=py -3"
)
%PY% "%ROOT%\webui\launcher.py" start %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo The web UI or the engine did not start. See the message above and %ROOT%\logs\.
  pause
)
endlocal & exit /b %RC%
