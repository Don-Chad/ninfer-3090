@echo off
rem Stop only what scripts\start-ui.bat started. A server or UI you launched
rem yourself (scripts\run.bat in its own console, another checkout) is left alone.
setlocal
set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
where py >nul 2>&1
if errorlevel 1 (
  set "PY=python"
) else (
  set "PY=py -3"
)
%PY% "%ROOT%\webui\launcher.py" stop %*
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
