@echo off
setlocal enabledelayedexpansion

rem Downloads one pinned model artifact, verifies it, and only then promotes it to its final name.
rem
rem   download-model.bat <model>
rem
rem   qwen38-27b       Qwen3.8-27B, 19.0 GiB. The default 27B for every benchmark in this repository
rem                    and the one docs\config-calculator.html's "27b" rows are measured against. It
rem                    is the official v3 artifact with the DFlash2 bundle, and it carries the MTP
rem                    weights too, so one file serves both --spec mtp and --spec dflash2. Published
rem                    measurements were taken against the v2 pin 18dfc887, whose weight bytes the v3
rem                    container preserves.
rem   qwen36-27b       Qwen3.6-27B groupwise-int, 16.3 GiB, target_key qwen3_6_27b. A different model
rem                    family from qwen3_8_27b, which is why having the latter does not satisfy the
rem                    former: four real-model tests -- ninfer_qwen3_6_27b_prefix_real_test,
rem                    _score_real_test, _load_plan_test and the Qwen3.6 27B half of the engine suite
rem                    -- skip without it.
rem   qwen36-35b-a3b   Qwen3.6-35B-A3B, 21.2 GiB, the RTX 3090-compatible vision model. The official
rem                    v3 artifact revision: its artifact-manifest.json lists the text, vision, mtp
rem                    and dflash components, and the DFlash bundle (from
rem                    z-lab/Qwen3.6-35B-A3B-DFlash) is what --spec dflash needs.
rem
rem To repin a model, edit its block below: take the size and sha256 from what HuggingFace reports for
rem the new revision (X-Linked-Size and X-Linked-ETag on the resolve URL, or the revision's
rem artifact-manifest.json), and for qwen36-35b-a3b check the manifest still lists a dflash component.
rem Changing a pin makes the staging hazard below live rather than hypothetical.
rem
rem Run with no argument (for instance by double-clicking it) it shows a numbered menu of the three
rem models instead; an unknown name still prints the usage and exits 2. download-model.sh has no
rem menu and always requires the name.
rem
rem Environment: NINFER_MODEL_DIR (default: models\ at the repo root in a checkout, or beside this
rem file in the release archive), NINFER_SKIP_SHA256=1 to accept a file on size alone.
rem
rem Uses aria2c (16 parallel ranges) when it is on PATH, since single-stream curl against these
rem artifacts has been observed to throttle to ~1 MB/s or stall outright; falls back to curl
rem otherwise. Install it with `winget install aria2.aria2` or `choco install aria2` to opt in.

if "%~1"=="" goto :menu
if /i "%~1"=="qwen38-27b" goto :model_qwen38_27b
if /i "%~1"=="qwen36-27b" goto :model_qwen36_27b
if /i "%~1"=="qwen36-35b-a3b" goto :model_qwen36_35b_a3b
if /i "%~1"=="-h" goto :usage_help
if /i "%~1"=="--help" goto :usage_help
goto :usage_error

:menu
echo Which model do you want to download?
echo.
echo   1 = Qwen3.8-27B (recommended)       qwen38-27b
echo   2 = Qwen3.6-35B-A3B                 qwen36-35b-a3b
echo   3 = Qwen3.6-27B                     qwen36-27b
echo.
choice /c 123 /n /m "Choose 1-3: "
rem choice reports 255 when it cannot read a key (no console, closed stdin). Refuse rather than
rem let that fall through and pick a model nobody chose.
if errorlevel 255 exit /b 2
if errorlevel 3 goto :model_qwen36_27b
if errorlevel 2 goto :model_qwen36_35b_a3b
if errorlevel 1 goto :model_qwen38_27b
exit /b 2

:usage_help
call :usage
exit /b 0

:usage_error
call :usage
exit /b 2

:usage
echo usage: %~nx0 ^<model^>
echo.
echo   qwen38-27b (recommended)
echo   qwen36-27b
echo   qwen36-35b-a3b
exit /b 0

:model_qwen38_27b
set "ARTIFACT=qwen3_8_27b.ninfer"
set "REPO=Qwen3.8-27B-NInfer"
set "REVISION=1cbd84e7221e51186bd7f093a149912d2489625b"
set "EXPECTED_SIZE=20437521664"
set "EXPECTED_SHA256=81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da"
set "LABEL=the Qwen3.8-27B model (19.0 GiB)"
set "TESTS_VARIABLE=NINFER_QWEN3_8_27B_WEIGHTS"
goto :model_selected

:model_qwen36_27b
set "ARTIFACT=qwen3_6_27b.ninfer"
set "REPO=Qwen3.6-27B-NInfer"
set "REVISION=3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829"
set "EXPECTED_SIZE=17495538688"
set "EXPECTED_SHA256=9b610a7d051e7c4dbf89adb604bd269d248b643c8f6c7ef75bdb871af92c6f6b"
set "LABEL=the Qwen3.6-27B model (16.3 GiB)"
set "TESTS_VARIABLE=NINFER_QWEN3_6_27B_WEIGHTS"
goto :model_selected

:model_qwen36_35b_a3b
set "ARTIFACT=qwen3_6_35b_a3b.ninfer"
set "REPO=Qwen3.6-35B-A3B-NInfer"
set "REVISION=ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6"
set "EXPECTED_SIZE=22790484480"
set "EXPECTED_SHA256=3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84"
set "LABEL=the RTX 3090-compatible Qwen3.6-35B-A3B vision model (21.2 GiB)"
set "TESTS_VARIABLE=NINFER_QWEN3_6_35B_A3B_WEIGHTS"
goto :model_selected

:model_selected
rem Two layouts share this script. In the release archive it sits at the archive root beside
rem models\. In a checkout it sits in scripts\, one level under the repo root, and models\ (see
rem .gitignore) is beside the repo root, not beside this script -- a directory named scripts with a
rem CMakeLists.txt above it tells the two apart (a bare CMakeLists.txt probe would misfire on an
rem archive unpacked beneath any source tree), matching run.bat's own default so a model
rem downloaded here is found there.
set "ROOT=%~dp0"
for %%I in ("%ROOT%..") do set "REPO_ROOT=%%~fI"
for %%I in ("%ROOT%.") do set "SCRIPT_DIRNAME=%%~nxI"
set "MODEL_DIR=%ROOT%models"
if /i "%SCRIPT_DIRNAME%"=="scripts" if exist "%REPO_ROOT%\CMakeLists.txt" set "MODEL_DIR=%REPO_ROOT%\models"
if defined NINFER_MODEL_DIR set "MODEL_DIR=%NINFER_MODEL_DIR%"
set "MODEL=%MODEL_DIR%\%ARTIFACT%"

rem Staged under a revision-scoped name so that curl -C - can only ever resume the same artifact.
rem curl -C - resumes by appending at the current file length, without checking what wrote those
rem bytes, so a leftover partial from a different revision -- for instance the previous pin's, on any
rem machine that ran an older script -- would be spliced onto the head of this one: a file of
rem entirely plausible size that is corrupt throughout. A name that carries the revision means a
rem resume can only ever continue the same artifact, and the checks below are what promote it to
rem the final name.
set "PART=%MODEL%.%REVISION%.part"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if exist "%MODEL%" (
  call :verify "%MODEL%"
  if "!VERIFY_OK!"=="1" (
    echo Model already present: %MODEL%
    exit /b 0
  )
  echo Existing %MODEL% did not verify against revision %REVISION%; fetching the pinned one.
  echo If it is the v2 file from an earlier release, you can upgrade it instead of downloading:
  echo   python tools\upgrade_ninfer_v2_to_v3.py "%MODEL%" "%MODEL%.v3" ^&^& move /y "%MODEL%.v3" "%MODEL%"
)

echo Downloading %LABEL%...
set "URL=https://huggingface.co/neroued/%REPO%/resolve/%REVISION%/%ARTIFACT%"
set "PART_NAME=%ARTIFACT%.%REVISION%.part"
where /q aria2c
if not errorlevel 1 (
  rem 16 parallel ranges over one file, not single-stream curl. On the Vast rental boxes this repo
  rem downloads at ~98 MB/s over aria2c against the same throttled-to-~1MB/s (and once outright
  rem stalled at 11 of 21 GiB) single-stream curl -- see scripts\multi-gpu-testing\README.md. -c
  rem resumes into the same .part name curl would use, and HuggingFace serves ranges fine.
  aria2c -x16 -s16 -c --file-allocation=none -d "%MODEL_DIR%" -o "%PART_NAME%" "%URL%"
) else (
  curl.exe -L -C - --fail --output "%PART%" "%URL%"
)
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)

call :verify "%PART%"
if not "!VERIFY_OK!"=="1" (
  echo Downloaded file at "%PART%" failed verification against revision %REVISION%. Delete it and run this file again.
  exit /b 1
)

move /y "%PART%" "%MODEL%" >nul
if errorlevel 1 (
  echo Failed to move "%PART%" to "%MODEL%". Delete "%PART%" and run this file again.
  exit /b 1
)
echo Model ready: %MODEL%
echo Point the tests at it with:  set %TESTS_VARIABLE%=%MODEL%
exit /b 0

rem Verifies %1 against EXPECTED_SIZE and, unless NINFER_SKIP_SHA256=1, EXPECTED_SHA256, setting
rem VERIFY_OK to 1 or 0. Used both for an existing MODEL (so a same-sized-but-corrupt file is not
rem accepted forever just because it happened to pass once, or was replaced out from under this
rem script) and for a freshly downloaded PART -- one check that cannot drift out of sync with
rem itself. ACTUAL_SHA256 is cleared before the for /f loop below: setlocal inherits existing
rem environment variables, so a stale ACTUAL_SHA256 left over from outside this script would
rem otherwise survive "if not defined" and be compared unchanged.
:verify
set "VERIFY_OK=0"
set "VERIFY_PATH=%~1"
for %%A in ("%VERIFY_PATH%") do set "VERIFY_SIZE=%%~zA"
if not "!VERIFY_SIZE!"=="%EXPECTED_SIZE%" exit /b 0
if "%NINFER_SKIP_SHA256%"=="1" (
  set "VERIFY_OK=1"
  exit /b 0
)
set "ACTUAL_SHA256="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%VERIFY_PATH%" SHA256') do (
  if not defined ACTUAL_SHA256 set "ACTUAL_SHA256=%%H"
)
set "ACTUAL_SHA256=!ACTUAL_SHA256: =!"
if /i "!ACTUAL_SHA256!"=="%EXPECTED_SHA256%" set "VERIFY_OK=1"
exit /b 0
