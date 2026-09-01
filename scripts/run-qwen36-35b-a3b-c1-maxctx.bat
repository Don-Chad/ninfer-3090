@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Qwen3.6-35B-A3B on one RTX 3090 -- single user. Runs the Ninja build directly out of
rem build-ninja\apps, not a packaged release.
rem
rem Two profiles below. Both were measured on this machine; pick by editing which command at the
rem bottom is commented out. The trade is almost exactly "half the context for 1.5x the decode".
rem
rem   profile                         context    decode    free VRAM after startup
rem   ---------------------------------------------------------------------------
rem   A  MTP3 + draft head (default)  131,072   240 tok/s          ~228 MiB
rem   B  no speculation               262,144   183 tok/s          ~256 MiB
rem
rem WHY THE CONTEXT HALVES: the MTP head is 856 MiB of resident weights and the optimized draft
rem head another 136 MiB. At rk8v4's 7,969 bytes per token that is about 124,000 tokens of KV, so
rem turning speculation on costs very close to half the 262,144 native context. 163,840 does not
rem fit -- it asks for 1.69 GiB of runtime reservation against 1.56 GiB available.
rem
rem HOW MUCH MTP IS ACTUALLY WORTH HERE, measured with ninfer_bench tg512, rk8v4, 8K context:
rem
rem   Qwen3.6-35B-A3B (MoE, ~3B active)      no spec 183.49  MTP3 253.50 (1.38x)  +draft 280.38 (1.53x)
rem   Qwen3.8-27B     (dense)                no spec  39.67  MTP3  63.21 (1.59x)
rem
rem So speculation helps this MoE *less* than it helps the dense 27B -- 1.38x against 1.59x --
rem even though its acceptance rate is higher (66.1% against 54.6%). That is the MoE structure,
rem not a tuning problem: speculation pays by amortising one weight read across several accepted
rem tokens, and in a dense model every token reads the same weights. Here each drafted token routes
rem to its own 8 of 256 experts, so verifying four tokens can touch far more expert weight than
rem verifying one, and the amortisation is weaker. The 35B is also already fast unspeculated
rem (183 vs 40 tok/s) because only ~3B of 35B parameters are active per token, which leaves
rem speculation less memory wall to hide.
rem
rem --lm-head-draft is worth including: it took 253.50 to 280.38 tok/s, +10.6%, while acceptance
rem did not move (0.6615 -> 0.6602). It is not drafting better, it is drafting more cheaply -- a
rem 131,072-row draft head instead of the full 248,320-row output head.
rem
rem WHY rk8v4 KV: 7,969 bytes/token against int8's ~10,560, roughly 32% more context for +0.082%
rem perplexity. Neither profile above fits on int8.
rem
rem VRAM headroom is thin in both profiles because the request was for the largest context that
rem fits. The binding constraint is whatever else Windows puts on the GPU. Each 32,768 tokens is
rem worth about 261 MiB, so if the server fails to start or dies when you open something
rem GPU-hungry, drop CONTEXT by 32768 and try again.
rem
rem Vision is off in both: the vision tower is another 261 MiB and there is no room for it.
rem ---------------------------------------------------------------------------------------------

set "MODEL=C:\Ninefer-3090\models\qwen3_6_35b_a3b.ninfer"

rem Profile A (active): speculation on, 131,072 context.
set "CONTEXT=131072"
rem Profile B: comment out the line above and uncomment this one, then swap the commands below.
rem set "CONTEXT=262144"

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

echo Qwen3.6-35B-A3B  ^|  C1  ^|  context %CONTEXT%  ^|  rk8v4 KV  ^|  MTP3 + draft head
echo API: http://%HOST%:%PORT%/v1
echo.

rem --- Profile A: speculation on. 131,072 context, ~240 tok/s decode. ---
"%SERVER%" "%MODEL%" ^
  --host %HOST% --port %PORT% ^
  --max-concurrency 1 ^
  --max-context %CONTEXT% ^
  --kv-capacity %CONTEXT% ^
  --kv-dtype rk8v4 ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --prefill-chunk 512 ^
  --max-pending-requests 16

rem --- Profile B: maximum context. 262,144 tokens, ~183 tok/s decode. Set CONTEXT=262144 above. ---
rem "%SERVER%" "%MODEL%" ^
rem   --host %HOST% --port %PORT% ^
rem   --max-concurrency 1 ^
rem   --max-context %CONTEXT% ^
rem   --kv-capacity %CONTEXT% ^
rem   --kv-dtype rk8v4 ^
rem   --prefill-chunk 512 ^
rem   --max-pending-requests 16

endlocal
