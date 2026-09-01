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
rem --lm-head-draft is worth including. It drafts over 131,072 rows instead of the full 248,320
rem output head: cheaper to draft, but it cannot propose anything outside that subset, so it trades
rem acceptance for speed. Whether that trade pays is entirely content-dependent, and the synthetic
rem benchmark corpus is a poor guide -- on the dense 27B it reverses the sign outright. Re-measured
rem on this model with real content, rk8v4, MTP3, greedy:
rem
rem   content                        without    with     delta   acceptance without -> with
rem   ---------------------------------------------------------------------------------------
rem   ninfer_bench tg512 (synthetic)  253.50   280.38   +10.6%     66.15% -> 66.02%
rem   pure code generation            320.32   326.32    +1.9%     89.81% -> 83.82%
rem   mixed prose + code              188.08   217.42   +15.6%     39.09% -> 41.12%
rem
rem So it helps on every content type here, though by very different margins, and on mixed content
rem it actually *raises* acceptance rather than costing any. Note how far apart the content types
rem are in absolute terms: 320 tok/s on pure code against 188 on mixed prose, because code is much
rem more predictable and speculation gets far more out of it. Any single decode figure for this
rem model is really a statement about the text being generated.
rem
rem For contrast, the same flag on the dense Qwen3.8-27B loses 7.1% on the synthetic corpus but
rem gains 2.6% on code and 6.6% on mixed -- see run-qwen38-c1.bat. Benchmark-corpus numbers for
rem this flag should not be trusted on either model.
rem
rem WHY rk8v4 KV: 7,969 bytes/token against int8's ~10,560, for +0.082% perplexity. Measured max
rem context on this machine, both KV profiles, C1:
rem
rem   KV       speculation        max context   free after startup
rem   ------------------------------------------------------------
rem   rk8v4    none                   262,144         ~256 MiB       <- profile B, native maximum
rem   rk8v4    MTP3 + draft head      131,072         ~184 MiB       <- profile A
rem   int8     none                   196,608         ~344 MiB
rem   int8     MTP3 + draft head       94,208         ~292 MiB
rem
rem rk8v4 is worth +33% context unspeculated and +39% with speculation. int8 cannot reach the
rem model's native 262,144 at all: 204,800 already asks for more runtime reservation than exists.
rem Switch by changing --kv-dtype below and setting CONTEXT to the matching row.
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
