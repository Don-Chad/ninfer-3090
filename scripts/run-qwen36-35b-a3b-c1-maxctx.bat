@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Qwen3.6-35B-A3B on one RTX 3090 -- single user. Runs the Ninja build directly out of
rem build-ninja\apps, not a packaged release.
rem
rem READ THIS FIRST: the maximum context on this machine is not a fixed number. The engine sizes
rem its runtime reservation from whatever VRAM is free after the weights land, and Windows' own
rem GPU usage moved by 650-900 MB during a single afternoon of measuring -- explorer, SearchHost,
rem ShellHost and CrossDeviceResume come and go. That is worth 40,000-50,000 tokens of context.
rem Two consecutive probes of the same configuration reported 1,938 MB and 1,078 MB available.
rem
rem The defaults below are therefore sized with margin, not at the cliff. The ladder shows what
rem was measured under a busy desktop (~21.9 GiB free) and what the same profile reached under a
rem quiet one (~22.6 GiB free). If a value fails to start, drop one rung.
rem
rem   profile                    default   free @busy   reached @quiet   decode
rem   ----------------------------------------------------------------------------
rem   A  MTP3 + draft head        81,920      193 MiB          131,072   ~240 tok/s
rem   B  no speculation          196,608      375 MiB          262,144   ~183 tok/s
rem
rem WHAT SPECULATION COSTS IN MEMORY, exact, read from the artifact directory:
rem
rem   MTP head            856 MiB   (--spec mtp)
rem   draft head          136 MiB   (--lm-head-draft, requires MTP)
rem   both                992 MiB
rem
rem   converted to context:        int8 (10,560 B/tok)   rk8v4 (7,969 B/tok)
rem   draft head alone                   13,504 tokens         17,896 tokens
rem   MTP head alone                     85,003 tokens        112,646 tokens
rem   both                               98,507 tokens        130,542 tokens
rem
rem So speculation does not buy context, it spends it: turning MTP off is worth roughly half the
rem native 262,144 back. The draft head is the cheap half of that pair -- 136 MiB for a measured
rem +1.9% to +15.6% decode, against the MTP head's 856 MiB for +38%.
rem
rem MEASURED MAX CONTEXT, all four combinations, taken under the quiet-desktop condition:
rem
rem   KV       speculation        max context   free after startup
rem   ------------------------------------------------------------
rem   rk8v4    none                   262,144         ~256 MiB      native maximum
rem   rk8v4    MTP3 + draft head      131,072         ~184 MiB
rem   int8     none                   196,608         ~344 MiB
rem   int8     MTP3 + draft head       94,208         ~292 MiB
rem
rem rk8v4 is worth +33% context unspeculated and +39% with speculation, for +0.082% perplexity.
rem int8 cannot reach the native 262,144 at all -- 204,800 already over-runs the reservation.
rem
rem HOW MUCH MTP IS WORTH, ninfer_bench tg512, rk8v4, 8K context:
rem
rem   Qwen3.6-35B-A3B (MoE, ~3B active)   no spec 183.49   MTP3 253.50 (1.38x)   +draft 280.38
rem   Qwen3.8-27B     (dense)             no spec  39.67   MTP3  63.21 (1.59x)
rem
rem Speculation helps this MoE *less* than the dense 27B -- 1.38x against 1.59x -- despite higher
rem acceptance (66.1% against 54.6%). That is structural: speculation pays by amortising one weight
rem read across several accepted tokens, and in a dense model every token reads the same weights.
rem Here each drafted token routes to its own 8 of 256 experts, so verifying four touches far more
rem expert weight than verifying one. The 35B also starts far less memory-bound, 183 against 40
rem tok/s, because only ~3B of 35B parameters are active per token.
rem
rem --lm-head-draft drafts over 131,072 rows instead of the full 248,320 output head: cheaper to
rem draft, but it cannot propose anything outside that subset. Whether that pays is entirely
rem content-dependent and the synthetic benchmark corpus is a bad guide -- on the dense 27B it
rem reverses the sign. Re-measured on this model with real content, rk8v4, MTP3, greedy:
rem
rem   content                        without    with     delta   acceptance without -> with
rem   ---------------------------------------------------------------------------------------
rem   ninfer_bench tg512 (synthetic)  253.50   280.38   +10.6%     66.15% -> 66.02%
rem   pure code generation            320.32   326.32    +1.9%     89.81% -> 83.82%
rem   mixed prose + code              188.08   217.42   +15.6%     39.09% -> 41.12%
rem
rem It helps on every content type here, by very different margins, and on mixed content it raises
rem acceptance rather than costing any. Note the spread in absolute terms: 320 tok/s on pure code
rem against 188 on mixed prose, because code is far more predictable. Any single decode figure for
rem this model is really a statement about the text being generated.
rem
rem Vision is off in both profiles: the vision tower is another 261 MiB and there is no room.
rem ---------------------------------------------------------------------------------------------

set "MODEL=C:\Ninefer-3090\models\qwen3_6_35b_a3b.ninfer"

rem Profile A (active): speculation on. Rungs: 81920 / 90112 / 98304 / 114688 / 131072.
rem set "CONTEXT=81920"
set "CONTEXT=114688"
rem Profile B: comment out the line above, uncomment this, and swap the commands at the bottom.
rem Rungs: 196608 / 212992 / 229376 / 245760 / 262144.
rem set "CONTEXT=196608"

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

rem --- Profile A: speculation on. ~240 tok/s decode. ---
"%SERVER%" "%MODEL%" ^
  --host %HOST% --port %PORT% ^
  --max-concurrency 1 ^
  --max-context %CONTEXT% ^
  --kv-capacity %CONTEXT% ^
  --kv-dtype rk8v4 ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --prefill-chunk 512 ^
  --max-pending-requests 16 ^
  --max-private-continuations 8 --max-shared-prefixes 4 --host-state-slots 16 --host-kv-mib 8192

rem --- Profile B: maximum context, no speculation. ~183 tok/s decode. ---
rem "%SERVER%" "%MODEL%" ^
rem   --host %HOST% --port %PORT% ^
rem   --max-concurrency 1 ^
rem   --max-context %CONTEXT% ^
rem   --kv-capacity %CONTEXT% ^
rem   --kv-dtype rk8v4 ^
rem   --prefill-chunk 512 ^
rem   --max-pending-requests 16

endlocal
