$ErrorActionPreference = "Stop"

$Repo = "G:\python\custom-kernel-3090\ninfer-mtp300"
$Output = Join-Path $Repo "benchmark-results\prompt-lookup-delayed"
$Bench = Join-Path $Repo "build-mtp300\bench\Release\ninfer_bench.exe"
$Weights = Join-Path $Repo "models\qwen3_6_35b_a3b.ninfer"
$Corpus = Join-Path $Repo "benchmark-results\prompt-lookup-code\python_code_edit_repetition.ids"
$Test = Join-Path $Repo "build-mtp300\tests\Release\ninfer_prompt_lookup_test.exe"
$MinimumFreeMiB = 22500
$MaximumGpuWaitMinutes = 120
$ProcessTimeoutSeconds = 180

New-Item -ItemType Directory -Path $Output -Force | Out-Null
$Log = Join-Path $Output "scheduled-run.log"

function Write-RunLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $Log -Value $line
}

function Invoke-Limited {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [hashtable]$Environment = @{}
    )
    foreach ($key in $Environment.Keys) {
        [Environment]::SetEnvironmentVariable($key, $Environment[$key], "Process")
    }
    try {
        $startParameters = @{
            FilePath = $Executable
            WorkingDirectory = $Repo
            WindowStyle = "Hidden"
            PassThru = $true
        }
        if ($Arguments.Count -gt 0) {
            $startParameters.ArgumentList = $Arguments
        }
        $process = Start-Process @startParameters
        if (-not $process.WaitForExit($ProcessTimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -Force
            throw "process timed out after $ProcessTimeoutSeconds seconds: $Executable"
        }
        if ($process.ExitCode -ne 0) {
            throw "process exited $($process.ExitCode): $Executable"
        }
    }
    finally {
        foreach ($key in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable($key, $null, "Process")
        }
    }
}

Write-RunLog "scheduled prompt-lookup run started"
$deadline = (Get-Date).AddMinutes($MaximumGpuWaitMinutes)
while ((Get-Date) -lt $deadline) {
    $query = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    $freeMiB = [int](($query | Select-Object -First 1).Trim())
    if ($freeMiB -ge $MinimumFreeMiB) {
        Write-RunLog "GPU ready with $freeMiB MiB free"
        break
    }
    Write-RunLog "waiting for GPU: $freeMiB MiB free"
    Start-Sleep -Seconds 60
}
if ($freeMiB -lt $MinimumFreeMiB) {
    throw "GPU did not reach $MinimumFreeMiB MiB free within $MaximumGpuWaitMinutes minutes"
}

Invoke-Limited -Executable $Test -Arguments @()

$common = @(
    "--weights", $Weights,
    "--corpus", $Corpus,
    "-pg", "128,128",
    "-r", "5",
    "--warmup", "2",
    "--max-ctx", "384",
    "--prefill-chunk", "128",
    "--kv-dtype", "int8",
    "--text-only",
    "--output", "json"
)

Invoke-Limited -Executable $Bench -Arguments (
    $common + @("--output-file", (Join-Path $Output "ordinary-k0.json"))
)
Invoke-Limited -Executable $Bench -Arguments (
    $common + @(
        "--mtp-draft-tokens", "3",
        "--lm-head-draft",
        "--output-file", (Join-Path $Output "mtp-k3.json")
    )
)
Invoke-Limited -Executable $Bench -Arguments (
    $common + @(
        "--mtp-draft-tokens", "3",
        "--lm-head-draft",
        "--output-file", (Join-Path $Output "lookup-k3-min4.json")
    )
) -Environment @{"NINFER_PROMPT_LOOKUP_MIN_MATCH" = "4"}

Write-RunLog "scheduled prompt-lookup run completed"
