$ErrorActionPreference = 'Stop'

# Qwen3.8-27B C1/64K defaults for one 24 GB RTX 3090.
$ModelRepository = 'neroued/Qwen3.8-27B-NInfer'
$ModelRevision = '3526913004b1cf552cb57b88d6a5c6f5e4a89a70'
$ModelFilename = 'qwen3_8_27b.ninfer'
$ExpectedSha256 = 'eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e'
$HostAddress = '127.0.0.1'
$Port = 8080
$ContextTokens = 65536
$PrefillChunkTokens = 512
$DraftTokens = 3

$ScriptRoot = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptRoot
$IsPackagedRelease = Test-Path -LiteralPath (Join-Path $ScriptRoot 'ninfer-serve.exe')
$InstallRoot = if ($IsPackagedRelease) { $ScriptRoot } else { $RepoRoot }
$ModelDirectory = Join-Path $InstallRoot 'models'
$ModelPath = Join-Path $ModelDirectory $ModelFilename
$PartialPath = "$ModelPath.partial"
$DownloadUrl = "https://huggingface.co/$ModelRepository/resolve/$ModelRevision/$ModelFilename"

$ServerCandidates = @(
    (Join-Path $InstallRoot 'ninfer-serve.exe'),
    (Join-Path $RepoRoot 'build-sm86-qwen38\apps\Release\ninfer-serve.exe'),
    (Join-Path $RepoRoot 'build-windows\apps\Release\ninfer-serve.exe')
)
$ServerPath = $ServerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $ServerPath) {
    throw "ninfer-serve.exe was not found. Extract the Windows release fully, or build the server first."
}

New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null

$ModelIsValid = $false
if (Test-Path -LiteralPath $ModelPath) {
    Write-Host 'Checking the existing Qwen3.8-27B artifact...'
    $ActualSha256 = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $ModelIsValid = $ActualSha256 -eq $ExpectedSha256
    if (-not $ModelIsValid) {
        throw "The existing model has the wrong SHA-256. Move or delete '$ModelPath', then run this launcher again."
    }
}

if (-not $ModelIsValid) {
    Write-Host 'Downloading the pinned Qwen3.8-27B NInfer artifact (about 16.96 GB)...'
    Write-Host 'An interrupted download is kept and resumed the next time you run this command.'
    & curl.exe --location --fail --retry 5 --retry-delay 5 --continue-at - --output $PartialPath $DownloadUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed with curl exit code $LASTEXITCODE. The partial file was preserved."
    }

    Write-Host 'Verifying the model download...'
    $ActualSha256 = (Get-FileHash -LiteralPath $PartialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $ExpectedSha256) {
        throw "Downloaded model SHA-256 mismatch. Expected $ExpectedSha256 but received $ActualSha256."
    }
    Move-Item -LiteralPath $PartialPath -Destination $ModelPath
}

Write-Host "Starting Qwen3.8-27B at http://${HostAddress}:$Port/v1"
Write-Host 'Profile: C1, 64K shared context, INT8 KV, MTP3, CUDA Graphs'
Write-Host 'Press Ctrl+C to stop the server.'

& $ServerPath $ModelPath `
    --host $HostAddress --port $Port `
    --max-context $ContextTokens --kv-capacity $ContextTokens --max-concurrency 1 `
    --prefill-chunk $PrefillChunkTokens --kv-dtype int8 `
    --spec mtp --draft-tokens $DraftTokens --lm-head-draft
exit $LASTEXITCODE
