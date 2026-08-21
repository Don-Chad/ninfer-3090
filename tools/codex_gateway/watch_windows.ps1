# Keeps the owned Huihui Codex gateway available on the loopback interface.
# The gateway itself owns the lazy NInfer child through a kill-on-close Job Object.

$ErrorActionPreference = 'Continue'

$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = 'C:\Program Files\Python313\pythonw.exe'
$logDirectory = (Resolve-Path -LiteralPath (Join-Path $repository '..\logs')).Path
$watchLog = Join-Path $logDirectory 'huihui-gateway-watchdog.log'
$stdoutLog = Join-Path $logDirectory 'huihui-gateway.out.log'
$stderrLog = Join-Path $logDirectory 'huihui-gateway.err.log'
$port = 8081
$model = 'huihui-qwen3.8-27b-abliterated'

$mutex = [Threading.Mutex]::new($false, 'Local\HuihuiCodexGatewayWatchdog')
if (-not $mutex.WaitOne(0)) {
    $mutex.Dispose()
    exit 0
}

try {
    Add-Content -LiteralPath $watchLog -Value "$(Get-Date -Format o) watchdog started"
    while ($true) {
        $healthy = $false
        try {
            $status = Invoke-RestMethod -Uri "http://127.0.0.1:$port/gateway/health" `
                -TimeoutSec 2
            $healthy = $status.status -eq 'ok' -and $status.model -eq $model
        }
        catch {
            $healthy = $false
        }

        if (-not $healthy) {
            $listener = Get-NetTCPConnection -State Listen -LocalPort $port `
                -ErrorAction SilentlyContinue
            if ($listener) {
                Add-Content -LiteralPath $watchLog -Value `
                    "$(Get-Date -Format o) port $port occupied by non-ready pid=$($listener.OwningProcess -join ',')"
            }
            else {
                try {
                    $env:HUIHUI_GATEWAY_PORT = "$port"
                    $env:HUIHUI_IDLE_TIMEOUT_SECONDS = '600'
                    $process = Start-Process -FilePath $python -ArgumentList @('-m', 'tools.codex_gateway') `
                        -WorkingDirectory $repository -WindowStyle Hidden -PassThru `
                        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
                    Add-Content -LiteralPath $watchLog -Value `
                        "$(Get-Date -Format o) started gateway pid=$($process.Id)"
                }
                catch {
                    Add-Content -LiteralPath $watchLog -Value `
                        "$(Get-Date -Format o) start failed: $($_.Exception.Message)"
                }
            }
        }
        Start-Sleep -Seconds 5
    }
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
