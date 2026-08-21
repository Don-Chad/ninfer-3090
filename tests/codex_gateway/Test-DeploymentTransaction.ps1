[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:Install = (Resolve-Path (Join-Path $PSScriptRoot '..\..\tools\codex_gateway\Install-HuihuiCodexGateway.ps1')).Path
$script:Uninstall = (Resolve-Path (Join-Path $PSScriptRoot '..\..\tools\codex_gateway\Uninstall-HuihuiCodexGateway.ps1')).Path
$script:Root = Join-Path ([IO.Path]::GetTempPath()) ('huihui-deployment-' + [guid]::NewGuid().ToString('N'))
$global:HuihuiTaskEnabled = $true
$global:HuihuiTaskPresent = $true
$global:HuihuiWatcherAlive = $false
$global:HuihuiGatewayAlive = $false
$global:HuihuiStoppedWatcher = $false
$global:HuihuiTestSid = 'S-1-5-21-111-222-333-1001'

function Assert-True([bool]$Condition,[string]$Message) { if(-not$Condition){throw "ASSERT: $Message"} }
function Assert-Throws([scriptblock]$Action,[string]$Pattern) {
    try{&$Action;throw 'Expected failure did not occur.'}catch{if($_.Exception.Message-notmatch$Pattern){throw "Wrong failure: $($_.Exception.Message)"}}
}
function Get-Paths([string]$Root){
    return [ordered]@{Codex=Join-Path $Root 'user\.codex';Config=Join-Path $Root 'user\.codex\config.toml';Catalog=Join-Path $Root 'user\.codex\model-catalogs\huihui-combined.json';State=Join-Path $Root 'user\.codex\huihui-gateway-deployment-state.json';Startup=Join-Path $Root 'Startup\Huihui-Codex-Gateway.cmd';Shim=Join-Path $Root 'Startup\NInfer-Codex-Shim.cmd';Disabled=Join-Path $Root 'Startup\NInfer-Codex-Shim.cmd.disabled-huihui-gateway';DatedDisabled=Join-Path $Root 'Startup\NInfer-Codex-Shim.cmd.disabled-20260821';Backup=Join-Path $Root 'user\.codex\config.toml.backup-huihui-gateway-install';DatedBackup=Join-Path $Root 'user\.codex\config.toml.backup-20260821-huihui-gateway';TaskBackup=Join-Path $Root 'user\.codex\NInfer-Huihui-Candidate-Server.pre-gateway.xml';Engine=Join-Path $Root 'repo\build-huihui-sm86-vcpkg\apps\ninfer-serve.exe';Artifact=Join-Path $Root 'models\huihui_qwen3_8_27b_abliterated.ninfer';Watcher=Join-Path $Root 'repo\tools\codex_gateway\watch_windows.ps1';Python=Join-Path $Root 'Python313\pythonw.exe'}
}
function New-TaskXml([string]$Root,[bool]$Enabled){
    $p=Get-Paths $Root;$requestLog=Join-Path $Root 'logs\huihui-candidate.requests.jsonl';$arguments=@($p.Artifact,'--host','127.0.0.1','--port','8090','--max-context','65536','--kv-capacity','auto','--max-concurrency','1','--max-pending-requests','16','--prefill-chunk','1024','--kv-dtype','rk8v4','--spec','mtp','--draft-tokens','3','--lm-head-draft','--vision','--no-cuda-graph','--request-log-jsonl',$requestLog)-join' '
    $enabledText=if($Enabled){'true'}else{'false'}
    return "<Task><Principals><Principal><UserId>$global:HuihuiTestSid</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals><Settings><Enabled>$enabledText</Enabled></Settings><Actions><Exec><Command>$($p.Engine)</Command><Arguments>$arguments</Arguments><WorkingDirectory>$(Split-Path $p.Engine -Parent)</WorkingDirectory></Exec></Actions></Task>"
}
function New-Fixture([string]$Root){
    if(Test-Path -LiteralPath $Root){Remove-Item -LiteralPath $Root -Recurse -Force}
    $p=Get-Paths $Root
    foreach($dir in $p.Codex,(Split-Path $p.Catalog -Parent),(Split-Path $p.Engine -Parent),(Split-Path $p.Watcher -Parent),(Split-Path $p.Artifact -Parent),(Split-Path $p.Python -Parent),(Split-Path $p.Startup -Parent)){New-Item -ItemType Directory -Path $dir -Force|Out-Null}
    [IO.File]::WriteAllText($p.Config,"model = `"baseline`"`r`n[features]`r`ntool_search = true`r`n",[Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($p.Catalog,'{"models":[{"slug":"huihui-qwen3.8-27b-abliterated","input_modalities":["text","image"],"supports_image_detail_original":true,"supports_search_tool":true}]}',[Text.UTF8Encoding]::new($false))
    foreach($file in $p.Engine,$p.Artifact,$p.Watcher,$p.Python,$p.Shim){[IO.File]::WriteAllText($file,"fixture-$file",[Text.UTF8Encoding]::new($false))}
    $global:HuihuiTaskEnabled=$true;$global:HuihuiTaskPresent=$true;$global:HuihuiWatcherAlive=$false;$global:HuihuiGatewayAlive=$false;$global:HuihuiStoppedWatcher=$false
}

function global:Get-ScheduledTask { param([string]$TaskName,$ErrorAction);if($global:HuihuiTaskPresent){return [pscustomobject]@{Settings=[pscustomobject]@{Enabled=[bool]$global:HuihuiTaskEnabled}}};return $null }
function global:Export-ScheduledTask { param([string]$TaskName);return New-TaskXml $env:HUIHUI_DEPLOYMENT_TEST_ROOT $global:HuihuiTaskEnabled }
function global:Disable-ScheduledTask { param([string]$TaskName);$global:HuihuiTaskEnabled=$false;return [pscustomobject]@{} }
function global:Enable-ScheduledTask { param([string]$TaskName);$global:HuihuiTaskEnabled=$true;return [pscustomobject]@{} }
function global:Register-ScheduledTask { param([string]$TaskName,[string]$Xml,[switch]$Force);$global:HuihuiTaskPresent=$true;[xml]$doc=$Xml;$global:HuihuiTaskEnabled=([string]$doc.Task.Settings.Enabled)-eq'true';return [pscustomobject]@{} }
function global:Get-NetTCPConnection { param($State,[int]$LocalPort,$ErrorAction);if($LocalPort-eq8081-and$global:HuihuiGatewayAlive){return [pscustomobject]@{OwningProcess=200}};return @() }
function global:Get-CimInstance { param([string]$ClassName,[string]$Filter,$ErrorAction)
    if($Filter-match'ProcessId = 200' -and $global:HuihuiGatewayAlive){$p=Get-Paths $env:HUIHUI_DEPLOYMENT_TEST_ROOT;return [pscustomobject]@{ProcessId=200;ParentProcessId=100;ExecutablePath=$p.Python;CommandLine="$($p.Python) -m tools.codex_gateway"}}
    if($Filter-match'ProcessId = 100' -and $global:HuihuiWatcherAlive){return [pscustomobject]@{ProcessId=100;ExecutablePath='C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe';CommandLine="powershell.exe -File `"$((Get-Paths $env:HUIHUI_DEPLOYMENT_TEST_ROOT).Watcher)`""}}
    if($Filter-match"Name = 'powershell.exe'" -and $global:HuihuiWatcherAlive){return Get-CimInstance -ClassName Win32_Process -Filter 'ProcessId = 100'}
    return $null
}
function global:Invoke-RestMethod { return [pscustomobject]@{status='ok';model='huihui-qwen3.8-27b-abliterated';state='stopped'} }
function global:Stop-Process { param([int]$Id,[switch]$Force);if($Id-eq100){$global:HuihuiWatcherAlive=$false;$global:HuihuiStoppedWatcher=$true};if($Id-eq200){$global:HuihuiGatewayAlive=$false} }
function global:Wait-Process { param([int]$Id,[int]$Timeout,$ErrorAction) }
function global:Start-Sleep { param([int]$Milliseconds,[int]$Seconds) }

function Invoke-Install([string]$Root,[hashtable]$Extra=@{}){$env:HUIHUI_DEPLOYMENT_TEST_ROOT=$Root;$env:HUIHUI_DEPLOYMENT_TEST_MODE='1';$env:HUIHUI_DEPLOYMENT_TEST_SID=$global:HuihuiTestSid;&$script:Install -Confirm:$false @Extra}
function Invoke-Uninstall([string]$Root,[hashtable]$Extra=@{}){$env:HUIHUI_DEPLOYMENT_TEST_ROOT=$Root;$env:HUIHUI_DEPLOYMENT_TEST_MODE='1';$env:HUIHUI_DEPLOYMENT_TEST_SID=$global:HuihuiTestSid;&$script:Uninstall -Confirm:$false @Extra}
function Initialize-AdoptionFixture([string]$Root,[byte[]]$PreConfig,[byte[]]$ManagedConfig,[byte[]]$StartupBytes){
    New-Fixture $Root;$paths=Get-Paths $Root
    [IO.File]::WriteAllBytes($paths.Config,$ManagedConfig);[IO.File]::WriteAllBytes($paths.DatedBackup,$PreConfig);[IO.File]::WriteAllBytes($paths.Startup,$StartupBytes)
    Move-Item $paths.Shim $paths.DatedDisabled;$global:HuihuiTaskEnabled=$false
}

try{
    $p=Get-Paths $script:Root
    New-Fixture $script:Root;$original=[IO.File]::ReadAllBytes($p.Config)
    Invoke-Install $script:Root;Assert-True(Test-Path $p.State)'fresh state';Assert-True(-not$global:HuihuiTaskEnabled)'task disabled';Invoke-Install $script:Root
    Invoke-Uninstall $script:Root;Assert-True(-not(Test-Path $p.State))'state removed';Assert-True($global:HuihuiTaskEnabled)'task restored';Assert-True(-not(Test-Path $p.Backup))'owned config backup cleaned';Assert-True(-not(Test-Path $p.TaskBackup))'owned task backup cleaned';Assert-True(([Convert]::ToBase64String([IO.File]::ReadAllBytes($p.Config)))-eq([Convert]::ToBase64String($original)))'config restored'
    Invoke-Install $script:Root;Invoke-Uninstall $script:Root
    Write-Host 'PASS fresh install -> uninstall -> reinstall'

    New-Fixture $script:Root;$pre=[IO.File]::ReadAllBytes($p.Config);Invoke-Install $script:Root;$managed=[IO.File]::ReadAllBytes($p.Config);$startup=[IO.File]::ReadAllBytes($p.Startup);Invoke-Uninstall $script:Root
    [IO.File]::WriteAllBytes($p.Config,$managed);[IO.File]::WriteAllBytes($p.DatedBackup,$pre);[IO.File]::WriteAllBytes($p.Startup,$startup);Move-Item $p.Shim $p.DatedDisabled;$global:HuihuiTaskEnabled=$false
    Invoke-Install $script:Root @{AdoptExistingDeployment=$true;PreGatewayConfigBackupPath=$p.DatedBackup;ExistingDisabledShimPath=$p.DatedDisabled;LegacyTaskPreStagingEnabled=$true}
    Invoke-Uninstall $script:Root;Assert-True(Test-Path $p.DatedBackup)'adopted dated backup preserved';Assert-True($global:HuihuiTaskEnabled)'adopted task enabled restored'
    Write-Host 'PASS adoption -> uninstall'

    foreach($fault in 1..8){New-Fixture $script:Root;$env:HUIHUI_DEPLOYMENT_FAULT_AFTER="$fault";try{Invoke-Install $script:Root}catch{};Remove-Item Env:HUIHUI_DEPLOYMENT_FAULT_AFTER -ErrorAction SilentlyContinue;Invoke-Install $script:Root;Assert-True((Get-Content $p.State -Raw|ConvertFrom-Json).phase-eq'installed')"install resume $fault";Invoke-Uninstall $script:Root}
    Write-Host 'PASS install interruption after every mutation -> resume'

    foreach($fault in 1..9){New-Fixture $script:Root;Invoke-Install $script:Root;$env:HUIHUI_DEPLOYMENT_FAULT_AFTER="$fault";try{Invoke-Uninstall $script:Root}catch{};Remove-Item Env:HUIHUI_DEPLOYMENT_FAULT_AFTER -ErrorAction SilentlyContinue;if(Test-Path $p.State){Invoke-Uninstall $script:Root};Assert-True(-not(Test-Path $p.State))"uninstall resume $fault"}
    Write-Host 'PASS uninstall interruption after every mutation -> resume'

    foreach($fault in 1..9){
        Initialize-AdoptionFixture $script:Root $pre $managed $startup
        Invoke-Install $script:Root @{AdoptExistingDeployment=$true;PreGatewayConfigBackupPath=$p.DatedBackup;ExistingDisabledShimPath=$p.DatedDisabled;LegacyTaskPreStagingEnabled=$true}
        $env:HUIHUI_DEPLOYMENT_FAULT_AFTER="$fault";try{Invoke-Uninstall $script:Root}catch{};Remove-Item Env:HUIHUI_DEPLOYMENT_FAULT_AFTER -ErrorAction SilentlyContinue
        if(Test-Path $p.State){Invoke-Uninstall $script:Root}
        Assert-True(-not(Test-Path $p.State))"adoption uninstall resume $fault";Assert-True($global:HuihuiTaskEnabled)"adoption task restored $fault";Assert-True(Test-Path $p.DatedBackup)"adoption backup preserved $fault"
    }
    Write-Host 'PASS adoption uninstall interruption after every mutation -> resume'

    New-Fixture $script:Root;Invoke-Install $script:Root;$global:HuihuiWatcherAlive=$true;Invoke-Uninstall $script:Root;Assert-True($global:HuihuiStoppedWatcher)'watcher-only process stopped'
    Write-Host 'PASS watcher alive + gateway absent stopped'

    New-Fixture $script:Root;Invoke-Install $script:Root;$state=Get-Content $p.State -Raw|ConvertFrom-Json;$state.paths.startup='C:\arbitrary.cmd';[IO.File]::WriteAllText($p.State,($state|ConvertTo-Json -Depth 8));Assert-Throws {Invoke-Install $script:Root}'non-canonical'
    New-Fixture $script:Root;Invoke-Install $script:Root;$state=Get-Content $p.State -Raw|ConvertFrom-Json;$state.schema='3';[IO.File]::WriteAllText($p.State,($state|ConvertTo-Json -Depth 8));Assert-Throws {Invoke-Install $script:Root}'integer 3'
    New-Fixture $script:Root;Invoke-Install $script:Root;$state=Get-Content $p.State -Raw|ConvertFrom-Json;$state.hashes.configBackup=('0'*64);[IO.File]::WriteAllText($p.State,($state|ConvertTo-Json -Depth 8));Assert-Throws {Invoke-Install $script:Root}'hashes differ'
    New-Fixture $script:Root;Invoke-Install $script:Root;Add-Content $p.Startup 'tamper';Assert-Throws {Invoke-Install $script:Root}'Startup'
    New-Fixture $script:Root;Invoke-Install $script:Root;Add-Content $p.Config 'user_drift = true';Assert-Throws {Invoke-Uninstall $script:Root}'Config';Invoke-Uninstall $script:Root @{ForceRestoreConfig=$true}
    New-Fixture $script:Root;(Get-Content $p.Catalog -Raw).Replace('"supports_search_tool":true','"supports_search_tool":false')|Set-Content $p.Catalog -NoNewline;Assert-Throws {Invoke-Install $script:Root}'Catalog'
    New-Fixture $script:Root;$global:HuihuiTaskEnabled=$true;function global:Export-ScheduledTask{return (New-TaskXml $env:HUIHUI_DEPLOYMENT_TEST_ROOT $global:HuihuiTaskEnabled).Replace('</Actions>','<Exec><Command>evil</Command></Exec></Actions>')};Assert-Throws {Invoke-Install $script:Root}'exactly one Exec'
    Remove-Item function:global:Export-ScheduledTask;function global:Export-ScheduledTask { param([string]$TaskName);return New-TaskXml $env:HUIHUI_DEPLOYMENT_TEST_ROOT $global:HuihuiTaskEnabled }
    New-Fixture $script:Root;$target="$($p.Watcher).target";Move-Item $p.Watcher $target;New-Item -ItemType SymbolicLink -Path $p.Watcher -Target $target|Out-Null;Assert-Throws {Invoke-Install $script:Root}'reparse point'
    Write-Host 'PASS tampered path/hash/type/reparse/Startup/catalog/XML failures'

    New-Fixture $script:Root;$before=(Get-ChildItem $script:Root -Recurse|ForEach-Object FullName)-join"`n";Invoke-Install $script:Root @{WhatIf=$true};$after=(Get-ChildItem $script:Root -Recurse|ForEach-Object FullName)-join"`n";Assert-True($before-eq$after)'WhatIf zero writes'
    Write-Host 'PASS WhatIf zero writes'
}finally{
    Remove-Item Env:HUIHUI_DEPLOYMENT_TEST_MODE,Env:HUIHUI_DEPLOYMENT_TEST_ROOT,Env:HUIHUI_DEPLOYMENT_TEST_SID,Env:HUIHUI_DEPLOYMENT_FAULT_AFTER -ErrorAction SilentlyContinue
    if(Test-Path $script:Root){Remove-Item -LiteralPath $script:Root -Recurse -Force}
}
