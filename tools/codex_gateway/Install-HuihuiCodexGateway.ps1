[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High', DefaultParameterSetName = 'Fresh')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Adopt')][switch]$AdoptExistingDeployment,
    [Parameter(Mandatory = $true, ParameterSetName = 'Adopt')][string]$PreGatewayConfigBackupPath,
    [Parameter(Mandatory = $true, ParameterSetName = 'Adopt')][string]$ExistingDisabledShimPath,
    [Parameter(Mandatory = $true, ParameterSetName = 'Adopt')][switch]$LegacyTaskPreStagingEnabled,
    [Parameter(DontShow = $true, ParameterSetName = 'Rollback')][switch]$InternalUninstall,
    [Parameter(DontShow = $true, ParameterSetName = 'Rollback')][switch]$ForceRestoreConfig
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:ModelSlug = 'huihui-qwen3.8-27b-abliterated'
$script:LegacyTaskName = 'NInfer-Huihui-Candidate-Server'
$script:LegacyShimName = 'NInfer-Codex-Shim.cmd'
$script:TestMode = $env:HUIHUI_DEPLOYMENT_TEST_MODE -eq '1'
$script:MutationNumber = 0

if ($script:TestMode) {
    if ([string]::IsNullOrWhiteSpace($env:HUIHUI_DEPLOYMENT_TEST_ROOT)) { throw 'Test mode requires HUIHUI_DEPLOYMENT_TEST_ROOT.' }
    $testRoot = [IO.Path]::GetFullPath($env:HUIHUI_DEPLOYMENT_TEST_ROOT)
    $repository = Join-Path $testRoot 'repo'
    $workstationRoot = $testRoot
    $codexHome = Join-Path $testRoot 'user\.codex'
    $startupDirectory = Join-Path $testRoot 'Startup'
    $pythonPath = Join-Path $testRoot 'Python313\pythonw.exe'
}
else {
    $repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
    $workstationRoot = (Resolve-Path -LiteralPath (Join-Path $repository '..')).Path
    $codexHome = Join-Path $env:USERPROFILE '.codex'
    $startupDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::Startup)
    $pythonPath = 'C:\Program Files\Python313\pythonw.exe'
}

$script:Paths = [ordered]@{
    Repository=[IO.Path]::GetFullPath($repository); CodexHome=[IO.Path]::GetFullPath($codexHome)
    Config=[IO.Path]::GetFullPath((Join-Path $codexHome 'config.toml'))
    Catalog=[IO.Path]::GetFullPath((Join-Path $codexHome 'model-catalogs\huihui-combined.json'))
    State=[IO.Path]::GetFullPath((Join-Path $codexHome 'huihui-gateway-deployment-state.json'))
    FreshConfigBackup=[IO.Path]::GetFullPath((Join-Path $codexHome 'config.toml.backup-huihui-gateway-install'))
    AdoptedConfigBackup=[IO.Path]::GetFullPath((Join-Path $codexHome 'config.toml.backup-20260821-huihui-gateway'))
    TaskBackup=[IO.Path]::GetFullPath((Join-Path $codexHome 'NInfer-Huihui-Candidate-Server.pre-gateway.xml'))
    StartupDirectory=[IO.Path]::GetFullPath($startupDirectory)
    Startup=[IO.Path]::GetFullPath((Join-Path $startupDirectory 'Huihui-Codex-Gateway.cmd'))
    LegacyShim=[IO.Path]::GetFullPath((Join-Path $startupDirectory $script:LegacyShimName))
    FreshDisabledShim=[IO.Path]::GetFullPath((Join-Path $startupDirectory "$($script:LegacyShimName).disabled-huihui-gateway"))
    AdoptedDisabledShim=[IO.Path]::GetFullPath((Join-Path $startupDirectory "$($script:LegacyShimName).disabled-20260821"))
    Watcher=[IO.Path]::GetFullPath((Join-Path $repository 'tools\codex_gateway\watch_windows.ps1'))
    Python=[IO.Path]::GetFullPath($pythonPath)
    Engine=[IO.Path]::GetFullPath((Join-Path $repository 'build-huihui-sm86-vcpkg\apps\ninfer-serve.exe'))
    Artifact=[IO.Path]::GetFullPath((Join-Path $workstationRoot 'models\huihui_qwen3_8_27b_abliterated.ninfer'))
}
$script:Utf8NoBom = [Text.UTF8Encoding]::new($false)
if (-not ('HuihuiDeployment.NativeFile' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
namespace HuihuiDeployment {
    public static class NativeFile {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool MoveFileEx(string source, string destination, int flags);
        public static void AtomicReplace(string source, string destination) {
            if (!MoveFileEx(source, destination, 0x1 | 0x8))
                throw new Win32Exception(Marshal.GetLastWin32Error());
        }
    }
}
'@
}
$script:StartupBytes = [Text.Encoding]::ASCII.GetBytes("@echo off`nstart `"`" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$($script:Paths.Watcher)`"`n")
$script:TaskWorkingDirectory = Split-Path -Parent $script:Paths.Engine
$script:TaskRequestLog = [IO.Path]::GetFullPath((Join-Path $workstationRoot 'logs\huihui-candidate.requests.jsonl'))
$script:TaskArgumentTokens = @(
    $script:Paths.Artifact,'--host','127.0.0.1','--port','8090','--max-context','65536','--kv-capacity','auto',
    '--max-concurrency','1','--max-pending-requests','16','--prefill-chunk','1024','--kv-dtype','rk8v4',
    '--spec','mtp','--draft-tokens','3','--lm-head-draft','--vision','--no-cuda-graph','--request-log-jsonl',$script:TaskRequestLog
)

function Invoke-FaultBoundary([string]$Name) {
    if (-not $script:TestMode) { return }
    $script:MutationNumber++
    if ($env:HUIHUI_DEPLOYMENT_TRACE_FILE) { Add-Content -LiteralPath $env:HUIHUI_DEPLOYMENT_TRACE_FILE -Value "$($script:MutationNumber):$Name" }
    $after = 0
    if ([int]::TryParse($env:HUIHUI_DEPLOYMENT_FAULT_AFTER, [ref]$after) -and $after -eq $script:MutationNumber) {
        throw "Injected interruption after mutation $after ($Name)."
    }
}

function Test-ScalarType($Value,[Type]$Type) { return $null -ne $Value -and $Value.GetType() -eq $Type }
function Assert-Properties($Object,[string[]]$Expected,[string]$Label) {
    if ($null -eq $Object -or $Object -isnot [pscustomobject]) { throw "$Label must be a JSON object." }
    if ((@($Object.PSObject.Properties.Name|Sort-Object)-join "`n") -ne (@($Expected|Sort-Object)-join "`n")) { throw "$Label has missing or unexpected properties." }
}
function Assert-StringProperty($Object,[string]$Name,[string]$Label) {
    if (-not (Test-ScalarType $Object.$Name ([string])) -or [string]::IsNullOrWhiteSpace($Object.$Name)) { throw "$Label.$Name must be a non-empty JSON string." }
}
function Assert-BoolProperty($Object,[string]$Name,[string]$Label) {
    if (-not (Test-ScalarType $Object.$Name ([bool]))) { throw "$Label.$Name must be a JSON boolean." }
}

function Assert-NoReparsePath([string]$Path,[switch]$AllowMissingLeaf) {
    $full=[IO.Path]::GetFullPath($Path); $probe=$full
    if (-not (Test-Path -LiteralPath $probe)) {
        if (-not $AllowMissingLeaf) { throw "Required path is missing: $full" }
        $probe=Split-Path -Parent $probe
    }
    while (-not [string]::IsNullOrEmpty($probe)) {
        if (Test-Path -LiteralPath $probe) {
            $item=Get-Item -LiteralPath $probe -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or $null -ne $item.LinkType) { throw "Managed path traverses a reparse point: $probe" }
        }
        $parent=Split-Path -Parent $probe; if($parent -eq $probe){break}; $probe=$parent
    }
}
function Require-File([string]$Path,[string]$Label) {
    Assert-NoReparsePath $Path
    if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){throw "$Label is missing: $Path"}
}
function Get-BytesHash([byte[]]$Bytes) {
    $sha=[Security.Cryptography.SHA256]::Create(); try{return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-','')}finally{$sha.Dispose()}
}
function Get-FileSha([string]$Path) {
    Require-File $Path 'Hashed file'
    $stream=[IO.File]::OpenRead($Path);$sha=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-','')}finally{$sha.Dispose();$stream.Dispose()}
}
function Test-FileHash([string]$Path,[string]$Hash) { return (Test-Path -LiteralPath $Path -PathType Leaf) -and (Get-FileSha $Path) -ceq $Hash }
function Write-AtomicBytes([string]$Path,[byte[]]$Bytes) {
    Assert-NoReparsePath $Path -AllowMissingLeaf
    $parent=Split-Path -Parent $Path
    if(-not(Test-Path -LiteralPath $parent -PathType Container)){throw "Destination directory is missing: $parent"}
    $temporary=Join-Path $parent ('.huihui-tmp-'+[guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllBytes($temporary,$Bytes)
        [HuihuiDeployment.NativeFile]::AtomicReplace($temporary,$Path)
    } finally {
        if(Test-Path -LiteralPath $temporary){Remove-Item -LiteralPath $temporary -Force}
    }
}
function Write-State($State,[switch]$Prepared) {
    Write-AtomicBytes $script:Paths.State $script:Utf8NoBom.GetBytes(($State|ConvertTo-Json -Depth 8))
    if($Prepared){Invoke-FaultBoundary 'prepared-journal'}
}
function Set-Step($State,[string]$Name,[string]$Value='done') { $State.steps.$Name=$Value; Write-State $State }
function Set-Phase($State,[string]$Phase,[string]$FaultName) { $State.phase=$Phase; Write-State $State; Invoke-FaultBoundary $FaultName }
function Get-CurrentSid {
    if($script:TestMode -and $env:HUIHUI_DEPLOYMENT_TEST_SID){return $env:HUIHUI_DEPLOYMENT_TEST_SID}
    return [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
}

function Get-TaskEnabledFromXml([string]$TaskXml) {
    [xml]$document=$TaskXml
    $nodes=@($document.SelectNodes("/*[local-name()='Task']/*[local-name()='Settings']/*[local-name()='Enabled']"))
    if($nodes.Count -ne 1 -or @('true','false') -cnotcontains $nodes[0].InnerText.ToLowerInvariant()){throw 'Task XML needs one boolean Settings/Enabled.'}
    return $nodes[0].InnerText.ToLowerInvariant() -ceq 'true'
}
function Get-TaskIdentityHash([bool]$Enabled) {
    $identity=@(
        $script:Paths.Engine
        ($script:TaskArgumentTokens -join "`n")
        $script:TaskWorkingDirectory
        (Get-CurrentSid)
        'InteractiveToken'
        'LeastPrivilege'
        $Enabled.ToString().ToLowerInvariant()
    ) -join "`n"
    return Get-BytesHash $script:Utf8NoBom.GetBytes($identity)
}
function Assert-LegacyCandidateTaskXml([string]$TaskXml) {
    try{[xml]$document=$TaskXml}catch{throw 'Legacy task XML is not well formed.'}
    $actions=@($document.SelectNodes("/*[local-name()='Task']/*[local-name()='Actions']"))
    if($actions.Count-ne 1 -or $actions[0].ChildNodes.Count-ne 1 -or $actions[0].ChildNodes[0].LocalName-cne 'Exec'){throw 'Task needs exactly one Exec action.'}
    $exec=$actions[0].ChildNodes[0]; $commands=@($exec.SelectNodes("*[local-name()='Command']")); $arguments=@($exec.SelectNodes("*[local-name()='Arguments']"));$working=@($exec.SelectNodes("*[local-name()='WorkingDirectory']"))
    if($exec.ChildNodes.Count-ne 3 -or $commands.Count-ne 1 -or $arguments.Count-ne 1 -or $working.Count-ne 1 -or $commands[0].InnerText-cne $script:Paths.Engine -or $working[0].InnerText-cne $script:TaskWorkingDirectory){throw 'Task Exec is not exact.'}
    if($arguments[0].InnerText -match '["'']'){throw 'Task arguments may not contain quoting or embedded payloads.'}
    $tokens=@($arguments[0].InnerText.Trim()-split '[ \t]+'|Where-Object{$_-ne''})
    if(($tokens-join"`n")-cne($script:TaskArgumentTokens-join"`n")){throw 'Task arguments are not the exact Huihui 8090 command.'}
    $principals=@($document.SelectNodes("/*[local-name()='Task']/*[local-name()='Principals']/*[local-name()='Principal']"))
    if($principals.Count-ne 1){throw 'Task needs exactly one principal.'}
    $uid=@($principals[0].SelectNodes("*[local-name()='UserId']")); $logon=@($principals[0].SelectNodes("*[local-name()='LogonType']")); $level=@($principals[0].SelectNodes("*[local-name()='RunLevel']"))
    if($uid.Count-ne 1 -or $uid[0].InnerText-cne(Get-CurrentSid) -or $logon.Count-ne 1 -or $logon[0].InnerText-cne'InteractiveToken' -or $level.Count-gt 1 -or ($level.Count-eq1 -and $level[0].InnerText-cne'LeastPrivilege')){throw 'Task principal is not exact or has a disallowed run level.'}
    [void](Get-TaskEnabledFromXml $TaskXml)
}

function Assert-Catalog([string]$Path) {
    Require-File $Path 'Combined catalog'
    try{$catalog=Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json}catch{throw 'Combined catalog is invalid JSON.'}
    if($catalog-isnot[pscustomobject] -or $catalog.models-isnot[array]){throw 'Catalog models must be an array.'}
    $candidate=@($catalog.models|Where-Object{$_-is[pscustomobject] -and $_.slug-ceq$script:ModelSlug})
    if($candidate.Count-ne 1){throw 'Catalog must contain exactly one Huihui entry.'}; $entry=$candidate[0]
    foreach($property in 'slug','input_modalities','supports_image_detail_original','supports_search_tool'){if($entry.PSObject.Properties.Name-notcontains$property){throw "Catalog candidate missing $property."}}
    if(-not(Test-ScalarType $entry.slug ([string])) -or $entry.input_modalities-isnot[array] -or $entry.input_modalities.Count-ne 2 -or
       @($entry.input_modalities|Where-Object{$_-isnot[string]}).Count-ne 0 -or $entry.input_modalities-notcontains'text' -or $entry.input_modalities-notcontains'image' -or
       -not(Test-ScalarType $entry.supports_image_detail_original ([bool])) -or -not$entry.supports_image_detail_original -or
       -not(Test-ScalarType $entry.supports_search_tool ([bool])) -or -not$entry.supports_search_tool){throw 'Catalog modalities/original-detail/search support are not exact.'}
}

function Get-TomlLines([string]$Text){return @($Text -split "`r?`n")}
function Get-TomlValueWithoutComment([string]$Value){
    $inside=$false;$escaped=$false
    for($i=0;$i-lt$Value.Length;$i++){$c=$Value[$i];if($escaped){$escaped=$false;continue};if($inside-and$c-eq'\'){$escaped=$true;continue};if($c-eq'"'){$inside=-not$inside;continue};if(-not$inside-and$c-eq'#'){return $Value.Substring(0,$i).Trim()}}
    return $Value.Trim()
}
function Assert-ManagedConfigText([string]$Text){
    $table='';$base=0;$catalog=0;$feature=0;$featureTables=0
    foreach($line in (Get-TomlLines $Text)){
        if($line-match'^\s*\[([^\[\]]+)\]\s*(?:#.*)?$'){$table=$matches[1].Trim();if($table-ceq'features'){$featureTables++};continue}
        if($line-match'^\s*([A-Za-z0-9_.-]+)\s*=\s*(.*)$'){$key=$matches[1];$value=Get-TomlValueWithoutComment $matches[2]
            if($table-eq''-and$key-ceq'openai_base_url'){$base++;if($value-cne'"http://127.0.0.1:8081/v1"'){throw 'Top-level gateway URL is not exact.'}}
            if($table-eq''-and$key-ceq'model_catalog_json'){$catalog++;$expected='"'+$script:Paths.Catalog.Replace('\','\\')+'"';if($value-cne$expected){throw 'Top-level catalog path is not exact.'}}
            if($table-ceq'features'-and$key-ceq'remote_compaction_v2'){$feature++;if($value-cne'false'){throw 'remote_compaction_v2 must be false.'}}
            if($table-eq''-and$key-ceq'features.remote_compaction_v2'){throw 'Dotted root compaction key is forbidden.'}
        }
    }
    if($base-ne1-or$catalog-ne1-or$featureTables-ne1-or$feature-ne1){throw "Managed config semantic keys are missing or duplicated (base=$base catalog=$catalog featureTables=$featureTables feature=$feature)."}
}
function Assert-AdoptionConfigText([string]$Text){
    $table='';$base=0;$catalog=0
    foreach($line in (Get-TomlLines $Text)){
        if($line-match'^\s*\[([^\[\]]+)\]\s*(?:#.*)?$'){$table=$matches[1].Trim();continue}
        if($line-match'^\s*([A-Za-z0-9_.-]+)\s*=\s*(.*)$'){$key=$matches[1];$value=Get-TomlValueWithoutComment $matches[2]
            if($table-eq''-and$key-ceq'openai_base_url'){$base++;if($value-cne'"http://127.0.0.1:8081/v1"'){throw 'Adoption gateway URL is not exact.'}}
            if($table-eq''-and$key-ceq'model_catalog_json'){$catalog++;$expected='"'+$script:Paths.Catalog.Replace('\','\\')+'"';if($value-cne$expected){throw 'Adoption catalog path is not exact.'}}
        }
    }
    if($base-ne1-or$catalog-ne1){throw 'Adoption needs one exact semantic top-level gateway and catalog key.'}
}
function New-ManagedConfigText([string]$Text){
    $result=[Collections.Generic.List[string]]::new();$table='';$features=0
    foreach($line in (Get-TomlLines $Text)){
        if($line-match'^\s*\[([^\[\]]+)\]\s*(?:#.*)?$'){$table=$matches[1].Trim();if($table-ceq'features'){$features++;if($features-gt1){throw 'Multiple [features] tables.'};$result.Add($line);$result.Add('remote_compaction_v2 = false');continue}}
        if($line-match'^\s*([A-Za-z0-9_.-]+)\s*='){$key=$matches[1];if(($table-eq''-and@('openai_base_url','model_catalog_json','features.remote_compaction_v2')-ccontains$key)-or($table-ceq'features'-and$key-ceq'remote_compaction_v2')){continue}}
        $result.Add($line)
    }
    if($features-eq0){if($result.Count-gt0-and$result[$result.Count-1]-ne''){$result.Add('')};$result.Add('[features]');$result.Add('remote_compaction_v2 = false')}
    $literal=$script:Paths.Catalog.Replace('\','\\')
    $output=(@('openai_base_url = "http://127.0.0.1:8081/v1"',"model_catalog_json = `"$literal`"",'')+$result.ToArray()-join"`r`n").TrimEnd("`r","`n")+"`r`n"
    Assert-ManagedConfigText $output;return $output
}
function Assert-StartupExact([string]$Path){Require-File $Path 'Startup entry';if((Get-BytesHash([IO.File]::ReadAllBytes($Path)))-cne(Get-BytesHash $script:StartupBytes)){throw 'Startup bytes are not exact.'}}

function New-StepObject{return [ordered]@{
    configBackup='pending';configInstalled='pending';startupInstalled='pending';taskBackup='pending';taskDisabled='pending';shimDisabled='pending'
    runtimeStopped='pending';startupRemoved='pending';taskRestored='pending';shimRestored='pending';configRestored='pending';taskBackupRemoved='pending';configBackupRemoved='pending'
}}
function Assert-StepState($Steps){
    $names=@('configBackup','configInstalled','startupInstalled','taskBackup','taskDisabled','shimDisabled','runtimeStopped','startupRemoved','taskRestored','shimRestored','configRestored','taskBackupRemoved','configBackupRemoved')
    Assert-Properties $Steps $names 'state.steps'
    foreach($name in $names){Assert-StringProperty $Steps $name 'state.steps';if(@('pending','done','notApplicable')-cnotcontains$Steps.$name){throw "Invalid step state: $name"}}
}
function Assert-State($State){
    Assert-Properties $State @('schema','deploymentId','mode','phase','paths','hashes','ownership','legacy','steps') 'state'
    if(($State.schema-isnot[int]-and$State.schema-isnot[long])-or$State.schema-ne3){throw 'State schema must be integer 3.'}
    foreach($name in 'deploymentId','mode','phase'){Assert-StringProperty $State $name 'state'}
    try{$id=[guid]$State.deploymentId}catch{throw 'deploymentId is not a GUID.'};if($id-eq[guid]::Empty){throw 'deploymentId is empty.'}
    if(@('fresh','adopt')-cnotcontains$State.mode-or@('prepared','installing','installed','uninstalling')-cnotcontains$State.phase){throw 'State mode/phase is invalid.'}
    Assert-Properties $State.paths @('config','configBackup','catalog','state','taskBackup','startup','legacyShim','disabledShim','watcher','python','engine','artifact') 'state.paths'
    $backup=if($State.mode-ceq'adopt'){$script:Paths.AdoptedConfigBackup}else{$script:Paths.FreshConfigBackup}
    $disabled=if($State.mode-ceq'adopt'){$script:Paths.AdoptedDisabledShim}else{$script:Paths.FreshDisabledShim}
    $expected=[ordered]@{config=$script:Paths.Config;configBackup=$backup;catalog=$script:Paths.Catalog;state=$script:Paths.State;taskBackup=$script:Paths.TaskBackup;startup=$script:Paths.Startup;legacyShim=$script:Paths.LegacyShim;disabledShim=$disabled;watcher=$script:Paths.Watcher;python=$script:Paths.Python;engine=$script:Paths.Engine;artifact=$script:Paths.Artifact}
    foreach($name in $expected.Keys){Assert-StringProperty $State.paths $name 'state.paths';if($State.paths.$name-cne$expected[$name]-or[IO.Path]::GetFullPath($State.paths.$name)-cne$expected[$name]){throw "Non-canonical state path: $name"};Assert-NoReparsePath $State.paths.$name -AllowMissingLeaf}
    Assert-Properties $State.hashes @('configBefore','configAtPrepare','configAfter','configBackup','startup','taskXml','taskBackup','taskRestored','shim','disabledShim','catalog','watcher') 'state.hashes'
    foreach($name in $State.hashes.PSObject.Properties.Name){Assert-StringProperty $State.hashes $name 'state.hashes';if($State.hashes.$name-notmatch'^(?:[0-9A-F]{64}|NONE)$'){throw "Invalid state hash: $name"}}
    Assert-Properties $State.ownership @('state','configBackup','taskBackup','startup') 'state.ownership';foreach($name in $State.ownership.PSObject.Properties.Name){Assert-BoolProperty $State.ownership $name 'state.ownership'}
    Assert-Properties $State.legacy @('taskWasPresent','taskWasEnabled','shimWasPresent') 'state.legacy';foreach($name in $State.legacy.PSObject.Properties.Name){Assert-BoolProperty $State.legacy $name 'state.legacy'}
    if(-not$State.ownership.state-or-not$State.ownership.startup-or$State.ownership.configBackup-ne($State.mode-ceq'fresh')-or$State.ownership.taskBackup-ne$State.legacy.taskWasPresent){throw 'Ownership fields are inconsistent.'}
    if($State.hashes.configBefore-cne$State.hashes.configBackup){throw 'Config pre/backup hashes differ.'}
    if($State.legacy.taskWasPresent){if($State.hashes.taskXml-cne$State.hashes.taskBackup-or$State.hashes.taskRestored-cne(Get-TaskIdentityHash $State.legacy.taskWasEnabled)){throw 'Task hashes differ from their exact captured/restored identities.'}}elseif($State.hashes.taskXml-cne'NONE'-or$State.hashes.taskBackup-cne'NONE'-or$State.hashes.taskRestored-cne'NONE'){throw 'Absent task has hashes.'}
    if($State.legacy.shimWasPresent){if($State.hashes.shim-cne$State.hashes.disabledShim){throw 'Shim hashes differ.'}}elseif($State.hashes.shim-cne'NONE'-or$State.hashes.disabledShim-cne'NONE'){throw 'Absent shim has hashes.'}
    Assert-StepState $State.steps
}
function Read-State{Require-File $script:Paths.State 'Deployment state';try{$state=Get-Content -LiteralPath $script:Paths.State -Raw|ConvertFrom-Json}catch{throw 'State is invalid JSON.'};Assert-State $state;return $state}

function Get-TaskSnapshot{
    $task=Get-ScheduledTask -TaskName $script:LegacyTaskName -ErrorAction SilentlyContinue;if(-not$task){return $null}
    $xml=Export-ScheduledTask -TaskName $script:LegacyTaskName;if([string]::IsNullOrWhiteSpace($xml)){throw 'Task export is empty.'};Assert-LegacyCandidateTaskXml $xml
    $enabled=Get-TaskEnabledFromXml $xml;if($task.Settings.Enabled-isnot[bool]-or$task.Settings.Enabled-ne$enabled){throw 'Task enabled state disagrees with XML.'}
    return [pscustomobject]@{Xml=$xml;Enabled=$enabled;Hash=Get-BytesHash $script:Utf8NoBom.GetBytes($xml);IdentityHash=Get-TaskIdentityHash $enabled}
}
function Assert-BaseAssets{
    foreach($pair in @(@($script:Paths.Watcher,'Watcher'),@($script:Paths.Python,'Python'),@($script:Paths.Engine,'Engine'),@($script:Paths.Artifact,'Artifact'),@($script:Paths.Config,'Config'),@($script:Paths.Catalog,'Catalog'))){Require-File $pair[0] $pair[1]}
    foreach($path in $script:Paths.State,$script:Paths.Startup,$script:Paths.TaskBackup){Assert-NoReparsePath $path -AllowMissingLeaf}
    Assert-Catalog $script:Paths.Catalog
    foreach($command in 'Get-ScheduledTask','Export-ScheduledTask','Disable-ScheduledTask','Register-ScheduledTask','Enable-ScheduledTask'){if(-not(Get-Command $command -ErrorAction SilentlyContinue)){throw "ScheduledTasks command unavailable: $command"}}
}
function New-PreparedState{
    Assert-BaseAssets;if(Test-Path -LiteralPath $script:Paths.State){throw 'Deployment state already exists.'}
    $configBytes=[IO.File]::ReadAllBytes($script:Paths.Config);$configText=$script:Utf8NoBom.GetString($configBytes);$installedBytes=$script:Utf8NoBom.GetBytes((New-ManagedConfigText $configText));$task=Get-TaskSnapshot
    if($AdoptExistingDeployment){
        if(-not$LegacyTaskPreStagingEnabled.IsPresent){throw 'Adoption requires LegacyTaskPreStagingEnabled.'}
        if([IO.Path]::GetFullPath($PreGatewayConfigBackupPath)-cne$script:Paths.AdoptedConfigBackup-or[IO.Path]::GetFullPath($ExistingDisabledShimPath)-cne$script:Paths.AdoptedDisabledShim){throw 'Adoption paths are not canonical.'}
        Require-File $script:Paths.AdoptedConfigBackup 'Adopted config backup';Require-File $script:Paths.AdoptedDisabledShim 'Adopted shim';Assert-StartupExact $script:Paths.Startup
        if(Test-Path -LiteralPath $script:Paths.LegacyShim){throw 'Active shim exists during adoption.'};if(-not$task-or$task.Enabled){throw 'Adoption needs exact disabled task.'};Assert-AdoptionConfigText $configText
        $before=Get-FileSha $script:Paths.AdoptedConfigBackup;$atPrepare=Get-BytesHash $configBytes;$after=Get-BytesHash $installedBytes
        $mode='adopt';$backup=$script:Paths.AdoptedConfigBackup;$disabled=$script:Paths.AdoptedDisabledShim;$shimHash=Get-FileSha $disabled;$shimPresent=$true
    }else{
        foreach($collision in $script:Paths.FreshConfigBackup,$script:Paths.TaskBackup,$script:Paths.Startup,$script:Paths.FreshDisabledShim){if(Test-Path -LiteralPath $collision){throw "Fresh install collision: $collision"}}
        $before=Get-BytesHash $configBytes;$atPrepare=$before;$after=Get-BytesHash $installedBytes;$mode='fresh';$backup=$script:Paths.FreshConfigBackup;$disabled=$script:Paths.FreshDisabledShim
        $shimPresent=Test-Path -LiteralPath $script:Paths.LegacyShim -PathType Leaf;$shimHash=if($shimPresent){Get-FileSha $script:Paths.LegacyShim}else{'NONE'}
    }
    if($task-and(Test-Path -LiteralPath $script:Paths.TaskBackup)){throw 'Task backup collision.'}
    $state=[ordered]@{schema=3;deploymentId=[guid]::NewGuid().ToString('D');mode=$mode;phase='prepared'
        paths=[ordered]@{config=$script:Paths.Config;configBackup=$backup;catalog=$script:Paths.Catalog;state=$script:Paths.State;taskBackup=$script:Paths.TaskBackup;startup=$script:Paths.Startup;legacyShim=$script:Paths.LegacyShim;disabledShim=$disabled;watcher=$script:Paths.Watcher;python=$script:Paths.Python;engine=$script:Paths.Engine;artifact=$script:Paths.Artifact}
        hashes=[ordered]@{configBefore=$before;configAtPrepare=$atPrepare;configAfter=$after;configBackup=$before;startup=Get-BytesHash $script:StartupBytes;taskXml=if($task){$task.Hash}else{'NONE'};taskBackup=if($task){$task.Hash}else{'NONE'};taskRestored=if($task){Get-TaskIdentityHash $(if($mode-ceq'adopt'){$true}else{$task.Enabled})}else{'NONE'};shim=$shimHash;disabledShim=$shimHash;catalog=Get-FileSha $script:Paths.Catalog;watcher=Get-FileSha $script:Paths.Watcher}
        ownership=[ordered]@{state=$true;configBackup=($mode-ceq'fresh');taskBackup=($null-ne$task);startup=$true}
        legacy=[ordered]@{taskWasPresent=($null-ne$task);taskWasEnabled=if($mode-ceq'adopt'){$true}elseif($task){$task.Enabled}else{$false};shimWasPresent=$shimPresent};steps=New-StepObject}
    if(-not$task){foreach($name in 'taskBackup','taskDisabled','taskRestored','taskBackupRemoved'){$state.steps.$name='notApplicable'}};if(-not$shimPresent){foreach($name in 'shimDisabled','shimRestored'){$state.steps.$name='notApplicable'}}
    $roundTrip=$state|ConvertTo-Json -Depth 8|ConvertFrom-Json;Assert-State $roundTrip
    return [pscustomobject]@{State=$state;InstalledBytes=$installedBytes;Task=$task}
}

function Assert-StateAssets($State,[switch]$InstalledStrict,[switch]$AllowConfigDrift){
    Assert-BaseAssets
    if((Get-FileSha $State.paths.catalog)-cne$State.hashes.catalog-or(Get-FileSha $State.paths.watcher)-cne$State.hashes.watcher){throw 'Catalog or watcher changed.'};Assert-Catalog $State.paths.catalog
    $configHash=Get-FileSha $State.paths.config
    if($InstalledStrict){if($configHash-cne$State.hashes.configAfter){throw 'Config drifted.'};Assert-ManagedConfigText(Get-Content -LiteralPath $State.paths.config -Raw);if(-not(Test-FileHash $State.paths.configBackup $State.hashes.configBackup)){throw 'Config backup missing/tampered.'};Assert-StartupExact $State.paths.startup}
    elseif(-not $AllowConfigDrift -and @($State.hashes.configBefore,$State.hashes.configAtPrepare,$State.hashes.configAfter)-cnotcontains$configHash){throw 'Config is not a prepared pre/post image.'}
    if($State.legacy.taskWasPresent){
        $task=Get-TaskSnapshot
        if(-not$task){throw 'Legacy task disappeared.'}
        if($InstalledStrict-and$task.Enabled){throw 'Legacy task is enabled.'}
        if(Test-Path -LiteralPath $State.paths.taskBackup){
            if(-not(Test-FileHash $State.paths.taskBackup $State.hashes.taskBackup)){throw 'Task backup tampered.'}
            Assert-LegacyCandidateTaskXml(Get-Content -LiteralPath $State.paths.taskBackup -Raw)
        }
        elseif($InstalledStrict){throw 'Task backup missing.'}
        elseif($task.Hash-cne$State.hashes.taskXml -and -not($State.phase-ceq'uninstalling' -and $task.IdentityHash-ceq$State.hashes.taskRestored)){
            throw 'Legacy task is neither the captured XML nor the exact restored uninstall postimage.'
        }
    }
    if($State.legacy.shimWasPresent){$active=Test-FileHash $State.paths.legacyShim $State.hashes.shim;$off=Test-FileHash $State.paths.disabledShim $State.hashes.disabledShim;if($active-eq$off){throw 'Shim state is ambiguous.'};if($InstalledStrict -and -not $off){throw 'Shim is not disabled.'}}
}

function Complete-Install($State,[byte[]]$PreparedInstalledBytes,$PreparedTask,[switch]$AlreadyApproved){
    if($State.phase-ceq'uninstalling'){throw 'Rollback is in progress; resume uninstall.'}
    if($State.phase-ceq'installed'){Assert-StateAssets $State -InstalledStrict;Write-Host 'Existing Huihui gateway deployment validated; no changes were made.';return}
    Assert-StateAssets $State
    # Stage and hash-check every pending post-image before this invocation's
    # single transaction approval and before advancing the journal phase.
    if($State.steps.configInstalled-ceq'pending' -and $null-eq$PreparedInstalledBytes){
        $currentConfigHash=Get-FileSha $State.paths.config
        if($currentConfigHash-ceq$State.hashes.configAtPrepare){$PreparedInstalledBytes=$script:Utf8NoBom.GetBytes((New-ManagedConfigText(Get-Content -LiteralPath $State.paths.config -Raw)));if((Get-BytesHash $PreparedInstalledBytes)-cne$State.hashes.configAfter){throw 'Staged config post-image hash mismatch.'}}
    }
    if(-not$AlreadyApproved -and -not$PSCmdlet.ShouldProcess('Huihui Codex gateway deployment','resume the complete schema-3 installation transaction')){Write-Host 'WhatIf/declined: all preconditions validated; zero changes made.';return}
    if($State.phase-ceq'prepared'){$State.phase='installing';Write-State $State}
    if($State.steps.configBackup-ceq'pending'){
        if(-not(Test-FileHash $State.paths.configBackup $State.hashes.configBackup)){if((Get-FileSha $State.paths.config)-cne$State.hashes.configBefore){throw 'Config pre-image unavailable.'};Write-AtomicBytes $State.paths.configBackup ([IO.File]::ReadAllBytes($State.paths.config));Invoke-FaultBoundary 'config-backup'};Set-Step $State 'configBackup'
    }
    if($State.steps.configInstalled-ceq'pending'){
        $hash=Get-FileSha $State.paths.config
        if($hash-cne$State.hashes.configAfter){if($hash-cne$State.hashes.configAtPrepare){throw 'Config changed during install.'};if($null-eq$PreparedInstalledBytes){$PreparedInstalledBytes=$script:Utf8NoBom.GetBytes((New-ManagedConfigText(Get-Content -LiteralPath $State.paths.config -Raw)))};if((Get-BytesHash $PreparedInstalledBytes)-cne$State.hashes.configAfter){throw 'Prepared config hash mismatch.'};Write-AtomicBytes $State.paths.config $PreparedInstalledBytes;Invoke-FaultBoundary 'config-installed'}
        Assert-ManagedConfigText(Get-Content -LiteralPath $State.paths.config -Raw);Set-Step $State 'configInstalled'
    }
    if($State.steps.startupInstalled-ceq'pending'){
        if(-not(Test-FileHash $State.paths.startup $State.hashes.startup)){if(Test-Path -LiteralPath $State.paths.startup){throw 'Startup collision.'};Write-AtomicBytes $State.paths.startup $script:StartupBytes;Invoke-FaultBoundary 'startup-installed'};Assert-StartupExact $State.paths.startup;Set-Step $State 'startupInstalled'
    }
    if($State.steps.taskBackup-ceq'pending'){
        if(-not(Test-FileHash $State.paths.taskBackup $State.hashes.taskBackup)){$task=if($PreparedTask){$PreparedTask}else{Get-TaskSnapshot};if(-not$task-or$task.Hash-cne$State.hashes.taskXml){throw 'Task changed before backup.'};Write-AtomicBytes $State.paths.taskBackup $script:Utf8NoBom.GetBytes($task.Xml);Invoke-FaultBoundary 'task-backup'};Set-Step $State 'taskBackup'
    }
    if($State.steps.taskDisabled-ceq'pending'){
        $task=Get-TaskSnapshot;if(-not$task){throw 'Task disappeared before disable.'};if($task.Enabled){Disable-ScheduledTask -TaskName $script:LegacyTaskName|Out-Null;Invoke-FaultBoundary 'task-disabled'};if((Get-TaskSnapshot).Enabled){throw 'Task remained enabled.'};Set-Step $State 'taskDisabled'
    }
    if($State.steps.shimDisabled-ceq'pending'){
        $active=Test-FileHash $State.paths.legacyShim $State.hashes.shim;$off=Test-FileHash $State.paths.disabledShim $State.hashes.disabledShim
        if(-not $off){if(-not $active -or (Test-Path -LiteralPath $State.paths.disabledShim)){throw 'Shim cannot be disabled.'};Move-Item -LiteralPath $State.paths.legacyShim -Destination $State.paths.disabledShim;Invoke-FaultBoundary 'shim-disabled'}
        if((Test-Path -LiteralPath $State.paths.legacyShim) -or -not (Test-FileHash $State.paths.disabledShim $State.hashes.disabledShim)){throw 'Shim disable post-image failed.'};Set-Step $State 'shimDisabled'
    }
    Assert-StateAssets $State -InstalledStrict;Set-Phase $State 'installed' 'phase-installed'
    Write-Host 'Installed Huihui Codex gateway deployment. Restart Codex/app-server when its Remote session permits it.'
}

function Get-ListenerProcess([int]$Port){
    $listeners=@(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue);if($listeners.Count-gt1){throw "Multiple listeners on $Port."};if($listeners.Count-eq0){return $null}
    $process=Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $($listeners[0].OwningProcess)";if(-not$process){throw "Cannot resolve listener on $Port."};return $process
}
function Get-ManagedRuntime{
    $pattern='(?i)(?:^|\s)-File\s+(?:"'+[regex]::Escape($script:Paths.Watcher)+'"|'+[regex]::Escape($script:Paths.Watcher)+')(?=\s|$)'
    $watchers=@(Get-CimInstance -ClassName Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue|Where-Object{$null-ne$_ -and $_.PSObject.Properties.Name-contains'ExecutablePath' -and $_.PSObject.Properties.Name-contains'CommandLine' -and $_.ExecutablePath-match'(?i)\\powershell\.exe$' -and $_.CommandLine-match$pattern})
    if($watchers.Count-gt1){throw 'Multiple exact managed watchers.'};$gateway=Get-ListenerProcess 8081
    if($gateway){
        if($gateway.ExecutablePath-cne$script:Paths.Python-or$gateway.CommandLine-notmatch'(?i)(?:^|\s)-m\s+tools\.codex_gateway(?:\s|$)'){throw '8081 is not exact Python gateway.'}
        if($watchers.Count-ne1-or$gateway.ParentProcessId-ne$watchers[0].ProcessId){throw 'Gateway parent is not exact watcher.'}
        try{$health=Invoke-RestMethod -Uri 'http://127.0.0.1:8081/gateway/health' -TimeoutSec 2}catch{throw 'Gateway health identity unavailable.'}
        if($health.status-cne'ok'-or$health.model-cne$script:ModelSlug-or@('stopped','starting','running','stopping')-cnotcontains$health.state){throw 'Gateway health identity is wrong.'}
    }
    return [pscustomobject]@{Watcher=if($watchers.Count-eq1){$watchers[0]}else{$null};Gateway=$gateway}
}
function Test-ProcessExists([int]$Id){return $null-ne(Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $Id" -ErrorAction SilentlyContinue)}
function Stop-ExactRuntime($Runtime){
    if($Runtime.Watcher){Stop-Process -Id $Runtime.Watcher.ProcessId -Force;try{Wait-Process -Id $Runtime.Watcher.ProcessId -Timeout 10 -ErrorAction Stop}catch{};if(Test-ProcessExists $Runtime.Watcher.ProcessId){throw 'Watcher remained live.'}}
    if($Runtime.Gateway-and(Test-ProcessExists $Runtime.Gateway.ProcessId)){Stop-Process -Id $Runtime.Gateway.ProcessId -Force;try{Wait-Process -Id $Runtime.Gateway.ProcessId -Timeout 10 -ErrorAction Stop}catch{};if(Test-ProcessExists $Runtime.Gateway.ProcessId){throw 'Gateway remained live.'}}
    if(Get-ListenerProcess 8081){throw '8081 remained occupied.'};$deadline=[DateTime]::UtcNow.AddSeconds(10)
    while(Get-ListenerProcess 8080){if([DateTime]::UtcNow-ge$deadline){throw 'Job-owned 8080 child did not disappear.'};Start-Sleep -Milliseconds 250}
    Start-Sleep -Milliseconds 6500;if(Get-ListenerProcess 8081){throw '8081 returned within watchdog interval.'}
}
function Assert-UninstallPreconditions($State){
    Assert-StateAssets $State -AllowConfigDrift:$ForceRestoreConfig;$hash=Get-FileSha $State.paths.config
    if((@($State.hashes.configBefore,$State.hashes.configAtPrepare,$State.hashes.configAfter) -cnotcontains $hash) -and -not $ForceRestoreConfig){throw 'Config changed; use ForceRestoreConfig only after review.'}
    if($hash -cne $State.hashes.configBefore -and -not (Test-FileHash $State.paths.configBackup $State.hashes.configBackup)){throw 'Config backup unavailable.'}
    if((Test-Path -LiteralPath $State.paths.startup) -and -not (Test-FileHash $State.paths.startup $State.hashes.startup)){throw 'Startup tampered.'}
    if($State.legacy.taskWasPresent){if(-not (Get-TaskSnapshot)){throw 'Task missing.'};if((Test-Path -LiteralPath $State.paths.taskBackup) -and -not (Test-FileHash $State.paths.taskBackup $State.hashes.taskBackup)){throw 'Task backup tampered.'}}
    if($State.legacy.shimWasPresent){$active=Test-FileHash $State.paths.legacyShim $State.hashes.shim;$off=Test-FileHash $State.paths.disabledShim $State.hashes.disabledShim;if($active-eq$off){throw 'Shim restore state is ambiguous.'}}
    return Get-ManagedRuntime
}
function Complete-Uninstall($State){
    $runtime=Assert-UninstallPreconditions $State
    if(-not$PSCmdlet.ShouldProcess('Huihui Codex gateway deployment','resume the complete schema-3 rollback transaction')){Write-Host 'WhatIf/declined: rollback and runtime validated; zero changes made.';return}
    if($State.phase-cne'uninstalling'){Set-Phase $State 'uninstalling' 'uninstalling-journal'}
    if($State.steps.runtimeStopped-ceq'pending'){Stop-ExactRuntime $runtime;Invoke-FaultBoundary 'runtime-stopped';Set-Step $State 'runtimeStopped'}
    if($State.steps.startupRemoved-ceq'pending'){if(Test-Path -LiteralPath $State.paths.startup){if(-not(Test-FileHash $State.paths.startup $State.hashes.startup)){throw 'Startup changed.'};Remove-Item -LiteralPath $State.paths.startup;Invoke-FaultBoundary 'startup-removed'};Set-Step $State 'startupRemoved'}
    if($State.steps.taskRestored-ceq'pending'){
        $task=Get-TaskSnapshot;if(-not$task){throw 'Task disappeared.'}
        if($task.Enabled-ne$State.legacy.taskWasEnabled){if(Test-FileHash $State.paths.taskBackup $State.hashes.taskBackup){$xml=Get-Content -LiteralPath $State.paths.taskBackup -Raw;Assert-LegacyCandidateTaskXml $xml;Register-ScheduledTask -TaskName $script:LegacyTaskName -Xml $xml -Force|Out-Null};if($State.legacy.taskWasEnabled){Enable-ScheduledTask -TaskName $script:LegacyTaskName|Out-Null}else{Disable-ScheduledTask -TaskName $script:LegacyTaskName|Out-Null};Invoke-FaultBoundary 'task-restored'}
        if((Get-TaskSnapshot).Enabled-ne$State.legacy.taskWasEnabled){throw 'Task restore failed.'};Set-Step $State 'taskRestored'
    }
    if($State.steps.shimRestored-ceq'pending'){
        if(-not(Test-FileHash $State.paths.legacyShim $State.hashes.shim)){if(-not(Test-FileHash $State.paths.disabledShim $State.hashes.disabledShim)-or(Test-Path -LiteralPath $State.paths.legacyShim)){throw 'Shim cannot restore.'};Move-Item -LiteralPath $State.paths.disabledShim -Destination $State.paths.legacyShim;Invoke-FaultBoundary 'shim-restored'}
        if(-not(Test-FileHash $State.paths.legacyShim $State.hashes.shim)-or(Test-Path -LiteralPath $State.paths.disabledShim)){throw 'Shim restore post-image failed.'};Set-Step $State 'shimRestored'
    }
    if($State.steps.configRestored-ceq'pending'){
        if((Get-FileSha $State.paths.config)-cne$State.hashes.configBefore){if(-not(Test-FileHash $State.paths.configBackup $State.hashes.configBackup)){throw 'Config backup disappeared.'};Write-AtomicBytes $State.paths.config ([IO.File]::ReadAllBytes($State.paths.configBackup));Invoke-FaultBoundary 'config-restored'}
        if((Get-FileSha $State.paths.config)-cne$State.hashes.configBefore){throw 'Config restore failed.'};Set-Step $State 'configRestored'
    }
    if($State.steps.taskBackupRemoved-ceq'pending'){if($State.ownership.taskBackup-and(Test-Path -LiteralPath $State.paths.taskBackup)){if(-not(Test-FileHash $State.paths.taskBackup $State.hashes.taskBackup)){throw 'Owned task backup changed.'};Remove-Item -LiteralPath $State.paths.taskBackup;Invoke-FaultBoundary 'task-backup-removed'};Set-Step $State 'taskBackupRemoved'}
    if($State.steps.configBackupRemoved-ceq'pending'){
        if($State.ownership.configBackup-and(Test-Path -LiteralPath $State.paths.configBackup)){if(-not(Test-FileHash $State.paths.configBackup $State.hashes.configBackup)){throw 'Owned config backup changed.'};Remove-Item -LiteralPath $State.paths.configBackup;Invoke-FaultBoundary 'config-backup-removed'}
        elseif(-not$State.ownership.configBackup){Invoke-FaultBoundary 'config-backup-preserved'}
        Set-Step $State 'configBackupRemoved'
    }
    Remove-Item -LiteralPath $script:Paths.State;Invoke-FaultBoundary 'state-removed';Write-Host 'Rolled back Huihui Codex gateway deployment. Restart Codex/app-server when its Remote session permits it.'
}

if($InternalUninstall){if(-not(Test-Path -LiteralPath $script:Paths.State -PathType Leaf)){throw "No deployment state at $($script:Paths.State)."};Complete-Uninstall(Read-State);return}
if(Test-Path -LiteralPath $script:Paths.State -PathType Leaf){Complete-Install (Read-State) $null $null;return}
$prepared=New-PreparedState
if(-not$PSCmdlet.ShouldProcess('Huihui Codex gateway deployment','begin the complete schema-3 installation transaction')){Write-Host 'WhatIf/declined: all preconditions and post-images validated; zero changes made.';return}
Write-State $prepared.State -Prepared
# The transaction gate above is the sole ShouldProcess decision for a fresh run.
Complete-Install $prepared.State $prepared.InstalledBytes $prepared.Task -AlreadyApproved
