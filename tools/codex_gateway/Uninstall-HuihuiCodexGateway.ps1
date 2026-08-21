[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param([switch]$ForceRestoreConfig)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$invoke = @{ InternalUninstall = $true; ForceRestoreConfig = $ForceRestoreConfig.IsPresent }
if ($PSBoundParameters.ContainsKey('WhatIf')) { $invoke.WhatIf = $PSBoundParameters['WhatIf'] }
if ($PSBoundParameters.ContainsKey('Confirm')) { $invoke.Confirm = $PSBoundParameters['Confirm'] }
& (Join-Path $PSScriptRoot 'Install-HuihuiCodexGateway.ps1') @invoke
