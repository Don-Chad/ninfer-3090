$ErrorActionPreference = 'Stop'

# Editable release configuration.
$ReleaseTag = '0.2.0-rtx3090-v2'
$WslDistro = 'Ubuntu-24.04'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$DistRoot = Join-Path $RepoRoot 'dist'
$WindowsName = "ninfer-rtx3090-windows-x64-$ReleaseTag"
$LinuxName = "ninfer-rtx3090-linux-x64-$ReleaseTag"
$WindowsDir = Join-Path $DistRoot $WindowsName
$LinuxDir = Join-Path $DistRoot $LinuxName

function Reset-PackageDirectory([string] $Path) {
    $absolute = [System.IO.Path]::GetFullPath($Path)
    $distAbsolute = [System.IO.Path]::GetFullPath($DistRoot) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($distAbsolute, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to reset a path outside dist: $absolute"
    }
    if (Test-Path -LiteralPath $absolute) {
        Remove-Item -LiteralPath $absolute -Recurse -Force
    }
    New-Item -ItemType Directory -Path $absolute | Out-Null
}

function Write-PackageHashes([string] $Directory) {
    $lines = Get-ChildItem -LiteralPath $Directory -File |
        Sort-Object Name |
        ForEach-Object {
            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            "$hash  $($_.Name)"
        }
    Set-Content -LiteralPath (Join-Path $Directory 'SHA256SUMS.txt') -Value $lines -Encoding ascii
}

$windowsBuild = Join-Path $RepoRoot 'build-windows\apps\Release'
$windowsBench = Join-Path $RepoRoot 'build-windows\bench\Release\ninfer_bench.exe'
$linuxBuild = Join-Path $RepoRoot 'build-sm86'

foreach ($required in @(
    (Join-Path $windowsBuild 'ninfer.exe'),
    (Join-Path $windowsBuild 'ninfer-serve.exe'),
    $windowsBench,
    (Join-Path $linuxBuild 'apps\ninfer'),
    (Join-Path $linuxBuild 'apps\ninfer-serve'),
    (Join-Path $linuxBuild 'bench\ninfer_bench')
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required release product is missing: $required"
    }
}

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
# Remove stale bundles produced by an earlier invocation, but never anything outside dist.
$currentProducts = @($WindowsName, $LinuxName, "$WindowsName.zip", "$LinuxName.tar.gz")
Get-ChildItem -LiteralPath $DistRoot |
    Where-Object {
        ($_.Name -like 'ninfer-rtx3090-windows-x64-*' -or
         $_.Name -like 'ninfer-rtx3090-linux-x64-*') -and
        $_.Name -notin $currentProducts
    } |
    ForEach-Object {
        $absolute = [System.IO.Path]::GetFullPath($_.FullName)
        $distAbsolute = [System.IO.Path]::GetFullPath($DistRoot) +
            [System.IO.Path]::DirectorySeparatorChar
        if (-not $absolute.StartsWith($distAbsolute, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove a stale product outside dist: $absolute"
        }
        Remove-Item -LiteralPath $absolute -Recurse -Force
    }
Reset-PackageDirectory $WindowsDir
Reset-PackageDirectory $LinuxDir

Copy-Item -LiteralPath (Join-Path $windowsBuild 'ninfer.exe') -Destination $WindowsDir
Copy-Item -LiteralPath (Join-Path $windowsBuild 'ninfer-serve.exe') -Destination $WindowsDir
Copy-Item -LiteralPath $windowsBench -Destination $WindowsDir
Get-ChildItem -LiteralPath $windowsBuild -Filter '*.dll' -File |
    Copy-Item -Destination $WindowsDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $WindowsDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'VERSION') -Destination $WindowsDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'docs\rtx-3090-windows.md') `
    -Destination (Join-Path $WindowsDir 'README.md')
Write-PackageHashes $WindowsDir

Copy-Item -LiteralPath (Join-Path $linuxBuild 'apps\ninfer') -Destination $LinuxDir
Copy-Item -LiteralPath (Join-Path $linuxBuild 'apps\ninfer-serve') -Destination $LinuxDir
Copy-Item -LiteralPath (Join-Path $linuxBuild 'bench\ninfer_bench') -Destination $LinuxDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $LinuxDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'VERSION') -Destination $LinuxDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'docs\rtx-3090-wsl.md') `
    -Destination (Join-Path $LinuxDir 'README.md')
Write-PackageHashes $LinuxDir

$windowsArchive = Join-Path $DistRoot "$WindowsName.zip"
$linuxArchive = Join-Path $DistRoot "$LinuxName.tar.gz"
if (Test-Path -LiteralPath $windowsArchive) { Remove-Item -LiteralPath $windowsArchive -Force }
if (Test-Path -LiteralPath $linuxArchive) { Remove-Item -LiteralPath $linuxArchive -Force }

Compress-Archive -Path $WindowsDir -DestinationPath $windowsArchive -CompressionLevel Optimal
$drive = $RepoRoot.Substring(0, 1).ToLowerInvariant()
$tail = $RepoRoot.Substring(3).Replace([char] 92, '/')
$wslRepo = "/mnt/$drive/$tail"
& wsl.exe -d $WslDistro -- bash -lc `
    "cd '$wslRepo' && chmod +x 'dist/$LinuxName/ninfer' 'dist/$LinuxName/ninfer-serve' 'dist/$LinuxName/ninfer_bench' && tar -C dist -czf 'dist/$LinuxName.tar.gz' '$LinuxName'"
if ($LASTEXITCODE -ne 0) { throw 'Linux archive creation failed.' }

$archiveHashes = foreach ($archive in @($windowsArchive, $linuxArchive)) {
    $item = Get-Item -LiteralPath $archive
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $item.FullName).Hash.ToLowerInvariant()
    "$hash  $($item.Name)"
}
Set-Content -LiteralPath (Join-Path $DistRoot 'SHA256SUMS.txt') `
    -Value $archiveHashes -Encoding ascii

Get-Item -LiteralPath $windowsArchive, $linuxArchive |
    Select-Object Name, @{Name = 'SizeMB'; Expression = {[math]::Round($_.Length / 1MB, 2)}}
