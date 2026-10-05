# Install and configure ccache for NInfer builds. Run once per machine; safe to run again.
#
# Why: a full rebuild is 10+ minutes and a public-header edit recompiles nearly everything. The
# cache is shared by every git worktree of this checkout, so a worktree that changes a few files
# rebuilds only those. cmake/CompilerCache.cmake turns it on automatically when ccache is found;
# this script installs ccache and writes the settings that make sharing work:
#
#   base_dir  the main checkout. ccache rewrites absolute paths under it to be relative, so the same
#             source in two worktrees produces the same command and hits the same entry. Without it
#             every worktree has its own, unshared entries.
#   hash_dir  false. The working directory is not hashed (it only matters for debug info).
#   max_size  the cache is one per-user directory shared by all trees.
#
# ccache.exe is installed flat into %LOCALAPPDATA%\ccache-bin (no admin, no package manager) and
# that directory is added to the user PATH. A ccache already on PATH is used as is.
#
#   .\scripts\setup-ccache.ps1                 install if needed, then configure
#   .\scripts\setup-ccache.ps1 -MaxSize 50G    larger cache
#
# After a build: ccache -s     (hit rate; a second worktree should hit nearly everything)
[CmdletBinding()]
param(
    [string]$MaxSize = "30G"
)

$ErrorActionPreference = "Stop"

# Pinned: the version and the SHA-256 of its Windows x86_64 zip.
$Version = "4.14.1"
$Sha256  = "6219f3865ca59aec41ee4b678df171d5d35855ecb2b6dbbbd20690b3a68af7b4"
$Url     = "https://github.com/ccache/ccache/releases/download/v$Version/ccache-$Version-windows-x86_64.zip"

$BinDir = Join-Path $env:LOCALAPPDATA "ccache-bin"
$Exe    = Join-Path $BinDir "ccache.exe"

$onPath = Get-Command ccache -ErrorAction SilentlyContinue
if ($onPath) {
    $Exe = $onPath.Source
    Write-Host "Using ccache on PATH: $Exe"
} elseif (-not (Test-Path $Exe)) {
    Write-Host "Installing ccache $Version into $BinDir"
    New-Item -ItemType Directory -Force $BinDir | Out-Null
    $zip = Join-Path ([System.IO.Path]::GetTempPath()) "ccache-$Version.zip"
    Invoke-WebRequest $Url -OutFile $zip
    $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $Sha256) {
        Remove-Item $zip -Force
        throw "ccache download checksum mismatch (expected $Sha256, got $actual)"
    }
    $extract = Join-Path ([System.IO.Path]::GetTempPath()) "ccache-$Version"
    if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
    Expand-Archive $zip -DestinationPath $extract
    Copy-Item (Get-ChildItem $extract -Recurse -Filter ccache.exe | Select-Object -First 1).FullName $Exe -Force
    Remove-Item $zip, $extract -Recurse -Force
}

if (-not $onPath) {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (($userPath -split ";") -notcontains $BinDir) {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$BinDir", "User")
        Write-Host "Added $BinDir to the user PATH (open a new shell to see it)."
    }
}

# The main checkout is the parent of git's common directory, which also holds every linked worktree.
$common = git -C $PSScriptRoot rev-parse --path-format=absolute --git-common-dir
if ($LASTEXITCODE -ne 0 -or -not $common) { throw "Not inside a git checkout; cannot find the main checkout." }
$Base = (Resolve-Path (Join-Path $common "..")).Path -replace "\\", "/"

& $Exe --set-config "base_dir=$Base"
& $Exe --set-config "hash_dir=false"
& $Exe --set-config "max_size=$MaxSize"

Write-Host ""
& $Exe --version | Select-Object -First 1
& $Exe --show-config | Select-String "cache_dir|base_dir|hash_dir|max_size"
Write-Host ""
Write-Host "Reconfigure a build tree to pick it up; CMake prints 'Compiler cache: ...' when it is on."
