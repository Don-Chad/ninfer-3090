$ErrorActionPreference = 'Stop'

$ReleaseTag = '0.5.0-rtx3090'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $RepoRoot 'build-sm86-qwen38'
$DistRoot = Join-Path $RepoRoot 'dist'
$ProductName = "ninfer-rtx3090-windows-x64-$ReleaseTag"
$ProductRoot = Join-Path $DistRoot $ProductName
$ArchivePath = Join-Path $DistRoot "$ProductName.zip"
$ChecksumPath = Join-Path $DistRoot 'SHA256SUMS-v0.5.0.txt'

$Products = @(
    @{ Source = 'apps\Release\ninfer.exe'; Destination = 'ninfer.exe' },
    @{ Source = 'apps\Release\ninfer-serve.exe'; Destination = 'ninfer-serve.exe' },
    @{ Source = 'bench\Release\ninfer_bench.exe'; Destination = 'ninfer_bench.exe' }
)

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
$resolvedDist = (Resolve-Path -LiteralPath $DistRoot).Path
$resolvedProductParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $ProductRoot))
if ($resolvedProductParent -ne $resolvedDist -or (Split-Path -Leaf $ProductRoot) -ne $ProductName) {
    throw "Refusing to package outside the expected dist directory: $ProductRoot"
}
if (Test-Path -LiteralPath $ProductRoot) {
    Remove-Item -LiteralPath $ProductRoot -Recurse -Force
}
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
New-Item -ItemType Directory -Path $ProductRoot | Out-Null

foreach ($product in $Products) {
    $source = Join-Path $BuildRoot $product.Source
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing release product: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $ProductRoot $product.Destination)
}

Get-ChildItem -LiteralPath (Join-Path $BuildRoot 'apps\Release') -Filter '*.dll' | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $ProductRoot
}
Copy-Item -LiteralPath (Join-Path $RepoRoot 'VERSION') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'docs\rtx-3090-windows.md') -Destination (Join-Path $ProductRoot 'README.md')
Copy-Item -LiteralPath (Join-Path $RepoRoot 'RELEASE_NOTES_0.5.0.md') -Destination $ProductRoot

$innerHashes = Get-ChildItem -LiteralPath $ProductRoot -File | Sort-Object Name | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($_.Name)"
}
$innerHashes | Set-Content -LiteralPath (Join-Path $ProductRoot 'SHA256SUMS.txt') -Encoding ascii

Compress-Archive -LiteralPath $ProductRoot -DestinationPath $ArchivePath -CompressionLevel Optimal
$archiveHash = Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256
"$($archiveHash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $ArchivePath)" |
    Set-Content -LiteralPath $ChecksumPath -Encoding ascii

Get-Item -LiteralPath $ArchivePath, $ChecksumPath |
    Select-Object Name, @{ Name = 'SizeMB'; Expression = { [math]::Round($_.Length / 1MB, 2) } }
