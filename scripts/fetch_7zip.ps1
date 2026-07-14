param(
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Destination) { $Destination = Join-Path $root "vendor\7zip" }
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null

$version = "26.02"
$bootstrapUrl = "https://github.com/ip7z/7zip/releases/download/$version/7zr.exe"
$bootstrapSha256 = "56b8cc9f4971cef253644fafe54063ed7fdca551d4dee0f8c6baa81b855acd72"
$extraUrl = "https://github.com/ip7z/7zip/releases/download/$version/7z2602-extra.7z"
$extraSha256 = "081df9e9311dfd9c9e0e98c1c80180b99bb51e4cb24156b5f3057fe3c259d70a"
$bootstrap = Join-Path $destinationPath "7zr.exe"
$extra = Join-Path $destinationPath "7z2602-extra.7z"

foreach ($asset in @(
    @{ Path = $bootstrap; Url = $bootstrapUrl; Hash = $bootstrapSha256 },
    @{ Path = $extra; Url = $extraUrl; Hash = $extraSha256 }
)) {
    if (-not (Test-Path -LiteralPath $asset.Path) -or
        (Get-FileHash -LiteralPath $asset.Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.Hash) {
        Invoke-WebRequest -UseBasicParsing -Uri $asset.Url -OutFile $asset.Path
    }
    $actual = (Get-FileHash -LiteralPath $asset.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $asset.Hash) { throw "7-Zip asset hash mismatch: $($asset.Path)" }
}

& $bootstrap x $extra "-o$destinationPath" -aoa | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Cannot extract the pinned 7-Zip package" }
$sevenZip = Join-Path $destinationPath "x64\7za.exe"
$license = Join-Path $destinationPath "License.txt"
if (-not (Test-Path -LiteralPath $sevenZip -PathType Leaf)) { throw "Pinned 7za.exe is missing" }
if (-not (Test-Path -LiteralPath $license -PathType Leaf)) { throw "7-Zip license is missing" }
Write-Output $sevenZip
