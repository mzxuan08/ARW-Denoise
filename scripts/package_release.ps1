param(
    [string]$Distribution = "dist\ArwDenoise",
    [string]$OutputDirectory = "outputs",
    [string]$Version = "0.3.1",
    [string]$Python = "python",
    [switch]$SkipRuntimeVerification
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$distributionPath = [System.IO.Path]::GetFullPath((Join-Path $root $Distribution))
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))
if (-not (Test-Path -LiteralPath (Join-Path $distributionPath "ArwDenoise.exe") -PathType Leaf)) {
    throw "Invalid distribution: $distributionPath"
}
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

& $Python (Join-Path $PSScriptRoot "release_manifest.py") $distributionPath --verify
if ($LASTEXITCODE -ne 0) { throw "Refusing to package an unverified distribution" }

$sevenZip = (& (Join-Path $PSScriptRoot "fetch_7zip.ps1") | Select-Object -Last 1).Trim()
if (-not (Test-Path -LiteralPath $sevenZip -PathType Leaf)) { throw "7za.exe is unavailable" }
$licenseSource = Join-Path (Split-Path -Parent (Split-Path -Parent $sevenZip)) "License.txt"
Copy-Item -LiteralPath $licenseSource -Destination (Join-Path $distributionPath "_internal\licenses\7-Zip-LGPL.txt") -Force
& $Python (Join-Path $PSScriptRoot "release_manifest.py") $distributionPath
if ($LASTEXITCODE -ne 0) { throw "Cannot refresh manifest after adding the 7-Zip license" }

$zip = Join-Path $outputPath "ArwDenoise-$Version-offline-win64.zip"
$solid = Join-Path $outputPath "ArwDenoise-$Version-offline-win64-solid.7z"
foreach ($archive in @($zip, $solid)) {
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
}
$parent = Split-Path -Parent $distributionPath
$folder = Split-Path -Leaf $distributionPath
Push-Location $parent
try {
    & $sevenZip a -tzip -mx=9 -mfb=258 -mpass=15 $zip $folder | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "ZIP creation failed" }
    & $sevenZip a -t7z -m0=lzma2 -mx=9 -ms=on -mmt=on -md=256m $solid $folder | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Solid 7z creation failed" }
}
finally {
    Pop-Location
}

$verificationRoot = Join-Path $outputPath ("verify-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $verificationRoot | Out-Null
try {
    foreach ($archive in @($zip, $solid)) {
        $target = Join-Path $verificationRoot ([IO.Path]::GetFileNameWithoutExtension($archive))
        New-Item -ItemType Directory -Path $target | Out-Null
        & $sevenZip x $archive "-o$target" -y | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Archive extraction failed: $archive" }
        $extracted = Join-Path $target $folder
        & $Python (Join-Path $PSScriptRoot "release_manifest.py") $extracted --verify
        if ($LASTEXITCODE -ne 0) { throw "Archive manifest mismatch: $archive" }
        if (-not $SkipRuntimeVerification) {
            & (Join-Path $PSScriptRoot "verify_offline_bundle.ps1") -Distribution $extracted
        }
    }
}
finally {
    $resolvedVerification = [System.IO.Path]::GetFullPath($verificationRoot)
    $resolvedOutput = [System.IO.Path]::GetFullPath($outputPath)
    if (-not $resolvedVerification.StartsWith($resolvedOutput + [IO.Path]::DirectorySeparatorChar)) {
        throw "Unsafe verification cleanup path: $resolvedVerification"
    }
    Remove-Item -LiteralPath $resolvedVerification -Recurse -Force
}

$hashes = foreach ($path in @($zip, $solid)) {
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($path))"
}
$hashPath = Join-Path $outputPath "SHA256SUMS-$Version.txt"
[IO.File]::WriteAllLines($hashPath, $hashes, [Text.UTF8Encoding]::new($false))
Write-Output "ZIP: $zip"
Write-Output "Solid: $solid"
Write-Output "Hashes: $hashPath"
