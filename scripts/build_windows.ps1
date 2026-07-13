param(
    [switch]$SkipDependencies
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $SkipDependencies) {
    python -m pip install -e ".[gui,raw,build]"
}

$dnglab = Join-Path $root "vendor\dnglab\dnglab.exe"
if (-not (Test-Path -LiteralPath $dnglab)) {
    & (Join-Path $PSScriptRoot "fetch_dnglab.ps1")
}

& (Join-Path $PSScriptRoot "fetch_licenses.ps1")
$licenses = Join-Path $root "vendor\licenses"

python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "ArwDenoise" `
    --collect-all rawpy `
    --add-binary "$dnglab;tools" `
    --add-data "$licenses;licenses" `
    (Join-Path $PSScriptRoot "launcher.py")

$distribution = Join-Path $root "dist\ArwDenoise"
Copy-Item -LiteralPath $licenses -Destination (Join-Path $distribution "licenses") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $root "README.md") -Destination $distribution -Force
Copy-Item -LiteralPath (Join-Path $root "THIRD_PARTY_NOTICES.md") -Destination $distribution -Force
Write-Output "Build complete: $(Join-Path $distribution 'ArwDenoise.exe')"
