param(
    [string]$Destination = "vendor/licenses"
)

$ErrorActionPreference = "Stop"
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Destination))
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null

$licenses = @{
    "LGPL-2.1.txt" = "https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt"
    "LGPL-3.0.txt" = "https://www.gnu.org/licenses/lgpl-3.0.txt"
    "dnglab-LICENSE.txt" = "https://raw.githubusercontent.com/dnglab/dnglab/main/LICENSE"
    "rawpy-LICENSE.txt" = "https://raw.githubusercontent.com/letmaik/rawpy/main/LICENSE"
    "numpy-LICENSE.txt" = "https://raw.githubusercontent.com/numpy/numpy/main/LICENSE.txt"
    "tifffile-LICENSE.txt" = "https://raw.githubusercontent.com/cgohlke/tifffile/master/LICENSE"
    "PyInstaller-COPYING.txt" = "https://raw.githubusercontent.com/pyinstaller/pyinstaller/develop/COPYING.txt"
}

foreach ($entry in $licenses.GetEnumerator()) {
    $target = Join-Path $destinationPath $entry.Key
    & curl.exe -L --fail --silent --show-error $entry.Value --output $target
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $target)) {
        throw "Failed to fetch license: $($entry.Key)"
    }
}

Copy-Item -LiteralPath (Join-Path (Get-Location) "LICENSE") -Destination (Join-Path $destinationPath "ARW-Denoise-GPL-3.0.txt") -Force
Copy-Item -LiteralPath (Join-Path (Get-Location) "THIRD_PARTY_NOTICES.md") -Destination $destinationPath -Force
Write-Output "License bundle ready: $destinationPath"
