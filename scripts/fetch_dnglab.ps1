param(
    [string]$Version = "0.7.2",
    [string]$Destination = "vendor/dnglab"
)

$ErrorActionPreference = "Stop"
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Destination))
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
$archive = Join-Path $destinationPath "dnglab-win-x64.zip"
$url = "https://github.com/dnglab/dnglab/releases/download/v$Version/dnglab-win-x64_v$Version.zip"
Invoke-WebRequest -Uri $url -OutFile $archive
Expand-Archive -LiteralPath $archive -DestinationPath $destinationPath -Force
Remove-Item -LiteralPath $archive
$executable = Join-Path $destinationPath "dnglab.exe"
if (-not (Test-Path -LiteralPath $executable)) {
    throw "Downloaded archive did not contain dnglab.exe"
}
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $executable).Hash
& $executable --version
Write-Output "Installed: $executable"
Write-Output "SHA256: $hash"

