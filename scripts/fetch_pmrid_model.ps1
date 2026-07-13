[CmdletBinding()]
param(
    [string]$Destination = "models\pmrid\torch_pretrained.ckp",
    [string]$SourceUrl = "https://raw.githubusercontent.com/MegEngine/PMRID/8ebb9e8e96559881dee957f34243933c5beb77dd/models/torch_pretrained.ckp",
    [string]$SourcePath,
    [ValidatePattern("^[0-9A-Fa-f]{64}$")]
    [string]$ExpectedSha256 = "9361614f3514d27351d81909f2215c0fdc38619c0288d936b7266485ac106c14"
)

$ErrorActionPreference = "Stop"
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$destinationDirectory = Split-Path -Parent $destinationPath

if (Test-Path -LiteralPath $destinationPath) {
    $existingHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash
    if ($existingHash -ieq $ExpectedSha256) {
        Write-Output "PMRID checkpoint already verified: $destinationPath"
        exit 0
    }
    throw "Refusing to replace an existing checkpoint with the wrong SHA-256: $destinationPath"
}

New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
$temporaryPath = Join-Path $destinationDirectory (".{0}.{1}.download" -f ([System.IO.Path]::GetFileName($destinationPath)), [guid]::NewGuid().ToString("N"))

try {
    if ($SourcePath) {
        Copy-Item -LiteralPath $SourcePath -Destination $temporaryPath
    }
    else {
        Invoke-WebRequest -Uri $SourceUrl -OutFile $temporaryPath -UseBasicParsing
    }

    $downloadHash = (Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256).Hash
    if ($downloadHash -ine $ExpectedSha256) {
        throw "PMRID checkpoint SHA-256 mismatch. Expected $ExpectedSha256, got $downloadHash"
    }

    Move-Item -LiteralPath $temporaryPath -Destination $destinationPath
    Write-Output "Verified PMRID checkpoint: $destinationPath"
}
finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
}

