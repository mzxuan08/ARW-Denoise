param(
    [string]$Python = "python",
    [switch]$SkipDependencies,
    [switch]$SkipLicenseFetch
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $SkipDependencies) {
    & $Python -m pip install -r (Join-Path $root "requirements-gpu-lock.txt")
    if ($LASTEXITCODE -ne 0) { throw "Runtime dependency installation failed" }
    & $Python -m pip install --no-deps -e $root
    if ($LASTEXITCODE -ne 0) { throw "Project installation failed" }
}

$dnglab = Join-Path $root "vendor\dnglab\dnglab.exe"
if (-not (Test-Path -LiteralPath $dnglab)) {
    & (Join-Path $PSScriptRoot "fetch_dnglab.ps1")
}
$modelRoot = Join-Path $root "models\pmrid"
$model = Join-Path $modelRoot "pmrid-fp16.onnx"
$manifest = Join-Path $modelRoot "manifest.json"
if (-not (Test-Path -LiteralPath $model) -or -not (Test-Path -LiteralPath $manifest)) {
    throw "PMRID offline model package is incomplete; run fetch_pmrid_model.ps1 and pmrid_to_onnx.py first"
}

$licenses = Join-Path $root "vendor\licenses"
if (-not $SkipLicenseFetch) {
    & (Join-Path $PSScriptRoot "fetch_licenses.ps1")
}
if (-not (Test-Path -LiteralPath $licenses)) {
    throw "License bundle is missing: $licenses"
}
Copy-Item -LiteralPath (Join-Path $root "vendor\pmrid\LICENSE") -Destination (Join-Path $licenses "PMRID-Apache-2.0.txt") -Force

$pythonExecutable = (Get-Command $Python -ErrorAction Stop).Source
$venvRoot = Split-Path -Parent (Split-Path -Parent $pythonExecutable)
$sitePackages = Join-Path $venvRoot "Lib\site-packages"
if (-not (Test-Path -LiteralPath $sitePackages)) {
    $sitePackages = (& $Python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
}
if (-not (Test-Path -LiteralPath $sitePackages)) { throw "Cannot locate site-packages" }
$ortLicense = Join-Path $sitePackages "onnxruntime\LICENSE"
if (Test-Path -LiteralPath $ortLicense) {
    Copy-Item -LiteralPath $ortLicense -Destination (Join-Path $licenses "ONNX-Runtime-MIT.txt") -Force
}
foreach ($pattern in @("pyside6-*.dist-info", "nvidia_cublas_cu12-*.dist-info", "nvidia_cuda_runtime_cu12-*.dist-info", "nvidia_cudnn_cu12-*.dist-info", "nvidia_cufft_cu12-*.dist-info", "nvidia_nvjitlink_cu12-*.dist-info")) {
    $package = Get-ChildItem -LiteralPath $sitePackages -Directory -Filter $pattern | Select-Object -First 1
    if ($package) {
        $licenseFiles = Get-ChildItem -LiteralPath $package.FullName -Recurse -File | Where-Object { $_.Name -match "License" }
        foreach ($licenseFile in $licenseFiles) {
            $safeName = ($package.Name -replace "\.dist-info$", "") + "-" + $licenseFile.Name
            Copy-Item -LiteralPath $licenseFile.FullName -Destination (Join-Path $licenses $safeName) -Force
        }
    }
}

$pyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm", "--clean", "--windowed", "--noupx",
    "--name", "ArwDenoise",
    "--collect-all", "rawpy",
    "--collect-binaries", "onnxruntime",
    "--hidden-import", "onnxruntime.capi._pybind_state",
    "--exclude-module", "torch",
    "--add-binary", "$dnglab;tools",
    "--add-data", "$modelRoot;models\pmrid",
    "--add-data", "$licenses;licenses"
)
foreach ($component in @("cublas", "cuda_runtime", "cudnn", "cufft", "nvjitlink")) {
    $bin = Join-Path $sitePackages "nvidia\$component\bin\*.dll"
    if (-not (Get-ChildItem -Path $bin -ErrorAction SilentlyContinue)) {
        throw "Required offline NVIDIA runtime DLLs are missing: $component"
    }
    $pyInstallerArgs += @("--add-binary", "$bin;nvidia\$component\bin")
}
$pyInstallerArgs += (Join-Path $PSScriptRoot "launcher.py")
& $Python @pyInstallerArgs
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$distribution = Join-Path $root "dist\ArwDenoise"
Copy-Item -LiteralPath (Join-Path $root "README.md") -Destination $distribution -Force
Copy-Item -LiteralPath (Join-Path $root "THIRD_PARTY_NOTICES.md") -Destination $distribution -Force
& $Python (Join-Path $PSScriptRoot "release_manifest.py") $distribution
if ($LASTEXITCODE -ne 0) { throw "Release manifest generation failed" }
Write-Output "Build complete: $(Join-Path $distribution 'ArwDenoise.exe')"
