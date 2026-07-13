param(
    [Parameter(Mandatory = $true)]
    [string]$Distribution
)

$ErrorActionPreference = "Stop"
$distributionPath = [System.IO.Path]::GetFullPath($Distribution)
$executable = Join-Path $distributionPath "ArwDenoise.exe"
$required = @(
    $executable,
    (Join-Path $distributionPath "_internal\tools\dnglab.exe"),
    (Join-Path $distributionPath "_internal\models\pmrid\manifest.json"),
    (Join-Path $distributionPath "_internal\models\pmrid\pmrid-fp16.onnx"),
    (Join-Path $distributionPath "_internal\onnxruntime\capi\onnxruntime_providers_cuda.dll")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Offline bundle file missing: $path" }
}
foreach ($pattern in @(
    "_internal\nvidia\cublas\bin\cublas64_12.dll",
    "_internal\nvidia\cuda_runtime\bin\cudart64_12.dll",
    "_internal\nvidia\cudnn\bin\cudnn64_9.dll",
    "_internal\nvidia\cufft\bin\cufft64_11.dll",
    "_internal\nvidia\nvjitlink\bin\nvJitLink_120_0.dll"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $distributionPath $pattern) -PathType Leaf)) {
        throw "Offline CUDA runtime file missing: $pattern"
    }
}

$savedPath = $env:PATH
$savedModel = $env:ARW_DENOISE_MODEL_DIR
$savedDngLab = $env:ARW_DENOISE_DNGLAB
$savedPythonPath = $env:PYTHONPATH
try {
    $env:PATH = [Environment]::GetFolderPath("System")
    Remove-Item Env:ARW_DENOISE_MODEL_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:ARW_DENOISE_DNGLAB -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

    $probe = Start-Process -FilePath $executable -ArgumentList "gpu-probe" -PassThru -WindowStyle Hidden
    if (-not $probe.WaitForExit(90000)) {
        $probe.Kill()
        throw "Offline GPU self-test timed out"
    }
    if ($probe.ExitCode -ne 0) { throw "Offline GPU self-test failed with exit code $($probe.ExitCode)" }

    $gui = Start-Process -FilePath $executable -PassThru -WindowStyle Hidden
    if ($gui.WaitForExit(5000)) {
        throw "Offline GUI exited during startup with code $($gui.ExitCode)"
    }
    $gui.Kill()
    $gui.WaitForExit()
}
finally {
    $env:PATH = $savedPath
    if ($null -eq $savedModel) { Remove-Item Env:ARW_DENOISE_MODEL_DIR -ErrorAction SilentlyContinue } else { $env:ARW_DENOISE_MODEL_DIR = $savedModel }
    if ($null -eq $savedDngLab) { Remove-Item Env:ARW_DENOISE_DNGLAB -ErrorAction SilentlyContinue } else { $env:ARW_DENOISE_DNGLAB = $savedDngLab }
    if ($null -eq $savedPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $savedPythonPath }
}
Write-Output "Offline bundle verified: $distributionPath"
