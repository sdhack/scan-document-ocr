param(
    [Parameter(Mandatory=$true)][string]$InputPdf,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Distro = 'Ubuntu-22.04',
    [int]$Workers = 3
)

$ErrorActionPreference = 'Stop'
$wslCheck = ((& wsl.exe -l -q 2>&1) -join "`n") -replace "`0", ""
if ($LASTEXITCODE -ne 0 -or $wslCheck -notmatch [regex]::Escape($Distro)) {
    throw "Required WSL distro not found: $Distro"
}
$gpuCheck = & wsl.exe -d $Distro -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ($LASTEXITCODE -ne 0 -or -not $gpuCheck) { throw 'NVIDIA GPU not available in WSL; CPU mode is refused' }
function Convert-ToWslPath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full -notmatch '^([A-Za-z]):\\(.*)$') { throw "Only local Windows drive paths are supported: $full" }
    return "/mnt/$($Matches[1].ToLower())/$(($Matches[2] -replace '\\','/') -replace ' ','%20')" -replace '%20',' '
}
$pdf = (Resolve-Path -LiteralPath $InputPdf).Path
$out = [System.IO.Path]::GetFullPath($OutputDir)
$pdfWsl = Convert-ToWslPath $pdf
$outWsl = Convert-ToWslPath $out
$scriptWsl = Convert-ToWslPath (Join-Path $PSScriptRoot 'persistent_ocr.py')
$ocrEngineWsl = Convert-ToWslPath (Join-Path $PSScriptRoot 'scan_pdf.py')
$verifyWsl = Convert-ToWslPath (Join-Path $PSScriptRoot 'verify_cuda.py')
if (-not $pdfWsl) { throw 'Failed to convert input PDF path to WSL path' }
if (-not $outWsl) { throw 'Failed to convert output path to WSL path' }

$wslHome = (& wsl.exe -d $Distro -- sh -c 'printf %s "$HOME"').Trim()
if ($LASTEXITCODE -ne 0 -or -not $wslHome) { throw 'Failed to resolve WSL home directory' }
$venv = if ($env:OCR_VENV_WSL) { $env:OCR_VENV_WSL } else { "$wslHome/.venvs/scan-document-ocr" }
$sitePackages = (& wsl.exe -d $Distro -- $venv/bin/python -c 'import site; print(site.getsitepackages()[0])').Trim()
if ($LASTEXITCODE -ne 0 -or -not $sitePackages) { throw 'Failed to resolve WSL Python site-packages' }
$lib = if ($env:OCR_CUDA_LIB_WSL) { $env:OCR_CUDA_LIB_WSL } else { "$sitePackages/nvidia" }
$ld = "$lib/cublas/lib:$lib/cudnn/lib:$lib/cuda_runtime/lib:$lib/cuda_nvrtc/lib:$lib/cufft/lib:$lib/curand/lib:$lib/cusolver/lib:$lib/cusparse/lib:$lib/nvtx/lib"
$pdf64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pdfWsl))
$out64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($outWsl))
$pythonWsl = "$venv/bin/python"
$verifyArgs = @('env', "LD_LIBRARY_PATH=$ld", $pythonWsl, $verifyWsl)
& wsl.exe -d $Distro -- @verifyArgs
if ($LASTEXITCODE -ne 0) { throw 'Real CUDA session verification failed; CPU fallback is refused' }
$envArgs = @(
    'env',
    "LD_LIBRARY_PATH=$ld",
    'OMP_NUM_THREADS=2',
    'MKL_NUM_THREADS=2',
    'OPENBLAS_NUM_THREADS=2',
    'OPENCV_FOR_THREADS_NUM=2',
    "OCR_ENGINE_SCRIPT=$ocrEngineWsl",
    "OCR_VENV_WSL=$venv",
    "OCR_CUDA_LIB_WSL=$lib",
    "OCR_WORKERS=$Workers",
    $pythonWsl,
    $scriptWsl,
    "b64:$pdf64",
    "b64:$out64",
    "$Workers"
)
& wsl.exe -d $Distro -- @envArgs
if ($LASTEXITCODE -ne 0) { throw "WSL GPU OCR failed with exit code: $LASTEXITCODE" }
