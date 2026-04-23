$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvYml = Join-Path $ProjectRoot "environment-win.yml"
$YoloConfigDir = Join-Path $ProjectRoot ".yolo"
$ModelDir = Join-Path $ProjectRoot "model"
$ModelPath = Join-Path $ModelDir "best.pt"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found in PATH. Please install Miniconda or Anaconda first."
}

Write-Host "Creating / updating conda environment from $EnvYml"
conda env create -f $EnvYml 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment may already exist, trying conda env update..."
    conda env update -f $EnvYml --prune
}

if (-not (Test-Path $YoloConfigDir)) {
    New-Item -ItemType Directory -Path $YoloConfigDir | Out-Null
}

if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir | Out-Null
}

Write-Host ""
Write-Host "Environment setup is ready."
Write-Host "Before running the scripts, set this in your terminal session:"
Write-Host "  `$env:YOLO_CONFIG_DIR = '$YoloConfigDir'"
Write-Host ""
if (Test-Path $ModelPath) {
    Write-Host "Model found: $ModelPath"
} else {
    Write-Host "Model not found: $ModelPath"
    Write-Host "Download the custom YOLO weights and place them at this path."
}
Write-Host ""
Write-Host "To run images:"
Write-Host "  conda activate dashanon-win"
Write-Host "  python blur_images_batch.py --input-root input_root --output-root blurred_root --issues-root issues_root"
Write-Host ""
Write-Host "To run a single folder:"
Write-Host "  python blur_images.py --config configs\img_blur.yaml"
