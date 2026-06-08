<#
  Hunyuan3D-2 EasySetup  -  one command to set up everything and launch the UI.

  Usage (from the repo folder):
      powershell -ExecutionPolicy Bypass -File .\setup.ps1

  Common options:
      .\setup.ps1 -NoLaunch              # install only, don't start the UI
      .\setup.ps1 -Port 7860             # serve on a different port
      .\setup.ps1 -Cuda cu121            # pick a CUDA wheel (cu121|cu124|cu128|cpu)
      .\setup.ps1 -NoText23d             # image->3D only (skip text->3D, smaller download)

  Prerequisites (the script checks and tells you if any are missing):
    * Miniconda / Anaconda          https://docs.conda.io/en/latest/miniconda.html
    * NVIDIA GPU + recent driver    (8GB+ VRAM recommended; CPU works but is slow)
    * Visual Studio Build Tools with the "Desktop development with C++" workload
                                    https://visualstudio.microsoft.com/downloads/
      (only needed to compile the texturing extensions; shape gen works without it)
#>
[CmdletBinding()]
param(
  [string]$EnvName   = "hunyuan3d",
  [int]   $Port      = 8080,
  [string]$Cuda      = "cu124",
  [string]$PyVersion = "3.11",
  [string]$Model     = "tencent/Hunyuan3D-2",
  [string]$Subfolder = "hunyuan3d-dit-v2-0",
  [switch]$NoLaunch,
  [switch]$NoText23d
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
function Info($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[warn] $m" -ForegroundColor Yellow }
function Die ($m){ Write-Host "[error] $m" -ForegroundColor Red; exit 1 }

# --- 1. conda -------------------------------------------------------------
if(-not (Get-Command conda -ErrorAction SilentlyContinue)){
  Die "conda not found on PATH. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
}

# --- 2. create / reuse the env -------------------------------------------
$envList = (conda env list) -join "`n"
if($envList -notmatch "[\\/]$([regex]::Escape($EnvName))(\s|$)"){
  Info "Creating conda env '$EnvName' (python $PyVersion)"
  conda create -y -n $EnvName "python=$PyVersion" | Out-Null
} else {
  Info "Reusing existing conda env '$EnvName'"
}

$py = (conda run -n $EnvName python -c "import sys; print(sys.executable)").Trim()
if(-not $py -or -not (Test-Path $py)){ Die "Could not resolve the python executable for env '$EnvName'." }
Info "Python: $py"

# --- 3. PyTorch -----------------------------------------------------------
Info "Installing PyTorch ($Cuda)"
if($Cuda -eq "cpu"){
  & $py -m pip install --upgrade torch torchvision
} else {
  & $py -m pip install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
}

# --- 4. Python dependencies ----------------------------------------------
Info "Installing Python requirements"
& $py -m pip install -r (Join-Path $repo "requirements.txt")
# tokenizers needed by the text->3D model; sentencepiece is commented in requirements
& $py -m pip install tiktoken sentencepiece

# --- 5. build the texturing CUDA extensions ------------------------------
$hasCuda = (& $py -c "import torch; print(torch.cuda.is_available())").Trim()
if($hasCuda -eq "True"){
  $cap = (& $py -c "import torch; m,n=torch.cuda.get_device_capability(); print(f'{m}.{n}')").Trim()
  Info "Building texturing extensions for CUDA arch $cap"
  $env:TORCH_CUDA_ARCH_LIST = $cap
  $env:DISTUTILS_USE_SDK    = "1"
  $env:MAX_JOBS             = "8"

  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  $vsPath  = $null
  if(Test-Path $vswhere){
    $vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath) 2>$null
  }
  if($vsPath){
    $devcmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
    $cr = Join-Path $repo "hy3dgen\texgen\custom_rasterizer"
    $dr = Join-Path $repo "hy3dgen\texgen\differentiable_renderer"
    Info "Using Visual Studio: $vsPath"
    cmd /c "`"$devcmd`" -arch=x64 -host_arch=x64 && `"$py`" -m pip install --no-build-isolation `"$cr`" && `"$py`" -m pip install --no-build-isolation `"$dr`""
    if($LASTEXITCODE -ne 0){ Warn "Texturing extension build failed. Shape generation will still work; texturing will be disabled." }
  } else {
    Warn "Visual Studio C++ build tools not found. Skipping texturing extensions."
    Warn "Install 'Desktop development with C++' to enable textured output, then re-run this script."
  }
} else {
  Warn "CUDA not available. Skipping texturing extensions (CPU shape gen only, and it will be slow)."
}

$rastOk = (& $py -c "import importlib.util as u; print('yes' if u.find_spec('custom_rasterizer') else 'no')").Trim()
Info "Texturing available: $rastOk"
Info "Setup complete."

# --- 6. launch ------------------------------------------------------------
$launchArgs = @(
  (Join-Path $repo "gradio_app.py"),
  "--model_path", $Model,
  "--subfolder", $Subfolder,
  "--texgen_model_path", "tencent/Hunyuan3D-2",
  "--port", "$Port"
)
if(-not $NoText23d){ $launchArgs += "--enable_t23d" }

if($NoLaunch){
  Write-Host ""
  Write-Host "Done. Launch the UI anytime with:" -ForegroundColor Green
  Write-Host "  conda run -n $EnvName python `"$(Join-Path $repo 'gradio_app.py')`" --model_path $Model --subfolder $Subfolder --texgen_model_path tencent/Hunyuan3D-2 --port $Port $(if(-not $NoText23d){'--enable_t23d'})"
} else {
  Info "Launching UI on http://localhost:$Port   (first run downloads ~10-15 GB of model weights)"
  & $py @launchArgs
}
