<#
  launch_ui.ps1 - hardened launcher for the Hunyuan3D-2 web UI.

  Handles every failure mode seen in practice:
    * kills stale gen3d/gradio python and frees the port
    * ensures enough free VRAM (auto-unloads Ollama models if a GPU is starved)
    * picks the GPU with the most free memory
    * launches the server and VERIFIES a real HTTP 200 serving the Hunyuan UI
    * opens the browser to 127.0.0.1 (NOT localhost - avoids the Windows IPv6/::1
      'connection refused' trap)
    * retries the whole launch if it fails

  Usage:   powershell -ExecutionPolicy Bypass -File .\launch_ui.ps1
           .\launch_ui.ps1 -Port 8080 -NoText23d -MinFreeMiB 16000
#>
[CmdletBinding()]
param(
  [int]$Port = 8080,
  [int]$MinFreeMiB = 16000,     # free VRAM a GPU needs before we trust it
  [int]$Attempts = 2,
  [int]$TimeoutSec = 300,
  [switch]$NoText23d
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$py = "C:\Users\reese\miniconda3\envs\gen3d\python.exe"
$urlIPv4 = "http://127.0.0.1:$Port/"
function Info($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[warn] $m" -ForegroundColor Yellow }

function Get-GpuFree {
  $rows = nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>$null
  $rows | ForEach-Object { $p = $_ -split ',\s*'; [pscustomobject]@{ idx=[int]$p[0]; free=([int]$p[2]-[int]$p[1]) } } | Sort-Object free -Descending
}

function Kill-Gen3d {
  Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*gen3d*' } | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
}

function Free-Ollama {
  Info "Freeing Ollama VRAM (models reload on demand)..."
  $names = (ollama ps 2>$null | Select-Object -Skip 1) | ForEach-Object { ($_ -split '\s{2,}')[0] } | Where-Object { $_ }
  foreach($n in $names){ ollama stop $n 2>&1 | Out-Null }
  Start-Sleep -Seconds 4
}

function Test-Up {
  try { $r = Invoke-WebRequest -Uri $urlIPv4 -UseBasicParsing -TimeoutSec 8
        return ($r.StatusCode -eq 200 -and $r.Content -match 'Hunyuan') } catch { return $false }
}

for($attempt=1; $attempt -le $Attempts; $attempt++){
  Info "Launch attempt $attempt of $Attempts"
  Kill-Gen3d
  # free the port if something else holds it
  try { (Get-NetTCPConnection -LocalPort $Port -ErrorAction Stop | Select-Object -Expand OwningProcess -Unique) | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } } catch {}

  # ensure a GPU has enough free VRAM
  $best = Get-GpuFree | Select-Object -First 1
  if($best.free -lt $MinFreeMiB){
    Warn "Best GPU only has $($best.free) MiB free (< $MinFreeMiB); freeing Ollama."
    Free-Ollama
    $best = Get-GpuFree | Select-Object -First 1
  }
  Info "Using GPU $($best.idx) ($($best.free) MiB free)"
  $env:CUDA_VISIBLE_DEVICES = "$($best.idx)"

  $log = "$repo\uiserver.log"; $errlog = "$repo\uiserver.err.log"
  Remove-Item $log,$errlog -Force -ErrorAction SilentlyContinue
  $a = @('gradio_app.py','--model_path','tencent/Hunyuan3D-2','--subfolder','hunyuan3d-dit-v2-0',
         '--texgen_model_path','tencent/Hunyuan3D-2','--host','0.0.0.0','--port',"$Port")
  if(-not $NoText23d){ $a += '--enable_t23d' }
  $proc = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo `
            -RedirectStandardOutput $log -RedirectStandardError $errlog -PassThru -WindowStyle Hidden
  Info "Server PID $($proc.Id); waiting up to ${TimeoutSec}s for a verified HTTP 200..."

  $deadline = (Get-Date).AddSeconds($TimeoutSec); $ok=$false
  while((Get-Date) -lt $deadline){
    if(Test-Up){ $ok=$true; break }
    if(-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)){ Warn "Process died during startup."; break }
    Start-Sleep -Seconds 5
  }

  if($ok){
    Start-Process $urlIPv4
    Write-Host ""
    Write-Host "  UI VERIFIED LIVE:  $urlIPv4   (PID $($proc.Id), GPU $($best.idx))" -ForegroundColor Green
    Write-Host "  Use 127.0.0.1 (NOT localhost) - localhost may resolve to IPv6 ::1 and refuse." -ForegroundColor Green
    exit 0
  }
  Warn "Attempt $attempt failed. Last error log:"
  Get-Content $errlog -Tail 8 -ErrorAction SilentlyContinue
  Kill-Gen3d
}

Write-Host "[error] Could not bring the UI up after $Attempts attempts. See $repo\uiserver.err.log" -ForegroundColor Red
exit 1
