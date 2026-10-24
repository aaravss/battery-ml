<#
  ONE command for the whole pipeline (data prep -> splits -> training -> export -> evaluation):

      powershell -ExecutionPolicy Bypass -File run_pipeline.ps1          (or double-click run_pipeline.bat)

  Re-run it whenever you add photos to data\own\<class>\. It is idempotent: downloads are cached, training resumes
  after a crash, and a run whose data/config changed is retrained from scratch automatically.
  Options:  -Background    start detached, log to logs\ and return immediately
            -RefreshWeb    also look for new Wikimedia button-cell files (they need a visual review!)
            -Fresh         delete previous training runs first
  Everything (Python, packages, caches, model weights) stays inside this folder.
#>
param([switch]$Background, [switch]$RefreshWeb, [switch]$Fresh)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
. "$PSScriptRoot\tools\env.ps1"
New-Item -ItemType Directory -Force logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "$PSScriptRoot\logs\pipeline_$stamp.log"

if ($Background) {
    $args2 = @("-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    if ($RefreshWeb) { $args2 += "-RefreshWeb" }
    if ($Fresh) { $args2 += "-Fresh" }
    $p = Start-Process powershell -ArgumentList $args2 -WindowStyle Hidden -PassThru
    Write-Host "Pipeline started in background (PID $($p.Id)). Follow progress in: logs\  (newest pipeline_*.log)"
    exit 0
}

function Log($msg) { $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"; Write-Host $line; Add-Content -Path $log -Value $line -Encoding utf8 }

function Step($name, [string[]]$pyArgs, [int]$retries = 0) {
    for ($attempt = 0; $attempt -le $retries; $attempt++) {
        Log "=== $name (attempt $($attempt + 1)) ==="
        & $env:PY -u @pyArgs 2>&1 | ForEach-Object { "$_" } |
            Where-Object { $_ -notmatch 'oneDNN custom operations|absl::InitializeLog|XNNPACK delegate|^\s*$' } |
            ForEach-Object { Write-Host $_; Add-Content -Path $log -Value $_ -Encoding utf8 }
        if ($LASTEXITCODE -eq 0) { return $true }
        Log "!!! $name failed with exit code $LASTEXITCODE"
    }
    return $false
}

# ---- bootstrap the environment if this is a fresh copy of the folder ----
if (-not (Test-Path $env:PY)) {
    Log "Creating .venv (Python 3.12) with the bundled uv"
    & $env:UV venv .venv --python 3.12 --python-preference only-managed 2>&1 | ForEach-Object { Log "$_" }
    & $env:UV pip install --python $env:PY -r requirements.txt 2>&1 | ForEach-Object { Log "$_" }
    & $env:UV pip install --python $env:PY torch --torch-backend=auto 2>&1 | ForEach-Object { Log "$_" }
}
if ($Fresh -and (Test-Path runs)) { Log "Fresh: removing runs\"; Remove-Item -Recurse -Force runs }

$ok = $true
$fetchArgs = @("pipeline\fetch_public.py"); if ($RefreshWeb) { $fetchArgs += "--refresh-web" }
if (-not (Step "fetch public data" $fetchArgs 1)) { Log "fetch failed; continuing with data already on disk" }
$ok = $ok -and (Step "prepare public data" @("pipeline\prepare_public.py"))
$ok = $ok -and (Step "build leak-free splits" @("pipeline\build_splits.py"))
if ($ok) { if (-not (Step "synthetic button cells (experiment)" @("pipeline\synth_buttons.py"))) { Log "synthetic step failed; baseline only" } }

$mode = (& $env:PY -c "import sys; sys.path.insert(0,'pipeline'); import config; print(config.SYNTHETIC_MODE)").Trim()
$variants = @("baseline"); if ($mode -ne "off") { $variants += "synthetic" }
$trained = @()
foreach ($v in $variants) {
    if (-not $ok) { break }
    # retries resume from the last epoch checkpoint (BackupAndRestore)
    if (Step "train $v" @("pipeline\train.py", "--variant", $v) 2) {
        if (Step "export $v to TFLite" @("pipeline\export_tflite.py", "--variant", $v) 1) { $trained += $v }
    }
}
if ($trained.Count -gt 0) { $ok = (Step "evaluate + calibrate + publish models" @("pipeline\evaluate.py")) -and $ok } else { $ok = $false }

if ($ok) { Log "PIPELINE OK. Deliverables: models\  Metrics: reports\RESULTS.md  Log: $log" }
else { Log "PIPELINE FINISHED WITH ERRORS - see $log"; exit 1 }
