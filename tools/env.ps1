# Keeps every cache/download inside the project folder. Dot-source before running anything:
#   . .\tools\env.ps1
$Root = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR          = "$Root\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$Root\.cache\python"
$env:UV_PYTHON_BIN_DIR     = "$Root\.cache\python-bin"
$env:UV_TOOL_DIR           = "$Root\.cache\uv-tools"
$env:PIP_CACHE_DIR         = "$Root\.cache\pip"
$env:KERAS_HOME            = "$Root\.cache\keras"
$env:TORCH_HOME            = "$Root\.cache\torch"
$env:MPLCONFIGDIR          = "$Root\.cache\matplotlib"
$env:TF_CPP_MIN_LOG_LEVEL  = "2"
$env:PYTHONUTF8            = "1"
$env:UV = "$Root\tools\uv\uv.exe"
$env:PY = "$Root\.venv\Scripts\python.exe"
