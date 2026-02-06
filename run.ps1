# Diagram Converter - Run web app and API service
# Usage: .\run.ps1 [-Gpu] [-NoPreload] [-Reload]
# Run from project root: .\run.ps1

param(
    [switch]$Gpu,      # Use GPU for VLM
    [switch]$NoPreload, # Skip VLM preload (faster start, model loads on first request)
    [switch]$Reload    # Enable --reload for development
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Ensure venv exists
if (-not (Test-Path ".venv\Scripts\activate.ps1")) {
    Write-Host "Creating virtual environment..."
    $py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }
    & $py -m venv .venv
}

# Activate and check dependencies
.\.venv\Scripts\Activate.ps1
if (-not (Get-Command "uvicorn" -ErrorAction SilentlyContinue)) {
    Write-Host "Installing dependencies..."
    pip install -r requirements.txt
}

# Env
if ($Gpu) { $env:USE_GPU = "1" }
if ($NoPreload) { $env:SKIP_PRELOAD = "1" }

Write-Host ""
Write-Host "Diagram Converter - Web UI + API Service" -ForegroundColor Cyan
Write-Host "  Web UI:    http://127.0.0.1:8000/" -ForegroundColor Green
Write-Host "  API docs:  http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "  API root:  http://127.0.0.1:8000/api" -ForegroundColor Green
Write-Host ""
if (-not $NoPreload) {
    Write-Host "VLM preload enabled - first start may take 2-5 min" -ForegroundColor Yellow
}
Write-Host ""

$uvicornArgs = @("src.api:app", "--host", "0.0.0.0", "--port", "8000")
if ($Reload) { $uvicornArgs += "--reload" }

& uvicorn @uvicornArgs
