#Requires -Version 5.1
# Thin wrapper so `.\scripts\deploy.ps1 [local|azure|status|down|teardown] [flags]` works
# from PowerShell. All logic lives in scripts/deploy.py -- one implementation for
# Windows, macOS, Linux and CI. See `.\scripts\deploy.ps1 --help`.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: uv is not installed -- https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
    exit 1
}
& uv run "$PSScriptRoot\deploy.py" @args
exit $LASTEXITCODE
