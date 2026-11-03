# Cross-platform entry (Windows): .\run.ps1 --method hybrid --split dev --output output\member_b\dev_hybrid
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
python scripts/run_baseline.py @args
exit $LASTEXITCODE
