# Cross-platform entry (Windows): .\run.ps1 --method bm25 --split dev --output output\bm25\dev_bm25
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
python scripts/run_baseline.py @args
exit $LASTEXITCODE
