Push-Location $PSScriptRoot
try {
    python scripts/run_baseline.py @args
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
