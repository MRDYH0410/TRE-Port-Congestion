param(
    [Parameter(Mandatory = $true)]
    [int]$GateRunnerPid
)

$ErrorActionPreference = "Stop"
$experimentDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeRoot = (Resolve-Path (Join-Path $experimentDir "..\..")).Path
$python = Join-Path $codeRoot ".venv\Scripts\python.exe"
$stdout = Join-Path $experimentDir "run_layered.stdout.log"
$stderr = Join-Path $experimentDir "run_layered.stderr.log"
$gateResult = Join-Path $experimentDir "gate_results\eight_path_computational_gate.json"
$continuation = Join-Path $experimentDir "run_layered.continuation.json"

Wait-Process -Id $GateRunnerPid

$payload = [ordered]@{
    gate_runner_pid = $GateRunnerPid
    gate_completed_at = (Get-Date).ToUniversalTime().ToString("o")
    formal_started = $false
    formal_exit_code = $null
    status = "GATE_NOT_ACCEPTED"
}

if (-not (Test-Path -LiteralPath $gateResult)) {
    Add-Content -LiteralPath $stderr -Value "[5.3.2] Gate result is missing; formal run was not started."
    $payload | ConvertTo-Json | Set-Content -LiteralPath $continuation -Encoding utf8
    exit 1
}

$gate = Get-Content -LiteralPath $gateResult -Raw | ConvertFrom-Json
if ($gate.status -ne "PASS") {
    Add-Content -LiteralPath $stderr -Value "[5.3.2] Eight-path gate did not pass; formal run was not started."
    $payload | ConvertTo-Json | Set-Content -LiteralPath $continuation -Encoding utf8
    exit 1
}

$payload.formal_started = $true
$payload.status = "FORMAL_RUNNING"
$payload | ConvertTo-Json | Set-Content -LiteralPath $continuation -Encoding utf8
Add-Content -LiteralPath $stdout -Value "[5.3.2] Eight-path gate accepted; starting the 88-path formal run."

& $python "experiments\5.3-2\run_5_3_2.py" "--phase" "formal" "--workers" "12" 1>> $stdout 2>> $stderr
$payload.formal_exit_code = $LASTEXITCODE
$payload.formal_completed_at = (Get-Date).ToUniversalTime().ToString("o")
$payload.status = if ($LASTEXITCODE -eq 0) { "FORMAL_COMPLETE" } else { "FORMAL_FAILED" }
$payload | ConvertTo-Json | Set-Content -LiteralPath $continuation -Encoding utf8
exit $LASTEXITCODE
