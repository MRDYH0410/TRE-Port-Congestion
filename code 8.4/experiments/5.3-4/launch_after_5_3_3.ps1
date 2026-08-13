$ErrorActionPreference = 'Stop'

$codeRoot = 'C:\MRDYH\paper\TRE - Port Congestion\8.4 - revise\code 8.4'
$experimentRoot = Join-Path $codeRoot 'experiments\5.3-4'
$logRoot = Join-Path $experimentRoot 'logs'
$statusPath = Join-Path $logRoot 'continuation_status.json'
$stdoutPath = Join-Path $logRoot 'formal_run_v2.stdout.log'
$stderrPath = Join-Path $logRoot 'formal_run_v2.stderr.log'
$pythonPath = Join-Path $codeRoot '.venv\Scripts\python.exe'
$runnerPath = Join-Path $experimentRoot 'run_5_3_4.py'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Write-ContinuationStatus {
    param(
        [string]$Status,
        [string]$Message,
        [Nullable[int]]$ChildPid = $null,
        [Nullable[int]]$ExitCode = $null
    )
    $payload = [ordered]@{
        status = $Status
        message = $Message
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
        watcher_pid = $PID
        child_pid = $ChildPid
        exit_code = $ExitCode
        wall_clock_limit_seconds = 28800
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
    }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

Write-ContinuationStatus -Status 'WAITING_FOR_5_3_3' -Message 'Checking that every run_5_3_3.py process has exited; no 5.3.4 simulation has started.'
while ($true) {
    $activeGateway = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run_5_3_3[.]py'
    }
    if (-not $activeGateway) {
        break
    }
    Start-Sleep -Seconds 60
}

$existingRobustness = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run_5_3_4[.]py'
}
if ($existingRobustness) {
    Write-ContinuationStatus -Status 'NOT_STARTED_DUPLICATE_GUARD' -Message 'A 5.3.4 process already exists; this watcher did not start another simulation.'
    exit 0
}

$gatewayAcceptance = Join-Path $codeRoot 'output\5.3.3_gateway_network_sensitivity\acceptance_5_3_3.json'
$gatewayNote = if (Test-Path -LiteralPath $gatewayAcceptance) {
    '5.3.3 process exited and its acceptance JSON exists.'
} else {
    '5.3.3 process exited without an acceptance JSON; 5.3.4 remains scientifically independent and is starting with its own upstream locks.'
}
Write-ContinuationStatus -Status 'STARTING_5_3_4' -Message $gatewayNote

$runnerArguments = '"' + $runnerPath + '" --phase all'
$child = Start-Process -FilePath $pythonPath `
    -ArgumentList $runnerArguments `
    -WorkingDirectory $codeRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru
Write-ContinuationStatus -Status 'RUNNING_5_3_4' -Message 'The single authorised 5.3.4 matched-training, gate, and formal process is running under the eight-hour contract.' -ChildPid $child.Id
$child.WaitForExit()
$child.Refresh()
$exitCode = $child.ExitCode
$finalStatus = if ($exitCode -eq 0) { 'FINISHED_5_3_4' } else { 'FAILED_5_3_4' }
Write-ContinuationStatus -Status $finalStatus -Message 'The 5.3.4 process exited; inspect formal logs and acceptance outputs.' -ChildPid $child.Id -ExitCode $exitCode
exit $exitCode
