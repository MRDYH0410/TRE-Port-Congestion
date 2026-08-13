$ErrorActionPreference = 'Stop'

$experimentRoot = 'C:\MRDYH\paper\TRE - Port Congestion\8.4 - revise\code 8.4\experiments\5.3-4'
$statusPath = Join-Path $experimentRoot 'logs\continuation_status.json'
$guardPath = Join-Path $experimentRoot 'logs\runtime_guard_status.json'

function Write-GuardStatus {
    param([string]$Status, [string]$Message, [datetime]$Deadline, [int]$ChildPid)
    [ordered]@{
        status = $Status
        message = $Message
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
        deadline_utc = $Deadline.ToUniversalTime().ToString('o')
        child_pid = $ChildPid
    } | ConvertTo-Json | Set-Content -LiteralPath $guardPath -Encoding UTF8
}

$status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
$childPid = [int]$status.child_pid
$limitSeconds = [int]$status.wall_clock_limit_seconds
$started = [datetime]::Parse([string]$status.updated_at).ToUniversalTime()
$deadline = $started.AddSeconds($limitSeconds)
$command = Get-CimInstance Win32_Process -Filter "ProcessId=$childPid" -ErrorAction SilentlyContinue
if (-not $command -or $command.CommandLine -notmatch 'run_5_3_4[.]py') {
    Write-GuardStatus -Status 'GUARD_NOT_ARMED' -Message 'The recorded child is absent or does not match the authorised 5.3.4 runner.' -Deadline $deadline -ChildPid $childPid
    exit 2
}

Write-GuardStatus -Status 'GUARD_ARMED' -Message 'The independent eight-hour wall-clock guard is active.' -Deadline $deadline -ChildPid $childPid
while ((Get-Date).ToUniversalTime() -lt $deadline) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$childPid" -ErrorAction SilentlyContinue
    if (-not $current) {
        Write-GuardStatus -Status 'FINISHED_BEFORE_LIMIT' -Message 'The authorised runner exited before the eight-hour deadline.' -Deadline $deadline -ChildPid $childPid
        exit 0
    }
    Start-Sleep -Seconds 60
}

$all = Get-CimInstance Win32_Process
$targets = New-Object System.Collections.Generic.List[int]
$targets.Add($childPid)
$changed = $true
while ($changed) {
    $changed = $false
    foreach ($process in $all) {
        if ($targets.Contains([int]$process.ParentProcessId) -and -not $targets.Contains([int]$process.ProcessId)) {
            $targets.Add([int]$process.ProcessId)
            $changed = $true
        }
    }
}
foreach ($pidToStop in ($targets | Sort-Object -Descending)) {
    Stop-Process -Id $pidToStop -Force -ErrorAction SilentlyContinue
}
Write-GuardStatus -Status 'TERMINATED_AT_LIMIT' -Message 'The authorised runner and its descendants were terminated at the eight-hour deadline; no duplicate run was started.' -Deadline $deadline -ChildPid $childPid
exit 3
