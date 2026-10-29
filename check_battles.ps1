# check_battles.ps1 -- GPU-free check that self-play battles are actually running.
# Run from anywhere:   powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\JustinCase\Documents\Projects\showdown AI\check_battles.ps1"
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stats = Join-Path $root 'bot\checkpoints\stats.jsonl'
$log   = Join-Path $root 'bot\run_gpu.log'

if (-not (Test-Path $stats)) { Write-Host "[NO STATS] $stats not found -- has a run started yet?" -ForegroundColor Yellow; exit 1 }

$s   = Get-Item $stats
$age = [int]((Get-Date) - $s.LastWriteTime).TotalSeconds
$last = (Get-Content $stats | Select-Object -Last 1) | ConvertFrom-Json
$b   = $last.battles
$node = (Get-Process node -ErrorAction SilentlyContinue).Count
$py   = (Get-Process python -ErrorAction SilentlyContinue).Count

Write-Host "battles so far : $b"
Write-Host "stats age      : ${age}s  (updated at $($s.LastWriteTime.ToString('HH:mm:ss')))"
Write-Host "node simulators: $node   (expect ~16 while running)"
Write-Host "python procs   : $py"
Write-Host ""

if ($age -lt 90) {
    Write-Host "[LIVE] battles are running" -ForegroundColor Green
} elseif ($age -lt 600) {
    Write-Host "[SLOW] no stats update for ${age}s -- could be a long battle or a stall; tail the log:" -ForegroundColor Yellow
    Write-Host "  Get-Content '$log' -Tail 15"
} else {
    Write-Host "[STOPPED?] stats stale for ${age}s -- the run likely ended or crashed. Check:" -ForegroundColor Red
    Write-Host "  Get-Content '$log' -Tail 25"
}
