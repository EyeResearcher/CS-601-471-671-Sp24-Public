$ErrorActionPreference = 'Stop'
$taskRunName = Get-Date -Format 'yyyyMMdd-HHmmss'
$taskRunDir = Join-Path $PSScriptRoot "runs\$taskRunName"
New-Item -ItemType Directory -Path $taskRunDir -Force | Out-Null
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskProcess = Start-Process -FilePath $taskPython -WorkingDirectory $PSScriptRoot `
    -ArgumentList @('-u', 'train_hw3.py', '--output', "runs/$taskRunName") `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRunDir 'training.log') `
    -RedirectStandardError (Join-Path $taskRunDir 'progress.log') -PassThru
$taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskRunDir 'process.pid')
$taskRunDir | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'runs\latest.txt')
Write-Output "Started process $($taskProcess.Id). Results: $taskRunDir"
