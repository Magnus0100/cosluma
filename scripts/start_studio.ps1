<# Launch a packaged desktop or a configured source checkout. Env files are data. #>
[CmdletBinding()]
param(
    [switch]$Qwen,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
$LaunchRoot = Split-Path -Parent $PSScriptRoot
foreach ($Name in @('.env', '.env.local')) {
    $EnvFile = Join-Path $LaunchRoot $Name
    if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) { continue }
    foreach ($Line in [System.IO.File]::ReadAllLines($EnvFile)) {
        $Entry = $Line.Trim()
        if (-not $Entry -or $Entry.StartsWith('#')) { continue }
        if ($Entry -notmatch '^((?:COSPLAY_[A-Z0-9_]+|OPENAI_API_KEY))\s*=(.*)$') {
            throw "Invalid environment entry in ${EnvFile}. Use literal KEY=VALUE without shell commands."
        }
        $Key = $Matches[1]
        $Value = $Matches[2].Trim()
        if ($Value.Length -ge 2 -and (($Value.StartsWith('"') -and $Value.EndsWith('"')) -or ($Value.StartsWith("'") -and $Value.EndsWith("'")))) {
            $Value = $Value.Substring(1, $Value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($Key, $Value, 'Process')
    }
}
$PythonCandidates = @(
    (Join-Path $LaunchRoot '.venv\Scripts\python.exe'),
    (Join-Path $LaunchRoot '.cache\desktop-runtime\Scripts\python.exe')
)
$Python = $PythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if ($Qwen) {
    if (-not $Python) { throw 'Qwen launcher requires the source Python environment; see docs/QWEN_IMAGE_LOCAL.md.' }
    & $Python -X utf8 (Join-Path $PSScriptRoot 'start_qwen_image.py') @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "Qwen launcher failed ($LASTEXITCODE)." }
    exit 0
}
$ExeCandidates = if ($env:COSPLAY_DESKTOP_EXE) { @($env:COSPLAY_DESKTOP_EXE) } else { @(
    (Join-Path $LaunchRoot 'cosplay-workbench.exe'),
    (Join-Path $LaunchRoot 'dist\cosplay-workbench\cosplay-workbench.exe')
) }
$DesktopExe = $ExeCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if ($env:COSPLAY_DESKTOP_EXE -and -not $DesktopExe) { throw 'COSPLAY_DESKTOP_EXE points to a missing executable.' }
$Workspace = if ($env:COSPLAY_WORKSPACE) { $env:COSPLAY_WORKSPACE } else {
    Join-Path $env:LOCALAPPDATA 'CosplayWorkbench\workspace'
}
$Workspace = [System.IO.Path]::GetFullPath($Workspace)
[System.IO.Directory]::CreateDirectory($Workspace) | Out-Null
$BackendUrl = if ($env:COSPLAY_BACKEND_URL) { $env:COSPLAY_BACKEND_URL } else { 'http://127.0.0.1:8787' }
$LaunchArgs = @('--root', $Workspace, '--backend-url', $BackendUrl, '--embedded-backend', '--auto-start-backend') + $ExtraArgs
if ($DesktopExe) {
    $Program = $DesktopExe
} elseif ($Python) {
    $Program = $Python
    $LaunchArgs = @('-X', 'utf8', (Join-Path $PSScriptRoot 'qt_workbench.py')) + $LaunchArgs
} else {
    throw 'Desktop executable is missing. Build with scripts/package_windows.ps1, or install scripts/requirements-desktop.txt in .venv.'
}
# Quote literal arguments for CreateProcess; environment values are never evaluated.
$QuotedArgs = $LaunchArgs | ForEach-Object {
    $Argument = [regex]::Replace([string]$_, '(\\*)"', '$1$1\"')
    $Argument = [regex]::Replace($Argument, '(\\+)$', '$1$1')
    '"' + $Argument + '"'
}
Start-Process -FilePath $Program -ArgumentList ($QuotedArgs -join ' ') -WorkingDirectory $Workspace -WindowStyle Hidden | Out-Null
