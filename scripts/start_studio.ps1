<# Start the source desktop with the user's selected existing Python. #>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = $env:COSLUMA_PYTHON,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
$DesktopExe = Join-Path (Split-Path $PSScriptRoot -Parent) 'CosLuma.exe'
if (Test-Path -LiteralPath $DesktopExe) {
    & $DesktopExe @ExtraArgs
    exit $LASTEXITCODE
}
if (-not $Python) {
    $SettingsPath = if ($env:COSLUMA_CONFIG) { $env:COSLUMA_CONFIG } else { Join-Path $env:USERPROFILE 'CosplayWorkbench\config\cosluma.json' }
    if (Test-Path -LiteralPath $SettingsPath) {
        $Python = (Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json).python
    }
}
if (-not $Python) {
    throw 'Source launch requires an existing Python. Set COSLUMA_PYTHON to its python.exe path, or pass -Python <path>. See README.'
}
& $Python -B (Join-Path $PSScriptRoot 'qt_workbench.py') @ExtraArgs
exit $LASTEXITCODE
