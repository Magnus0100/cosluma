<#
.SYNOPSIS
    Build the PySide6 one-folder Windows workbench with a verified desktop runtime.

.DESCRIPTION
    The release and debug specs deliberately exclude Poppler's optional ICU
    pair.  Those generic DLL names collide with Qt's ABI in a frozen process;
    document tooling is not part of this desktop package.
#>
[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$DebugOnly,
    [switch]$SkipSmoke,
    [string]$PythonExecutable,
    [string]$SmokeRoot,
    [string]$DistPath,
    [string]$WorkPath
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonCandidates = if ($PythonExecutable) { @($PythonExecutable) } else { @(
    (Join-Path $RepoRoot '.venv\Scripts\python.exe'),
    (Join-Path $RepoRoot '.cache\desktop-runtime\Scripts\python.exe')
) }
$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        continue
    }
    $ProbeOutput = & $Candidate -X utf8 (Join-Path $PSScriptRoot 'verify_desktop_dependencies.py') --root $RepoRoot 2>&1
    if ($LASTEXITCODE -eq 0) {
        $Python = (Resolve-Path -LiteralPath $Candidate).Path
        Write-Host "Verified desktop dependencies: $ProbeOutput"
        break
    }
    throw "The selected desktop runtime failed the pinned dependency/license gate: $ProbeOutput"
}
if (-not $Python) {
    throw 'No project desktop runtime found. Install scripts/requirements-desktop.txt into .venv (Python 3.12 x64), or pass -PythonExecutable.'
}

$BuildArgs = @('-m', 'PyInstaller', '--noconfirm')
if ($DistPath) { $BuildArgs += @('--distpath', $DistPath) }
if ($WorkPath) { $BuildArgs += @('--workpath', $WorkPath) }
$PackageDistRoot = if ($DistPath) { [System.IO.Path]::GetFullPath($DistPath) } else { Join-Path $RepoRoot 'dist' }
if ($Clean) { $BuildArgs += '--clean' }
$Specs = if ($DebugOnly) { @('cosplay-workbench-debug.spec') } else { @('cosplay-workbench-debug.spec', 'cosplay-workbench.spec') }
foreach ($Spec in $Specs) {
    Write-Host "Building $Spec with the verified Python 3.12 x64 runtime"
    & $Python @BuildArgs $Spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed for $Spec ($LASTEXITCODE)" }
}

if (-not $SkipSmoke) {
    $SmokeProjectRoot = if ($SmokeRoot) { [System.IO.Path]::GetFullPath($SmokeRoot) } else { $RepoRoot }
    [System.IO.Directory]::CreateDirectory($SmokeProjectRoot) | Out-Null
    $DebugExe = Join-Path $PackageDistRoot 'cosplay-workbench-debug\cosplay-workbench-debug.exe'
    if (-not (Test-Path -LiteralPath $DebugExe)) { throw "Missing debug executable: $DebugExe" }
    Write-Host 'Running frozen Debug Qt/native smoke'
    & $DebugExe '--root' $SmokeProjectRoot '--smoke'
    if ($LASTEXITCODE -ne 0) { throw "Frozen Qt smoke failed ($LASTEXITCODE)" }
    if (-not $DebugOnly) {
        $ReleaseExe = Join-Path $PackageDistRoot 'cosplay-workbench\cosplay-workbench.exe'
        if (-not (Test-Path -LiteralPath $ReleaseExe)) { throw "Missing release executable: $ReleaseExe" }
        Write-Host 'Running frozen Release Qt/native smoke'
        $ReleaseSmoke = Start-Process -FilePath $ReleaseExe -ArgumentList @('--root', ('"' + $SmokeProjectRoot + '"'), '--smoke') -WindowStyle Hidden -Wait -PassThru
        if ($ReleaseSmoke.ExitCode -ne 0) { throw "Frozen Release Qt/native smoke failed ($($ReleaseSmoke.ExitCode))" }
        Write-Host 'Frozen Release Qt/native smoke exit: 0'
    }
}

Write-Host 'Windows package complete.'

