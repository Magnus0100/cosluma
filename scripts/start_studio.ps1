<# Launch a packaged desktop or a configured source checkout. Env files are data. #>
[CmdletBinding()]
param(
    [switch]$Qwen,
    [switch]$PassThru,
    [switch]$Describe,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
# Windows shortcut launches can inherit a minimal module path where utility
# cmdlets are not auto-loaded.  Import the module explicitly before integrity
# checks so the packaged launcher behaves the same as an interactive shell.
Import-Module Microsoft.PowerShell.Utility -ErrorAction Stop

function Get-Sha256Hex([string]$Path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try { return ([System.BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $sha.Dispose() }
}

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
# Never leak the development/agent Python search path into the frozen app or
# its model workers.  Each worker owns its interpreter and package set.
foreach ($Name in @('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE')) {
    [Environment]::SetEnvironmentVariable($Name, $null, 'Process')
}
$ConfigRoot = if ($env:COSPLAY_CONFIG_ROOT) { $env:COSPLAY_CONFIG_ROOT } else { Join-Path $env:USERPROFILE 'CosplayWorkbench\config' }
$ConfigRoot = [System.IO.Path]::GetFullPath($ConfigRoot)
[Environment]::SetEnvironmentVariable('COSPLAY_CONFIG_ROOT', $ConfigRoot, 'Process')
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
$SourceCheckout = $Python -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'qt_workbench.py') -PathType Leaf)
$ExeCandidates = if ($env:COSPLAY_DESKTOP_EXE) { @($env:COSPLAY_DESKTOP_EXE) } elseif ($SourceCheckout) { @() } else { @(
    (Join-Path $LaunchRoot 'cosplay-workbench.exe')
) }
$DesktopExe = $ExeCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if ($env:COSPLAY_DESKTOP_EXE -and -not $DesktopExe) { throw 'COSPLAY_DESKTOP_EXE points to a missing executable.' }
$StartupHealth = $null
if ($DesktopExe) {
    $InstallRoot = Split-Path -Parent $DesktopExe
    [Environment]::SetEnvironmentVariable('COSPLAY_PACKAGE_ROOT', $InstallRoot, 'Process')
    $InstallManifestPath = Join-Path $InstallRoot 'install-manifest.json'
    if (-not (Test-Path -LiteralPath $InstallManifestPath -PathType Leaf)) {
        throw "Standalone installation is incomplete: missing $InstallManifestPath. Reinstall the complete version directory."
    }
    try { $InstallManifest = Get-Content -LiteralPath $InstallManifestPath -Raw | ConvertFrom-Json }
    catch { throw "Standalone installation manifest cannot be read: $InstallManifestPath" }
    if ($InstallManifest.schema -ne 'cosplay_desktop_installation.v1' -or
        $InstallManifest.executable -ne 'cosplay-workbench.exe') {
        throw 'Standalone installation manifest version or executable identity does not match.'
    }
    $ActualExeSha256 = Get-Sha256Hex $DesktopExe
    if ($ActualExeSha256 -ne ([string]$InstallManifest.executable_sha256).ToLowerInvariant()) {
        throw 'Standalone executable integrity check failed; reinstall the complete current directory.'
    }
    if ($InstallManifest.build_identity_sha256) {
        $BuildComments = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($DesktopExe).Comments
        $ExpectedComments = 'Cosplay build ' + $InstallManifest.source_commit + ' ' + $InstallManifest.build_identity_sha256
        if ($BuildComments -ne $ExpectedComments) {
            throw 'Executable build identity differs from the installed source manifest. Reinstall the complete current package.'
        }
    }
    if ($InstallManifest.software_files_sha256) {
        foreach ($Entry in $InstallManifest.software_files_sha256.PSObject.Properties) {
            $Target = [System.IO.Path]::GetFullPath((Join-Path $InstallRoot $Entry.Name))
            if (-not $Target.StartsWith($InstallRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'Installation manifest contains an escaped software path.'
            }
            if (-not (Test-Path -LiteralPath $Target -PathType Leaf) -or (Get-Sha256Hex $Target) -ne $Entry.Value) {
                throw "Installed software file differs from the complete build: $($Entry.Name). Reinstall the current package."
            }
        }
    }
    $RequiredDirectories = @($InstallManifest.required_directories)
    if ($RequiredDirectories.Count -eq 0) { $RequiredDirectories = @('_internal') }
    foreach ($Required in $RequiredDirectories) {
        if (-not (Test-Path -LiteralPath (Join-Path $InstallRoot $Required))) {
            throw "Standalone installation is incomplete: missing directory $Required."
        }
    }
    $PackageRegistry = Join-Path $InstallRoot 'model\local-models.json'
    $SavedRegistrySelection = if ($env:COSPLAY_MODEL_SETTINGS) { $env:COSPLAY_MODEL_SETTINGS } else { Join-Path $ConfigRoot 'model-selection.json' }
    if (-not $env:COSPLAY_MODEL_REGISTRY -and
        -not (Test-Path -LiteralPath $SavedRegistrySelection -PathType Leaf) -and
        (Test-Path -LiteralPath $PackageRegistry -PathType Leaf)) {
        [Environment]::SetEnvironmentVariable('COSPLAY_MODEL_REGISTRY', $PackageRegistry, 'Process')
    }
    $PackageBeautyConfig = Join-Path $InstallRoot 'model\beautygrpo.json'
    $SavedBeautySelection = Join-Path $ConfigRoot 'beautygrpo-selection.json'
    if (-not $env:COSPLAY_BEAUTYGRPO_CONFIG -and
        -not (Test-Path -LiteralPath $SavedBeautySelection -PathType Leaf) -and
        (Test-Path -LiteralPath $PackageBeautyConfig -PathType Leaf)) {
        [Environment]::SetEnvironmentVariable('COSPLAY_BEAUTYGRPO_CONFIG', $PackageBeautyConfig, 'Process')
    }
    # Machine-specific external deployments are data, never baked-in drives.
    if (-not $env:COSPLAY_QWEN_DEPLOY_ROOT) {
        $DeploymentPath = Join-Path $InstallRoot 'model\deployment.json'
        $ConfiguredQwenRoot = $null
        if (Test-Path -LiteralPath $DeploymentPath -PathType Leaf) {
            $Deployment = Get-Content -LiteralPath $DeploymentPath -Raw | ConvertFrom-Json
            if ($Deployment.schema -ne 'cosplay_local_deployment.v1') { throw 'Unsupported local deployment configuration.' }
            if ($Deployment.qwen_deploy_root) {
                $ConfiguredQwenRoot = ([string]$Deployment.qwen_deploy_root).Replace('${PACKAGE_ROOT}', $InstallRoot)
                if (-not [System.IO.Path]::IsPathRooted($ConfiguredQwenRoot)) {
                    $ConfiguredQwenRoot = Join-Path (Split-Path $DeploymentPath) $ConfiguredQwenRoot
                }
            }
        }
        $QwenCandidates = @($ConfiguredQwenRoot, (Join-Path $InstallRoot 'models\qwen-image-2.1')) | Where-Object { $_ }
        $QwenRoot = $QwenCandidates | Where-Object {
            Test-Path -LiteralPath (Join-Path $_ 'ComfyUI') -PathType Container
        } | Select-Object -First 1
        if ($QwenRoot) {
            [Environment]::SetEnvironmentVariable('COSPLAY_QWEN_DEPLOY_ROOT', $QwenRoot, 'Process')
        }
    }
}
function Get-WorkspaceAbsolute([string]$Value, [string]$Label) {
    if (-not $Value -or $Value -match '[\x00-\x1f]') { throw "$Label requires a nonempty filesystem path." }
    $Resolved = [System.IO.Path]::GetFullPath($Value)
    if ((Test-Path -LiteralPath $Resolved) -and -not (Test-Path -LiteralPath $Resolved -PathType Container)) {
        throw "$Label points to a file instead of a workspace directory: $Resolved"
    }
    return $Resolved
}
# Parse before opening the saved pointer: an explicit root must also own the
# health receipt and child working directory, even if the pointer is invalid.
$ExtraLaunchArgs = @($ExtraArgs | Where-Object { $_ -ne $null })
$NormalizedExtraArgs = @()
$ExplicitWorkspace = $null
for ($Index = 0; $Index -lt $ExtraLaunchArgs.Count; $Index++) {
    $Argument = $ExtraLaunchArgs[$Index]
    if ($Argument -eq '--root') {
        if ($Index + 1 -ge $ExtraLaunchArgs.Count -or -not $ExtraLaunchArgs[$Index + 1] -or $ExtraLaunchArgs[$Index + 1].StartsWith('--')) {
            throw 'Explicit --root requires a workspace path.'
        }
        $Index++
        $ExplicitWorkspace = Get-WorkspaceAbsolute $ExtraLaunchArgs[$Index] 'Explicit --root'
        $NormalizedExtraArgs += @('--root', $ExplicitWorkspace)
    } elseif ($Argument.StartsWith('--root=')) {
        $ExplicitWorkspace = Get-WorkspaceAbsolute $Argument.Substring(7) 'Explicit --root'
        $NormalizedExtraArgs += @('--root', $ExplicitWorkspace)
    } else { $NormalizedExtraArgs += $Argument }
}
$ExtraLaunchArgs = $NormalizedExtraArgs
$WorkspacePointer = Join-Path $ConfigRoot 'workspace-location.json'
$Workspace = if ($ExplicitWorkspace) { $ExplicitWorkspace } elseif ($env:COSPLAY_WORKSPACE) {
    Get-WorkspaceAbsolute $env:COSPLAY_WORKSPACE 'COSPLAY_WORKSPACE'
} elseif (Test-Path -LiteralPath $WorkspacePointer) {
    try {
        $Location = [System.IO.File]::ReadAllText($WorkspacePointer) | ConvertFrom-Json
        $LocationFields = @($Location.PSObject.Properties.Name)
        if ($Location -is [array] -or $LocationFields.Count -ne 2 -or
            $LocationFields -cnotcontains 'schema' -or $LocationFields -cnotcontains 'path' -or
            $Location.schema -isnot [string] -or $Location.schema -cne 'cosplay_workspace_location.v1' -or
            $Location.path -isnot [string] -or -not $Location.path -or $Location.path.Trim() -cne $Location.path -or
            ($Location.path -notmatch '^[A-Za-z]:[\\/]' -and $Location.path -notmatch '^\\\\[^\\/]+[\\/][^\\/]+(?:[\\/]|$)')) {
            throw 'Expected schema cosplay_workspace_location.v1 and an absolute path.'
        }
        Get-WorkspaceAbsolute $Location.path 'Saved workspace'
    } catch { throw "Invalid workspace location configuration ${WorkspacePointer}: $($_.Exception.Message)" }
} else { Join-Path $env:USERPROFILE 'CosplayWorkbench\workspace' }
$Workspace = [System.IO.Path]::GetFullPath($Workspace)
[System.IO.Directory]::CreateDirectory($Workspace) | Out-Null
if ($DesktopExe) {
    # Write a cheap, user-readable installation health receipt before launch.
    # It never loads weights or blocks the UI; the backend performs the same
    # capability checks before a task starts.
    $Health = [ordered]@{
        schema = 'cosplay_desktop_startup_health.v1'
        status = 'configured'
        verification_scope = 'software_hashes_and_configuration_paths'
        inference_verified = $false
        package_root = $InstallRoot
        executable = $DesktopExe
        manifest = $InstallManifestPath
        registry = $null
        beauty_config = $null
        missing = @()
        source_checkout_required = $false
        source_commit = $InstallManifest.source_commit
        qwen_deployment = $env:COSPLAY_QWEN_DEPLOY_ROOT
        external_paths = @()
    }
    $SavedRegistrySelection = if ($env:COSPLAY_MODEL_SETTINGS) { $env:COSPLAY_MODEL_SETTINGS } else { Join-Path $ConfigRoot 'model-selection.json' }
    $RegistryPath = $env:COSPLAY_MODEL_REGISTRY
    if (-not $RegistryPath -and (Test-Path -LiteralPath $SavedRegistrySelection -PathType Leaf)) {
        try {
            $SavedRegistry = Get-Content $SavedRegistrySelection -Raw | ConvertFrom-Json
            if ($SavedRegistry.prefer_user -eq $true -or -not (Test-Path -LiteralPath $PackageRegistry -PathType Leaf)) {
                $RegistryPath = $SavedRegistry.registry_path
            }
        } catch { }
    }
    if (-not $RegistryPath) { $RegistryPath = Join-Path $InstallRoot 'model\local-models.json' }
    $Health.registry = $RegistryPath
    $ConfigPaths = @($RegistryPath, $env:COSPLAY_BEAUTYGRPO_CONFIG)
    $SavedBeautySelection = Join-Path $ConfigRoot 'beautygrpo-selection.json'
    if (-not $ConfigPaths[1] -and (Test-Path -LiteralPath $SavedBeautySelection -PathType Leaf)) {
        try { $ConfigPaths[1] = (Get-Content $SavedBeautySelection -Raw | ConvertFrom-Json).model_config } catch { }
    }
    if (-not $ConfigPaths[1]) { $ConfigPaths[1] = Join-Path $InstallRoot 'model\beautygrpo.json' }
    $Health.beauty_config = $ConfigPaths[1]
    foreach ($ConfigPath in $ConfigPaths) {
        if (-not $ConfigPath -or -not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
            if ($ConfigPath) { $Health.missing += $ConfigPath }
            continue
        }
        try {
            $Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
            if ($Config.schema -eq 'beautygrpo_installation_pointer.v1') {
                $Pointer = ([string]$Config.installation).Replace('${PACKAGE_ROOT}', $InstallRoot)
                if (-not [System.IO.Path]::IsPathRooted($Pointer)) { $Pointer = Join-Path (Split-Path $ConfigPath) $Pointer }
                $ConfigPath = [System.IO.Path]::GetFullPath($Pointer)
                $Health.beauty_config = $ConfigPath
                $Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
            }
            if ($Config.schema -notin @('cosplay_local_models.current.v1', 'beautygrpo_models.v1') -or $Config.ready -ne $true) {
                $Health.missing += "Invalid model configuration: $ConfigPath"
            }
            $ResourceKeys = @('python', 'semantics_checkpoint', 'semantics_repo_dir', 'vitmatte_model_root',
                'dinov3_model_root', 'scene_python', 'scene_model_root', 'birefnet_model_path',
                'lama_checkpoint', 'background_sr_checkpoint', 'face_model_path', 'face_python',
                'onnx_runtime_root', 'base_model_root', 'lora_path', 'base_manifest',
                'regions_python', 'regions_model_root', 'face_registry')
            foreach ($Property in $Config.PSObject.Properties) {
                if ($Property.Value -is [string] -and $Property.Name -in $ResourceKeys) {
                    $Resource = $Property.Value.Replace('${PACKAGE_ROOT}', $InstallRoot)
                    if (-not [System.IO.Path]::IsPathRooted($Resource)) { $Resource = Join-Path (Split-Path $ConfigPath) $Resource }
                    $Resource = [System.IO.Path]::GetFullPath($Resource)
                    if (-not (Test-Path -LiteralPath $Resource)) { $Health.missing += $Resource }
                    if (-not $Resource.StartsWith($InstallRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                        $Health.external_paths += $Resource
                    }
                }
            }
        } catch { $Health.missing += $ConfigPath }
    }
    if ($Health.missing.Count -gt 0) { $Health.status = 'degraded' }
    elseif ($Health.external_paths.Count -gt 0) { $Health.status = 'configured_external_models' }
    $Health | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Workspace 'installation-health.json') -Encoding utf8
    $StartupHealth = $Health
}
$BackendUrl = if ($env:COSPLAY_BACKEND_URL) { $env:COSPLAY_BACKEND_URL } else { 'http://127.0.0.1:8787' }
# An omitted ValueFromRemainingArguments parameter can be represented as a
# one-element null array by Windows PowerShell.  Start-Process rejects null
# argument elements, so normalise it before constructing the launch vector.
$LaunchArgs = @()
if (-not ($ExtraLaunchArgs | Where-Object { $_ -eq '--root' -or $_.StartsWith('--root=') })) {
    $LaunchArgs += @('--root', $Workspace)
}
if (-not ($ExtraLaunchArgs | Where-Object { $_ -eq '--backend-url' -or $_.StartsWith('--backend-url=') })) {
    $LaunchArgs += @('--backend-url', $BackendUrl)
}
$LaunchArgs += @('--embedded-backend', '--auto-start-backend', '--prewarm') + $ExtraLaunchArgs
if ($DesktopExe) {
    $Program = $DesktopExe
} elseif ($Python) {
    $Program = $Python
    $LaunchArgs = @('-X', 'utf8', (Join-Path $PSScriptRoot 'qt_workbench.py')) + $LaunchArgs
} else {
    throw 'Desktop executable is missing. Build with scripts/package_windows.ps1, or install scripts/requirements-desktop.txt in .venv.'
}
# Start-Process joins its array with spaces, without quoting each element.
# Apply Windows CRT quoting explicitly, including quotes and trailing slashes.
function ConvertTo-NativeArgument([string]$Value) {
    $Escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $Escaped = [regex]::Replace($Escaped, '(\\+)$', '$1$1')
    return '"' + $Escaped + '"'
}
$CommandLine = ($LaunchArgs | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
if ($Describe) {
    [pscustomobject]@{ program = $Program; arguments = $LaunchArgs; workspace = $Workspace; installation_health = $StartupHealth } | ConvertTo-Json -Depth 6
    return
}
# Qt restores the selected task/provider and owns service initialization.
# This is the user-facing Qt application.  Hiding the process window here
# leaves the executable alive with no visible main window, which makes the
# desktop shortcut appear broken.  Keep the application window visible.
$LaunchedProcess = Start-Process -FilePath $Program -ArgumentList $CommandLine -WorkingDirectory $Workspace -PassThru
if ($PassThru) { $LaunchedProcess }
