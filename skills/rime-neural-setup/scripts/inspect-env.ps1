<#
.SYNOPSIS
    Read-only environment audit for rime-neural installation on Windows Weasel.
.DESCRIPTION
    Detects OS architecture, Weasel install directory & version, librime Lua 5.4 export readiness,
    RimeUserDir (registry-first, %APPDATA%\Rime fallback), active schemas, installed .gram models,
    existing rime-neural files, and CPU thread budget. Outputs structured JSON for agent consumption.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "SilentlyContinue"

function Get-WeaselDir {
    foreach ($regPath in @("HKLM:\SOFTWARE\Rime\Weasel", "HKLM:\SOFTWARE\WOW6432Node\Rime\Weasel", "HKCU:\SOFTWARE\Rime\Weasel")) {
        $root = (Get-ItemProperty -Path $regPath -Name "WeaselRoot" -ErrorAction SilentlyContinue).WeaselRoot
        if ($root -and (Test-Path (Join-Path $root "rime.dll"))) {
            return (Get-Item $root).FullName
        }
    }
    $candidates = @()
    foreach ($base in @("C:\Program Files\Rime", "C:\Program Files (x86)\Rime")) {
        if (Test-Path $base) {
            $candidates += Get-ChildItem $base -Directory -Filter "weasel-*" |
                Where-Object { Test-Path (Join-Path $_.FullName "rime.dll") }
        }
    }
    $best = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($best) { return $best.FullName }
    return $null
}

function Get-RimeUserDir {
    $regDir = (Get-ItemProperty -Path "HKCU:\SOFTWARE\Rime\Weasel" -Name "RimeUserDir" -ErrorAction SilentlyContinue).RimeUserDir
    if ($regDir -and $regDir.Trim()) {
        return $regDir.Trim()
    }
    return (Join-Path $env:APPDATA "Rime")
}

function Test-RimeDllExportsLua54 {
    param([string]$DllPath)
    if (-not $DllPath -or -not (Test-Path $DllPath)) { return $false }
    try {
        # Lightweight binary string scan for exported Lua 5.4 C API symbols in rime.dll
        $bytes = [System.IO.File]::ReadAllBytes($DllPath)
        $ascii = [System.Text.Encoding]::ASCII.GetString($bytes)
        return ($ascii.Contains("lua_pcallk") -and $ascii.Contains("lua_tolstring") -and $ascii.Contains("lua_createtable"))
    } catch {
        return $false
    }
}

$weaselDir = Get-WeaselDir
$rimeDll = if ($weaselDir) { Join-Path $weaselDir "rime.dll" } else { $null }
$weaselVersion = if ($weaselDir) { (Split-Path $weaselDir -Leaf) -replace '^weasel-', '' } else { $null }
$dllVerInfo = if ($rimeDll -and (Test-Path $rimeDll)) { (Get-Item $rimeDll).VersionInfo.FileVersion } else { $null }
$luaExportsOk = Test-RimeDllExportsLua54 -DllPath $rimeDll

$userDir = Get-RimeUserDir
$userDirExists = Test-Path $userDir

# Detect active schema list from default.custom.yaml / default.yaml and last selected schema from user.yaml
$activeSchemas = @()
foreach ($cfgFile in @("default.custom.yaml", "default.yaml")) {
    $p = Join-Path $userDir $cfgFile
    if (Test-Path $p) {
        $lines = Get-Content $p -Encoding UTF8
        $inList = $false
        foreach ($line in $lines) {
            if ($line -match '^\s*"?schema_list(?:/\+)?"?\s*:') { $inList = $true; continue }
            if ($inList) {
                if ($line -match '^\s*-\s*(?:\{\s*)?schema\s*:\s*"?([A-Za-z0-9_\-]+)"?') {
                    if ($activeSchemas -notcontains $Matches[1]) { $activeSchemas += $Matches[1] }
                } elseif ($line -match '^\s*[A-Za-z_"]' -and $line -notmatch '^\s*#') {
                    $inList = $false
                }
            }
        }
        if ($activeSchemas.Count -gt 0) { break }
    }
}

$selectedSchema = $null
$userYaml = Join-Path $userDir "user.yaml"
if (Test-Path $userYaml) {
    $m = Select-String -Path $userYaml -Pattern 'previously_selected_schema\s*:\s*"?([A-Za-z0-9_\-]+)"?' | Select-Object -First 1
    if ($m) { $selectedSchema = $m.Matches[0].Groups[1].Value }
}

# Detect installed schemas, custom yamls, .gram models, and existing rime-neural files
$schemaFiles = if ($userDirExists) {
    @(Get-ChildItem $userDir -Filter "*.schema.yaml" | ForEach-Object { $_.Name -replace '\.schema\.yaml$', '' })
} else { @() }

$customYamls = if ($userDirExists) {
    @(Get-ChildItem $userDir -Filter "*.custom.yaml" | ForEach-Object { $_.Name })
} else { @() }

$gramFiles = [System.Collections.Generic.List[object]]::new()
if ($userDirExists) {
    foreach ($gf in (Get-ChildItem $userDir -Filter "*.gram")) {
        $gramFiles.Add([ordered]@{
            name    = $gf.Name
            size_mb = [math]::Round($gf.Length / 1MB, 2)
        })
    }
}

$neuralDir = Join-Path $userDir "neural"
$neuralFiles = [System.Collections.Generic.List[object]]::new()
if (Test-Path $neuralDir) {
    foreach ($nf in (Get-ChildItem $neuralDir -File)) {
        $neuralFiles.Add([ordered]@{
            name    = $nf.Name
            size_mb = [math]::Round($nf.Length / 1MB, 2)
        })
    }
}

$luaFilterPath = Join-Path $userDir "lua\rime_neural.lua"
$weaselProcs = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })

# Check if rime-wanxiang algebra hints exist (for explicit pinyin_layout guidance)
$wanxiangHints = @()
foreach ($wf in @("wanxiang.custom.yaml", "wanxiang_pro.custom.yaml", "wanxiang.schema.yaml")) {
    $wp = Join-Path $userDir $wf
    if (Test-Path $wp) {
        $hits = @(Select-String -Path $wp -Pattern 'algebra|double_pinyin|flypy|zrm|mspy|sogou|abc|ziguang|pyjj|自然码|小鹤|微软|搜狗' |
            Select-Object -First 8 | ForEach-Object { "$($wf):$($_.LineNumber): $($_.Line.Trim())" })
        if ($hits.Count -gt 0) { $wanxiangHints += $hits }
    }
}

$result = [ordered]@{
    os_arch                  = $env:PROCESSOR_ARCHITECTURE
    is_64bit_os              = [Environment]::Is64BitOperatingSystem
    logical_processors       = [Environment]::ProcessorCount
    weasel_dir               = $weaselDir
    weasel_folder_version    = $weaselVersion
    rime_dll_version         = $dllVerInfo
    rime_dll_lua54_exports   = $luaExportsOk
    weasel_server_pids       = $weaselProcs
    rime_user_dir            = $userDir
    rime_user_dir_exists     = $userDirExists
    previously_selected      = $selectedSchema
    active_schema_list       = $activeSchemas
    installed_schemas        = $schemaFiles
    existing_custom_yamls    = $customYamls
    installed_gram_models    = $gramFiles
    existing_neural_files    = $neuralFiles
    lua_filter_installed     = (Test-Path $luaFilterPath)
    wanxiang_layout_hints    = $wanxiangHints
    has_curl                 = [bool](Get-Command curl.exe -ErrorAction SilentlyContinue)
    has_python               = [bool](Get-Command python -ErrorAction SilentlyContinue)
    has_cargo                = [bool](Get-Command cargo -ErrorAction SilentlyContinue)
}

$result | ConvertTo-Json -Depth 5
