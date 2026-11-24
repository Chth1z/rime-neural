<#
.SYNOPSIS
    Read-only environment audit for rime-neural installation on Windows Weasel.
.DESCRIPTION
    Detects OS architecture, Weasel install directory & version, PE32+ librime Lua 5.4 export readiness,
    RimeUserDir (registry-first, %APPDATA%\Rime fallback) + non-ASCII warning flag, active schemas,
    installed .gram models, existing rime-neural files, and CPU thread budget. Outputs JSON.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

function Get-WeaselDir {
    $proc = Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($proc) {
        try {
            if ($proc.Path) {
                $pdir = Split-Path $proc.Path -Parent
                if ($pdir -and (Test-Path -LiteralPath (Join-Path $pdir "rime.dll"))) { return (Get-Item -LiteralPath $pdir).FullName }
            }
        } catch {}
    }
    foreach ($regPath in @("HKLM:\SOFTWARE\Rime\Weasel", "HKLM:\SOFTWARE\WOW6432Node\Rime\Weasel", "HKCU:\SOFTWARE\Rime\Weasel")) {
        $prop = Get-ItemProperty -Path $regPath -Name "WeaselRoot" -ErrorAction SilentlyContinue
        if ($prop -and $prop.PSObject.Properties["WeaselRoot"] -and $prop.WeaselRoot) {
            $root = $prop.WeaselRoot
            if (Test-Path -LiteralPath (Join-Path $root "rime.dll")) {
                return (Get-Item -LiteralPath $root).FullName
            }
        }
    }
    $bases = @()
    if ($env:ProgramFiles) { $bases += (Join-Path $env:ProgramFiles "Rime") }
    if (${env:ProgramFiles(x86)}) { $bases += (Join-Path ${env:ProgramFiles(x86)} "Rime") }
    $candidates = @()
    foreach ($base in ($bases | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $base) {
            $candidates += Get-ChildItem -LiteralPath $base -Directory -Filter "weasel-*" -ErrorAction SilentlyContinue |
                Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "rime.dll") }
        }
    }
    $best = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($best) { return $best.FullName }
    return $null
}

function Get-RimeUserDir {
    $prop = Get-ItemProperty -Path "HKCU:\SOFTWARE\Rime\Weasel" -Name "RimeUserDir" -ErrorAction SilentlyContinue
    if ($prop -and $prop.PSObject.Properties["RimeUserDir"] -and $prop.RimeUserDir -and $prop.RimeUserDir.Trim()) {
        return $prop.RimeUserDir.Trim()
    }
    return (Join-Path $env:APPDATA "Rime")
}

function Test-RimeDllExportsLua54 {
    param([string]$DllPath)
    if (-not $DllPath -or -not (Test-Path -LiteralPath $DllPath -PathType Leaf)) { return $false }
    try {
        # Parse PE32+ Export Directory (`AddressOfNames`) with strict bounds checks to verify
        # x64 Lua 5.4 (`lua_newuserdatauv` + all 21 symbols required by `src/lua_api.rs`).
        $bytes = [System.IO.File]::ReadAllBytes($DllPath)
        if ($bytes.Length -lt 0x100 -or $bytes[0] -ne 0x4D -or $bytes[1] -ne 0x5A) { return $false }
        $peOff = [BitConverter]::ToInt32($bytes, 0x3C)
        if ($peOff -lt 0 -or ($peOff + 24) -gt $bytes.Length) { return $false }
        if ($bytes[$peOff] -ne 0x50 -or $bytes[$peOff + 1] -ne 0x45 -or $bytes[$peOff + 2] -ne 0 -or $bytes[$peOff + 3] -ne 0) { return $false }
        $machine = [BitConverter]::ToUInt16($bytes, $peOff + 4)
        if ($machine -ne 0x8664) { return $false } # require x64
        $numSec = [BitConverter]::ToUInt16($bytes, $peOff + 6)
        $optSize = [BitConverter]::ToUInt16($bytes, $peOff + 20)
        $optOff = $peOff + 24
        if ($optSize -lt 120 -or ($optOff + $optSize) -gt $bytes.Length) { return $false }
        $magic = [BitConverter]::ToUInt16($bytes, $optOff)
        if ($magic -ne 0x20B) { return $false } # PE32+
        $numRva = [BitConverter]::ToUInt32($bytes, $optOff + 108)
        if ($numRva -lt 1) { return $false }
        $expRva = [BitConverter]::ToUInt32($bytes, $optOff + 112)
        if ($expRva -eq 0) { return $false }

        $secOff = $optOff + $optSize
        if (($secOff + $numSec * 40) -gt $bytes.Length) { return $false }
        $rvaToOff = {
            param([uint32]$rva)
            for ($i = 0; $i -lt $numSec; $i++) {
                $s = $secOff + $i * 40
                $vAddr = [BitConverter]::ToUInt32($bytes, $s + 12)
                $rSize = [BitConverter]::ToUInt32($bytes, $s + 16)
                $rPtr  = [BitConverter]::ToUInt32($bytes, $s + 20)
                if ($rSize -gt 0 -and $rPtr -gt 0 -and $rva -ge $vAddr) {
                    $delta = [uint64]$rva - [uint64]$vAddr
                    if ($delta -lt [uint64]$rSize) {
                        $off64 = [uint64]$rPtr + $delta
                        if ($off64 -lt [uint64]$bytes.Length) {
                            return [int]$off64
                        }
                    }
                }
            }
            return -1
        }

        $expOff = & $rvaToOff $expRva
        if ($expOff -lt 0 -or ($expOff + 40) -gt $bytes.Length) { return $false }
        $numNames = [BitConverter]::ToUInt32($bytes, $expOff + 24)
        $namesRva = [BitConverter]::ToUInt32($bytes, $expOff + 32)
        $namesOff = & $rvaToOff $namesRva
        if ($namesOff -lt 0) { return $false }

        $needed = [System.Collections.Generic.HashSet[string]]::new([string[]]@(
            "lua_newuserdatauv",
            "lua_version", "lua_checkstack", "lua_gettop",
            "lua_createtable", "lua_pushcclosure",
            "lua_rawget", "lua_rawset", "lua_type",
            "lua_tolstring", "lua_pushlstring",
            "lua_pushboolean", "lua_pushinteger", "lua_pushnumber", "lua_pushnil",
            "lua_toboolean", "lua_tointegerx", "lua_tonumberx",
            "lua_rawlen", "lua_rawgeti", "lua_rawseti", "lua_settop"
        ))
        for ($i = 0; $i -lt $numNames -and $needed.Count -gt 0; $i++) {
            $pos = $namesOff + $i * 4
            if (($pos + 4) -gt $bytes.Length) { break }
            $nameRva = [BitConverter]::ToUInt32($bytes, $pos)
            $nOff = & $rvaToOff $nameRva
            if ($nOff -lt 0 -or $nOff -ge $bytes.Length) { continue }
            $end = $nOff
            while ($end -lt $bytes.Length -and $bytes[$end] -ne 0) { $end++ }
            if ($end -gt $nOff) {
                $sym = [System.Text.Encoding]::ASCII.GetString($bytes, $nOff, $end - $nOff)
                [void]$needed.Remove($sym)
            }
        }
        return ($needed.Count -eq 0)
    } catch {
        return $false
    }
}

$weaselDir = Get-WeaselDir
$rimeDll = if ($weaselDir) { Join-Path $weaselDir "rime.dll" } else { $null }
$sharedDataDir = if ($weaselDir -and (Test-Path -LiteralPath (Join-Path $weaselDir "data"))) { Join-Path $weaselDir "data" } else { $null }
$weaselVersion = if ($weaselDir) { (Split-Path $weaselDir -Leaf) -replace '^weasel-', '' } else { $null }
$luaExportsOk = Test-RimeDllExportsLua54 -DllPath $rimeDll

$userDir = Get-RimeUserDir
$userDirExists = Test-Path -LiteralPath $userDir
$userDirNonAscii = [bool]($userDir -match '[^\x00-\x7F]')

$distVersion = ""
$installYaml = Join-Path $userDir "installation.yaml"
if (Test-Path -LiteralPath $installYaml) {
    $vm = Select-String -LiteralPath $installYaml -Pattern '^\s*distribution_version\s*:\s*[''"]?([^''"#]+)[''"]?' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($vm) { $distVersion = $vm.Matches[0].Groups[1].Value.Trim() }
}

# Detect active schema list (prefer compiled build\default.yaml; fallback to merging default.custom.yaml + default.yaml)
$parseSchemaList = {
    param([string]$filePath)
    $found = [System.Collections.Generic.List[string]]::new()
    if (-not (Test-Path -LiteralPath $filePath)) { return @() }
    $lines = Get-Content -LiteralPath $filePath -Encoding UTF8 -ErrorAction SilentlyContinue
    $inList = $false
    foreach ($line in $lines) {
        if ($line -match '^\s*[''"]?schema_list(?:/\+)?[''"]?\s*:\s*(.*)$') {
            $rest = $Matches[1].Trim()
            if ($rest -match '^\[') {
                foreach ($m in [regex]::Matches($rest, 'schema\s*:\s*[''"]?([A-Za-z0-9_\-]+)[''"]?')) {
                    if (-not $found.Contains($m.Groups[1].Value)) { $found.Add($m.Groups[1].Value) }
                }
                $inList = $false
            } else {
                $inList = $true
            }
            continue
        }
        if ($inList) {
            if ($line -match '^\s*-\s*(?:\{\s*)?schema\s*:\s*[''"]?([A-Za-z0-9_\-]+)[''"]?') {
                if (-not $found.Contains($Matches[1])) { $found.Add($Matches[1]) }
            } elseif ($line -match '^\s*[A-Za-z_''"]' -and $line -notmatch '^\s*#') {
                $inList = $false
            }
        }
    }
    return $found.ToArray()
}

$activeSchemas = @(& $parseSchemaList (Join-Path $userDir "build\default.yaml"))
if ($activeSchemas.Count -eq 0) {
    $customSchemas = @(& $parseSchemaList (Join-Path $userDir "default.custom.yaml"))
    $baseSchemas   = @(& $parseSchemaList (Join-Path $userDir "default.yaml"))
    $customRaw = if (Test-Path -LiteralPath (Join-Path $userDir "default.custom.yaml")) {
        Get-Content -LiteralPath (Join-Path $userDir "default.custom.yaml") -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    } else { "" }
    if ($customRaw -match '(?m)^\s*[''"]?schema_list/\+[''"]?\s*:') {
        $activeSchemas = @(@($baseSchemas) + @($customSchemas) | Where-Object { $_ } | Select-Object -Unique)
    } elseif ($customSchemas.Count -gt 0) {
        $activeSchemas = @($customSchemas)
    } else {
        $activeSchemas = @($baseSchemas)
    }
}

$selectedSchema = $null
$userYaml = Join-Path $userDir "user.yaml"
if (Test-Path -LiteralPath $userYaml) {
    $m = Select-String -LiteralPath $userYaml -Pattern 'previously_selected_schema\s*:\s*[''"]?([A-Za-z0-9_\-]+)[''"]?' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { $selectedSchema = $m.Matches[0].Groups[1].Value }
}

$schemaFiles = if ($userDirExists) {
    @(Get-ChildItem -LiteralPath $userDir -Filter "*.schema.yaml" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name -replace '\.schema\.yaml$', '' })
} else { @() }

$customYamls = if ($userDirExists) {
    @(Get-ChildItem -LiteralPath $userDir -Filter "*.custom.yaml" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
} else { @() }

$gramFiles = [System.Collections.Generic.List[object]]::new()
if ($userDirExists) {
    foreach ($gf in (Get-ChildItem -LiteralPath $userDir -Filter "*.gram" -ErrorAction SilentlyContinue)) {
        $gramFiles.Add([ordered]@{
            name    = $gf.Name
            size_mb = [math]::Round($gf.Length / 1MB, 2)
        })
    }
}

$neuralDir = Join-Path $userDir "neural"
if (-not (Test-Path -LiteralPath $neuralDir) -and $sharedDataDir -and (Test-Path -LiteralPath (Join-Path $sharedDataDir "neural"))) {
    $neuralDir = Join-Path $sharedDataDir "neural"
}
$neuralFiles = [System.Collections.Generic.List[object]]::new()
if (Test-Path -LiteralPath $neuralDir) {
    foreach ($nf in (Get-ChildItem -LiteralPath $neuralDir -File -ErrorAction SilentlyContinue)) {
        $neuralFiles.Add([ordered]@{
            name    = $nf.Name
            size_mb = [math]::Round($nf.Length / 1MB, 2)
        })
    }
}

$luaFilterPath = Join-Path $userDir "lua\rime_neural.lua"
$weaselProcs = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })

$wanxiangHints = @()
foreach ($wf in @("wanxiang.custom.yaml", "wanxiang_pro.custom.yaml", "wanxiang.schema.yaml")) {
    $wp = Join-Path $userDir $wf
    if (Test-Path -LiteralPath $wp) {
        $hits = @(Select-String -LiteralPath $wp -Pattern 'algebra|double_pinyin|flypy|zrm|mspy|sogou|abc|ziguang|pyjj|\u81ea\u7136\u7801|\u5c0f\u9e64|\u5fae\u8f6f|\u641c\u72d7' -ErrorAction SilentlyContinue |
            Select-Object -First 8 | ForEach-Object { "$($wf):$($_.LineNumber): $($_.Line.Trim())" })
        if ($hits.Count -gt 0) { $wanxiangHints += $hits }
    }
}

$result = [ordered]@{
    os_arch                  = $env:PROCESSOR_ARCHITECTURE
    is_64bit_os              = [Environment]::Is64BitOperatingSystem
    logical_processors       = [Environment]::ProcessorCount
    weasel_dir               = $weaselDir
    weasel_shared_data_dir   = $sharedDataDir
    weasel_folder_version    = $weaselVersion
    weasel_dist_version      = $distVersion
    rime_dll_lua54_exports   = $luaExportsOk
    weasel_server_pids       = @($weaselProcs)
    rime_user_dir            = $userDir
    rime_user_dir_exists     = $userDirExists
    rime_user_dir_non_ascii  = $userDirNonAscii
    previously_selected      = $selectedSchema
    active_schema_list       = @($activeSchemas)
    installed_schemas        = @($schemaFiles)
    existing_custom_yamls    = @($customYamls)
    installed_gram_models    = @($gramFiles)
    detected_neural_dir      = $neuralDir
    existing_neural_files    = @($neuralFiles)
    lua_filter_installed     = (Test-Path -LiteralPath $luaFilterPath -PathType Leaf)
    wanxiang_layout_hints    = @($wanxiangHints)
    has_curl                 = [bool](Get-Command curl.exe -ErrorAction SilentlyContinue)
    has_python               = [bool](Get-Command python -ErrorAction SilentlyContinue)
    has_cargo                = [bool](Get-Command cargo -ErrorAction SilentlyContinue)
}

$result | ConvertTo-Json -Depth 5
