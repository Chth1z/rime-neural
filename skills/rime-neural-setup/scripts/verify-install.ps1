<#
.SYNOPSIS
    Post-deployment verification for rime-neural on Windows Weasel.
.DESCRIPTION
    Checks installed binaries & models (respecting custom rime_neural/data_dir and Weasel
    shared_data_dir\neural fallback, reporting DLL FileVersion, and warning on non-ASCII Lua
    loadlib paths), WeaselServer process health, active-session Weasel logs (filtering by
    LastWriteTime and per-line glog timestamps), and compiled build/<schema>.schema.yaml filter
    order, switch reset state, and beam settings.
#>
[CmdletBinding()]
param(
    [string]$RimeDir = "",
    [string]$SchemaId = ""
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if (-not $RimeDir) {
    $prop = Get-ItemProperty -Path "HKCU:\SOFTWARE\Rime\Weasel" -Name "RimeUserDir" -ErrorAction SilentlyContinue
    $RimeDir = if ($prop -and $prop.PSObject.Properties["RimeUserDir"] -and $prop.RimeUserDir -and $prop.RimeUserDir.Trim()) {
        $prop.RimeUserDir.Trim()
    } else {
        Join-Path $env:APPDATA "Rime"
    }
}

function Get-WeaselSharedDataDir {
    $proc = Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($proc) {
        try {
            if ($proc.Path) {
                $d = Join-Path (Split-Path $proc.Path -Parent) "data"
                if (Test-Path -LiteralPath $d) { return $d }
            }
        } catch {}
    }
    foreach ($regPath in @("HKLM:\SOFTWARE\Rime\Weasel", "HKLM:\SOFTWARE\WOW6432Node\Rime\Weasel", "HKCU:\SOFTWARE\Rime\Weasel")) {
        $prop = Get-ItemProperty -Path $regPath -Name "WeaselRoot" -ErrorAction SilentlyContinue
        if ($prop -and $prop.PSObject.Properties["WeaselRoot"] -and $prop.WeaselRoot) {
            $d = Join-Path $prop.WeaselRoot "data"
            if (Test-Path -LiteralPath $d) { return $d }
        }
    }
    return $null
}

function Unescape-YamlScalar {
    param([string]$Raw)
    $s = $Raw.Trim()
    if ($s.Length -ge 2 -and $s.StartsWith('"') -and $s.EndsWith('"')) {
        $inner = $s.Substring(1, $s.Length - 2)
        return ($inner -replace '\\\\', '\')
    }
    if ($s.Length -ge 2 -and $s.StartsWith("'") -and $s.EndsWith("'")) {
        $inner = $s.Substring(1, $s.Length - 2)
        return ($inner -replace "''", "'")
    }
    return $s
}

function Parse-GlogLineTimestamp {
    param([string]$Line, [datetime]$ReferenceNow)
    # Matches both `W20261124 21:05:12.123456` (8-digit YYYYMMDD) and `W1124 21:05:12.123456` (4-digit MMDD)
    if ($Line -match '^[IWEF](\d{8})\s+(\d{2}):(\d{2}):(\d{2})') {
        try {
            return [datetime]::ParseExact("$($Matches[1]) $($Matches[2]):$($Matches[3]):$($Matches[4])", "yyyyMMdd HH:mm:ss", $null)
        } catch { return $null }
    }
    if ($Line -match '^[IWEF](\d{2})(\d{2})\s+(\d{2}):(\d{2}):(\d{2})') {
        $month = [int]$Matches[1]; $day = [int]$Matches[2]
        $hour = [int]$Matches[3]; $min = [int]$Matches[4]; $sec = [int]$Matches[5]
        try {
            $dt = New-Object datetime($ReferenceNow.Year, $month, $day, $hour, $min, $sec)
            # Handle Dec -> Jan year rollover
            if ($dt -gt $ReferenceNow.AddDays(1)) { $dt = $dt.AddYears(-1) }
            return $dt
        } catch { return $null }
    }
    return $null
}

$checks = [System.Collections.Generic.List[object]]::new()
$allOk  = $true

function Add-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail)
    if (-not $Ok) { $script:allOk = $false }
    $checks.Add([ordered]@{ name = $Name; ok = $Ok; detail = $Detail })
}

# Resolve target SchemaId before file checks so custom rime_neural/data_dir & model_file can be discovered
if (-not $SchemaId) {
    $userYaml = Join-Path $RimeDir "user.yaml"
    if (Test-Path -LiteralPath $userYaml) {
        $m = Select-String -LiteralPath $userYaml -Pattern 'previously_selected_schema\s*:\s*[''"]?([A-Za-z0-9_\-]+)[''"]?' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($m) { $SchemaId = $m.Matches[0].Groups[1].Value }
    }
}

$builtSchema = if ($SchemaId) { Join-Path $RimeDir "build\$SchemaId.schema.yaml" } else { "" }
$builtRaw = if ($builtSchema -and (Test-Path -LiteralPath $builtSchema)) {
    Get-Content -LiteralPath $builtSchema -Raw -Encoding UTF8
} else { "" }

# Resolve active neural data_dir, optional model_file, and option_name from compiled schema
$neuralDir = Join-Path $RimeDir "neural"
$hasCustomDataDir = $false
$configuredModelFile = ""
$configuredOptionName = "neural_rerank"
if ($builtRaw) {
    $inNeural = $false
    foreach ($line in ($builtRaw -split "`r?`n")) {
        if ($line -match '^(rime_neural|neural_rerank|zhiyi_rerank):\s*$') { $inNeural = $true; continue }
        if ($inNeural -and $line -match '^\S') { $inNeural = $false }
        if ($inNeural -and $line -match '^\s{2}data_dir\s*:\s*(.+?)\s*(?:#.*)?$') {
            $customDir = Unescape-YamlScalar -Raw $Matches[1]
            if ($customDir) {
                $neuralDir = $customDir.Replace("/", "\")
                $hasCustomDataDir = $true
            }
        } elseif ($inNeural -and $line -match '^\s{2}model_file\s*:\s*(.+?)\s*(?:#.*)?$') {
            $mf = Unescape-YamlScalar -Raw $Matches[1]
            if ($mf) { $configuredModelFile = $mf }
        } elseif ($inNeural -and $line -match '^\s{2}option_name\s*:\s*(.+?)\s*(?:#.*)?$') {
            $configuredOptionName = Unescape-YamlScalar -Raw $Matches[1]
        }
    }
}
if (-not $hasCustomDataDir -and -not (Test-Path -LiteralPath (Join-Path $neuralDir "rime_neural.dll") -PathType Leaf)) {
    $sharedDir = Get-WeaselSharedDataDir
    if ($sharedDir -and (Test-Path -LiteralPath (Join-Path $sharedDir "neural\rime_neural.dll") -PathType Leaf)) {
        $neuralDir = Join-Path $sharedDir "neural"
    }
}
$luaFile = Join-Path $RimeDir "lua\rime_neural.lua"

# 1. Required files, non-zero size, DLL VersionInfo, and ASCII loadlib path check
if ($neuralDir -match '[^\x00-\x7F]') {
    Add-Check -Name "path:ascii_loadlib" -Ok $false -Detail "Non-ASCII neural data_dir '$neuralDir' will fail Lua ANSI package.loadlib; configure rime_neural/data_dir to an ASCII path"
}
foreach ($req in @("rime_neural.dll", "onnxruntime.dll", "tokenizer.json", "unified_vocab.json")) {
    $p = Join-Path $neuralDir $req
    $item = if (Test-Path -LiteralPath $p -PathType Leaf) { Get-Item -LiteralPath $p } else { $null }
    $okFile = ($null -ne $item) -and ($item.Length -gt 0)
    $detail = if ($okFile) {
        $mb = "{0:F2} MB" -f ($item.Length / 1MB)
        if ($req -like "*.dll" -and $item.VersionInfo -and $item.VersionInfo.FileVersion) {
            "v$($item.VersionInfo.FileVersion) ($mb)"
        } else {
            $mb
        }
    } elseif ($null -ne $item) {
        "EMPTY (0 bytes): $p"
    } else {
        "MISSING: $p"
    }
    Add-Check -Name "file:$req" -Ok $okFile -Detail $detail
}

$supportedOnnx = @("laya.one.fused.int8g.onnx", "laya.one.int8g.onnx", "laya.one.w4.onnx", "laya.int8g.onnx")
$onnxFiles = @(Get-ChildItem -LiteralPath $neuralDir -Filter "*.onnx" -File -ErrorAction SilentlyContinue | Where-Object { $_.Length -gt 0 })
if ($configuredModelFile) {
    $targetModelPath = Join-Path $neuralDir $configuredModelFile
    $mItem = if (Test-Path -LiteralPath $targetModelPath -PathType Leaf) { Get-Item -LiteralPath $targetModelPath } else { $null }
    $mOk = ($null -ne $mItem) -and ($mItem.Length -gt 0)
    $mDetail = if ($mOk) { "$configuredModelFile ({0:F1} MB)" -f ($mItem.Length / 1MB) } else { "MISSING configured model_file: $targetModelPath" }
    Add-Check -Name "file:onnx_model" -Ok $mOk -Detail $mDetail
} else {
    $onnxDetail = if ($onnxFiles.Count -ge 1) {
        ($onnxFiles | ForEach-Object { "$($_.Name) ({0:F1} MB)" -f ($_.Length / 1MB) }) -join ", "
    } else {
        "MISSING: no non-empty .onnx model in $neuralDir"
    }
    $onnxOk = ($onnxFiles.Count -eq 1) -and ($supportedOnnx -contains $onnxFiles[0].Name)
    if ($onnxFiles.Count -gt 1) {
        $onnxDetail = "CONFLICT ($($onnxFiles.Count) .onnx files may mismatch tokenizer/vocab): $onnxDetail"
    } elseif ($onnxFiles.Count -eq 1 -and ($supportedOnnx -notcontains $onnxFiles[0].Name)) {
        $onnxDetail = "UNSUPPORTED auto-detect filename '$($onnxFiles[0].Name)' (set rime_neural/model_file explicitly or install pumpkin/feather)"
    }
    Add-Check -Name "file:onnx_model" -Ok $onnxOk -Detail $onnxDetail
}

$luaItem = if (Test-Path -LiteralPath $luaFile -PathType Leaf) { Get-Item -LiteralPath $luaFile } else { $null }
$luaOk = ($null -ne $luaItem) -and ($luaItem.Length -gt 0)
$luaDetail = if ($luaOk) { $luaFile } else { "MISSING or empty: $luaFile" }
Add-Check -Name "file:rime_neural.lua" -Ok $luaOk -Detail $luaDetail

# 2. WeaselServer running cleanly as single instance
$procs = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
$pidText = ($procs | ForEach-Object { $_.Id }) -join ","
Add-Check -Name "process:WeaselServer" -Ok ($procs.Count -eq 1) -Detail "count=$($procs.Count) pids=$pidText"

# 3. Inspect compiled build/<SchemaId>.schema.yaml
if (-not $SchemaId) {
    Add-Check -Name "build:schema_resolved" -Ok $false -Detail "Could not infer active schema from user.yaml; pass -SchemaId <schema_id> explicitly"
} elseif ($builtRaw) {
    $filters = [System.Collections.Generic.List[string]]::new()
    $inEngine = $false
    $inFilters = $false
    $inTranslator = $false
    $inSwitches = $false
    $maxSentencesVal = -1
    $switchDeclared = $false
    $switchHasReset1 = $false
    $curSwitchMatches = $false
    foreach ($line in ($builtRaw -split "`r?`n")) {
        if ($line -match '^engine:\s*$') { $inEngine = $true; $inFilters = $false; $inTranslator = $false; $inSwitches = $false; continue }
        if ($line -match '^translator:\s*$') { $inTranslator = $true; $inEngine = $false; $inFilters = $false; $inSwitches = $false; continue }
        if ($line -match '^switches:\s*$') { $inSwitches = $true; $inEngine = $false; $inFilters = $false; $inTranslator = $false; continue }
        if ($line -match '^\S') { $inEngine = $false; $inFilters = $false; $inTranslator = $false; $inSwitches = $false }
        if ($inEngine) {
            if ($line -match '^\s{2}filters:\s*$') { $inFilters = $true; continue }
            if ($inFilters) {
                if ($line -match '^\s{4}-\s*[''"]?([^''"#]+?)[''"]?\s*(?:#.*)?$') {
                    $filters.Add($Matches[1].Trim())
                } elseif ($line -match '^\s{2}\S') {
                    $inFilters = $false
                }
            }
        }
        if ($inTranslator -and $line -match '^\s{2}max_sentences\s*:\s*[''"]?(\d+)[''"]?') {
            $maxSentencesVal = [int]$Matches[1]
        }
        if ($inSwitches -and $configuredOptionName) {
            if ($line -match '^\s{2}-\s*(.*)$') {
                $curSwitchMatches = $false
                if ($Matches[1] -match ('^name\s*:\s*[''"]?' + [regex]::Escape($configuredOptionName) + '[''"]?\s*$')) {
                    $switchDeclared = $true
                    $curSwitchMatches = $true
                }
            } elseif ($line -match ('^\s{4}name\s*:\s*[''"]?' + [regex]::Escape($configuredOptionName) + '[''"]?\s*$')) {
                $switchDeclared = $true
                $curSwitchMatches = $true
            } elseif ($curSwitchMatches -and $line -match '^\s{4}reset\s*:\s*1\b') {
                $switchHasReset1 = $true
            }
        }
    }

    $neuralIndices = @()
    for ($i = 0; $i -lt $filters.Count; $i++) {
        if ($filters[$i] -match '^lua_filter@\*?rime_neural(?:\b|$)') { $neuralIndices += $i }
    }
    $hasFilter = ($neuralIndices.Count -eq 1)
    $filterMsg = if ($neuralIndices.Count -eq 1) {
        "$($filters[$neuralIndices[0]]) mounted once at engine/filters[$($neuralIndices[0])]"
    } elseif ($neuralIndices.Count -gt 1) {
        "DUPLICATE mount: rime_neural appears $($neuralIndices.Count) times in engine/filters"
    } else {
        "NOT FOUND in build/$SchemaId.schema.yaml engine/filters (did you run WeaselDeployer /deploy?)"
    }
    Add-Check -Name "build:${SchemaId}:filter_mounted" -Ok $hasFilter -Detail $filterMsg

    # Verify rime_neural appears BEFORE OpenCC / Traditionalization filters (excluding emoji/dictionary simplifiers)
    if ($neuralIndices.Count -ge 1) {
        $firstNeural = $neuralIndices[0]
        $firstOpencc = -1
        $firstOpenccName = ""
        for ($i = 0; $i -lt $filters.Count; $i++) {
            if ($filters[$i] -match '^(simplifier(?:@(?:traditionalize|trad|s2t\w*|s2hk\w*|t2s\w*|zh_trad|zh_tw|zh_hk|opencc))?$|lua_filter@\*?tw_dual_filter\b|lua_filter@\*?wanxiang\.charset_filter\b)') {
                $firstOpencc = $i
                $firstOpenccName = $filters[$i]
                break
            }
        }
        if ($firstOpencc -ge 0) {
            $orderOk = $firstNeural -lt $firstOpencc
            $orderMsg = if ($orderOk) {
                "rime_neural (idx $firstNeural) precedes $firstOpenccName (idx $firstOpencc)"
            } else {
                "WARNING: rime_neural (idx $firstNeural) appears AFTER $firstOpenccName (idx $firstOpencc); move it before OpenCC!"
            }
            Add-Check -Name "build:${SchemaId}:filter_before_opencc" -Ok $orderOk -Detail $orderMsg
        }
    }

    if ($switchDeclared -and -not $switchHasReset1) {
        Add-Check -Name "build:${SchemaId}:switch_reset" -Ok $true -Detail "Note: switch '$configuredOptionName' has no 'reset: 1'; state persists via user.yaml (ensure it is enabled in F4 menu)"
    }

    $maxSentOk = ($maxSentencesVal -ge 6)
    $curSentLabel = if ($maxSentencesVal -ge 0) { [string]$maxSentencesVal } else { "unset" }
    $maxSentMsg = if ($maxSentOk) {
        "translator/max_sentences=$maxSentencesVal (>= 6)"
    } else {
        "Recommend setting translator/max_sentences: 8 and translator/sentence_cutoff_threshold: 0.3 (current=$curSentLabel)"
    }
    Add-Check -Name "build:${SchemaId}:max_sentences" -Ok $maxSentOk -Detail $maxSentMsg
} else {
    Add-Check -Name "build:${SchemaId}" -Ok $false -Detail "Missing compiled file: $builtSchema (run WeaselDeployer /deploy)"
}

# 4. Scan Weasel logs by LastWriteTime + per-line glog timestamp (covers active sessions > 10 min old)
$logDir = Join-Path $env:TEMP "rime.weasel"
$recentHits = [System.Collections.Generic.List[string]]::new()
$seenLines  = [System.Collections.Generic.HashSet[string]]::new()
if (Test-Path -LiteralPath $logDir) {
    $now = Get-Date
    $cutoff = $now.AddMinutes(-10)
    $activePidRegex = ""
    if ($procs.Count -ge 1) {
        try {
            $startAnchor = ($procs | Sort-Object StartTime -Descending | Select-Object -First 1).StartTime.AddSeconds(-60)
            $cutoff = $startAnchor
        } catch {}
        $activePidRegex = '\.(' + (($procs | ForEach-Object { [regex]::Escape([string]$_.Id) }) -join '|') + ')$'
    }
    $allLogs = @(Get-ChildItem -LiteralPath $logDir -File -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.LastWriteTime -ge $cutoff) -or ($activePidRegex -and $_.BaseName -match $activePidRegex)
        } | Sort-Object LastWriteTime -Descending)

    $warnLogs = @($allLogs | Where-Object { $_.Name -match '\.log\.(WARNING|ERROR|FATAL)\.' } | Select-Object -First 6)
    foreach ($lf in $warnLogs) {
        $hits = Select-String -LiteralPath $lf.FullName -Pattern '^[WEF]\d{4,8}\s.*(rime_neural|lua_filter.*error|loadlib failed)' -ErrorAction SilentlyContinue
        foreach ($h in $hits) {
            $line = $h.Line.Trim()
            $ts = Parse-GlogLineTimestamp -Line $line -ReferenceNow $now
            if ($null -ne $ts -and $ts -lt $cutoff) { continue }
            if ($seenLines.Add($line)) {
                $recentHits.Add("$($lf.Name):$($h.LineNumber): $line")
            }
        }
    }
    $infoLogs = @($allLogs | Where-Object { $_.Name -match '\.log\.INFO\.' } | Select-Object -First 4)
    foreach ($lf in $infoLogs) {
        $hits = Select-String -LiteralPath $lf.FullName -Pattern '^[WEF]\d{4,8}\s.*(rime_neural|lua_filter.*error|loadlib failed)' -ErrorAction SilentlyContinue
        foreach ($h in $hits) {
            $line = $h.Line.Trim()
            $ts = Parse-GlogLineTimestamp -Line $line -ReferenceNow $now
            if ($null -ne $ts -and $ts -lt $cutoff) { continue }
            if ($seenLines.Add($line)) {
                $recentHits.Add("$($lf.Name):$($h.LineNumber): $line")
            }
        }
    }
}
$logMsg = if ($recentHits.Count -eq 0) {
    "No rime_neural errors/warnings in active Weasel session logs"
} else {
    ($recentHits | Select-Object -First 5) -join " | "
}
Add-Check -Name "logs:weasel_warnings" -Ok ($recentHits.Count -eq 0) -Detail $logMsg

$out = [ordered]@{
    ok         = $allOk
    rime_dir   = $RimeDir
    neural_dir = $neuralDir
    schema_id  = $SchemaId
    checks     = @($checks)
}
$out | ConvertTo-Json -Depth 5
if (-not $allOk) { exit 1 }
