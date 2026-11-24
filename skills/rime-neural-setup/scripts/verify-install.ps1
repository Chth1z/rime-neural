<#
.SYNOPSIS
    Post-deployment verification for rime-neural on Windows Weasel.
.DESCRIPTION
    Checks installed binaries & models (respecting custom rime_neural/data_dir and warning on
    non-ASCII Lua loadlib paths), WeaselServer process health, recent Weasel logs anchored to the
    active WeaselServer session, and compiled build/<schema>.schema.yaml filter order & beam settings.
#>
[CmdletBinding()]
param(
    [string]$RimeDir = "",
    [string]$SchemaId = ""
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if (-not $RimeDir) {
    $regDir = (Get-ItemProperty -Path "HKCU:\SOFTWARE\Rime\Weasel" -Name "RimeUserDir" -ErrorAction SilentlyContinue).RimeUserDir
    $RimeDir = if ($regDir -and $regDir.Trim()) { $regDir.Trim() } else { Join-Path $env:APPDATA "Rime" }
}

$checks = [System.Collections.Generic.List[object]]::new()
$allOk  = $true

function Add-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail)
    if (-not $Ok) { $script:allOk = $false }
    $checks.Add([ordered]@{ name = $Name; ok = $Ok; detail = $Detail })
}

# Resolve target SchemaId before file checks so custom rime_neural/data_dir can be discovered
if (-not $SchemaId) {
    $userYaml = Join-Path $RimeDir "user.yaml"
    if (Test-Path -LiteralPath $userYaml) {
        $m = Select-String -LiteralPath $userYaml -Pattern 'previously_selected_schema\s*:\s*[''"]?([A-Za-z0-9_\-]+)[''"]?' | Select-Object -First 1
        if ($m) { $SchemaId = $m.Matches[0].Groups[1].Value }
    }
}

$builtSchema = if ($SchemaId) { Join-Path $RimeDir "build\$SchemaId.schema.yaml" } else { "" }
$builtRaw = if ($builtSchema -and (Test-Path -LiteralPath $builtSchema)) {
    Get-Content -LiteralPath $builtSchema -Raw -Encoding UTF8
} else { "" }

# Resolve active neural data_dir (respecting rime_neural/data_dir override if configured)
$neuralDir = Join-Path $RimeDir "neural"
if ($builtRaw) {
    $inNeural = $false
    foreach ($line in ($builtRaw -split "`r?`n")) {
        if ($line -match '^(rime_neural|neural_rerank|zhiyi_rerank):\s*$') { $inNeural = $true; continue }
        if ($inNeural -and $line -match '^\S') { $inNeural = $false }
        if ($inNeural -and $line -match '^\s{2}data_dir\s*:\s*[''"]?([^''"#]+?)[''"]?\s*(?:#.*)?$') {
            $customDir = $Matches[1].Trim()
            if ($customDir) { $neuralDir = $customDir }
            break
        }
    }
}
$luaFile = Join-Path $RimeDir "lua\rime_neural.lua"

# 1. Required files & ASCII loadlib path check
if ($neuralDir -match '[^\x00-\x7F]') {
    Add-Check -Name "path:ascii_loadlib" -Ok $false -Detail "Non-ASCII neural data_dir '$neuralDir' will fail Lua ANSI package.loadlib; configure rime_neural/data_dir to an ASCII path"
}
foreach ($req in @("rime_neural.dll", "onnxruntime.dll", "tokenizer.json", "unified_vocab.json")) {
    $p = Join-Path $neuralDir $req
    $exists = Test-Path -LiteralPath $p
    $detail = if ($exists) { "{0:F2} MB" -f ((Get-Item -LiteralPath $p).Length / 1MB) } else { "MISSING: $p" }
    Add-Check -Name "file:$req" -Ok $exists -Detail $detail
}

$onnxFiles = @(Get-ChildItem -LiteralPath $neuralDir -Filter "*.onnx" -ErrorAction SilentlyContinue)
$onnxDetail = if ($onnxFiles.Count -ge 1) {
    ($onnxFiles | ForEach-Object { "$($_.Name) ({0:F1} MB)" -f ($_.Length / 1MB) }) -join ", "
} else {
    "MISSING: no .onnx model in $neuralDir"
}
Add-Check -Name "file:onnx_model" -Ok ($onnxFiles.Count -eq 1) -Detail $(if ($onnxFiles.Count -gt 1) { "CONFLICT ($($onnxFiles.Count) .onnx files may mismatch tokenizer/vocab): $onnxDetail" } else { $onnxDetail })

$luaDetail = if (Test-Path -LiteralPath $luaFile) { $luaFile } else { "MISSING: $luaFile" }
Add-Check -Name "file:rime_neural.lua" -Ok (Test-Path -LiteralPath $luaFile) -Detail $luaDetail

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
    $maxSentencesVal = -1
    foreach ($line in ($builtRaw -split "`r?`n")) {
        if ($line -match '^engine:\s*$') { $inEngine = $true; $inFilters = $false; $inTranslator = $false; continue }
        if ($line -match '^translator:\s*$') { $inTranslator = $true; $inEngine = $false; $inFilters = $false; continue }
        if ($line -match '^\S') { $inEngine = $false; $inFilters = $false; $inTranslator = $false }
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
            if ($filters[$i] -match '^(simplifier(?:@(?:traditionalize|s2t\w*|t2s\w*))?$|lua_filter@\*?tw_dual_filter\b|lua_filter@\*?wanxiang\.charset_filter\b)') {
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

    $maxSentOk = ($maxSentencesVal -ge 6)
    $maxSentMsg = if ($maxSentOk) {
        "translator/max_sentences=$maxSentencesVal (>= 6)"
    } else {
        "Recommend setting translator/max_sentences: 8 and translator/sentence_cutoff_threshold: 0.3 (current=$maxSentencesVal)"
    }
    Add-Check -Name "build:${SchemaId}:max_sentences" -Ok $maxSentOk -Detail $maxSentMsg
} else {
    Add-Check -Name "build:${SchemaId}" -Ok $false -Detail "Missing compiled file: $builtSchema (run WeaselDeployer /deploy)"
}

# 4. Scan Weasel logs anchored to active WeaselServer start time (minus 20s for preceding deploy)
$logDir = Join-Path $env:TEMP "rime.weasel"
$recentHits = [System.Collections.Generic.List[string]]::new()
$seenLines  = [System.Collections.Generic.HashSet[string]]::new()
if (Test-Path -LiteralPath $logDir) {
    $cutoff = (Get-Date).AddMinutes(-10)
    if ($procs.Count -ge 1) {
        try {
            $startAnchor = ($procs | Sort-Object StartTime -Descending | Select-Object -First 1).StartTime.AddSeconds(-20)
            if ($startAnchor -gt $cutoff) { $cutoff = $startAnchor }
        } catch {}
    }
    $warnLogs = Get-ChildItem -LiteralPath $logDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.CreationTime -ge $cutoff -and ($_.Name -match 'WARNING|ERROR') } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 4
    foreach ($lf in $warnLogs) {
        $hits = Select-String -LiteralPath $lf.FullName -Pattern 'rime_neural|lua_filter.*error|loadlib failed' -ErrorAction SilentlyContinue
        foreach ($h in $hits) {
            $line = $h.Line.Trim()
            if ($seenLines.Add($line)) {
                $recentHits.Add("$($lf.Name):$($h.LineNumber): $line")
            }
        }
    }
    $infoLogs = Get-ChildItem -LiteralPath $logDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.CreationTime -ge $cutoff -and ($_.Name -match '\.INFO\.') } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 3
    foreach ($lf in $infoLogs) {
        $hits = Select-String -LiteralPath $lf.FullName -Pattern '^[WEF]\d{4,8}\s.*(rime_neural|lua_filter.*error|loadlib failed)' -ErrorAction SilentlyContinue
        foreach ($h in $hits) {
            $line = $h.Line.Trim()
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
    checks     = $checks
}
$out | ConvertTo-Json -Depth 5
if (-not $allOk) { exit 1 }
