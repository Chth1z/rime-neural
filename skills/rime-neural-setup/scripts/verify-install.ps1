<#
.SYNOPSIS
    Post-deployment verification for rime-neural on Windows Weasel.
.DESCRIPTION
    Checks installed binaries & models, WeaselServer process health, recent Weasel logs for
    rime_neural warnings/errors, and compiled build/<schema>.schema.yaml filter order & beam settings.
#>
[CmdletBinding()]
param(
    [string]$RimeDir = "",
    [string]$SchemaId = ""
)

$ErrorActionPreference = "Stop"

if (-not $RimeDir) {
    $regDir = (Get-ItemProperty -Path "HKCU:\SOFTWARE\Rime\Weasel" -Name "RimeUserDir" -ErrorAction SilentlyContinue).RimeUserDir
    $RimeDir = if ($regDir -and $regDir.Trim()) { $regDir.Trim() } else { Join-Path $env:APPDATA "Rime" }
}

$neuralDir = Join-Path $RimeDir "neural"
$luaFile   = Join-Path $RimeDir "lua\rime_neural.lua"
$checks    = [System.Collections.Generic.List[object]]::new()
$allOk     = $true

function Add-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail)
    if (-not $Ok) { $script:allOk = $false }
    $checks.Add([ordered]@{ name = $Name; ok = $Ok; detail = $Detail })
}

# 1. Required files
foreach ($req in @("rime_neural.dll", "onnxruntime.dll", "tokenizer.json", "unified_vocab.json")) {
    $p = Join-Path $neuralDir $req
    $exists = Test-Path $p
    $detail = if ($exists) { "{0:F2} MB" -f ((Get-Item $p).Length / 1MB) } else { "MISSING: $p" }
    Add-Check -Name "file:$req" -Ok $exists -Detail $detail
}

$onnxFiles = @(Get-ChildItem $neuralDir -Filter "*.onnx" -ErrorAction SilentlyContinue)
$onnxDetail = if ($onnxFiles.Count -ge 1) {
    ($onnxFiles | ForEach-Object { "$($_.Name) ({0:F1} MB)" -f ($_.Length / 1MB) }) -join ", "
} else {
    "MISSING: no .onnx model in $neuralDir"
}
Add-Check -Name "file:onnx_model" -Ok ($onnxFiles.Count -ge 1) -Detail $onnxDetail

$luaDetail = if (Test-Path $luaFile) { $luaFile } else { "MISSING: $luaFile" }
Add-Check -Name "file:rime_neural.lua" -Ok (Test-Path $luaFile) -Detail $luaDetail

# 2. WeaselServer running cleanly as single instance
$procs = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
$pidText = ($procs | ForEach-Object { $_.Id }) -join ","
Add-Check -Name "process:WeaselServer" -Ok ($procs.Count -eq 1) -Detail "count=$($procs.Count) pids=$pidText"

# 3. Inspect compiled build/<SchemaId>.schema.yaml if SchemaId specified (or infer from user.yaml)
if (-not $SchemaId) {
    $userYaml = Join-Path $RimeDir "user.yaml"
    if (Test-Path $userYaml) {
        $m = Select-String -Path $userYaml -Pattern 'previously_selected_schema\s*:\s*"?([A-Za-z0-9_\-]+)"?' | Select-Object -First 1
        if ($m) { $SchemaId = $m.Matches[0].Groups[1].Value }
    }
}

if ($SchemaId) {
    $builtSchema = Join-Path $RimeDir "build\$SchemaId.schema.yaml"
    if (Test-Path $builtSchema) {
        $raw = Get-Content $builtSchema -Raw -Encoding UTF8
        $hasFilter = $raw -match 'lua_filter@\*rime_neural'
        $filterMsg = if ($hasFilter) {
            "lua_filter@*rime_neural present in build/$SchemaId.schema.yaml"
        } else {
            "NOT FOUND in build/$SchemaId.schema.yaml (did you run WeaselDeployer /deploy?)"
        }
        Add-Check -Name "build:${SchemaId}:filter_mounted" -Ok $hasFilter -Detail $filterMsg

        # Verify rime_neural appears BEFORE simplifier (OpenCC) if simplifier exists
        $idxNeural = $raw.IndexOf("lua_filter@*rime_neural")
        $idxSimp   = $raw.IndexOf("simplifier")
        if ($idxNeural -ge 0 -and $idxSimp -ge 0) {
            $orderOk = $idxNeural -lt $idxSimp
            $orderMsg = if ($orderOk) {
                "rime_neural precedes simplifier (OpenCC)"
            } else {
                "WARNING: rime_neural appears AFTER simplifier; move it before OpenCC!"
            }
            Add-Check -Name "build:${SchemaId}:filter_before_opencc" -Ok $orderOk -Detail $orderMsg
        }

        $maxSentOk = $raw -match 'max_sentences:\s*([6-9]|[1-9]\d)'
        $maxSentMsg = if ($maxSentOk) {
            "max_sentences >= 6 configured"
        } else {
            "Recommend setting translator/max_sentences: 8 and translator/sentence_cutoff_threshold: 0.3"
        }
        Add-Check -Name "build:${SchemaId}:max_sentences" -Ok $maxSentOk -Detail $maxSentMsg
    } else {
        Add-Check -Name "build:${SchemaId}" -Ok $false -Detail "Missing compiled file: $builtSchema (run WeaselDeployer /deploy)"
    }
}

# 4. Scan latest Weasel warning/error logs from the last 10 minutes
$logDir = Join-Path $env:TEMP "rime.weasel"
$recentHits = @()
if (Test-Path $logDir) {
    $cutoff = (Get-Date).AddMinutes(-10)
    $logs = Get-ChildItem $logDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $cutoff -and ($_.Name -match 'WARNING|ERROR') } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 4
    foreach ($lf in $logs) {
        $hits = Select-String -Path $lf.FullName -Pattern 'rime_neural|lua_filter.*error|loadlib failed' -ErrorAction SilentlyContinue
        foreach ($h in $hits) {
            $recentHits += "$($lf.Name):$($h.LineNumber): $($h.Line.Trim())"
        }
    }
}
$logMsg = if ($recentHits.Count -eq 0) {
    "No rime_neural errors/warnings in recent Weasel logs"
} else {
    ($recentHits | Select-Object -First 5) -join " | "
}
Add-Check -Name "logs:weasel_warnings" -Ok ($recentHits.Count -eq 0) -Detail $logMsg

$out = [ordered]@{
    ok        = $allOk
    rime_dir  = $RimeDir
    schema_id = $SchemaId
    checks    = $checks
}
$out | ConvertTo-Json -Depth 5
if (-not $allOk) { exit 1 }
