<#
.SYNOPSIS
    Install rime-neural (DLL + Lua filter + ONNX Runtime + verified ONNX model) into Rime user dir.
.DESCRIPTION
    1. Downloads & cryptographically verifies the selected Apache-2.0 ONNX reranker model
       (.onnx + paired tokenizer.json + unified_vocab.json):
         - pumpkin (default): Zhiyi ML-pumpkin-1103 (12L-384d fused KV-cache, highest Chinese accuracy)
         - feather          : Zhiyi zhen-feather-preview-1121 (8L-256d compact KV-cache, ultra-light)
       Automatically removes conflicting .onnx models on model switch so tokenizer.json and
       unified_vocab.json always stay paired with the active .onnx file.
    2. Downloads pinned & SHA-512/SHA-256-verified Microsoft.ML.OnnxRuntime win-x64 onnxruntime.dll
       (or uses an explicitly passed -OrtDllPath after PE32+ x64 validation).
    3. Installs rime_neural.dll + lua/rime_neural.lua from:
         - -Prebuilt (downloads latest official GitHub Release from Chth1z/rime-neural & verifies SHA-256)
         - -ReleaseTag <tag> (downloads a specific GitHub Release tag, e.g. v1.2.0)
         - -PrebuiltDll <path> (uses a local prebuilt DLL)
         - local `cargo build --release --locked` (default when running from a source checkout)
    4. Stops WeaselServer cleanly (failing before mutation if it cannot exit), backs up replaced
       files and tracks newly created files for full rollback on failure, installs all staged files,
       and restarts WeaselServer detached.
.PARAMETER RimeDir
    Target Rime user directory (defaults to HKCU\SOFTWARE\Rime\Weasel RimeUserDir or %APPDATA%\Rime).
.PARAMETER Model
    ONNX model preset to install: "pumpkin" (default, 12L) or "feather" (8L).
.PARAMETER OrtDllPath
    Optional path to an existing win-x64 onnxruntime.dll (skips NuGet download).
.PARAMETER LocalModelArchive
    Optional path to a local model zip archive (skips model download; still enforces SHA-256).
.PARAMETER PrebuiltDll
    Optional path to a local prebuilt rime_neural.dll (mutually exclusive with -Prebuilt / -ReleaseTag).
.PARAMETER ReleaseTag
    Download a specific GitHub Release tag (e.g. v1.2.0) with SHA256SUMS.txt verification.
.PARAMETER DataDir
    Optional ASCII directory for neural binaries and models (useful for non-ASCII Windows usernames).
.PARAMETER Prebuilt
    Download the latest official GitHub Release bundle instead of building with Cargo.
.PARAMETER ForceDownload
    Re-download ONNX model and ONNX Runtime even if verified files already exist locally.
.PARAMETER SkipBuild
    Reuse an existing target\release\rime_neural.dll from a source checkout without running cargo.
#>
param(
    [string]$RimeDir = "",
    [ValidateSet("pumpkin", "feather")]
    [string]$Model = "pumpkin",
    [string]$OrtDllPath = "",
    [string]$LocalModelArchive = "",
    [string]$PrebuiltDll = "",
    [string]$ReleaseTag = "",
    [string]$DataDir = "",
    [switch]$Prebuilt,
    [switch]$ForceDownload,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.IO.Compression.FileSystem

if ($ReleaseTag -and ($ReleaseTag -notmatch '^[A-Za-z0-9._-]+$')) {
    throw "Invalid -ReleaseTag format: '$ReleaseTag'"
}
if (($Prebuilt -or $ReleaseTag) -and ($PrebuiltDll -or $SkipBuild)) {
    throw "Conflicting options: -Prebuilt / -ReleaseTag cannot be combined with -PrebuiltDll or -SkipBuild."
}
if ($PrebuiltDll -and $SkipBuild) {
    throw "Conflicting options: -PrebuiltDll cannot be combined with -SkipBuild."
}

function Invoke-NativeSafe {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    $resolved = Get-Command -Name $FilePath -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $resolved) {
        return [pscustomobject]@{
            ExitCode = 127
            Output   = @("Executable not found: $FilePath")
        }
    }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $global:LASTEXITCODE = 0
        $output = & $resolved.Source @ArgumentList 2>&1
        $ok = $?
        $code = $LASTEXITCODE
        $exitCode = if ($null -ne $code -and [int]$code -ne 0) {
            [int]$code
        } elseif (-not $ok) {
            1
        } else {
            0
        }
        return [pscustomobject]@{
            ExitCode = $exitCode
            Output   = @($output | ForEach-Object { [string]$_ })
        }
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

function Invoke-CurlDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [string]$Label = "file"
    )
    $res = Invoke-NativeSafe -FilePath "curl.exe" -ArgumentList @(
        "-fsSL",
        "--proto", "=https",
        "--proto-redir", "=https",
        "--retry", "3",
        "--connect-timeout", "20",
        "--max-time", "300",
        "-o", $OutFile,
        "--",
        $Url
    )
    if ($res.ExitCode -ne 0) {
        $tail = ($res.Output | Select-Object -Last 5) -join " | "
        throw "$Label download failed (ExitCode=$($res.ExitCode)): $tail"
    }
}

function Test-Pe64Dll {
    param([string]$DllPath)
    if (-not $DllPath -or -not (Test-Path -LiteralPath $DllPath -PathType Leaf)) { return $false }
    try {
        $fs = [System.IO.File]::OpenRead($DllPath)
        try {
            $head = New-Object byte[] 4096
            $read = $fs.Read($head, 0, $head.Length)
            if ($read -lt 0x100 -or $head[0] -ne 0x4D -or $head[1] -ne 0x5A) { return $false }
            $peOff = [BitConverter]::ToInt32($head, 0x3C)
            if ($peOff -lt 0 -or ($peOff + 26) -gt $read) { return $false }
            if ($head[$peOff] -ne 0x50 -or $head[$peOff + 1] -ne 0x45 -or $head[$peOff + 2] -ne 0 -or $head[$peOff + 3] -ne 0) { return $false }
            $machine = [BitConverter]::ToUInt16($head, $peOff + 4)
            $characteristics = [BitConverter]::ToUInt16($head, $peOff + 22)
            $magic = [BitConverter]::ToUInt16($head, $peOff + 24)
            return ($machine -eq 0x8664 -and $magic -eq 0x20B -and ($characteristics -band 0x2000) -ne 0)
        } finally {
            $fs.Dispose()
        }
    } catch {
        return $false
    }
}

function Expand-ZipAllowlisted {
    param(
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$DestDir,
        [Parameter(Mandatory = $true)][string[]]$AllowNames,
        [string]$RequiredSubPath = "",
        [long]$MaxEntryBytes = 536870912
    )
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    $destFull = [System.IO.Path]::GetFullPath($DestDir)
    if (-not $destFull.EndsWith("\")) { $destFull += "\" }

    $allowSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($n in $AllowNames) { [void]$allowSet.Add($n) }
    $reqNorm = if ($RequiredSubPath) { $RequiredSubPath.Replace("\", "/").TrimStart("/") } else { "" }

    $extracted = @{}
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        foreach ($entry in $zip.Entries) {
            if (-not $entry.Name) { continue } # directory entry
            $entryNorm = $entry.FullName.Replace("\", "/").TrimStart("/")
            if ($entryNorm -match '(^|/)\.\.(/|$)' -or $entry.FullName.StartsWith("/") -or $entry.FullName.StartsWith("\") -or $entry.FullName -match '^[A-Za-z]:') {
                throw "Refusing unsafe zip entry path: $($entry.FullName)"
            }
            if ($reqNorm) {
                if (-not $entryNorm.Equals($reqNorm, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
            } else {
                if (-not $allowSet.Contains($entry.Name)) { continue }
            }
            if ($entry.Length -lt 0 -or $entry.Length -gt $MaxEntryBytes) {
                throw "Zip entry '$($entry.FullName)' exceeds allowed size limit ($MaxEntryBytes bytes)."
            }
            $outPath = [System.IO.Path]::GetFullPath((Join-Path $DestDir $entry.Name))
            if (-not $outPath.StartsWith($destFull, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "ZipSlip path escape blocked for entry: $($entry.FullName)"
            }
            $inStream = $entry.Open()
            try {
                $outStream = [System.IO.File]::Create($outPath)
                try {
                    $inStream.CopyTo($outStream)
                } finally {
                    $outStream.Dispose()
                }
            } finally {
                $inStream.Dispose()
            }
            $extracted[$entry.Name.ToLowerInvariant()] = $outPath
        }
    } finally {
        $zip.Dispose()
    }
    return $extracted
}

function Get-DefaultRimeDir {
    $prop = Get-ItemProperty -Path "HKCU:\SOFTWARE\Rime\Weasel" -Name "RimeUserDir" -ErrorAction SilentlyContinue
    if ($prop -and $prop.PSObject.Properties["RimeUserDir"] -and $prop.RimeUserDir -and $prop.RimeUserDir.Trim()) {
        return $prop.RimeUserDir.Trim()
    }
    return (Join-Path $env:APPDATA "Rime")
}

function Resolve-WeaselServerExe {
    $proc = Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($proc) {
        try {
            if ($proc.Path -and (Test-Path -LiteralPath $proc.Path)) { return $proc.Path }
        } catch {}
    }
    foreach ($regPath in @("HKLM:\SOFTWARE\Rime\Weasel", "HKLM:\SOFTWARE\WOW6432Node\Rime\Weasel", "HKCU:\SOFTWARE\Rime\Weasel")) {
        $prop = Get-ItemProperty -Path $regPath -Name "WeaselRoot" -ErrorAction SilentlyContinue
        if ($prop -and $prop.PSObject.Properties["WeaselRoot"] -and $prop.WeaselRoot) {
            $cand = Join-Path $prop.WeaselRoot "WeaselServer.exe"
            if (Test-Path -LiteralPath $cand) { return $cand }
        }
    }
    $bases = @()
    if ($env:ProgramFiles) { $bases += (Join-Path $env:ProgramFiles "Rime") }
    if (${env:ProgramFiles(x86)}) { $bases += (Join-Path ${env:ProgramFiles(x86)} "Rime") }
    $dirs = @()
    foreach ($base in ($bases | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $base) {
            $dirs += Get-ChildItem -LiteralPath $base -Directory -Filter "weasel-*" -ErrorAction SilentlyContinue |
                Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "WeaselServer.exe") }
        }
    }
    $best = $dirs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($best) { return (Join-Path $best.FullName "WeaselServer.exe") }
    return $null
}

function Stop-WeaselServerClean {
    param([string]$ServerExe)
    $running = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
    if ($running.Count -eq 0) { return $false }
    if ($ServerExe -and (Test-Path -LiteralPath $ServerExe)) {
        [void](Invoke-NativeSafe -FilePath $ServerExe -ArgumentList @("/q"))
        for ($i = 0; $i -lt 15; $i++) {
            if (-not (Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)) { return $true }
            Start-Sleep -Milliseconds 250
        }
    }
    Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 15; $i++) {
        if (-not (Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 200
    }
    $still = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
    if ($still.Count -gt 0) {
        $pids = ($still | ForEach-Object { $_.Id }) -join ","
        throw "WeaselServer.exe (PID=$pids) is still running and could not be stopped; close Weasel manually (or run elevated) before installing."
    }
    return $true
}

function Start-WeaselServerDetached {
    param([string]$ServerExe)
    if (-not $ServerExe -or -not (Test-Path -LiteralPath $ServerExe)) { return }
    $started = $false
    try {
        $cim = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$ServerExe`"" }
        if ($null -ne $cim -and $cim.ReturnValue -eq 0) { $started = $true }
    } catch {}
    if (-not $started) {
        Start-Process -FilePath $ServerExe
    }
    Start-Sleep -Milliseconds 700
    $procs = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
    Write-Output "Restarted WeaselServer (count=$($procs.Count) PID=$(($procs | ForEach-Object { $_.Id }) -join ','))."
}

function Get-FileSha512Base64 {
    param([string]$Path)
    $sha = [System.Security.Cryptography.SHA512]::Create()
    try {
        $fs = [System.IO.File]::OpenRead($Path)
        try {
            return [Convert]::ToBase64String($sha.ComputeHash($fs))
        } finally {
            $fs.Dispose()
        }
    } finally {
        $sha.Dispose()
    }
}

function Assert-Sha256InSums {
    param(
        [string]$SumsPath,
        [string]$EntryName,
        [string]$ActualPath
    )
    if (-not (Test-Path -LiteralPath $SumsPath)) {
        throw "SHA256SUMS.txt missing: $SumsPath"
    }
    if (-not (Test-Path -LiteralPath $ActualPath)) {
        throw "File missing for SHA-256 verification: $ActualPath"
    }
    $pat = '(?i)^([0-9a-f]{64})\s+\*?' + [regex]::Escape($EntryName) + '$'
    $matchLine = Get-Content -LiteralPath $SumsPath -Encoding UTF8 | ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match $pat } | Select-Object -First 1
    if (-not $matchLine) {
        throw "No SHA-256 checksum entry found for '$EntryName' in $SumsPath"
    }
    $null = $matchLine -match $pat
    $expected = $Matches[1].ToLowerInvariant()
    $got = (Get-FileHash -LiteralPath $ActualPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($got -ne $expected) {
        throw "SHA-256 mismatch on $EntryName! Expected=$expected Got=$got"
    }
    Write-Output "Verified $EntryName (SHA-256: $($got.Substring(0,12))...)."
}

function Test-SamePath {
    param([string]$PathA, [string]$PathB)
    try {
        return [System.IO.Path]::GetFullPath($PathA).Equals(
            [System.IO.Path]::GetFullPath($PathB),
            [System.StringComparison]::OrdinalIgnoreCase
        )
    } catch {
        return $false
    }
}

function Resolve-LatestReleaseTagViaRedirect {
    $res = Invoke-NativeSafe -FilePath "curl.exe" -ArgumentList @(
        "-fsSIL",
        "--proto", "=https",
        "--proto-redir", "=https",
        "--connect-timeout", "15",
        "--max-time", "30",
        "-o", "NUL",
        "-w", "%{url_effective}",
        "--",
        "https://github.com/Chth1z/rime-neural/releases/latest"
    )
    if ($res.ExitCode -eq 0) {
        $finalUrl = ($res.Output -join "").Trim()
        if ($finalUrl -match '/releases/tag/([A-Za-z0-9._-]+)$') {
            return $Matches[1]
        }
    }
    return ""
}

if (-not $RimeDir) { $RimeDir = Get-DefaultRimeDir }

$repoRoot  = if ($PSScriptRoot) { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } else { "" }
$neuralDir = if ($DataDir) { $DataDir.Trim() } else { Join-Path $RimeDir "neural" }
$luaDir    = Join-Path $RimeDir "lua"
$baseHeaders = @{ "User-Agent" = "rime-neural-installer/1.2" }
$authHeaders = @{ "User-Agent" = "rime-neural-installer/1.2" }
$ghToken   = if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } elseif ($env:GH_TOKEN) { $env:GH_TOKEN } else { "" }
if ($ghToken) { $authHeaders["Authorization"] = "Bearer $ghToken" }

if ($neuralDir -match '[^\x00-\x7F]') {
    Write-Warning "Target neural directory '$neuralDir' contains non-ASCII characters; Lua 5.4 package.loadlib on Windows uses ANSI ACP and may fail. Consider passing -DataDir to an ASCII path (e.g. C:\ProgramData\rime-neural) and setting rime_neural/data_dir."
}

$models = @{
    pumpkin = @{
        Name        = "ML-pumpkin-1103 (12L fused KV-cache, Apache-2.0)"
        Url         = "https://github.com/dd1000001000/zhiyi_models/releases/download/model-ml-pumpkin-1103/laya-ml-pumpkin-1103-int8g.zip"
        File        = "laya.one.fused.int8g.onnx"
        Sha256      = "c273b2cfc46179e9df2a94aaa381bb6fb1d55862fe8834b4631950180d521ca5"
        TokSha256   = "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"
        VocabSha256 = "5b2be1746816ecf73152496f4797e3bff9ba9fee27a36408a46e265d93008e72"
    }
    feather = @{
        Name        = "zhen-feather-preview-1121 (8L compact KV-cache, Apache-2.0)"
        Url         = "https://github.com/dd1000001000/zhiyi_models/releases/download/model-zhen-feather-preview-1121/laya-zhen-feather-preview-1121-slim.zip"
        File        = "laya.one.w4.onnx"
        Sha256      = "98c615a1c2dfeb82de5c9f749e7910e3e900261a760ae82d43d2b63e9164a35c"
        TokSha256   = "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"
        VocabSha256 = "196812c096c7d2a54e5d1be020ca0850601ad21dca849f92dc53a766643cfb53"
    }
}
$mcfg = $models[$Model]
foreach ($hashKey in @("Sha256", "TokSha256", "VocabSha256")) {
    if (-not $mcfg[$hashKey] -or ($mcfg[$hashKey] -notmatch '^[0-9a-f]{64}$')) {
        throw "Internal configuration error: missing SHA-256 pin for $Model.$hashKey"
    }
}

# Pinned official Microsoft.ML.OnnxRuntime NuGet package (verified against NuGet v3 SHA-512 + win-x64 DLL SHA-256)
$pinnedOrtVersion   = "1.31.0"
$pinnedOrtUrl       = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$pinnedOrtVersion/microsoft.ml.onnxruntime.$pinnedOrtVersion.nupkg"
$pinnedOrtSha512    = "J7arTn+7nrWdZ+PF2Sl9aNW1/Ba5DCt1ywZ3r5l3LKqzj3w8oC7Cl/uagG3Xp1ivuYJYQAMSuew0ef/74f7XCw=="
$pinnedOrtDllSha256 = "90497e3d406c09f4d59d364c696a2f93dc02f6ef13741b3b89eec95bdb4a3806"

$runTag         = (Get-Date -Format "yyyyMMddHHmmss") + "-" + ([guid]::NewGuid().ToString("N").Substring(0, 8))
$stageRoot      = Join-Path $env:TEMP ("rime-neural-install-" + $runTag)
$backupDir      = Join-Path $stageRoot "backup"
New-Item -ItemType Directory -Force -Path $stageRoot, $backupDir | Out-Null

$stagedModelDir = $null
$pruneOtherOnnx = $false
$stagedOrtDll   = $null
$builtDll       = ""
$luaSrc         = if ($repoRoot) { Join-Path $repoRoot "lua\rime_neural.lua" } else { "" }
$serverExe      = Resolve-WeaselServerExe
$wasRunning     = [bool](Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
$stoppedWeasel  = $false
$rollbackFailed = $false
$backedUpFiles  = [System.Collections.Generic.List[hashtable]]::new()
$createdFiles   = [System.Collections.Generic.List[string]]::new()

function Backup-TargetForRollback {
    param([string]$TargetPath)
    if (Test-Path -LiteralPath $TargetPath) {
        $bak = Join-Path $backupDir (([guid]::NewGuid().ToString("N").Substring(0, 8)) + "_" + (Split-Path $TargetPath -Leaf))
        Copy-Item -LiteralPath $TargetPath -Destination $bak -Force
        $backedUpFiles.Add(@{ Target = $TargetPath; Backup = $bak })
    }
}

function Copy-StagedTracked {
    param([string]$Source, [string]$Destination)
    if (Test-SamePath -PathA $Source -PathB $Destination) { return }
    $existed = Test-Path -LiteralPath $Destination
    if (-not $existed) {
        $createdFiles.Add($Destination)
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

try {
    # 1. Stage & SHA-256 verify ONNX model + paired tokenizer/vocab (fail-closed)
    $modelTarget = Join-Path $neuralDir $mcfg.File
    $tokTarget   = Join-Path $neuralDir "tokenizer.json"
    $vocabTarget = Join-Path $neuralDir "unified_vocab.json"
    $otherOnnxPresent = (Test-Path -LiteralPath $neuralDir) -and (
        @(Get-ChildItem -LiteralPath $neuralDir -Filter "*.onnx" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne $mcfg.File }).Count -gt 0
    )

    $needModel = $ForceDownload -or `
        -not (Test-Path -LiteralPath $modelTarget) -or `
        -not (Test-Path -LiteralPath $tokTarget) -or `
        -not (Test-Path -LiteralPath $vocabTarget)

    if (-not $needModel) {
        if ((Get-FileHash -LiteralPath $modelTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mcfg.Sha256) {
            Write-Output "Existing $($mcfg.File) hash differs; refreshing..."
            $needModel = $true
        } elseif ((Get-FileHash -LiteralPath $tokTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mcfg.TokSha256) {
            Write-Output "Existing tokenizer.json does not match $($mcfg.File); refreshing..."
            $needModel = $true
        } elseif ((Get-FileHash -LiteralPath $vocabTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mcfg.VocabSha256) {
            Write-Output "Existing unified_vocab.json does not match $($mcfg.File); refreshing..."
            $needModel = $true
        } elseif ($otherOnnxPresent) {
            $pruneOtherOnnx = $true
        }
    }

    if ($needModel) {
        $modelStage = Join-Path $stageRoot "model"
        New-Item -ItemType Directory -Force -Path $modelStage | Out-Null
        if ($LocalModelArchive) {
            if (-not (Test-Path -LiteralPath $LocalModelArchive)) {
                throw "Specified -LocalModelArchive not found: $LocalModelArchive"
            }
            $archive = (Resolve-Path -LiteralPath $LocalModelArchive).Path
            Write-Output "Using local model archive: $archive"
        } else {
            $archive = Join-Path $modelStage "model.zip"
            Write-Output "Downloading $($mcfg.Name)..."
            Invoke-CurlDownload -Url $mcfg.Url -OutFile $archive -Label "Model archive"
        }

        $extDir = Join-Path $modelStage "extracted"
        $extMap = Expand-ZipAllowlisted -ArchivePath $archive -DestDir $extDir -AllowNames @($mcfg.File, "tokenizer.json", "unified_vocab.json")
        $modelPath = $extMap[$mcfg.File.ToLowerInvariant()]
        $tokPath   = $extMap["tokenizer.json"]
        $vocabPath = $extMap["unified_vocab.json"]
        if (-not $modelPath) { throw "$($mcfg.File) not found in archive." }
        if (-not $tokPath)   { throw "tokenizer.json missing from model archive." }
        if (-not $vocabPath) { throw "unified_vocab.json missing from model archive." }

        $actualHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $mcfg.Sha256) {
            throw "SHA-256 mismatch on $($mcfg.File)! Expected=$($mcfg.Sha256) Got=$actualHash"
        }
        $hTok = (Get-FileHash -LiteralPath $tokPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hTok -ne $mcfg.TokSha256) { throw "SHA-256 mismatch on tokenizer.json! Expected=$($mcfg.TokSha256) Got=$hTok" }
        $hVoc = (Get-FileHash -LiteralPath $vocabPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hVoc -ne $mcfg.VocabSha256) { throw "SHA-256 mismatch on unified_vocab.json! Expected=$($mcfg.VocabSha256) Got=$hVoc" }

        Write-Output "Verified $($mcfg.File) + paired vocab/tokenizer (SHA-256: $($actualHash.Substring(0,12))...)."
        $stagedModelDir = $extDir
    } else {
        Write-Output "Model $($mcfg.File) + paired vocab/tokenizer already verified in $neuralDir."
    }

    # 2. Stage onnxruntime.dll (explicit -OrtDllPath or pinned SHA-512/SHA-256 Microsoft NuGet win-x64)
    $targetOrt = Join-Path $neuralDir "onnxruntime.dll"
    if ($OrtDllPath) {
        if (-not (Test-Path -LiteralPath $OrtDllPath -PathType Leaf)) {
            throw "Specified -OrtDllPath not found: $OrtDllPath"
        }
        $resolvedOrt = (Resolve-Path -LiteralPath $OrtDllPath).Path
        if (-not (Test-Pe64Dll -DllPath $resolvedOrt)) {
            throw "Specified -OrtDllPath is not a valid 64-bit PE32+ DLL: $resolvedOrt"
        }
        $stagedOrtDll = $resolvedOrt
    } else {
        $needOrt = $ForceDownload -or -not (Test-Path -LiteralPath $targetOrt -PathType Leaf)
        if (-not $needOrt) {
            $gotExistingOrt = (Get-FileHash -LiteralPath $targetOrt -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($gotExistingOrt -ne $pinnedOrtDllSha256) {
                Write-Output "Existing onnxruntime.dll hash differs from pinned $pinnedOrtVersion; refreshing..."
                $needOrt = $true
            }
        }
        if ($needOrt) {
            $ortStage = Join-Path $stageRoot "ort"
            New-Item -ItemType Directory -Force -Path $ortStage | Out-Null
            $nupkg = Join-Path $ortStage "ort.nupkg"
            Write-Output "Downloading pinned Microsoft.ML.OnnxRuntime $pinnedOrtVersion (win-x64) from NuGet..."
            Invoke-CurlDownload -Url $pinnedOrtUrl -OutFile $nupkg -Label "ONNX Runtime"
            $gotSha512 = Get-FileSha512Base64 -Path $nupkg
            if ($gotSha512 -cne $pinnedOrtSha512) {
                throw "SHA-512 mismatch on Microsoft.ML.OnnxRuntime $pinnedOrtVersion! Expected=$pinnedOrtSha512 Got=$gotSha512"
            }
            Write-Output "Verified Microsoft.ML.OnnxRuntime $pinnedOrtVersion NuGet SHA-512."
            $ortEx = Join-Path $ortStage "ex"
            $ortMap = Expand-ZipAllowlisted -ArchivePath $nupkg -DestDir $ortEx -AllowNames @("onnxruntime.dll") -RequiredSubPath "runtimes/win-x64/native/onnxruntime.dll"
            $dll = $ortMap["onnxruntime.dll"]
            if (-not $dll -or -not (Test-Path -LiteralPath $dll)) { throw "onnxruntime.dll not found inside NuGet package." }
            $gotOrtSha = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($gotOrtSha -ne $pinnedOrtDllSha256) {
                throw "SHA-256 mismatch on extracted onnxruntime.dll! Expected=$pinnedOrtDllSha256 Got=$gotOrtSha"
            }
            $stagedOrtDll = $dll
        } else {
            Write-Output "onnxruntime.dll ($pinnedOrtVersion) already verified in $neuralDir."
        }
    }

    # 3. Resolve & verify rime_neural.dll + lua/rime_neural.lua
    if ($Prebuilt -or $ReleaseTag) {
        $relStage = Join-Path $stageRoot "release"
        New-Item -ItemType Directory -Force -Path $relStage | Out-Null

        $zipUrl    = ""
        $zipName   = ""
        $zipDigest = ""
        $sumsUrl   = ""

        $apiUrl = if ($ReleaseTag) {
            "https://api.github.com/repos/Chth1z/rime-neural/releases/tags/$ReleaseTag"
        } else {
            "https://api.github.com/repos/Chth1z/rime-neural/releases/latest"
        }
        $rel = $null
        try {
            Write-Output "Fetching official release metadata from $apiUrl..."
            $rel = Invoke-RestMethod $apiUrl -Headers $authHeaders -TimeoutSec 20
        } catch {
            if ($ghToken) {
                # Retry unauthenticated in case GITHUB_TOKEN / GH_TOKEN is stale or for GHES
                try {
                    Write-Warning "Authenticated GitHub API call failed ($($_.Exception.Message)); retrying anonymously..."
                    $rel = Invoke-RestMethod $apiUrl -Headers $baseHeaders -TimeoutSec 20
                } catch {}
            }
            if (-not $rel) {
                $fallbackTag = if ($ReleaseTag) { $ReleaseTag } else { Resolve-LatestReleaseTagViaRedirect }
                if ($fallbackTag) {
                    Write-Warning "GitHub API unavailable; falling back to direct release URLs for $fallbackTag."
                    $zipName = "rime-neural-$fallbackTag-win-x64.zip"
                    $zipUrl  = "https://github.com/Chth1z/rime-neural/releases/download/$fallbackTag/$zipName"
                    $sumsUrl = "https://github.com/Chth1z/rime-neural/releases/download/$fallbackTag/SHA256SUMS.txt"
                } else {
                    throw
                }
            }
        }
        if ($rel) {
            $zipAsset = $rel.assets | Where-Object { $_.name -match 'win-x64\.zip$' } | Select-Object -First 1
            $sumAsset = $rel.assets | Where-Object { $_.name -eq 'SHA256SUMS.txt' } | Select-Object -First 1
            if (-not $zipAsset) { throw "No win-x64.zip asset found in release $($rel.tag_name)." }
            $zipUrl  = $zipAsset.browser_download_url
            $zipName = $zipAsset.name
            if ($zipAsset.PSObject.Properties["digest"] -and $zipAsset.digest -match '^sha256:([0-9a-fA-F]{64})$') {
                $zipDigest = $Matches[1].ToLowerInvariant()
            }
            if ($sumAsset) { $sumsUrl = $sumAsset.browser_download_url }
        }

        if (-not $sumsUrl) {
            throw "Release SHA256SUMS.txt asset not found; refusing unverified binary download."
        }
        $extSums = Join-Path $relStage "SHA256SUMS.txt"
        Invoke-CurlDownload -Url $sumsUrl -OutFile $extSums -Label "Release SHA256SUMS.txt"

        $relZip = Join-Path $relStage $zipName
        Write-Output "Downloading $zipName..."
        Invoke-CurlDownload -Url $zipUrl -OutFile $relZip -Label $zipName

        # Fail-closed verification against external Release SHA256SUMS.txt (and GitHub API digest if present)
        Assert-Sha256InSums -SumsPath $extSums -EntryName $zipName -ActualPath $relZip
        if ($zipDigest) {
            $gotZip = (Get-FileHash -LiteralPath $relZip -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($gotZip -ne $zipDigest) {
                throw "GitHub asset digest mismatch on $zipName! Expected=$zipDigest Got=$gotZip"
            }
        }

        $relEx = Join-Path $relStage "ex"
        $relMap = Expand-ZipAllowlisted -ArchivePath $relZip -DestDir $relEx -AllowNames @("rime_neural.dll", "rime_neural.lua")
        $builtDll = $relMap["rime_neural.dll"]
        $relLua   = $relMap["rime_neural.lua"]
        if (-not $builtDll -or -not $relLua) { throw "Release zip missing rime_neural.dll or rime_neural.lua." }
        Assert-Sha256InSums -SumsPath $extSums -EntryName "rime_neural.dll" -ActualPath $builtDll
        Assert-Sha256InSums -SumsPath $extSums -EntryName "rime_neural.lua" -ActualPath $relLua
        $luaSrc = $relLua
    } elseif ($PrebuiltDll) {
        if (-not (Test-Path -LiteralPath $PrebuiltDll -PathType Leaf)) {
            throw "Specified -PrebuiltDll not found: $PrebuiltDll"
        }
        $builtDll = (Resolve-Path -LiteralPath $PrebuiltDll).Path
        if (-not (Test-Pe64Dll -DllPath $builtDll)) {
            throw "Specified -PrebuiltDll is not a valid 64-bit PE32+ DLL: $builtDll"
        }
    } else {
        if (-not $repoRoot) {
            throw "Local build requires running from a repository checkout; pass -Prebuilt for standalone installs."
        }
        $builtDll = Join-Path $repoRoot "target\release\rime_neural.dll"
        if (-not $SkipBuild -or -not (Test-Path -LiteralPath $builtDll)) {
            if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
                throw "cargo not found. Pass -Prebuilt to download the official Release DLL, or install Rust (https://rustup.rs)."
            }
            Write-Output "Building rime_neural (release --locked)..."
            $cargoRes = Invoke-NativeSafe -FilePath "cargo" -ArgumentList @("build", "--release", "--locked", "--manifest-path", (Join-Path $repoRoot "Cargo.toml"))
            if ($cargoRes.ExitCode -ne 0) {
                $tail = ($cargoRes.Output | Select-Object -Last 10) -join "`n"
                throw "cargo build failed (ExitCode=$($cargoRes.ExitCode)):`n$tail"
            }
        }
    }

    if (-not (Test-Path -LiteralPath $builtDll -PathType Leaf)) { throw "Native DLL missing: $builtDll" }
    if (-not $luaSrc -or -not (Test-Path -LiteralPath $luaSrc -PathType Leaf)) {
        throw "Lua filter missing: $luaSrc (run from a repository checkout or pass -Prebuilt)."
    }

    # 4. Create target directories only after all staging & verification succeeded; stop WeaselServer cleanly
    New-Item -ItemType Directory -Force -Path $neuralDir, $luaDir | Out-Null

    $serverExe = if ($serverExe) { $serverExe } else { Resolve-WeaselServerExe }
    $runningNow = [bool](Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
    $shouldRestart = $wasRunning -or $runningNow
    if ($runningNow) {
        $stoppedWeasel = Stop-WeaselServerClean -ServerExe $serverExe
    }

    # Back up existing files before mutating so catch can roll back cleanly on failure
    if ($stagedModelDir -or $pruneOtherOnnx) {
        foreach ($existingOnnx in (Get-ChildItem -LiteralPath $neuralDir -Filter "*.onnx" -ErrorAction SilentlyContinue)) {
            Backup-TargetForRollback -TargetPath $existingOnnx.FullName
        }
    }
    if ($stagedModelDir) {
        foreach ($f in @("tokenizer.json", "unified_vocab.json")) {
            Backup-TargetForRollback -TargetPath (Join-Path $neuralDir $f)
        }
    }
    if ($stagedOrtDll) { Backup-TargetForRollback -TargetPath $targetOrt }
    Backup-TargetForRollback -TargetPath (Join-Path $neuralDir "rime_neural.dll")
    Backup-TargetForRollback -TargetPath (Join-Path $luaDir "rime_neural.lua")

    # Install lock-prone DLLs first so any residual file lock fails before mutating model assets
    Copy-StagedTracked -Source $builtDll -Destination (Join-Path $neuralDir "rime_neural.dll")
    Write-Output ("Installed rime_neural.dll ({0:F2} MB) -> {1}" -f ((Get-Item -LiteralPath $builtDll).Length / 1MB), $neuralDir)

    if ($stagedOrtDll) {
        Copy-StagedTracked -Source $stagedOrtDll -Destination $targetOrt
        Write-Output "Installed onnxruntime.dll -> $targetOrt."
    }

    if ($stagedModelDir -or $pruneOtherOnnx) {
        foreach ($oldOnnx in (Get-ChildItem -LiteralPath $neuralDir -Filter "*.onnx" -ErrorAction SilentlyContinue)) {
            if ($oldOnnx.Name -ne $mcfg.File) {
                Remove-Item -LiteralPath $oldOnnx.FullName -Force
                Write-Output "Removed conflicting model $($oldOnnx.Name) from $neuralDir."
            }
        }
    }
    if ($stagedModelDir) {
        foreach ($f in @($mcfg.File, "tokenizer.json", "unified_vocab.json")) {
            $src = Join-Path $stagedModelDir $f
            if (Test-Path -LiteralPath $src) {
                Copy-StagedTracked -Source $src -Destination (Join-Path $neuralDir $f)
            }
        }
        Write-Output "Installed model $($mcfg.File) -> $neuralDir."
    }

    Copy-StagedTracked -Source $luaSrc -Destination (Join-Path $luaDir "rime_neural.lua")
    Write-Output "Installed rime_neural.lua -> $luaDir"

    if ($shouldRestart) {
        Start-WeaselServerDetached -ServerExe $serverExe
        $stoppedWeasel = $false
    }
}
catch {
    # Roll back: remove newly created files first, then restore backed-up originals
    foreach ($newPath in $createdFiles) {
        try {
            if (Test-Path -LiteralPath $newPath) {
                Remove-Item -LiteralPath $newPath -Force
            }
        } catch {
            $rollbackFailed = $true
            Write-Warning "Rollback could not remove newly created file '$newPath': $($_.Exception.Message)"
        }
    }
    foreach ($entry in $backedUpFiles) {
        try {
            if (Test-Path -LiteralPath $entry.Backup) {
                Copy-Item -LiteralPath $entry.Backup -Destination $entry.Target -Force
            }
        } catch {
            $rollbackFailed = $true
            Write-Warning "Rollback failed for '$($entry.Target)' (backup preserved at '$($entry.Backup)'): $($_.Exception.Message)"
        }
    }
    if ($stoppedWeasel) {
        try { Start-WeaselServerDetached -ServerExe $serverExe } catch {}
    }
    throw
}
finally {
    if ($stageRoot -and (Test-Path -LiteralPath $stageRoot)) {
        if ($rollbackFailed) {
            Write-Warning "Preserving backup directory due to incomplete rollback: $backupDir"
        } else {
            Remove-Item -LiteralPath $stageRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Output 'Done! Mount `lua_filter@*rime_neural` in your schema custom.yaml and run Weasel Deploy.'
exit 0
