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
    2. Downloads pinned & SHA-512-verified Microsoft.ML.OnnxRuntime win-x64 onnxruntime.dll
       (or uses an explicitly passed -OrtDllPath).
    3. Installs rime_neural.dll + lua/rime_neural.lua from:
         - -Prebuilt (downloads latest official GitHub Release from Chth1z/rime-neural & verifies SHA-256)
         - -ReleaseTag <tag> (downloads a specific GitHub Release tag, e.g. v1.1.0)
         - -PrebuiltDll <path> (uses a local prebuilt DLL)
         - local `cargo build --release --locked` (default when running from a source checkout)
    4. Stops WeaselServer cleanly before mutating any DLL/model, backs up replaced files for
       rollback on failure, installs all staged files, and restarts WeaselServer detached.
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

if ($ReleaseTag -and ($ReleaseTag -notmatch '^[A-Za-z0-9._-]+$')) {
    throw "Invalid -ReleaseTag format: '$ReleaseTag'"
}

function Invoke-NativeSafe {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $FilePath @ArgumentList 2>&1
        $code = $LASTEXITCODE
        return [pscustomobject]@{
            ExitCode = if ($null -eq $code) { 0 } else { [int]$code }
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
    $res = Invoke-NativeSafe -FilePath "curl.exe" -ArgumentList @("-fsSL", "--retry", "3", "--connect-timeout", "20", "-o", $OutFile, $Url)
    if ($res.ExitCode -ne 0) {
        $tail = ($res.Output | Select-Object -Last 5) -join " | "
        throw "$Label download failed (ExitCode=$($res.ExitCode)): $tail"
    }
}

function Get-DefaultRimeDir {
    $regDir = (Get-ItemProperty -Path "HKCU:\SOFTWARE\Rime\Weasel" -Name "RimeUserDir" -ErrorAction SilentlyContinue).RimeUserDir
    if ($regDir -and $regDir.Trim()) {
        return $regDir.Trim()
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
        $root = (Get-ItemProperty -Path $regPath -Name "WeaselRoot" -ErrorAction SilentlyContinue).WeaselRoot
        if ($root -and (Test-Path -LiteralPath (Join-Path $root "WeaselServer.exe"))) {
            return (Join-Path $root "WeaselServer.exe")
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
        if (-not (Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 200
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
    $matchLine = Get-Content -LiteralPath $SumsPath -Encoding ASCII | ForEach-Object { $_.Trim() } |
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

if (-not $RimeDir) { $RimeDir = Get-DefaultRimeDir }

$repoRoot  = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$neuralDir = if ($DataDir) { $DataDir.Trim() } else { Join-Path $RimeDir "neural" }
$luaDir    = Join-Path $RimeDir "lua"
$headers   = @{ "User-Agent" = "rime-neural-installer/1.1" }
$ghToken   = if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } elseif ($env:GH_TOKEN) { $env:GH_TOKEN } else { "" }
if ($ghToken) { $headers["Authorization"] = "Bearer $ghToken" }

if ($neuralDir -match '[^\x00-\x7F]') {
    Write-Warning "Target neural directory '$neuralDir' contains non-ASCII characters; Lua 5.4 package.loadlib on Windows uses ANSI ACP and may fail. Consider passing -DataDir to an ASCII path (e.g. C:\ProgramData\rime-neural) and setting rime_neural/data_dir."
}

$knownOnnxFiles = @(
    "laya.one.fused.int8g.onnx",
    "laya.one.int8g.onnx",
    "laya.one.w4.onnx",
    "laya.int8g.onnx"
)

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

# Pinned official Microsoft.ML.OnnxRuntime NuGet package (verified against NuGet v3 catalog SHA-512)
$pinnedOrtVersion = "1.31.0"
$pinnedOrtUrl     = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$pinnedOrtVersion/microsoft.ml.onnxruntime.$pinnedOrtVersion.nupkg"
$pinnedOrtSha512  = "J7arTn+7nrWdZ+PF2Sl9aNW1/Ba5DCt1ywZ3r5l3LKqzj3w8oC7Cl/uagG3Xp1ivuYJYQAMSuew0ef/74f7XCw=="

New-Item -ItemType Directory -Force -Path $neuralDir, $luaDir | Out-Null

$runTag    = (Get-Date -Format "yyyyMMddHHmmss") + "-" + ([guid]::NewGuid().ToString("N").Substring(0, 8))
$stageRoot = Join-Path $env:TEMP ("rime-neural-install-" + $runTag)
$backupDir = Join-Path $stageRoot "backup"
New-Item -ItemType Directory -Force -Path $stageRoot, $backupDir | Out-Null

$stagedModelDir = $null
$stagedOrtDll   = $null
$builtDll       = ""
$luaSrc         = Join-Path $repoRoot "lua\rime_neural.lua"
$serverExe      = Resolve-WeaselServerExe
$wasRunning     = [bool](Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
$stoppedWeasel  = $false
$backedUpFiles  = [System.Collections.Generic.List[hashtable]]::new()

function Backup-TargetForRollback {
    param([string]$TargetPath)
    if (Test-Path -LiteralPath $TargetPath) {
        $bak = Join-Path $backupDir (([guid]::NewGuid().ToString("N").Substring(0, 8)) + "_" + (Split-Path $TargetPath -Leaf))
        Copy-Item -LiteralPath $TargetPath -Destination $bak -Force
        $backedUpFiles.Add(@{ Target = $TargetPath; Backup = $bak })
    }
}

try {
    # 1. Stage & SHA-256 verify ONNX model + paired tokenizer/vocab (fail-closed)
    $modelTarget = Join-Path $neuralDir $mcfg.File
    $tokTarget   = Join-Path $neuralDir "tokenizer.json"
    $vocabTarget = Join-Path $neuralDir "unified_vocab.json"
    $otherOnnxPresent = @(Get-ChildItem -LiteralPath $neuralDir -Filter "*.onnx" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne $mcfg.File }).Count -gt 0

    $needModel = $ForceDownload -or $otherOnnxPresent -or `
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
        New-Item -ItemType Directory -Force -Path $extDir | Out-Null
        Expand-Archive -LiteralPath $archive -DestinationPath $extDir -Force

        $modelSrc = Get-ChildItem -LiteralPath $extDir -Recurse -Filter $mcfg.File | Select-Object -First 1
        if (-not $modelSrc) { throw "$($mcfg.File) not found in archive." }
        $actualHash = (Get-FileHash -LiteralPath $modelSrc.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $mcfg.Sha256) {
            throw "SHA-256 mismatch on $($mcfg.File)! Expected=$($mcfg.Sha256) Got=$actualHash"
        }
        $srcModelDir = $modelSrc.DirectoryName
        foreach ($reqAsset in @("tokenizer.json", "unified_vocab.json")) {
            $reqPath = Join-Path $srcModelDir $reqAsset
            if (-not (Test-Path -LiteralPath $reqPath)) { throw "$reqAsset missing from model archive." }
        }
        $hTok = (Get-FileHash -LiteralPath (Join-Path $srcModelDir "tokenizer.json") -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hTok -ne $mcfg.TokSha256) { throw "SHA-256 mismatch on tokenizer.json! Expected=$($mcfg.TokSha256) Got=$hTok" }
        $hVoc = (Get-FileHash -LiteralPath (Join-Path $srcModelDir "unified_vocab.json") -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hVoc -ne $mcfg.VocabSha256) { throw "SHA-256 mismatch on unified_vocab.json! Expected=$($mcfg.VocabSha256) Got=$hVoc" }

        Write-Output "Verified $($mcfg.File) + paired vocab/tokenizer (SHA-256: $($actualHash.Substring(0,12))...)."
        $stagedModelDir = $srcModelDir
    } else {
        Write-Output "Model $($mcfg.File) + paired vocab/tokenizer already verified in $neuralDir."
    }

    # 2. Stage onnxruntime.dll (uses explicit -OrtDllPath or pinned SHA-512-verified Microsoft NuGet win-x64)
    $targetOrt = Join-Path $neuralDir "onnxruntime.dll"
    if ($OrtDllPath) {
        if (-not (Test-Path -LiteralPath $OrtDllPath)) {
            throw "Specified -OrtDllPath not found: $OrtDllPath"
        }
        $stagedOrtDll = (Resolve-Path -LiteralPath $OrtDllPath).Path
    } elseif ($ForceDownload -or -not (Test-Path -LiteralPath $targetOrt)) {
        $ortStage = Join-Path $stageRoot "ort"
        New-Item -ItemType Directory -Force -Path $ortStage | Out-Null
        $nupkg = Join-Path $ortStage "ort.zip"
        Write-Output "Downloading pinned Microsoft.ML.OnnxRuntime $pinnedOrtVersion (win-x64) from NuGet..."
        Invoke-CurlDownload -Url $pinnedOrtUrl -OutFile $nupkg -Label "ONNX Runtime"
        $gotSha512 = Get-FileSha512Base64 -Path $nupkg
        if ($gotSha512 -ne $pinnedOrtSha512) {
            throw "SHA-512 mismatch on Microsoft.ML.OnnxRuntime $pinnedOrtVersion! Expected=$pinnedOrtSha512 Got=$gotSha512"
        }
        Write-Output "Verified Microsoft.ML.OnnxRuntime $pinnedOrtVersion NuGet SHA-512."
        $ortEx = Join-Path $ortStage "ex"
        Expand-Archive -LiteralPath $nupkg -DestinationPath $ortEx -Force
        $dll = Join-Path $ortEx "runtimes\win-x64\native\onnxruntime.dll"
        if (-not (Test-Path -LiteralPath $dll)) { throw "onnxruntime.dll not found inside NuGet package." }
        $stagedOrtDll = $dll
    } else {
        Write-Output "onnxruntime.dll already present in $neuralDir."
    }

    # 3. Resolve & verify rime_neural.dll + lua/rime_neural.lua
    if ($Prebuilt -or $ReleaseTag) {
        $relStage = Join-Path $stageRoot "release"
        New-Item -ItemType Directory -Force -Path $relStage | Out-Null

        $zipUrl     = ""
        $zipName    = ""
        $zipDigest  = ""
        $sumsUrl    = ""

        $apiUrl = if ($ReleaseTag) {
            "https://api.github.com/repos/Chth1z/rime-neural/releases/tags/$ReleaseTag"
        } else {
            "https://api.github.com/repos/Chth1z/rime-neural/releases/latest"
        }
        try {
            Write-Output "Fetching official release metadata from $apiUrl..."
            $rel = Invoke-RestMethod $apiUrl -Headers $headers -TimeoutSec 20
            $zipAsset = $rel.assets | Where-Object { $_.name -match 'win-x64\.zip$' } | Select-Object -First 1
            $sumAsset = $rel.assets | Where-Object { $_.name -eq 'SHA256SUMS.txt' } | Select-Object -First 1
            if (-not $zipAsset) { throw "No win-x64.zip asset found in release $($rel.tag_name)." }
            $zipUrl  = $zipAsset.browser_download_url
            $zipName = $zipAsset.name
            if ($zipAsset.digest -match '^sha256:([0-9a-fA-F]{64})$') { $zipDigest = $Matches[1].ToLowerInvariant() }
            if ($sumAsset) { $sumsUrl = $sumAsset.browser_download_url }
        } catch {
            if ($ReleaseTag) {
                Write-Warning "GitHub API unavailable ($($_.Exception.Message)); falling back to direct release URLs for $ReleaseTag."
                $zipName = "rime-neural-$ReleaseTag-win-x64.zip"
                $zipUrl  = "https://github.com/Chth1z/rime-neural/releases/download/$ReleaseTag/$zipName"
                $sumsUrl = "https://github.com/Chth1z/rime-neural/releases/download/$ReleaseTag/SHA256SUMS.txt"
            } else {
                throw
            }
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
        Expand-Archive -LiteralPath $relZip -DestinationPath $relEx -Force
        $builtDll = Join-Path $relEx "rime_neural.dll"
        $relLua   = Join-Path $relEx "rime_neural.lua"
        Assert-Sha256InSums -SumsPath $extSums -EntryName "rime_neural.dll" -ActualPath $builtDll
        Assert-Sha256InSums -SumsPath $extSums -EntryName "rime_neural.lua" -ActualPath $relLua
        $luaSrc = $relLua
    } elseif ($PrebuiltDll) {
        if (-not (Test-Path -LiteralPath $PrebuiltDll)) {
            throw "Specified -PrebuiltDll not found: $PrebuiltDll"
        }
        $builtDll = (Resolve-Path -LiteralPath $PrebuiltDll).Path
    } else {
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

    if (-not (Test-Path -LiteralPath $builtDll)) { throw "Native DLL missing: $builtDll" }
    if (-not (Test-Path -LiteralPath $luaSrc)) {
        throw "Lua filter missing: $luaSrc (run from a repository checkout or pass -Prebuilt)."
    }

    # 4. Re-check running WeaselServer right before mutation (prevents TOCTOU if TSF started it during downloads)
    $serverExe = if ($serverExe) { $serverExe } else { Resolve-WeaselServerExe }
    $runningNow = [bool](Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
    $shouldRestart = $wasRunning -or $runningNow
    if ($runningNow) {
        $stoppedWeasel = Stop-WeaselServerClean -ServerExe $serverExe
    }

    # Back up existing files before mutating so catch can roll back cleanly on failure
    if ($stagedModelDir) {
        foreach ($existingOnnx in (Get-ChildItem -LiteralPath $neuralDir -Filter "*.onnx" -ErrorAction SilentlyContinue)) {
            Backup-TargetForRollback -TargetPath $existingOnnx.FullName
        }
        foreach ($f in @("tokenizer.json", "unified_vocab.json", "rl_agent_config.json", "LICENSE")) {
            Backup-TargetForRollback -TargetPath (Join-Path $neuralDir $f)
        }
    }
    if ($stagedOrtDll) { Backup-TargetForRollback -TargetPath $targetOrt }
    Backup-TargetForRollback -TargetPath (Join-Path $neuralDir "rime_neural.dll")
    Backup-TargetForRollback -TargetPath (Join-Path $luaDir "rime_neural.lua")

    if ($stagedModelDir) {
        foreach ($oldOnnx in (Get-ChildItem -LiteralPath $neuralDir -Filter "*.onnx" -ErrorAction SilentlyContinue)) {
            if ($oldOnnx.Name -ne $mcfg.File) {
                Remove-Item -LiteralPath $oldOnnx.FullName -Force
                Write-Output "Removed conflicting model $($oldOnnx.Name) from $neuralDir."
            }
        }
        foreach ($f in @($mcfg.File, "tokenizer.json", "unified_vocab.json", "rl_agent_config.json", "LICENSE")) {
            $src = Join-Path $stagedModelDir $f
            if (Test-Path -LiteralPath $src) {
                Copy-Item -LiteralPath $src -Destination (Join-Path $neuralDir $f) -Force
            }
        }
        Write-Output "Installed model $($mcfg.File) -> $neuralDir."
    }

    if ($stagedOrtDll) {
        Copy-Item -LiteralPath $stagedOrtDll -Destination $targetOrt -Force
        Write-Output "Installed onnxruntime.dll -> $targetOrt."
    }

    Copy-Item -LiteralPath $builtDll -Destination (Join-Path $neuralDir "rime_neural.dll") -Force
    Write-Output ("Installed rime_neural.dll ({0:F2} MB) -> {1}" -f ((Get-Item -LiteralPath $builtDll).Length / 1MB), $neuralDir)

    Copy-Item -LiteralPath $luaSrc -Destination (Join-Path $luaDir "rime_neural.lua") -Force
    Write-Output "Installed rime_neural.lua -> $luaDir"

    if ($shouldRestart) {
        Start-WeaselServerDetached -ServerExe $serverExe
        $stoppedWeasel = $false
    }
}
catch {
    # Roll back any partially replaced files before restarting WeaselServer
    foreach ($entry in $backedUpFiles) {
        try {
            if (Test-Path -LiteralPath $entry.Backup) {
                Copy-Item -LiteralPath $entry.Backup -Destination $entry.Target -Force
            }
        } catch {}
    }
    if ($stoppedWeasel) {
        try { Start-WeaselServerDetached -ServerExe $serverExe } catch {}
    }
    throw
}
finally {
    if ($stageRoot -and (Test-Path -LiteralPath $stageRoot)) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Output "Done! Mount `lua_filter@*rime_neural` in your schema custom.yaml and run Weasel Deploy."
exit 0
