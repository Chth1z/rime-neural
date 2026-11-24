<#
.SYNOPSIS
    Install rime-neural (DLL + Lua filter + ONNX Runtime + verified ONNX model) into Rime user dir.
.DESCRIPTION
    1. Downloads & verifies the selected Apache-2.0 ONNX reranker model:
         - pumpkin (default): Zhiyi ML-pumpkin-1103 (12L-384d fused KV-cache, highest Chinese accuracy)
         - feather          : Zhiyi zhen-feather-preview-1121 (8L-256d compact KV-cache, ultra-light)
       Automatically removes conflicting .onnx models on model switch so tokenizer.json and
       unified_vocab.json always stay paired with the active .onnx file.
    2. Locates or downloads pinned & hash-verified Microsoft.ML.OnnxRuntime win-x64 onnxruntime.dll.
    3. Installs rime_neural.dll + lua/rime_neural.lua from:
         - -Prebuilt (downloads latest official GitHub Release from Chth1z/rime-neural & verifies SHA-256)
         - -ReleaseTag <tag> (downloads a specific GitHub Release tag, e.g. v1.1.0)
         - -PrebuiltDll <path> (uses a local prebuilt DLL)
         - local `cargo build --release --locked` (default when running from a source checkout)
    4. Stops WeaselServer cleanly before copying any DLL, installs all staged files, and restarts
       WeaselServer detached.
#>
param(
    [string]$RimeDir = "",
    [ValidateSet("pumpkin", "feather")]
    [string]$Model = "pumpkin",
    [string]$OrtDllPath = "",
    [string]$LocalModelArchive = "",
    [string]$PrebuiltDll = "",
    [string]$ReleaseTag = "",
    [switch]$Prebuilt,
    [switch]$ForceDownload,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

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
            if ($proc.Path -and (Test-Path $proc.Path)) { return $proc.Path }
        } catch {}
    }
    foreach ($regPath in @("HKLM:\SOFTWARE\Rime\Weasel", "HKLM:\SOFTWARE\WOW6432Node\Rime\Weasel", "HKCU:\SOFTWARE\Rime\Weasel")) {
        $root = (Get-ItemProperty -Path $regPath -Name "WeaselRoot" -ErrorAction SilentlyContinue).WeaselRoot
        if ($root -and (Test-Path (Join-Path $root "WeaselServer.exe"))) {
            return (Join-Path $root "WeaselServer.exe")
        }
    }
    $dirs = @()
    foreach ($base in @("C:\Program Files\Rime", "C:\Program Files (x86)\Rime")) {
        if (Test-Path $base) {
            $dirs += Get-ChildItem $base -Directory -Filter "weasel-*" -ErrorAction SilentlyContinue |
                Where-Object { Test-Path (Join-Path $_.FullName "WeaselServer.exe") }
        }
    }
    $best = $dirs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($best) { return (Join-Path $best.FullName "WeaselServer.exe") }
    return $null
}

function Stop-WeaselServerClean {
    param([string]$ServerExe)
    $running = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
    if ($running.Count -eq 0) { return }
    if ($ServerExe -and (Test-Path $ServerExe)) {
        [void](Invoke-NativeSafe -FilePath $ServerExe -ArgumentList @("/q"))
        for ($i = 0; $i -lt 15; $i++) {
            if (-not (Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)) { return }
            Start-Sleep -Milliseconds 250
        }
    }
    Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 15; $i++) {
        if (-not (Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 200
    }
}

function Start-WeaselServerDetached {
    param([string]$ServerExe)
    if (-not $ServerExe -or -not (Test-Path $ServerExe)) { return }
    try {
        Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$ServerExe`"" } | Out-Null
    } catch {
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
    if (-not (Test-Path $SumsPath)) {
        throw "SHA256SUMS.txt missing: $SumsPath"
    }
    if (-not (Test-Path $ActualPath)) {
        throw "File missing for SHA-256 verification: $ActualPath"
    }
    $pat = '(?i)^([0-9a-f]{64})\s+\*?' + [regex]::Escape($EntryName) + '$'
    $matchLine = Get-Content $SumsPath -Encoding ASCII | ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match $pat } | Select-Object -First 1
    if (-not $matchLine) {
        throw "No SHA-256 checksum entry found for '$EntryName' in $SumsPath"
    }
    $null = $matchLine -match $pat
    $expected = $Matches[1].ToLowerInvariant()
    $got = (Get-FileHash $ActualPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($got -ne $expected) {
        throw "SHA-256 mismatch on $EntryName! Expected=$expected Got=$got"
    }
    Write-Output "Verified $EntryName (SHA-256: $($got.Substring(0,12))...)."
}

if (-not $RimeDir) { $RimeDir = Get-DefaultRimeDir }

$repoRoot  = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$neuralDir = Join-Path $RimeDir "neural"
$luaDir    = Join-Path $RimeDir "lua"
$headers   = @{ "User-Agent" = "rime-neural-installer/1.1" }
$ghToken   = if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } elseif ($env:GH_TOKEN) { $env:GH_TOKEN } else { "" }
if ($ghToken) { $headers["Authorization"] = "Bearer $ghToken" }

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
        Url         = "https://github.com/dd1000001000/zhiyi_models/releases/download/model-zhen-feather-preview-1121/laya-zhen-feather-preview-1121-w4.zip"
        File        = "laya.one.w4.onnx"
        Sha256      = "98c615a1c2dfeb82de5c9f749e7910e3e900261a760ae82d43d2b63e9164a35c"
        TokSha256   = ""
        VocabSha256 = ""
    }
}
$mcfg = $models[$Model]

# Pinned official Microsoft.ML.OnnxRuntime NuGet package (verified against NuGet v3 catalog SHA-512)
$pinnedOrtVersion = "1.31.0"
$pinnedOrtUrl     = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$pinnedOrtVersion/microsoft.ml.onnxruntime.$pinnedOrtVersion.nupkg"
$pinnedOrtSha512  = "J7arTn+7nrWdZ+PF2Sl9aNW1/Ba5DCt1ywZ3r5l3LKqzj3w8oC7Cl/uagG3Xp1ivuYJYQAMSuew0ef/74f7XCw=="

New-Item -ItemType Directory -Force -Path $neuralDir, $luaDir | Out-Null

$runTag = (Get-Date -Format "yyyyMMddHHmmss") + "-" + ([guid]::NewGuid().ToString("N").Substring(0, 8))
$stageRoot = Join-Path $env:TEMP ("rime-neural-install-" + $runTag)
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null

$stagedModelDir = $null
$stagedOrtDll   = $null
$builtDll       = ""
$luaSrc         = Join-Path $repoRoot "lua\rime_neural.lua"
$serverExe      = Resolve-WeaselServerExe
$wasRunning     = [bool](Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
$stoppedWeasel  = $false

try {
    # 1. Stage & SHA-256 verify ONNX model + paired tokenizer/vocab (never leaving conflicting .onnx models)
    $modelTarget = Join-Path $neuralDir $mcfg.File
    $tokTarget   = Join-Path $neuralDir "tokenizer.json"
    $vocabTarget = Join-Path $neuralDir "unified_vocab.json"
    $otherOnnxPresent = @($knownOnnxFiles | Where-Object { $_ -ne $mcfg.File -and (Test-Path (Join-Path $neuralDir $_)) }).Count -gt 0

    $needModel = $ForceDownload -or $otherOnnxPresent -or `
        -not (Test-Path $modelTarget) -or `
        -not (Test-Path $tokTarget) -or `
        -not (Test-Path $vocabTarget)

    if (-not $needModel -and $mcfg.Sha256) {
        if ((Get-FileHash $modelTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mcfg.Sha256) {
            Write-Output "Existing $($mcfg.File) hash differs; refreshing..."
            $needModel = $true
        }
    }
    if (-not $needModel -and $mcfg.TokSha256) {
        if ((Get-FileHash $tokTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mcfg.TokSha256) {
            Write-Output "Existing tokenizer.json does not match $($mcfg.File); refreshing..."
            $needModel = $true
        }
    }
    if (-not $needModel -and $mcfg.VocabSha256) {
        if ((Get-FileHash $vocabTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mcfg.VocabSha256) {
            Write-Output "Existing unified_vocab.json does not match $($mcfg.File); refreshing..."
            $needModel = $true
        }
    }

    if ($needModel) {
        $modelStage = Join-Path $stageRoot "model"
        New-Item -ItemType Directory -Force -Path $modelStage | Out-Null
        if ($LocalModelArchive -and (Test-Path $LocalModelArchive)) {
            $archive = (Resolve-Path $LocalModelArchive).Path
            Write-Output "Using local model archive: $archive"
        } else {
            $archive = Join-Path $modelStage "model.zip"
            Write-Output "Downloading $($mcfg.Name)..."
            & curl.exe -fSL --retry 3 --connect-timeout 20 -o $archive $mcfg.Url
            if ($LASTEXITCODE -ne 0) { throw "Model download failed (ExitCode=$LASTEXITCODE)" }
        }

        $extDir = Join-Path $modelStage "extracted"
        New-Item -ItemType Directory -Force -Path $extDir | Out-Null
        Expand-Archive -Path $archive -DestinationPath $extDir -Force

        $modelSrc = Get-ChildItem $extDir -Recurse -Filter $mcfg.File | Select-Object -First 1
        if (-not $modelSrc) { throw "$($mcfg.File) not found in archive." }
        $actualHash = (Get-FileHash $modelSrc.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($mcfg.Sha256 -and ($actualHash -ne $mcfg.Sha256)) {
            throw "SHA-256 mismatch on $($mcfg.File)! Expected=$($mcfg.Sha256) Got=$actualHash"
        }
        $srcModelDir = $modelSrc.DirectoryName
        foreach ($reqAsset in @("tokenizer.json", "unified_vocab.json")) {
            $reqPath = Join-Path $srcModelDir $reqAsset
            if (-not (Test-Path $reqPath)) { throw "$reqAsset missing from model archive." }
        }
        if ($mcfg.TokSha256) {
            $hTok = (Get-FileHash (Join-Path $srcModelDir "tokenizer.json") -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($hTok -ne $mcfg.TokSha256) { throw "SHA-256 mismatch on tokenizer.json! Expected=$($mcfg.TokSha256) Got=$hTok" }
        }
        if ($mcfg.VocabSha256) {
            $hVoc = (Get-FileHash (Join-Path $srcModelDir "unified_vocab.json") -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($hVoc -ne $mcfg.VocabSha256) { throw "SHA-256 mismatch on unified_vocab.json! Expected=$($mcfg.VocabSha256) Got=$hVoc" }
        }
        Write-Output "Verified $($mcfg.File) + paired vocab/tokenizer (SHA-256: $($actualHash.Substring(0,12))...)."
        $stagedModelDir = $srcModelDir
    } else {
        Write-Output "Model $($mcfg.File) already verified in $neuralDir."
    }

    # 2. Stage onnxruntime.dll (copied ONLY after WeaselServer is stopped)
    $targetOrt = Join-Path $neuralDir "onnxruntime.dll"
    if (-not (Test-Path $targetOrt) -or $OrtDllPath) {
        $resolvedOrt = ""
        if ($OrtDllPath -and (Test-Path $OrtDllPath)) {
            $resolvedOrt = (Resolve-Path $OrtDllPath).Path
        }
        if (-not $resolvedOrt -and (Get-Command python -ErrorAction SilentlyContinue)) {
            $pyCode = "import sys, pathlib`ntry:`n  import onnxruntime`n  ver = tuple(int(x) for x in onnxruntime.__version__.split('.')[:2])`n  p = pathlib.Path(onnxruntime.__file__).parent / 'capi' / 'onnxruntime.dll'`n  if sys.maxsize > 2**32 and ver >= (1, 17) and p.is_file(): print(p)`nexcept Exception:`n  pass"
            $pyRes = Invoke-NativeSafe -FilePath "python" -ArgumentList @("-c", $pyCode)
            if ($pyRes.ExitCode -eq 0 -and $pyRes.Output.Count -gt 0) {
                $candPath = ($pyRes.Output | Select-Object -First 1).Trim()
                if ($candPath -and (Test-Path $candPath)) {
                    $resolvedOrt = $candPath
                }
            }
        }
        if (-not $resolvedOrt) {
            $ortStage = Join-Path $stageRoot "ort"
            New-Item -ItemType Directory -Force -Path $ortStage | Out-Null
            $nupkg = Join-Path $ortStage "ort.zip"
            Write-Output "Downloading pinned Microsoft.ML.OnnxRuntime $pinnedOrtVersion (win-x64) from NuGet..."
            & curl.exe -fSL --retry 3 --connect-timeout 20 -o $nupkg $pinnedOrtUrl
            if ($LASTEXITCODE -ne 0) { throw "ONNX Runtime download failed (ExitCode=$LASTEXITCODE)." }
            $gotSha512 = Get-FileSha512Base64 -Path $nupkg
            if ($gotSha512 -ne $pinnedOrtSha512) {
                throw "SHA-512 mismatch on Microsoft.ML.OnnxRuntime $pinnedOrtVersion! Expected=$pinnedOrtSha512 Got=$gotSha512"
            }
            Write-Output "Verified Microsoft.ML.OnnxRuntime $pinnedOrtVersion NuGet SHA-512."
            $ortEx = Join-Path $ortStage "ex"
            Expand-Archive -Path $nupkg -DestinationPath $ortEx -Force
            $dll = Join-Path $ortEx "runtimes\win-x64\native\onnxruntime.dll"
            if (-not (Test-Path $dll)) { throw "onnxruntime.dll not found inside NuGet package." }
            $stagedOrtDll = $dll
        } else {
            $stagedOrtDll = $resolvedOrt
        }
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
        & curl.exe -fSL --retry 3 --connect-timeout 20 -o $extSums $sumsUrl
        if ($LASTEXITCODE -ne 0) { throw "Release SHA256SUMS.txt download failed (ExitCode=$LASTEXITCODE)" }

        $relZip = Join-Path $relStage $zipName
        Write-Output "Downloading $zipName..."
        & curl.exe -fSL --retry 3 --connect-timeout 20 -o $relZip $zipUrl
        if ($LASTEXITCODE -ne 0) { throw "Release download failed (ExitCode=$LASTEXITCODE)" }

        # Fail-closed verification against external Release SHA256SUMS.txt (and GitHub API digest if present)
        Assert-Sha256InSums -SumsPath $extSums -EntryName $zipName -ActualPath $relZip
        if ($zipDigest) {
            $gotZip = (Get-FileHash $relZip -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($gotZip -ne $zipDigest) {
                throw "GitHub asset digest mismatch on $zipName! Expected=$zipDigest Got=$gotZip"
            }
        }

        $relEx = Join-Path $relStage "ex"
        Expand-Archive -Path $relZip -DestinationPath $relEx -Force
        $builtDll = Join-Path $relEx "rime_neural.dll"
        $relLua   = Join-Path $relEx "rime_neural.lua"
        Assert-Sha256InSums -SumsPath $extSums -EntryName "rime_neural.dll" -ActualPath $builtDll
        Assert-Sha256InSums -SumsPath $extSums -EntryName "rime_neural.lua" -ActualPath $relLua
        $luaSrc = $relLua
    } elseif ($PrebuiltDll -and (Test-Path $PrebuiltDll)) {
        $builtDll = (Resolve-Path $PrebuiltDll).Path
    } else {
        $builtDll = Join-Path $repoRoot "target\release\rime_neural.dll"
        if (-not $SkipBuild -or -not (Test-Path $builtDll)) {
            if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
                throw "cargo not found. Pass -Prebuilt to download the official Release DLL, or install Rust (https://rustup.rs)."
            }
            Write-Output "Building rime_neural (release --locked)..."
            & cargo build --release --locked --manifest-path (Join-Path $repoRoot "Cargo.toml")
            if ($LASTEXITCODE -ne 0) { throw "cargo build failed (ExitCode=$LASTEXITCODE)" }
        }
    }

    if (-not (Test-Path $builtDll)) { throw "Native DLL missing: $builtDll" }
    if (-not (Test-Path $luaSrc)) {
        throw "Lua filter missing: $luaSrc (run from a repository checkout or pass -Prebuilt)."
    }

    # 4. Stop WeaselServer cleanly BEFORE replacing any DLL or model files, then restart detached
    if ($wasRunning) {
        Stop-WeaselServerClean -ServerExe $serverExe
        $stoppedWeasel = $true
    }

    if ($stagedModelDir) {
        foreach ($oldOnnx in $knownOnnxFiles) {
            if ($oldOnnx -ne $mcfg.File) {
                $oldPath = Join-Path $neuralDir $oldOnnx
                if (Test-Path $oldPath) {
                    Remove-Item $oldPath -Force
                    Write-Output "Removed conflicting model $oldOnnx from $neuralDir."
                }
            }
        }
        foreach ($f in @($mcfg.File, "tokenizer.json", "unified_vocab.json", "rl_agent_config.json", "LICENSE")) {
            $src = Join-Path $stagedModelDir $f
            if (Test-Path $src) {
                Copy-Item $src (Join-Path $neuralDir $f) -Force
            }
        }
        Write-Output "Installed model $($mcfg.File) -> $neuralDir."
    }

    if ($stagedOrtDll) {
        Copy-Item $stagedOrtDll $targetOrt -Force
        Write-Output "Installed onnxruntime.dll -> $targetOrt."
    }

    Copy-Item $builtDll (Join-Path $neuralDir "rime_neural.dll") -Force
    Write-Output ("Installed rime_neural.dll ({0:F2} MB) -> {1}" -f ((Get-Item $builtDll).Length / 1MB), $neuralDir)

    Copy-Item $luaSrc (Join-Path $luaDir "rime_neural.lua") -Force
    Write-Output "Installed rime_neural.lua -> $luaDir"

    if ($wasRunning) {
        Start-WeaselServerDetached -ServerExe $serverExe
        $stoppedWeasel = $false
    }
}
catch {
    if ($stoppedWeasel -and $wasRunning) {
        try { Start-WeaselServerDetached -ServerExe $serverExe } catch {}
    }
    throw
}
finally {
    if ($stageRoot -and (Test-Path $stageRoot)) {
        Remove-Item $stageRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Output "Done! Mount `lua_filter@*rime_neural` in your schema custom.yaml and run Weasel Deploy."
exit 0
