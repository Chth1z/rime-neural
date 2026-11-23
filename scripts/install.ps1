<#
.SYNOPSIS
    Install rime-neural (DLL + Lua filter + ONNX Runtime + verified ONNX model) into %APPDATA%\Rime.
.DESCRIPTION
    1. Downloads & verifies the selected Apache-2.0 ONNX reranker model:
         - pumpkin (default): Zhiyi ML-pumpkin-1103 (12L-384d fused KV-cache, highest Chinese accuracy)
         - feather          : Zhiyi zhen-feather-preview-1121 (8L-256d compact KV-cache, ultra-light)
    2. Locates or downloads official Microsoft.ML.OnnxRuntime win-x64 onnxruntime.dll.
    3. Installs rime_neural.dll + lua/rime_neural.lua from:
         - -Prebuilt (downloads latest official GitHub Release from Chth1z/rime-neural & checks SHA-256)
         - -ReleaseTag <tag> (downloads a specific GitHub Release tag, e.g. v1.0.0)
         - -PrebuiltDll <path> (uses a local prebuilt DLL)
         - local `cargo build --release` (default when running from a source checkout)
    4. Stops WeaselServer cleanly, installs files to %APPDATA%\Rime\neural\ & %APPDATA%\Rime\lua\,
       and restarts WeaselServer.
#>
param(
    [string]$RimeDir = (Join-Path $env:APPDATA "Rime"),
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
$repoRoot  = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$neuralDir = Join-Path $RimeDir "neural"
$luaDir    = Join-Path $RimeDir "lua"
$headers   = @{ "User-Agent" = "rime-neural-installer/1.0" }

$models = @{
    pumpkin = @{
        Name   = "ML-pumpkin-1103 (12L fused KV-cache, Apache-2.0)"
        Url    = "https://github.com/dd1000001000/zhiyi_models/releases/download/model-ml-pumpkin-1103/laya-ml-pumpkin-1103-int8g.zip"
        File   = "laya.one.fused.int8g.onnx"
        Sha256 = "c273b2cfc46179e9df2a94aaa381bb6fb1d55862fe8834b4631950180d521ca5"
    }
    feather = @{
        Name   = "zhen-feather-preview-1121 (8L compact KV-cache, Apache-2.0)"
        Url    = "https://github.com/dd1000001000/zhiyi_models/releases/download/model-zhen-feather-preview-1121/laya-zhen-feather-preview-1121-w4.zip"
        File   = "laya.one.w4.onnx"
        Sha256 = "98c615a1c2dfeb82de5c9f749e7910e3e900261a760ae82d43d2b63e9164a35c"
    }
}
$mcfg = $models[$Model]

New-Item -ItemType Directory -Force -Path $neuralDir, $luaDir | Out-Null

# 1. Ensure ONNX model files exist & pass SHA-256 verification
$modelTarget = Join-Path $neuralDir $mcfg.File
$needModel = $ForceDownload -or `
    -not (Test-Path $modelTarget) -or `
    -not (Test-Path (Join-Path $neuralDir "tokenizer.json")) -or `
    -not (Test-Path (Join-Path $neuralDir "unified_vocab.json"))

if (-not $needModel -and $mcfg.Sha256) {
    $curHash = (Get-FileHash $modelTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($curHash -ne $mcfg.Sha256) {
        Write-Output "Existing $($mcfg.File) hash differs; refreshing..."
        $needModel = $true
    }
}

if ($needModel) {
    $stage = Join-Path $env:TEMP ("rime-neural-model-" + (Get-Date -Format "yyyyMMddHHmmss"))
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    try {
        if ($LocalModelArchive -and (Test-Path $LocalModelArchive)) {
            $archive = (Resolve-Path $LocalModelArchive).Path
            Write-Output "Using local model archive: $archive"
        } else {
            $archive = Join-Path $stage "model.zip"
            Write-Output "Downloading $($mcfg.Name)..."
            & curl.exe -fSL --retry 3 --connect-timeout 20 -o $archive $mcfg.Url
            if ($LASTEXITCODE -ne 0) { throw "Model download failed (ExitCode=$LASTEXITCODE)" }
        }

        $extDir = Join-Path $stage "extracted"
        New-Item -ItemType Directory -Force -Path $extDir | Out-Null
        Expand-Archive -Path $archive -DestinationPath $extDir -Force

        $modelSrc = (Get-ChildItem $extDir -Recurse -Filter $mcfg.File | Select-Object -First 1)
        if (-not $modelSrc) { throw "$($mcfg.File) not found in archive." }
        $actualHash = (Get-FileHash $modelSrc.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($mcfg.Sha256 -and ($actualHash -ne $mcfg.Sha256)) {
            throw "SHA-256 mismatch on $($mcfg.File)! Expected=$($mcfg.Sha256) Got=$actualHash"
        }
        Write-Output "Verified $($mcfg.File) (SHA-256: $($actualHash.Substring(0,12))...)."

        $srcModelDir = $modelSrc.DirectoryName
        foreach ($f in @($mcfg.File, "tokenizer.json", "unified_vocab.json", "rl_agent_config.json", "LICENSE")) {
            $src = Join-Path $srcModelDir $f
            if (Test-Path $src) {
                Copy-Item $src (Join-Path $neuralDir $f) -Force
            }
        }
        Write-Output "Installed model $($mcfg.File) -> $neuralDir."
    }
    finally {
        Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Output "Model $($mcfg.File) already verified in $neuralDir."
}

# 2. Resolve onnxruntime.dll
$targetOrt = Join-Path $neuralDir "onnxruntime.dll"
if (-not (Test-Path $targetOrt) -or $OrtDllPath) {
    $resolvedOrt = ""
    if ($OrtDllPath -and (Test-Path $OrtDllPath)) {
        $resolvedOrt = (Resolve-Path $OrtDllPath).Path
    }
    if (-not $resolvedOrt -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $pyOrt = (& python -c "import onnxruntime, pathlib; p=pathlib.Path(onnxruntime.__file__).parent/'capi'/'onnxruntime.dll'; print(p if p.is_file() else '')" 2>$null)
        if ($pyOrt -and (Test-Path $pyOrt.Trim())) {
            $resolvedOrt = $pyOrt.Trim()
        }
    }
    if (-not $resolvedOrt) {
        $stageOrt = Join-Path $env:TEMP ("rime-neural-ort-" + (Get-Date -Format "yyyyMMddHHmmss"))
        New-Item -ItemType Directory -Force -Path $stageOrt | Out-Null
        try {
            $nupkg = Join-Path $stageOrt "ort.zip"
            Write-Output "Downloading official Microsoft.ML.OnnxRuntime (win-x64) from NuGet..."
            & curl.exe -fSL --retry 3 --connect-timeout 20 -o $nupkg "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime"
            if ($LASTEXITCODE -ne 0) { throw "ONNX Runtime download failed." }
            $ortEx = Join-Path $stageOrt "ex"
            Expand-Archive -Path $nupkg -DestinationPath $ortEx -Force
            $dll = Join-Path $ortEx "runtimes\win-x64\native\onnxruntime.dll"
            if (-not (Test-Path $dll)) { throw "onnxruntime.dll not found inside NuGet package." }
            Copy-Item $dll $targetOrt -Force
            Write-Output "Installed onnxruntime.dll from NuGet."
        }
        finally {
            Remove-Item $stageOrt -Recurse -Force -ErrorAction SilentlyContinue
        }
    } else {
        Copy-Item $resolvedOrt $targetOrt -Force
        Write-Output "Copied onnxruntime.dll from $resolvedOrt."
    }
} else {
    Write-Output "onnxruntime.dll already present in $neuralDir."
}

# 3. Resolve rime_neural.dll + lua/rime_neural.lua (GitHub Release / Prebuilt / Local Cargo Build)
$stageRel = $null
$builtDll = ""
$luaSrc   = Join-Path $repoRoot "lua\rime_neural.lua"

try {
    if ($Prebuilt -or $ReleaseTag) {
        $apiUrl = if ($ReleaseTag) {
            "https://api.github.com/repos/Chth1z/rime-neural/releases/tags/$ReleaseTag"
        } else {
            "https://api.github.com/repos/Chth1z/rime-neural/releases/latest"
        }
        Write-Output "Fetching official release metadata from $apiUrl..."
        $rel = Invoke-RestMethod $apiUrl -Headers $headers -TimeoutSec 20
        $zipAsset = $rel.assets | Where-Object { $_.name -match 'win-x64\.zip$' } | Select-Object -First 1
        if (-not $zipAsset) { throw "No win-x64.zip asset found in release $($rel.tag_name)." }

        $stageRel = Join-Path $env:TEMP ("rime-neural-rel-" + (Get-Date -Format "yyyyMMddHHmmss"))
        New-Item -ItemType Directory -Force -Path $stageRel | Out-Null
        $relZip = Join-Path $stageRel $zipAsset.name
        Write-Output "Downloading $($zipAsset.name) ($($rel.tag_name))..."
        & curl.exe -fSL --retry 3 --connect-timeout 20 -o $relZip $zipAsset.browser_download_url
        if ($LASTEXITCODE -ne 0) { throw "Release download failed (ExitCode=$LASTEXITCODE)" }

        $relEx = Join-Path $stageRel "ex"
        Expand-Archive -Path $relZip -DestinationPath $relEx -Force
        $builtDll = Join-Path $relEx "rime_neural.dll"
        $relLua   = Join-Path $relEx "rime_neural.lua"
        $sumFile  = Join-Path $relEx "SHA256SUMS.txt"
        if (-not (Test-Path $builtDll)) { throw "rime_neural.dll missing from release bundle." }
        if (Test-Path $sumFile) {
            foreach ($entry in @(
                @{ File = "rime_neural.dll"; Path = $builtDll },
                @{ File = "rime_neural.lua"; Path = $relLua }
            )) {
                if (-not (Test-Path $entry.Path)) { continue }
                $pat = [regex]::Escape($entry.File) + '$'
                $line = Get-Content $sumFile | Where-Object { $_ -match $pat } | Select-Object -First 1
                if ($line) {
                    $expected = ($line -split '\s+' | Select-Object -First 1)
                    $got = (Get-FileHash $entry.Path -Algorithm SHA256).Hash.ToLowerInvariant()
                    if ($got -ne $expected.ToLowerInvariant()) {
                        throw "SHA-256 mismatch on $($entry.File)! Expected=$expected Got=$got"
                    }
                    Write-Output "Verified $($entry.File) (SHA-256: $($got.Substring(0,12))...)."
                }
            }
        }
        if (Test-Path $relLua) { $luaSrc = $relLua }
    } elseif ($PrebuiltDll -and (Test-Path $PrebuiltDll)) {
        $builtDll = (Resolve-Path $PrebuiltDll).Path
    } else {
        $builtDll = Join-Path $repoRoot "target\release\rime_neural.dll"
        if (-not $SkipBuild -or -not (Test-Path $builtDll)) {
            if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
                throw "cargo not found. Pass -Prebuilt to download the official Release DLL, or install Rust (https://rustup.rs)."
            }
            Write-Output "Building rime_neural (release)..."
            & cargo build --release --manifest-path (Join-Path $repoRoot "Cargo.toml")
            if ($LASTEXITCODE -ne 0) { throw "cargo build failed (ExitCode=$LASTEXITCODE)" }
        }
    }

    if (-not (Test-Path $builtDll)) { throw "Native DLL missing: $builtDll" }

    # 4. Stop WeaselServer cleanly (so pinned DLL is not locked), copy DLL + Lua, restart
    $weaselDir = (Get-ChildItem "C:\Program Files\Rime" -Directory -Filter "weasel-*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    $serverExe = if ($weaselDir) { Join-Path $weaselDir "WeaselServer.exe" } else { $null }
    $wasRunning = [bool](Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)

    if ($wasRunning -and $serverExe) {
        & $serverExe /q 2>$null
        for ($i = 0; $i -lt 15; $i++) {
            if (-not (Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 250
        }
        Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    }

    Copy-Item $builtDll (Join-Path $neuralDir "rime_neural.dll") -Force
    Write-Output ("Installed rime_neural.dll ({0:F2} MB) -> {1}" -f ((Get-Item $builtDll).Length / 1MB), $neuralDir)

    if (Test-Path $luaSrc) {
        Copy-Item $luaSrc (Join-Path $luaDir "rime_neural.lua") -Force
        Write-Output "Installed rime_neural.lua -> $luaDir"
    }

    if ($wasRunning -and $serverExe) {
        try {
            Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$serverExe`"" } | Out-Null
        } catch {
            Start-Process -FilePath $serverExe
        }
        Start-Sleep -Milliseconds 700
        $procs = @(Get-Process -Name WeaselServer -ErrorAction SilentlyContinue)
        Write-Output "Restarted WeaselServer (count=$($procs.Count) PID=$(($procs | ForEach-Object { $_.Id }) -join ','))."
    }
}
finally {
    if ($stageRel -and (Test-Path $stageRel)) {
        Remove-Item $stageRel -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Output "Done! Mount `lua_filter@*rime_neural` in your schema custom.yaml and run Weasel Deploy."
