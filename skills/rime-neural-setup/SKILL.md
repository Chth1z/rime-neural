---
name: rime-neural-setup
description: >-
  Install, configure, upgrade, or troubleshoot the rime-neural ONNX neural sentence
  reranker plugin for Windows Rime / Weasel (小狼毫). Use when the user asks to
  install rime-neural, add local AI/neural reranking to Weasel or Rime schemas
  (rime-ice, rime-frost, rime-wanxiang, or Double Pinyin), pair it with
  wanxiang-lts-zh-hans.gram, or fix rime-neural deployment issues.
---

# `rime-neural` Installation & Configuration Skill

Safely installs and configures [`Chth1z/rime-neural`](https://github.com/Chth1z/rime-neural) on **official Windows Weasel (`librime >= 1.17.0` / Weasel `>= 0.17.4`)** without modifying `rime.dll` or disturbing the user's dictionaries and `*.userdb`.

## Workflow

### Step 1: Audit the Environment (Read-Only)

Run [`scripts/inspect-env.ps1`](./scripts/inspect-env.ps1) first to gather facts before changing any file. If the skill folder is installed locally, invoke it directly; if you only read this `SKILL.md` from a URL, fetch the two helper scripts to `%TEMP%` first:

```powershell
# If running from a local skill installation:
powershell -ExecutionPolicy Bypass -File "<skill_dir>\scripts\inspect-env.ps1"

# Or if running without a local checkout:
$s = Join-Path $env:TEMP "rime-neural-inspect-env.ps1"
& curl.exe -fSL -o $s "https://raw.githubusercontent.com/Chth1z/rime-neural/main/skills/rime-neural-setup/scripts/inspect-env.ps1"
powershell -ExecutionPolicy Bypass -File $s
```

Verify the JSON output before proceeding:
- **OS**: Must be Windows x64 (`is_64bit_os: true`). If on macOS/Linux, stop and inform the user that `rime-neural` currently supports Windows Weasel x64 only.
- **Weasel & Lua 5.4 Exports**: `weasel_dir` must be non-null AND `rime_dll_lua54_exports` must be `true`. If `false`, stop and tell the user to upgrade to official **Weasel Nightly `0.17.4+`** (<https://github.com/rime/weasel/releases>) first — see [`references/troubleshooting.md`](./references/troubleshooting.md).
- **Real User Directory**: Always use `rime_user_dir` from the audit (respects custom `HKCU\Software\Rime\Weasel\RimeUserDir`, not just `%APPDATA%\Rime`).
- **Target Schema(s)**: Check `previously_selected`, `active_schema_list`, `installed_gram_models`, and `wanxiang_layout_hints`.

### Step 2: Confirm User Preferences (Brief, Single Prompt)

Unless the user already specified all choices (or asked for "recommended defaults"), briefly summarize what you detected (Weasel version, user dir, active schema(s), whether a `.gram` model is already installed) and confirm:
1. **Which schema(s)** to enable `rime-neural` on (default: `previously_selected` or active pinyin schema).
2. **ONNX Model**:
   - `pumpkin` (default, recommended): `ML-pumpkin-1103` (12L, ~82 MB, highest accuracy).
   - `feather`: `zhen-feather-preview-1121` (8L, ~38 MB, lighter for older or battery-constrained laptops).
3. **Optional Wanxiang `.gram` (`wanxiang-lts-zh-hans.gram`, ~373 MB)** — *only ask if `installed_gram_models` is empty*:
   - Explain clearly that `rime-neural` works standalone, **but pairing it with `wanxiang-lts-zh-hans.gram` improves both Top-8 candidate recall and reranking accuracy (`0 MB` extra RAM via shared memory mapping)**.
4. **Optional Privacy Preferences** (mention defaults so there are no surprises):
   - `enable_uia_context: true` (reads up to 64 chars before cursor locally; password fields & password managers are hard-blocked).
   - `log_feedback: true` (redacted local commit log capped at 5 MiB for optional personal weight tuning).

### Step 3: Install Binaries, ONNX Runtime & Verified Model

**Important**: Because `rime_neural.dll` pins itself in `WeaselServer.exe`, Weasel must be stopped before replacing the DLL.

1. Download or run the official installer `scripts/install.ps1` with `-Prebuilt` (verifies SHA-256 of both `rime_neural.dll` from `Chth1z/rime-neural` Releases and the Apache-2.0 ONNX model from `dd1000001000/zhiyi_models`, and fetches official `onnxruntime.dll` win-x64):

   ```powershell
   $installer = Join-Path $env:TEMP "install-rime-neural.ps1"
   & curl.exe -fSL -o $installer "https://raw.githubusercontent.com/Chth1z/rime-neural/main/scripts/install.ps1"
   powershell -ExecutionPolicy Bypass -File $installer -Prebuilt -Model pumpkin -RimeDir "<rime_user_dir>"
   ```

   (Use `-Model feather` if the user chose the lightweight 8-layer model.)

2. **If the user opted to install `wanxiang-lts-zh-hans.gram`** and it is not already in `<rime_user_dir>`:
   Download the latest LTS `.gram` asset from `amzxyz/rime-wanxiang-data` (`releases/tags/LTS`) into `<rime_user_dir>\wanxiang-lts-zh-hans.gram`:

   ```powershell
   $rel = Invoke-RestMethod "https://api.github.com/repos/amzxyz/rime-wanxiang-data/releases/tags/LTS" -Headers @{ "User-Agent" = "rime-neural-setup" }
   $asset = $rel.assets | Where-Object { $_.name -eq "wanxiang-lts-zh-hans.gram" } | Select-Object -First 1
   & curl.exe -fSL --retry 3 -o "<rime_user_dir>\wanxiang-lts-zh-hans.gram" $asset.browser_download_url
   ```

### Step 4: Patch `<schema_id>.custom.yaml` Safely

Read [`references/schema-patches.md`](./references/schema-patches.md) (raw URL if running without a local clone: `https://raw.githubusercontent.com/Chth1z/rime-neural/main/skills/rime-neural-setup/references/schema-patches.md`) before editing any YAML file.

Key invariants:
- **Back up** any existing `<schema_id>.custom.yaml` to `<schema_id>.custom.yaml.bak.<timestamp>` before modifying.
- **Never** edit `user.yaml`, `*.userdb`, `custom_phrase.txt`, or synced dict folders.
- Ensure `translator/max_sentences: 8` and `translator/sentence_cutoff_threshold: 0.3` are set.
- Mount `lua_filter@*rime_neural` **before** any `simplifier` / OpenCC filter (typically `engine/filters/@before 0: lua_filter@*rime_neural`), and never duplicate it.
- Set `rime_neural/pinyin_layout` explicitly whenever using **`rime-wanxiang` with Double Pinyin** or a custom `schema_id`.
- Wire `grammar/language: wanxiang-lts-zh-hans` if you just installed the Wanxiang `.gram` model and the schema does not already configure `grammar:`.

### Step 5: Deploy, Restart Detached & Verify End-to-End

Never end the task without deploying and verifying!

1. **Stop `WeaselServer.exe`** (if running), run **Weasel Deploy**, and **start `WeaselServer.exe` detached** (so it survives when your tool shell exits):

   ```powershell
   $weaselDir = "<weasel_dir>"
   & "$weaselDir\WeaselServer.exe" /q 2>$null
   Start-Sleep -Milliseconds 600
   Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

   $p = Start-Process -FilePath "$weaselDir\WeaselDeployer.exe" -ArgumentList "/deploy" -Wait -PassThru
   if ($p.ExitCode -ne 0) { throw "WeaselDeployer failed with exit code $($p.ExitCode)" }

   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$weaselDir\WeaselServer.exe`"" } | Out-Null
   Start-Sleep -Milliseconds 800
   ```

2. **Run [`scripts/verify-install.ps1`](./scripts/verify-install.ps1)** in a **separate command** (this also proves `WeaselServer.exe` stayed alive after the previous shell exited):

   ```powershell
   # Local skill installation:
   powershell -ExecutionPolicy Bypass -File "<skill_dir>\scripts\verify-install.ps1" -RimeDir "<rime_user_dir>" -SchemaId "<schema_id>"

   # Or without a local checkout:
   $v = Join-Path $env:TEMP "rime-neural-verify.ps1"
   & curl.exe -fSL -o $v "https://raw.githubusercontent.com/Chth1z/rime-neural/main/skills/rime-neural-setup/scripts/verify-install.ps1"
   powershell -ExecutionPolicy Bypass -File $v -RimeDir "<rime_user_dir>" -SchemaId "<schema_id>"
   ```

   Every check in the JSON output must report `"ok": true`. If any check fails, consult [`references/troubleshooting.md`](./references/troubleshooting.md) (raw URL: `https://raw.githubusercontent.com/Chth1z/rime-neural/main/skills/rime-neural-setup/references/troubleshooting.md`) and resolve it before finishing.

### Step 6: Report Clearly to the User

Give a short summary covering:
- Installed components (`rime_neural.dll` version, ONNX model, `onnxruntime.dll`, optional `.gram` model) and paths.
- Modified config file(s) and exact backup path(s).
- Verification status (`WeaselDeployer` exit 0, single live `WeaselServer` PID, `build/<schema>.schema.yaml` confirmed, 0 errors in Weasel logs).
- How to toggle `统计 / 神经` in the `F4` menu, and a reminder that the first sentence after Weasel restarts may take ~1–2 seconds while the ONNX model finishes async warm-up in the background.
