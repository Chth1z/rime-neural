# Troubleshooting & Rollback Reference

## 1. Weasel Version Too Old (`rime_dll_lua54_exports: false`)

- **Symptom**: `inspect-env.ps1` reports `rime_dll_lua54_exports: false`, or Weasel logs show `rime_neural: init failed` / missing Lua symbols, or Weasel version is older than `0.17.4` (`librime < 1.17.0`).
- **Cause**: Official stable Weasel `0.16.x` does not export Lua 5.4 C API symbols (`lua_pcallk`, `lua_tolstring`, etc.) from `rime.dll`.
- **Fix**: Direct the user to install **official Weasel Nightly `0.17.4+`** (`librime >= 1.17.0`) from <https://github.com/rime/weasel/releases>. Do NOT try to replace `rime.dll` manually with random builds.

## 2. File Locked When Copying `rime_neural.dll`

- **Symptom**: `Copy-Item` fails with `The process cannot access the file because it is being used by another process`.
- **Cause**: `rime_neural.dll` intentionally pins itself into `WeaselServer.exe` (`GET_MODULE_HANDLE_EX_FLAG_PIN`) so Rime redeploys never crash or reload ONNX Runtime.
- **Fix**: Always stop `WeaselServer.exe` cleanly first:
  ```powershell
  & "$weaselDir\WeaselServer.exe" /q 2>$null
  Start-Sleep -Milliseconds 600
  Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  ```
  Copy the DLL, deploy, and restart `WeaselServer.exe` detached.

## 3. `WeaselServer.exe` Exits Right After Agent Command Finishes

- **Symptom**: `Start-Process WeaselServer.exe` appears to succeed during the tool call, then `Get-Process WeaselServer` returns empty in the next shell call.
- **Cause**: Many agent shells run commands inside a Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`), which terminates child processes when the shell exits.
- **Fix**: Launch `WeaselServer.exe` detached via WMI/CIM:
  ```powershell
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$weaselDir\WeaselServer.exe`"" } | Out-Null
  ```
  Then verify in a separate check that exactly **1** `WeaselServer` process is running.

## 4. Reranking Doesn't Seem to Trigger After Deploy

Check these in order:
1. Run `verify-install.ps1 -SchemaId <schema>` to confirm `lua_filter@*rime_neural` appears in `build/<schema>.schema.yaml` and precedes `simplifier`.
2. Verify `translator/max_sentences: 8` and `translator/sentence_cutoff_threshold: 0.3` are present in `build/<schema>.schema.yaml` — if Rime only produces 1 sentence candidate, there is nothing to rerank.
3. Check the F4 switch: make sure `neural_rerank` is ON (`神经`, not `统计`). Note: if `switches` has no `reset: 1`, `user.yaml` may remember an older OFF state.
4. For **`rime-wanxiang` Double Pinyin** users: verify `rime_neural/pinyin_layout` is explicitly set (e.g. `flypy`, `zrm`, `mspy`), NOT `auto` (which defaults `wanxiang` to `full`!).
5. Wait ~1–2 seconds on first launch: `sync_init: false` loads the 82 MB ONNX model asynchronously in the background so Weasel starts with `0 ms` delay.

## 5. Traditional Chinese (`s2t` / `s2twp`) Garbled or Ignored

- **Cause**: `lua_filter@*rime_neural` was placed **after** `simplifier` in `engine/filters`, so the model received Traditional Chinese instead of Simplified.
- **Fix**: Move `lua_filter@*rime_neural` to position `0` (`engine/filters/@before 0`) and redeploy. `rime_neural` natively extracts genuine pre-OpenCC simplified text for both reranking and feedback logs.

## 6. Clean Uninstall / Rollback

To completely remove `rime-neural` without affecting the user's Rime dicts or `userdb`:
1. Restore the backed-up `<schema_id>.custom.yaml.bak.*` (or remove the `rime_neural`, `lua_filter@*rime_neural`, and `neural_rerank` switch entries).
2. Stop `WeaselServer.exe` (`/q`).
3. Remove `<RimeUserDir>\lua\rime_neural.lua` and `<RimeUserDir>\neural\`.
4. Run `"<WeaselDir>\WeaselDeployer.exe" /deploy` and restart `WeaselServer.exe`.
