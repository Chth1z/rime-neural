# AGENTS.md

Instructions for AI coding agents working with or installing `rime-neural`.

## Installing & Configuring `rime-neural` for a User

If the user asks you to install, configure, upgrade, or troubleshoot `rime-neural` on their Windows Rime / Weasel setup, follow [`skills/rime-neural-setup/SKILL.md`](./skills/rime-neural-setup/SKILL.md) strictly:

1. Run the read-only environment audit (`skills/rime-neural-setup/scripts/inspect-env.ps1`) before changing any file.
2. Confirm Weasel `>= 0.17.4` (`rime_dll_lua54_exports: true`), target schema(s), model (`pumpkin` vs `feather`), and optional `wanxiang-lts-zh-hans.gram`.
3. Back up any `*.custom.yaml` before editing; never modify `user.yaml`, `*.userdb`, `custom_phrase.txt`, or synced dictionaries.
4. Mount `lua_filter@*rime_neural` before `simplifier` (OpenCC), set `translator/max_sentences: 8` and `translator/sentence_cutoff_threshold: 0.3`, and set `rime_neural/pinyin_layout` explicitly for `rime-wanxiang` Double Pinyin users. If `rime_user_dir_non_ascii` is `true`, install to an ASCII `-DataDir` and quote `rime_neural/data_dir` with single quotes or forward slashes in YAML.
5. Deploy via `WeaselDeployer.exe /deploy`, restart `WeaselServer.exe` detached (`Win32_Process.Create`), and run `skills/rime-neural-setup/scripts/verify-install.ps1` in a separate command to confirm all checks pass.

## Developing / Building This Repository

- Language: Rust (`edition = "2024"`, `rust-version = "1.92"`, `cdylib` target `rime_neural.dll`) + Lua 5.4 (`lua/rime_neural.lua`).
- Validation commands (must all pass before committing code changes; matches `.github/workflows/ci-release.yml`):
  ```powershell
  # Rust formatting, strict Clippy, unit tests, and release DLL build
  cargo fmt --all -- --check
  cargo clippy --locked --all-targets -- -D warnings
  cargo test --locked
  cargo build --release --locked

  # Python tuner compilation & unit tests
  python -m py_compile scripts/tune-fusion.py tests/test_tune_fusion.py
  python -m unittest discover -s tests -v

  # YAML validation (examples + CI workflow)
  python -c "import glob, yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in glob.glob('examples/*.custom.yaml') + ['.github/workflows/ci-release.yml']]"

  # Windows PowerShell 5.1 AST + UTF-8 no-BOM + pure-ASCII check across all .ps1 scripts
  powershell.exe -NoProfile -NonInteractive -Command '$ErrorActionPreference="Stop"; foreach ($f in @("scripts\install.ps1","skills\rime-neural-setup\scripts\inspect-env.ps1","skills\rime-neural-setup\scripts\verify-install.ps1")) { $p=(Resolve-Path $f).Path; $b=[IO.File]::ReadAllBytes($p); if ($b.Length -ge 3 -and $b[0]-eq 0xEF -and $b[1]-eq 0xBB -and $b[2]-eq 0xBF) { throw "BOM: $f" }; if ($b | Where-Object { $_ -gt 127 } | Select-Object -First 1) { throw "Non-ASCII: $f" }; $t=$null;$e=$null; [void][Management.Automation.Language.Parser]::ParseFile($p,[ref]$t,[ref]$e); if ($e.Count -gt 0) { throw "AST: $f" } }'
  ```
- Lua 5.4 syntax check (CI runs `luac5.4 -p lua/rime_neural.lua` on Linux; on Windows, validate via `luac` or official Weasel `rime.dll`).
- Architecture notes: see [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).
- Security & privacy boundaries: see [`SECURITY.md`](./SECURITY.md).
