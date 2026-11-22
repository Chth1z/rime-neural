# AGENTS.md

Instructions for AI coding agents working with or installing `rime-neural`.

## Installing & Configuring `rime-neural` for a User

If the user asks you to install, configure, upgrade, or troubleshoot `rime-neural` on their Windows Rime / Weasel setup, follow [`skills/rime-neural-setup/SKILL.md`](./skills/rime-neural-setup/SKILL.md) strictly:

1. Run the read-only environment audit (`skills/rime-neural-setup/scripts/inspect-env.ps1`) before changing any file.
2. Confirm Weasel `>= 0.17.4` (`rime_dll_lua54_exports: true`), target schema(s), model (`pumpkin` vs `feather`), and optional `wanxiang-lts-zh-hans.gram`.
3. Back up any `*.custom.yaml` before editing; never modify `user.yaml`, `*.userdb`, `custom_phrase.txt`, or synced dictionaries.
4. Mount `lua_filter@*rime_neural` before `simplifier` (OpenCC), set `translator/max_sentences: 8` and `translator/sentence_cutoff_threshold: 0.3`, and set `rime_neural/pinyin_layout` explicitly for `rime-wanxiang` Double Pinyin users.
5. Deploy via `WeaselDeployer.exe /deploy`, restart `WeaselServer.exe` detached (`Win32_Process.Create`), and run `skills/rime-neural-setup/scripts/verify-install.ps1` in a separate command to confirm all checks pass.

## Developing / Building This Repository

- Language: Rust (`edition = "2021"`, `cdylib` target `rime_neural.dll`) + Lua 5.4 (`lua/rime_neural.lua`).
- Validation commands (must all pass before committing code changes):
  ```powershell
  cargo fmt --check
  cargo clippy --all-targets -- -D warnings
  cargo test --locked
  cargo build --release --locked
  ```
- Architecture notes: see [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).
