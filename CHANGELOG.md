# Changelog

All notable changes to `rime-neural` are documented in this file.

## [Unreleased]

## [v1.2.0] - 2026-11-24

> **Upgrade Note**: Users upgrading from `v1.0.0` or `v1.1.0` who previously enabled `log_feedback: true` should run `python scripts\tune-fusion.py --scrub-legacy-inplace` to convert historical `v1` records in `feedback.jsonl` / `feedback.jsonl.1` to the zero-plaintext `v2` schema in place.

### Security, Privacy & Soundness
- **Zero-Plaintext Feedback & Reparse-Safe Rotation (`src/feedback.rs`, `src/lib.rs`, `lua/rime_neural.lua`, `scripts/tune-fusion.py`, `tests/test_tune_fusion.py`)**:
  - Switched `feedback.jsonl` to schema `v: 2` numeric-only records (`model`, `has_gram`, `has_ctx`, `logits`, `gram_deltas`, `char_lps`, clamped `shown_top1_idx`/`chosen_idx` against `num_cands <= 64`; never persists user text, keystrokes, or UIA context).
  - Eliminated symlink/reparse-point TOCTOU races using `FILE_FLAG_OPEN_REPARSE_POINT` + `GetFileInformationByHandle` across the target file, `.1` backup, and parent directory; propagated hard rotation errors and enforced a 2× hard cap on transient sharing-violation locks during atomic `MoveFileExW` 5 MiB rotation.
  - Added `--scrub-legacy-inplace` (with Windows `PermissionError` retry and hybrid-record plaintext stripping), `--schema`, `--model`, `--has-gram`, preserved `--top1-sample-rate` on `v1` logs before and after scrubbing, scaled MAP prior regularization by Kish `N_eff`, froze collinear `w_ctx_lp` unless both `has_ctx` strata have `>= 20` samples, enforced strict integer/domain sanity checks, and added a dedicated `unittest` suite (`tests/test_tune_fusion.py`).
- **Fail-Closed UIA Hardening & Focus-Change Events (`src/uia.rs`, `lua/rime_neural.lua`)**:
  - Upgraded the background worker to MTA (`COINIT_MULTITHREADED`) with 200 ms `IUIAutomation6` timeouts and an `IUIAutomationFocusChangedEventHandler` for immediate snapshot invalidation and same-window password detection when focus moves; throttled refreshes schedule a trailing deferred probe.
  - Clamped `TextPattern` ranges strictly inside `[elem_range.Start, elem_range.End]` (`RangeFromChild(&focused)`, aborting without `ValuePattern` fallthrough on out-of-element carets or failed `MoveEndpointByUnit`), fixed blank-line paragraph bleed, NBSP tail trimming, and partial-preedit stripping, used dynamic buffer growth for `QueryFullProcessImageNameW`, refused unresolved PIDs without caching, and expanded the fail-closed denylist to cover password managers/authenticators (including versioned portable prefixes `mobaxterm_*` / `keepass-*`), credential/UAC UIs, remote-session clients, and local terminals (`WindowsTerminal`, `conhost`, `cmd`, `powershell`, `pwsh`, `mintty`, `alacritty`, `wezterm-gui`, `kitty`).
  - Froze UIA context per composition in Lua (`comp_ctx`), sliced `preedit_before_caret` via `caret_pos`, lazily toggled `set_uia_enabled`, and prevented post-commit overwrite races.
- **Lua 5.4 FFI Soundness & Embedded PE Metadata (`Cargo.toml`, `build.rs`, `src/lua_api.rs`, `src/lib.rs`)**:
  - Upgraded to Rust `edition = "2024"` (`rust-version = "1.92"`) with `[lints.clippy]` `undocumented_unsafe_blocks = "warn"` and `panic = "unwind"` in `[profile.release]`.
  - Probed `rime.dll` / `librime.dll` / `lua54.dll` / current process requiring Lua 5.4-exclusive `lua_newuserdatauv` before invoking `lua_version(L)`, enforced `(504.0..505.0)` prior to `pin_self_module()`, switched all Lua C bindings and entry points to `extern "C-unwind"` with `catch_unwind` boundaries, switched to strict `LUA_TSTRING` checks and metamethod-free `lua_rawget`/`lua_rawset`, fixed `read_weights` direct `f64` parsing, and exposed module `version` + per-call `.gram` / `has_ctx` / `has_gram` fields.
  - Added pure-Rust `build.rs` embedding Win32 `VS_VERSION_INFO` into `rime_neural.dll`.
- **Installer, Skill Scripts & CI Hardening (`scripts/install.ps1`, `skills/rime-neural-setup/`, `.github/workflows/ci-release.yml`, `SECURITY.md`)**:
  - Hardened `install.ps1` with `--proto =https --proto-redir =https --max-time 300`, allowlisted ZipSlip-checked extraction (`Expand-ZipAllowlisted`), pinned `onnxruntime.dll` 1.31.0 SHA-512/SHA-256 + PE32+ x64 validation, verified `WeaselServer` exit before mutation, tracked newly created files for full rollback, fixed `feather` model download URL and pinned model SHA-256 digests, added `-DataDir` support, and fixed `Invoke-NativeSafe` missing-binary exit code handling.
  - Hardened `inspect-env.ps1` and `verify-install.ps1` with strict PE32+ export table validation (`lua_newuserdatauv` + all 21 symbols), fixed `active_schema_list` array serialization, added `data_dir` YAML unescaping, DLL `FileVersion` reporting, and active-session glog timestamp filtering.
  - Hardened CI with Windows PowerShell 5.1 AST/ASCII/no-BOM validation, Linux `luac5.4 -p` syntax linting, Rust 1.92 MSRV checking, `--locked` Clippy, `unittest` discovery, real `SHA256SUMS.txt` regex self-verification, tag parity across `Cargo.toml`/`Cargo.lock`/`CHANGELOG.md`/PE `ProductVersion`, and `--prerelease` handling; added `SECURITY.md`.

### Performance & NLP Correctness
- **`.gram` Non-BMP Equivalence & Per-Schema Isolation (`src/gram.rs`, `src/engine.rs`, `lua/rime_neural.lua`)**:
  - Implemented a 3–4× faster single-pass forward double-array trie walk with `FILE_FLAG_RANDOM_ACCESS` and checked `Rime::Grammar/1.0` bounds, fixed non-BMP UTF-8 encoding (`〇`, CJK Ext A–I) to match upstream `librime-octagram` bit-for-bit, added configurable `collocation_max_length` (`2..=8`), cached `.gram` readers per `(path, penalties, max_length)` across schemas (with `shared_data_dir` fallback), and added synthetic trie unit tests.
- **Pinyin Alignment & Double Pinyin Fixes (`src/pinyin.rs`)**:
  - Split Full Pinyin alignment into 3 ordered phases (exact -> fuzzy transposition across readings and user aliases -> ASCII partial prefix) to eliminate polyphone partial-stealing, replaced the greedy fallback chunker with an `O(L)` dynamic-programming splitter, removed invalid Double Pinyin pair fallback to preserve 1:1 slot alignment, and added `ms`/`jj` schema token recognition.
- **Engine Robustness & Caching (`src/engine.rs`)**:
  - Fixed `AsyncInitGuard::drop` TOCTOU ordering, validated required ONNX outputs at initialization (eliminating unchecked output-index panics), handled `PreTokenizerWrapper::Sequence` in memory for no-prepend tokenizers, switched special-token sanitization to a dynamic first-byte trigger table preserving ordinary HTML vocabulary tokens and `<unk>`, merged `uv.ml.pinyin` with single-char guard and extended-char `syl_to_all_cids`, suppressed spurious `char_lps` penalties on equal-length alternate segmentations, added finiteness guards to Pass-2 P2C verification, and cached `(last_ctx_str, last_ctx_ids)` with `kv_idx` reuse across Pass-1/Pass-2.
- **Lua Filter Usability & Compatibility (`lua/rime_neural.lua`)**:
  - Auto-detected optional `switches` declaration (`require_option`), expanded Traditional Chinese detection for `rime-wanxiang`, added configurable `rime_neural/tags` allowlist (default `["abc"]`), normalized Windows paths before `package.loadlib`, preserved rerank snapshots across trailing punctuation auto-commits while clearing stale snapshots on partial-segment selection, and kept `feedback.jsonl` under writable `user_dir\neural\`.

## [v1.1.0] - 2026-11-23

### Improvements & Fixes
- **Hot-Path Allocation Reduction & Async Diagnostics (`src/engine.rs`, `src/lib.rs`)**:
  - Decodes `char_logits` directly from ONNX Runtime's borrowed output slice (removing a 300–768 KB per-keystroke copy), splits full/partial unit lookup tables to avoid per-syllable string formatting, reuses `i64` input scratch buffers across keystrokes, and reuses Pass-1 candidate token IDs during Pass-2 P2C verification.
  - Includes `p2c_verify_margin` in the evaluation LRU cache key.
  - Captures background `init_async` failures in `stats().last_error` and logs them once via Rime `log.warning`.
- **Full Pinyin `u`/`v` Alias Recovery (`src/pinyin.rs`)**:
  - Aligns common Full Pinyin user spellings (`lue`/`nue`, `[jqxy]v`, `[jqxy]ve`, `[jqxy]van`, `[jqxy]vn`) to canonical vocabulary units (`lve`/`nve`, `[jqxy]u*`) across memoized alignment, explicit separator fast path, and greedy fallback, while keeping ambiguous multi-syllable readings (`ce lu e` -> `侧录额`) intact.
- **UIA Context & Post-Commit Hardening (`src/uia.rs`, `src/lib.rs`)**:
  - Strips zero-width format anchors (`U+200B`, `U+200C`, `U+200D`, `U+2060`, `U+FEFF`) before preedit suffix stripping for Electron / Chromium rich-text editors (ProseMirror / Slate).
  - Supports stripping inline `preedit_type: preview` candidate text and bounds `ValuePattern` fallback strings to a 256-char tail.
  - Adds `kick_uia_after_commit()` to schedule a settled post-commit UIA refresh off the main thread without being dropped by the 35 ms keystroke throttle.
- **Lua Filter Context Normalization & Feedback Fixes (`lua/rime_neural.lua`)**:
  - Automatically applies `tw2sp.json` (falling back to `t2s.json`, overridable via `rime_neural/t2s_config`) when Traditional mode is active or Traditional characters appear, preserving Mainland Simplified phrasing (`程序`, `文件`) while normalizing Taiwan phrases (`程式最佳化`, `軟體`, `記憶體`, `看著`).
  - Normalizes UIA / `commit_history` text before deduplicating against multi-segment `comp_prefix`, strips `comp_prefix` before multi-segment feedback rank matching, and skips reranking on mixed uppercase/digit inputs in Full Pinyin as well as Double Pinyin.
- **Tuner & Verification Tooling (`scripts/tune-fusion.py`, `skills/rime-neural-setup/scripts/verify-install.ps1`)**:
  - Adds Armijo backtracking line search to the Newton-Raphson MAP solver and prints a rerank disagreement/acceptance breakdown on every run.
  - Extends `verify-install.ps1` to scan `W`/`E` records inside recent Weasel `.INFO.` logs in addition to `WARNING`/`ERROR` log files.

## [v1.0.0] - 2026-11-23

First standalone public release (Apache-2.0), audited across 1,600 in-domain/holdout sentences, 840 never-before-seen 5-domain OOD sentences, and all 3,008 derived 2-key codes across 7 Double Pinyin layouts.

### Highlights
- **Zero-Patch Official Weasel Support**: Dynamic Lua 5.4 C API symbol resolution from `rime.dll` + `GetModuleHandleExA(PIN)` module pinning (`0 ms` startup blocking via `init_async`, `0 ms` redeploy reuse, zero redeploy crashes).
- **Universal Pinyin Engine**: Full Pinyin (`O(N*L)` memoized alignment + explicit `'` separator preservation) and 7 Double Pinyin layouts (ZRM, Flypy, MSPY, Sogou, ABC, Ziguang, Jiajia/PYJJ) verified 3,008/3,008 against `rime-ice` / `rime-frost`.
- **Dynamic ONNX Introspection & Selective Output Fetching**: Automatically detects encoder/choice layers, attention heads, and `choice_compact` format (supports 12L `ML-pumpkin-1103` and 8L `zhen-feather-preview-1121` with the same DLL), and prunes unused ONNX outputs via `OutputSelector`.
- **Zero-Extra-RAM `.gram` Fusion + Pass-2 Verified P2C**: Win32 `CreateFileMappingW` shared-page-cache reader using the active schema's grammar penalties, zero-allocation `char_logits` decoding, and KV-cached Pass-2 cross-encoder + `.gram` OOV synthesis.
- **Privacy-Hardened Non-Blocking UIA Context**: Dedicated STA thread (`< 2 µs` hot path, keystroke-1 pre-warm), `RangeFromChild` + `U+FFFC` chrome clamping, synchronous `HWND` isolation, and hard password / password-manager blocking.
- **Unbiased Local Feedback & Anti-Overfitting MAP Tuner**: Pre-OpenCC genuine Simplified capture, automatic secret redaction, 5 MiB rotation, and Newton-Raphson MAP + 5-fold NLL CV + paired 95% CI tuner (`scripts/tune-fusion.py`).

[Unreleased]: https://github.com/Chth1z/rime-neural/compare/v1.2.0...HEAD
[v1.2.0]: https://github.com/Chth1z/rime-neural/compare/v1.1.0...v1.2.0
[v1.1.0]: https://github.com/Chth1z/rime-neural/compare/v1.0.0...v1.1.0
[v1.0.0]: https://github.com/Chth1z/rime-neural/releases/tag/v1.0.0
