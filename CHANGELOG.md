# Changelog

All notable changes to `rime-neural` are documented in this file.

## [Unreleased]

### Security, Privacy & Soundness
- **Zero-Plaintext Feedback & Legacy Scrubbing (`src/feedback.rs`, `src/lib.rs`, `lua/rime_neural.lua`, `scripts/tune-fusion.py`)**:
  - Switched `feedback.jsonl` to schema `v: 2` numeric-only records (`model`, `has_gram`, `has_ctx`, `logits`, `gram_deltas`, `char_lps`, candidate indices; never persists user text, keystrokes, or UIA context), with symlink rejection, input length caps, and atomic `MoveFileExW` 5 MiB rotation.
  - Added `--scrub-legacy-inplace`, `--schema`, `--model`, Kish `N_eff` gating, strict type/bound parsing, and domain sanity guards (`w_logit >= 0.1`, `w_rank <= 0.0`, `w_logit + w_ctx_lp >= 0.1`) to `scripts/tune-fusion.py`.
- **Fail-Closed UIA Hardening (`src/uia.rs`)**:
  - Upgraded background worker to MTA (`COINIT_MULTITHREADED`) with 200 ms `IUIAutomation6` timeouts, `(hwnd, pid)` denylist cache, post-read foreground `HWND` & `GetAncestor(GA_ROOT)` verification, deferred trailing probes when throttled by `MIN_REFRESH_INTERVAL`, fail-closed `ERROR_ACCESS_DENIED` / `CurrentIsPassword` / `MoveEndpointByUnit` / `end_inside < 0` checks, bullet-masked password detection, and expanded password-manager / remote-session denylist.
- **Lua 5.4 FFI & Rust 2024 Hardening (`Cargo.toml`, `src/lua_api.rs`, `src/lib.rs`)**:
  - Upgraded to Rust `edition = "2024"` (`rust-version = "1.92"`) with `#![warn(clippy::undocumented_unsafe_blocks)]` (0 warnings).
  - Verified `lua_version(L) >= 504.0`, checked `pin_self_module()` return value, switched to strict `LUA_TSTRING` checks before `lua_tolstring` and metamethod-free `lua_rawget`/`lua_rawset`, checked `tointegerx` validity (`isnum`) in `lua_log_feedback`, and wrapped all `extern "C"` exports in `catch_unwind`.
- **CI & Supply-Chain Hardening (`.github/workflows/ci-release.yml`, `scripts/install.ps1`)**:
  - Eliminated PowerShell `${{ github.ref_name }}` injection in `publish-release`, scoped `contents: write` strictly to `publish-release`, added non-Rust CI gates (PowerShell AST, Python tuner self-test, YAML examples, tag-version parity, `SHA256SUMS.txt` LF & self-verification).
  - Fixed `feather` model download URL (`laya-zhen-feather-preview-1121-slim.zip`) and pinned SHA-256 hashes for both `pumpkin` and `feather` `.onnx`/`tokenizer.json`/`unified_vocab.json`, pinned SHA-512 Microsoft NuGet `onnxruntime.dll` (1.31.0), added fail-closed external `SHA256SUMS.txt` checks, PS 5.1 stderr-safe `curl`/`cargo` execution, `-DataDir` support, and backup-and-rollback on install failure.

### Performance & NLP Correctness
- **Engine Hot-Path & Cold-Start Optimizations (`src/engine.rs`, `src/gram.rs`, `src/pinyin.rs`)**:
  - Implemented 3–4× faster single-pass forward double-array trie walk in `gram.rs` (bit-identical to `octagram.cc`) with `FILE_FLAG_RANDOM_ACCESS` and checked `Rime::Grammar/1.0` bounds; reused Pass-1 `.gram` scores in Pass-2.
  - Replaced 3× JSON re-parse of 32 MB `tokenizer.json` for `choice_compact` models with in-memory `tokenizer.clone()` + `Metaspace(PrependScheme::Never)`; switched to `encode_fast`, dynamic special-token filtering, monotonic-clock LRU, token-ID prefix KV-cache with shared `true_mask`, precomputed `slot_allowed`, and sequential inter-op execution (`with_parallel_execution(false)`, `with_inter_op_spinning(false)`).
  - Merged `uv.ml.pinyin` into `engine.readings` (+2,737 Traditional/variant/extended Hanzi), gated P2C synthesis strictly on `all_same_len`, guarded `shifted_lcp` 1:1 alignment, supported `'`-separated chunks in Double Pinyin and adjacent-transposition fuzzy recovery in Full Pinyin.
- **Lua Filter Lifecycle & Feedback Accuracy (`lua/rime_neural.lua`)**:
  - Preserved rerank snapshots across trailing punctuation auto-commits (`,`, `.`, `?`, `!`, `:`, `;`) with exact segment-range commit extraction while clearing stale snapshots on partial-segment selection; prevented Traditional UIA double-stripping via atomic `preedit_stripped` flag; fixed non-ASCII feedback path (always under writable `user_dir\neural\`); added per-path DLL failure tracking, lazy `.gram` attachment, and module-level OpenCC caching.

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
