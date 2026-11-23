# Changelog

All notable changes to `rime-neural` are documented in this file.

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
