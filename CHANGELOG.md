# Changelog

All notable changes to `rime-neural` are documented in this file.

## [v1.0.0] - 2026-11-23

First standalone public release (Apache-2.0), audited across 1,600 in-domain/holdout sentences, 840 never-before-seen 5-domain OOD sentences, and all 3,008 derived 2-key codes across 7 Double Pinyin layouts.

### Highlights
- **Zero-Patch Official Weasel Support**: Dynamic Lua 5.4 C API symbol resolution from `rime.dll` + `GetModuleHandleExA(PIN)` module pinning (`0 ms` startup blocking via `init_async`, `0 ms` redeploy reuse, zero redeploy crashes).
- **Universal Pinyin Engine**: Full Pinyin (`O(N*L)` memoized alignment + explicit `'` separator preservation) and 7 Double Pinyin layouts (ZRM, Flypy, MSPY, Sogou, ABC, Ziguang, Jiajia/PYJJ) verified 3,008/3,008 against `rime-ice` / `rime-frost`.
- **Dynamic ONNX Introspection & Selective Output Fetching**: Automatically detects encoder/choice layers, attention heads, and `choice_compact` format (supports 12L `ML-pumpkin-1103` and 8L `zhen-feather-preview-1121` with the same DLL), and prunes unused ONNX outputs via `OutputSelector`.
- **Zero-Extra-RAM `.gram` Fusion + Pass-2 Verified P2C**: Win32 `CreateFileMappingW` shared-page-cache reader using the active schema's grammar penalties, zero-allocation `char_logits` decoding, and KV-cached Pass-2 cross-encoder + `.gram` OOV synthesis.
- **Privacy-Hardened Non-Blocking UIA Context**: Dedicated STA thread (`< 2 µs` hot path, keystroke-1 pre-warm), `RangeFromChild` + `U+FFFC` chrome clamping, synchronous `HWND` isolation, and hard password / password-manager blocking.
- **Unbiased Local Feedback & Anti-Overfitting MAP Tuner**: Pre-OpenCC genuine Simplified capture, automatic secret redaction, 5 MiB rotation, and Newton-Raphson MAP + 5-fold NLL CV + paired 95% CI tuner (`scripts/tune-fusion.py`).
