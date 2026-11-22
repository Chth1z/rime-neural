# rime-neural

Zero-patch, in-process ONNX neural sentence reranker plugin for official Windows **Rime / Weasel** (`librime >= 1.17.0`).

`rime-neural` loads directly inside official unmodified `WeaselServer.exe` via Weasel's exported Lua 5.4 C API — no custom `rime.dll` build, no external daemon process, no cloud requests.

## Highlights

- **Zero Custom Weasel Build**: Dynamically binds the Lua 5.4 C API from `rime.dll` and pins `rime_neural.dll` in process so Rime redeploys never crash or reload the ONNX Runtime session (`0 ms` redeploy reuse, `0 ms` startup delay via `init_async`).
- **Universal Pinyin Support**: Auto-detects **Full Pinyin** and all **7 major Rime Double Pinyin layouts** — Natural (自然码), Flypy (小鹤), MSPY (微软, `; = ing`), Sogou (搜狗, `; = ing`), ABC (智能ABC), Ziguang (紫光, `; = ing`), and Jiajia/PYJJ (拼音加加). Exhaustively verified across all 3,008 derived 2-key codes in `rime-ice` / `rime-frost` (0 mismatches). Compatible with `rime-ice`, `rime-frost`, `rime-wanxiang`, and custom schemas.
- **Fused Prefix KV-Cache + Dynamic Model Introspection**: Uses 16-slot LRU prefix KV-cache and selective ONNX output fetching. Automatically introspects model layers/heads and `choice_compact` format, supporting both high-accuracy 12-layer **`ML-pumpkin-1103`** (default) and lightweight 8-layer **`zhen-feather-preview-1121`** without recompiling.
- **Zero-Extra-RAM `.gram` Fusion + Pass-2 Verified P2C**: Memory-maps Rime `.gram` files (such as `wanxiang-lts-zh-hans.gram`) read-only via Win32 `CreateFileMappingW` (`0 MB` extra physical RAM, `~0.01 ms`/sentence) using your schema's actual grammar penalties, and validates novel high-potential OOV `char_logits` candidates via KV-cached Pass-2 cross-encoder + `.gram` checks.
- **Non-Blocking, Privacy-Hardened Windows UIA Context**: Reads up to 64 characters strictly before the caret on a dedicated background STA thread (`< 2 µs` hot path, pre-warmed on keystrokes 1–2). Clamps ranges to `RangeFromChild(focused)` and `U+FFFC` boundaries so Chromium/Electron apps never bleed toolbar chrome, isolates foreground `HWND` to stop cross-app context bleed, and hard-blocks password fields (`IsPassword == true`) and password managers.
- **Strict Dictionary Protection**: Reranks only `sentence` candidates (`cand:get_genuine().type == "sentence"`), never overriding exact dictionary hits, userdb phrases, English words, dates, numbers, or punctuation.
- **Unbiased Local Feedback & Anti-Overfitting MAP Tuner**: Optional local `feedback.jsonl` (pre-OpenCC genuine Simplified text, auto-redacted secrets/digits/emails/URLs, 5 MiB auto-rotation) paired with `scripts/tune-fusion.py` (exact damped Newton-Raphson MAP shrinkage + 5-fold NLL CV + paired 95% CI gate).

## Quick Start (Windows / Weasel)

### 1. Install DLL, Lua Filter, ONNX Runtime & Model

Requires official Weasel with `librime >= 1.17.0` (e.g. Weasel Nightly `0.17.4+`).

Run the installer from PowerShell (downloads the official prebuilt `rime_neural.dll` + `rime_neural.lua` from GitHub Releases, verifies SHA-256, fetches `onnxruntime.dll`, and installs the Apache-2.0 `ML-pumpkin-1103` model into `%APPDATA%\Rime\neural\`):

```powershell
# Default: 12-layer ML-pumpkin-1103 (best accuracy)
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt

# Optional: 8-layer zhen-feather (lighter / faster on low-power CPUs)
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt -Model feather
```

Or build from source if you have Rust installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

### 2. Mount in Your Schema's `*.custom.yaml`

Ready-to-use patches are provided under [`examples/`](examples/) (`rime_ice`, `double_pinyin_flypy`, `double_pinyin_mspy`, `rime_frost_double_pinyin`, `wanxiang`). Minimal patch:

```yaml
patch:
  # 1. Expose Rime's top-8 sentence beam candidates to the neural reranker
  translator/max_sentences: 8
  translator/sentence_cutoff_threshold: 0.3

  # 2. Mount rime_neural FIRST (before simplifier / OpenCC filters)
  engine/filters/@before 0: lua_filter@*rime_neural

  # 3. Optional F4 switch (defaults to ON)
  switches/+:
    - name: neural_rerank
      states: [ 统计, 神经 ]
      reset: 1

  # 4. Reranker configuration (all fields optional; defaults shown)
  rime_neural:
    option_name: neural_rerank
    pinyin_layout: auto          # auto | full | zrm | flypy | mspy | sogou | abc | ziguang | jiajia
    max_candidates: 8
    fusion_weights: [1.0, -1.4, 0.05, 0.15, 0.0]
    p2c_verify_margin: 0.20      # <= 0 disables Pass-2 P2C synthesis
    p2c_top1_logit_diff: 2.0
    p2c_top1_min_gram_diff: -2.0
    p2c_top2_logit_diff: 0.5
    enable_uia_context: true
    threads: 0                   # 0 = auto-detect (clamped 4..8)
    log_feedback: true           # writes %APPDATA%\Rime\neural\feedback.jsonl (5 MiB cap)
    sync_init: false             # async background init -> 0 ms startup delay
```

Save and run **Weasel Deploy (重新部署)**. If the DLL or model is removed, or `neural_rerank` is toggled off, the filter degrades cleanly with zero overhead.

## Benchmarks (Live Official Weasel `0.17.4`)

Measured end-to-end through official unmodified `rime.dll` (`rime-frost` + `wanxiang-lts-zh-hans.gram`, `enable_user_dict: false` strict isolation, `gaboolic/rime-schema-compare` normalization protocol) on Intel Core Ultra 7 255HX:

### 1. In-Domain & Holdout Accuracy (1,600 Sentences, Double Pinyin, Live `commit_history`)

| Configuration | All 1,600 (Sent / Char) | Tech 200 | Zhihu Dev 400 | Zhihu Holdout 1,000 |
| :--- | :---: | :---: | :---: | :---: |
| Pure Statistical (`wanxiang` 372 MB `.gram`) | `75.12%` / `95.51%` | `81.00%` | `70.50%` | `75.80%` |
| **`rime-neural` v1.0.0 (`ML-pumpkin-1103`)** | **`82.19%`** / **`97.05%`** (`+7.07%`) | **`91.00%`** | **`79.75%`** | **`81.40%`** |

- Offline with punctuated preceding context (representing UIA in real editors/browsers): **`85.19%`** (`tech 92.00%`, `holdout 84.60%`).
- **Full Pinyin Tech-200** (`rime_frost`): pure statistical `80.50%` → **`rime-neural` `89.50%`** (strict single-sentence zero-context) / **`91.00%`** (continuous `commit_history`, char `98.72%`).
- **Lightweight 8L `zhen-feather` (`laya.one.w4.onnx`)** on Double Pinyin Tech-200: **`88.50%`** (char `98.23%`).

### 2. Never-Before-Seen Cross-Domain OOD Audit (840 Sentences, 5 Unseen Domains)

Tested on 5 completely untouched corpora (`news`, `novel`, `prose`, `old tech`, `2026-08-30 Zhihu`) with zero tuning on any of them:

| Configuration | OOD 840 All (Sent / Char) | News (152) | Novel (40) | Prose (211) | Tech Old (137) | Zhihu 08-30 (300) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Pure Statistical (`wanxiang` `.gram`) | `60.48%` / `90.94%` | `70.4%` | `65.0%` | `37.0%` | `66.4%` | `68.7%` |
| **`rime-neural` v1.0.0 (`[1.0, -1.4, 0.05, 0.15, 0.0]`)** | **`68.57%`** / **`93.52%`** (`+8.09%`) | **`78.3%`** | **`70.0%`** | **`46.4%`** | **`75.9%`** | **`75.7%`** |

### 3. Keystroke Latency (`bench-rime.ps1`, Ultra 7 255HX)

| Mode | AC (1st-key / rest avg / peak) | Battery (1st-key / rest avg / peak) |
| :--- | :---: | :---: |
| Pure Statistical (Double Pinyin) | `0.1 ms / 0.4 ms / 0.9 ms` | `0.4 ms / 1.6 ms / 5.2 ms` |
| **`rime-neural` v1.0.0 (Double Pinyin)** | **`0.1 ms / ~1.8 ms / ~10 ms`** | **`0.4 ms / ~4.1 ms / ~22 ms`** |
| **`rime-neural` v1.0.0 (Full Pinyin)** | **`0.1 ms / ~4.2 ms / ~13 ms`** | **`0.3 ms / ~9.2 ms / ~28 ms`** |

## Optional: Personal Weight Calibration

After collecting daily commits in `%APPDATA%\Rime\neural\feedback.jsonl` (100% unbiased logging, secrets redacted, 5 MiB rotation), run:

```powershell
python scripts\tune-fusion.py
```

The tuner fits exact damped Newton-Raphson MAP shrinkage around the OOD prior `[1.0, -1.4, 0.05, 0.15, 0.0]` and only suggests an update when `N >= 100` AND 5-fold out-of-fold weighted log-loss improvement passes a one-sided paired 95% confidence test (`z >= 1.645`, Kish `N_eff`) with non-decreasing Top-1 accuracy (verified `0/25` false positives under null, `88%–100%` power under genuine user shift).

## Architecture & Privacy

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full details on:
- `rime.dll` dynamic symbol resolution & module pinning
- Selective ORT output fetching & dynamic ONNX introspection
- UIA `RangeFromChild` / `U+FFFC` chrome clamping & password-manager denylist
- All 7 Double Pinyin layouts & memoized Full Pinyin alignment

## License & Credits

- **`rime-neural`** source code is licensed under [Apache-2.0](LICENSE).
- Default ONNX models (`ML-pumpkin-1103`, `zhen-feather-preview-1121`) are from [dd1000001000/zhiyi_models](https://github.com/dd1000001000/zhiyi_models) (Apache-2.0), downloaded on demand with SHA-256 verification.
- Optional `.gram` N-gram fusion is compatible with [amzxyz/rime-wanxiang-data](https://github.com/amzxyz/rime-wanxiang-data) (`Rime::Grammar/1.0` format).
- Built for [rime/weasel](https://github.com/rime/weasel) and [rime/librime](https://github.com/rime/librime).
