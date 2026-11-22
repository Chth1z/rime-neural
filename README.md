<div align="center">

# rime-neural

**Zero-patch, in-process ONNX neural sentence reranker for official Windows Rime / Weasel**

[![CI & Release](https://github.com/Chth1z/rime-neural/actions/workflows/ci-release.yml/badge.svg)](https://github.com/Chth1z/rime-neural/actions/workflows/ci-release.yml)
[![Release](https://img.shields.io/github/v/release/Chth1z/rime-neural?label=release)](https://github.com/Chth1z/rime-neural/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64%20(Weasel%20%E2%89%A5%200.17.4)-0078D6.svg)](https://github.com/rime/weasel)

**English** | [简体中文](README.zh-CN.md)

</div>

---

## Overview

Statistical N-gram models (such as 400 MB `.gram` language models) already give Rime strong sentence decoding, yet they remain fundamentally limited by local 2–5 character windows and cannot use long-range semantics, preceding sentence context, or neural cross-candidate comparison.

**`rime-neural`** bridges this gap as a drop-in Lua 5.4 + Rust native plugin for **official, unmodified Weasel** (`librime >= 1.17.0`):

- **Zero Custom Weasel / `librime` Build**: Dynamically resolves Lua 5.4 C API symbols from Weasel's loaded `rime.dll` and pins itself in process so Rime redeploys never crash or reload ONNX Runtime (`0 ms` redeploy reuse, `0 ms` startup blocking via `init_async`).
- **Offline & Lightweight**: Runs 100% locally on CPU via ONNX Runtime int8/w4 quantized models with a 16-slot prefix KV-cache — `~2.0 ms` average per keystroke on AC (~4.1 ms on battery), zero cloud dependency, zero GPU wakeup stutter.
- **Universal Schema Support**: Auto-detects **Full Pinyin** and **all 7 major Rime Double Pinyin layouts** (Natural/ZRM, Flypy/小鹤, MSPY/微软, Sogou/搜狗, ABC/智能ABC, Ziguang/紫光, Jiajia/PYJJ/拼音加加). Exhaustively verified against all 3,008 derived 2-key codes in official `rime-ice` / `rime-frost` schemas (**0 mismatches**). Works seamlessly with `rime-ice`, `rime-frost`, `rime-wanxiang`, and custom schemas.
- **Dynamic Model Introspection**: Reads model layer/head geometry and `choice_compact` format directly from the ONNX graph and `unified_vocab.json`, supporting both high-accuracy 12-layer **`ML-pumpkin-1103`** (default) and lightweight 8-layer **`zhen-feather-preview-1121`** with the same DLL.
- **Zero-Extra-RAM `.gram` Fusion + Pass-2 Verified P2C**: Memory-maps your existing Rime `.gram` file read-only (`0 MB` extra physical RAM, `~0.01 ms`/sentence) using your schema's actual grammar penalties, and validates novel high-potential OOV `char_logits` candidates via KV-cached Pass-2 cross-encoder + `.gram` checks.
- **Privacy-Hardened, Non-Blocking UIA Context**: Reads up to 64 characters strictly before the caret on a dedicated background STA thread (`< 2 µs` hot path, keystroke-1 pre-warm). Clamps `TextPattern` ranges to `RangeFromChild(focused)` and `U+FFFC` so Chromium/Electron apps never bleed toolbar chrome, isolates foreground `HWND` to block cross-app bleed, and hard-blocks password fields and password managers.
- **Strict Dictionary Protection**: Only reranks `sentence` candidates (`cand:get_genuine().type == "sentence"`), never overriding exact dictionary hits, userdb phrases, English words, dates, numbers, or punctuation.

## Architecture

```text
Official Weasel (WeaselServer.exe / rime.dll, librime >= 1.17.0, Lua 5.4)
  ├── lua/rime_neural.lua                # Universal Lua filter: auto layout, sentence-only gate, UIA pre-warm
  └── neural/rime_neural.dll (Rust)      # Dynamic Lua C API binding via GetModuleHandleA("rime.dll") + module pin
        ├── engine.rs                    # ONNX Runtime dynamic introspection, selective output fetch,
        │                                # 16-slot LRU KV-cache, 256-slot eval LRU, Pass-2 verified P2C
        ├── gram.rs                      # Zero-extra-RAM Win32 CreateFileMappingW reader for Rime::Grammar/1.0
        ├── pinyin.rs                    # Full Pinyin (O(N*L) memoized) + 7 Double Pinyin layouts (3,008/3,008)
        ├── uia.rs                       # Non-blocking background STA UIA reader (RangeFromChild + password block)
        └── feedback.rs                  # Local redacted unbiased commit log (5 MiB auto-rotation)
```

For full implementation details, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Requirements

- **OS**: Windows 10 / 11 (x64)
- **IME**: Official [Weasel](https://github.com/rime/weasel/releases) with `librime >= 1.17.0` (e.g. Weasel Nightly `0.17.4+`)
- **Optional (Recommended)**: A Rime `.gram` language model such as [`wanxiang-lts-zh-hans.gram`](https://github.com/amzxyz/rime-wanxiang-data/releases/tag/LTS) (works without `.gram` too — automatically falls back to pure neural reranking)

## Quick Start

### Step 1: Install DLL, Lua Filter, ONNX Runtime & Model

Run from PowerShell (no Rust toolchain required when using `-Prebuilt`):

```powershell
# Default: 12-layer ML-pumpkin-1103 (best Chinese accuracy)
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt

# Optional: 8-layer zhen-feather (lighter / faster on low-power CPUs)
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt -Model feather
```

The installer automatically:
1. Downloads the official prebuilt `rime-neural-*-win-x64.zip` from GitHub Releases and verifies `SHA256SUMS.txt`.
2. Downloads and SHA-256 verifies the selected Apache-2.0 ONNX model from `dd1000001000/zhiyi_models`.
3. Locates or fetches official `onnxruntime.dll` (win-x64) from Microsoft NuGet.
4. Safely stops Weasel, installs files into `%APPDATA%\Rime\neural\` and `%APPDATA%\Rime\lua\rime_neural.lua`, and restarts Weasel.

If you prefer building from source (Rust 1.80+):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

### Step 2: Mount in Your Schema's `*.custom.yaml`

Ready-to-copy templates are in [`examples/`](examples/) (`rime_ice`, `double_pinyin_flypy`, `double_pinyin_mspy`, `rime_frost_double_pinyin`, `wanxiang`). Minimal patch:

```yaml
patch:
  # 1. Expose Rime's top-8 sentence beam candidates to the neural reranker
  translator/max_sentences: 8
  translator/sentence_cutoff_threshold: 0.3

  # 2. Mount rime_neural FIRST (before simplifier / OpenCC filters)
  engine/filters/@before 0: lua_filter@*rime_neural

  # 3. Optional F4 toggle switch (defaults to ON)
  switches/+:
    - name: neural_rerank
      states: [ 统计, 神经 ]
      reset: 1

  # 4. Configuration (all fields optional; defaults shown)
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
    log_feedback: true           # %APPDATA%\Rime\neural\feedback.jsonl (5 MiB cap)
    sync_init: false             # async background init -> 0 ms startup delay
```

Save and click **Weasel Deploy (重新部署)**. If the DLL/model is missing or `neural_rerank` is toggled off, the filter degrades to a zero-overhead pass-through.

> **Note for `rime-wanxiang` Double Pinyin users**: Because `rime-wanxiang` keeps `schema_id: wanxiang` when switching double pinyin layouts via `speller/algebra`, set `pinyin_layout` explicitly (e.g. `flypy` or `zrm`).

## Configuration Reference

| Key (`rime_neural/...`) | Default | Description |
| :--- | :--- | :--- |
| `option_name` | `neural_rerank` | Rime boolean switch name that toggles neural reranking |
| `pinyin_layout` | `auto` | `auto`, `full`, `zrm`, `flypy`, `mspy`, `sogou`, `abc`, `ziguang`, `jiajia` (`pyjj`) |
| `max_candidates` | `8` | Max leading `sentence` candidates sent to the reranker |
| `fusion_weights` | `[1.0, -1.4, 0.05, 0.15, 0.0]` | 5D linear prior `[logit_lp, ln(rank+1), gram_delta, char_lp, logit_lp*has_ctx]` |
| `p2c_verify_margin` | `0.20` | Pass-1 `char_logits` margin to trigger KV-cached Pass-2 P2C check (`<= 0` disables P2C) |
| `p2c_top1_logit_diff` | `2.0` | Min Pass-2 logit lead over Rime's best candidate to promote P2C to #1 |
| `p2c_top1_min_gram_diff` | `-2.0` | Min `.gram` per-char score floor for #1 P2C promotion |
| `p2c_top2_logit_diff` | `0.5` | Min Pass-2 logit lead to safely insert P2C at #2 |
| `enable_uia_context` | `true` | Non-blocking Windows UIA before-caret context (per-schema setting) |
| `threads` | `0` | ONNX Runtime intra-op CPU threads (`0` = auto-detect clamped to `4..8`) |
| `min_input_len` | `4` (DP) / `6` (Full) | Min pinyin key count before invoking neural reranking |
| `model_file` | `""` (auto) | Optional explicit ONNX filename under `%APPDATA%\Rime\neural\` |
| `log_feedback` | `true` | Record unbiased, redacted multi-candidate commits locally for MAP tuning |
| `sync_init` | `false` | `false` = async background load (`0 ms` startup); `true` = blocking load (for benchmarks) |

## Benchmarks

All evaluations below were measured end-to-end through official unmodified `rime.dll` (`Weasel 0.17.4`, `librime 1.17.0`, `enable_user_dict: false` strict isolation, [`gaboolic/rime-schema-compare`](https://github.com/gaboolic/rime-schema-compare) normalization protocol) on an Intel Core Ultra 7 255HX laptop.

### 1. Horizontal Comparison vs. Major Rime Schemas (Public `public_articles_2026-11-19.txt`, Full Pinyin)

Same public corpus file (SHA-256 `1a8edab3...`) and same `wanxiang-lts-zh-hans.gram` (SHA-256 `e9ddd501...`) as [`rime-schema-compare/report/latest.md`](https://github.com/gaboolic/rime-schema-compare/blob/main/report/latest.md):

| Schema / Engine | Sentence Acc | Char Acc | Delta vs. Pure N-gram Leader |
| :--- | :---: | :---: | :---: |
| `mingyuepinyin` (no model) | `63.32%` | `92.15%` | `-19.59%` |
| `luna_pinyin_with_octagram` (stock Luna + essay-bgw) | `66.83%` | `93.78%` | `-16.08%` |
| `rime_ice` (no model) | `69.35%` | `93.93%` | `-13.56%` |
| `wanxiang` (no model) | `69.35%` | `94.02%` | `-13.56%` |
| `rime_frost` (no model) | `70.85%` | `94.27%` | `-12.06%` |
| `rime_ice_with_gram` (+ Wanxiang 400 MB) | `74.37%` | `95.95%` | `-8.54%` |
| `rime_wanxiang_with_gram` (+ Wanxiang 400 MB) | `78.89%` | `96.84%` | `-4.02%` |
| `rime_wubi_sentens_wubi86_with_gram` (Wubi + Wanxiang) | `79.40%` | `96.99%` | `-3.51%` |
| `rime_frost_with_gram` (pure N-gram leader) | `82.91%` | `97.38%` | baseline |
| **`rime_frost` + `rime-neural v1.0.0` (strict zero-context)** | **`89.50%`** | **`98.42%`** | **`+6.59%`** |
| **`rime_frost` + `rime-neural v1.0.0` (live `commit_history`)** | **`91.00%`** | **`98.72%`** | **`+8.09%`** |

### 2. Multi-Model Horizontal Comparison (1,600 Sentences, Double Pinyin, Same Rime Top-8 Beam)

| Reranker Model | Size | All 1,600 (Sent / Char) | Tech 200 | Zhihu Holdout 1,000 |
| :--- | :---: | :---: | :---: | :---: |
| Pure Statistical (`rime_frost` + Wanxiang `.gram`) | — | `75.12%` / `95.51%` | `81.00%` | `75.80%` |
| `msime-4M` (neural LM, offline optimal) | 4.5 MB | `76.12%` (`+1.00%`) | — | — |
| `msime-25M` (neural LM, offline optimal) | 25.5 MB | `78.81%` (`+3.69%`) | — | — |
| `Zhiyi 1.4V2-Long-L12` (no KV-cache, live `commit_history`) | 84 MB | `81.31%` / `96.83%` | `90.50%` | `80.00%` |
| `zhen-feather-preview-1121` (8L compact KV-cache, live) | 38 MB | — | `88.50%` | — |
| **`rime-neural v1.0.0` (`ML-pumpkin-1103`, live `commit_history`)** | **82 MB** | **`82.19%` / `97.05%`** | **`91.00%`** | **`81.40%`** |
| **`rime-neural v1.0.0` (with punctuated UIA-like context)** | **82 MB** | **`85.19%`** | **`92.00%`** | **`84.60%`** |

### 3. Never-Before-Seen Cross-Domain OOD Blind Audit (840 Sentences, 5 Untouched Domains)

Zero parameter tuning on any of these 5 corpora (`news`, `novel`, `prose`, `old tech`, `2026-08-30 Zhihu`):

| Configuration | OOD 840 All (Sent / Char) | News (152) | Novel (40) | Prose (211) | Tech Old (137) | Zhihu 08-30 (300) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Pure Statistical (`wanxiang` `.gram`) | `60.48%` / `90.94%` | `70.4%` | `65.0%` | `37.0%` | `66.4%` | `68.7%` |
| **`rime-neural v1.0.0` (`[1.0, -1.4, 0.05, 0.15, 0.0]`)** | **`68.57%` / `93.52%`** (`+8.09%`) | **`78.3%`** | **`70.0%`** | **`46.4%`** | **`75.9%`** | **`75.7%`** |

### 4. Keystroke Latency (Ultra 7 255HX, Isolated `rime.dll` Benchmark)

| Mode | AC (1st-key / rest avg / peak) | Battery (1st-key / rest avg / peak) |
| :--- | :---: | :---: |
| Pure Statistical (Double Pinyin) | `0.1 ms / 0.4 ms / 0.9 ms` | `0.4 ms / 1.6 ms / 5.2 ms` |
| **`rime-neural v1.0.0` (Double Pinyin)** | **`0.1 ms / 2.0 ms / 11.5 ms`** | **`0.4 ms / ~4.1 ms / ~22 ms`** |
| **`rime-neural v1.0.0` (Full Pinyin)** | **`0.2 ms / 4.7 ms / 14.7 ms`** | **`0.3 ms / ~9.2 ms / ~28 ms`** |

## Privacy & Security Guarantees

- **100% Local Execution**: No network calls at runtime; models and DLLs are downloaded only during explicit installation with SHA-256 verification.
- **Hard Password Blocking**: Controls reporting `IsPassword == true` or hosted by password managers / credential UIs (`KeePass`, `KeePassXC`, `1Password`, `Bitwarden`, `Enpass`, `LastPass`, `Dashlane`, `RoboForm`, `NordPass`, `ProtonPass`, `CredentialUIBroker`, `consent.exe`, `LockApp`, `mstsc.exe`) return empty context and suppress feedback logging.
- **No Chrome Bleed & No Cross-App Bleed**: UIA `TextPattern` ranges are clamped strictly within `RangeFromChild(&focused)` and `U+FFFC` separators, and switching foreground `HWND` immediately invalidates the prior window's `commit_history`.
- **Redacted, Bounded Feedback**: Local `feedback.jsonl` redacts 4+ digit numbers, emails, URLs, and 12+ char ASCII tokens to `<REDACTED>`, truncates context to 64 chars, and caps disk usage at 5 MiB. Set `log_feedback: false` to disable completely.

## Optional: Personal Weight Calibration

After accumulating daily commits in `%APPDATA%\Rime\neural\feedback.jsonl`, run:

```powershell
python scripts\tune-fusion.py
```

The tuner solves exact damped Newton-Raphson MAP shrinkage around `[1.0, -1.4, 0.05, 0.15, 0.0]`, picks regularization via 5-fold out-of-fold weighted log-loss (NLL), and only recommends new weights when `N >= 100` AND improvement passes a one-sided paired 95% confidence test (`z >= 1.645`, Kish effective `N_eff`) with non-decreasing Top-1 accuracy (verified `0/25` false positives under null, `88%–100%` power under true preference shift).

## FAQ / Troubleshooting

- **Nothing happens after deploy**: Check `%TEMP%\rime.weasel\*WARNING*` logs and confirm `%APPDATA%\Rime\neural\` contains `rime_neural.dll`, `onnxruntime.dll`, `laya.one.fused.int8g.onnx` (or `laya.one.w4.onnx`), `tokenizer.json`, and `unified_vocab.json`, and `%APPDATA%\Rime\lua\rime_neural.lua` exists.
- **Weasel version requirement**: Requires official Weasel with `librime >= 1.17.0` (Lua 5.4 C API exported from `rime.dll`). Older Weasel builds without exported Lua C symbols will safely fall back to statistical pass-through.
- **How to disable UIA context or feedback**: Set `enable_uia_context: false` and/or `log_feedback: false` under `rime_neural:` in your `*.custom.yaml`, then redeploy.

## License & Credits

- **`rime-neural`** is licensed under [Apache-2.0](LICENSE).
- Supported ONNX models (`ML-pumpkin-1103`, `zhen-feather-preview-1121`) are published by [dd1000001000/zhiyi_models](https://github.com/dd1000001000/zhiyi_models) (Apache-2.0), downloaded on demand with SHA-256 verification.
- `.gram` memory-mapped fusion supports [`Rime::Grammar/1.0`](https://github.com/lotem/librime-octagram) models such as [amzxyz/rime-wanxiang-data](https://github.com/amzxyz/rime-wanxiang-data).
- Built for [rime/weasel](https://github.com/rime/weasel) and [rime/librime](https://github.com/rime/librime).
