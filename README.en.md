<div align="center">

# rime-neural

**Local ONNX neural sentence reranker, typo-tolerant lattice & span repair plugin for official Rime / Weasel**

[![CI & Release](https://github.com/Chth1z/rime-neural/actions/workflows/ci-release.yml/badge.svg)](https://github.com/Chth1z/rime-neural/actions/workflows/ci-release.yml)
[![Release](https://img.shields.io/github/v/release/Chth1z/rime-neural?label=release)](https://github.com/Chth1z/rime-neural/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64%20%7C%20Linux%20%7C%20macOS%20(librime%20%E2%89%A5%201.17.0)-0078D6.svg)](https://github.com/rime/weasel)

[简体中文](README.md) | **English**

</div>

---

## Overview

`rime-neural` is an in-process ONNX neural sentence engine for official Rime hosts (Windows Weasel `>= 0.17.4` and Linux / macOS `librime >= 1.17.0` hosts exporting Lua 5.4 C API symbols), built as Lua 5.4 components backed by a Rust native module that mounts directly into existing Rime schemas.

### Features

- **Complements `.gram` language models with focus-bound context & document entity splice**: Works standalone or paired with a Chinese statistical grammar model (such as [`wanxiang-lts-zh-hans.gram`](https://github.com/amzxyz/rime-wanxiang-data/releases/tag/LTS)). Rime generates Top-8 sentence candidates; `rime-neural` shares a read-only `.gram` reader across schemas (kernel-pinned `MapViewOfFile` on Windows, immutable snapshot on Unix/macOS) and fuses neural logits, `.gram` transition scores, and rank priors. A batched KV-cached Pass-2 cross-encoder check verifies both high-confidence OOV `char_logits` candidates and maximal 2+ character homophone entity spans present in the active focus-bound context. Enabling opt-in `enable_document_entities: true` extends homophone entity scanning to up to 1,024 preceding characters within the same focused control without adding any ONNX KV prefix tokens.
- **Automatic 1-edit typo-tolerant lattice (`src/typo.rs` + `lua/rime_neural.lua`)**: Recovers staggered QWERTY physical neighbor slips, adjacent transpositions, doubled keys, and dropped keys across Full Pinyin and all 7 Double Pinyin layouts (`enable_typo_recovery`, enabled by default). Strictly gated by `is_live_pinyin_prefix`: clean inputs and valid mid-word prefixes pass through in ~1 µs with zero translator queries or extra ONNX calls; proven-dead keystreams are verified via a dedicated read-only `script_translator`, `.gram` fluency gating, and neural reranking.
- **Optional full-dictionary neural span repair & literal lock (`lua/rime_neural_repair.lua`)**: An opt-in standalone `lua_processor` mounted at the front of `engine/processors` (idle path costs at most two `KeyEvent:eq` comparisons). `Alt+r` steps across word boundaries (`Left`/`Right`) without moving the physical caret, queries the active schema's full dictionary plus 1-edit typo neighbors (e.g. ZRM `gsye -> 公园`, Full Pinyin `jintan -> 今天`), ranks alternatives in full-sentence context via the shared 12-layer neural model, supports chained multi-word pre-commit repairs, honors active OpenCC Traditional output conversion, and reverts cleanly on `Escape`/`Alt+r`. `Alt+l` locks verbatim ASCII identifiers (`GPUv2`, `OAuth2`, `v2.0.0`, `rime-neural`, `CI/CD`) as native confirmed `raw` segments inside Chinese compositions.
- **Sentence-only scope**: Reranks candidates with `cand:get_genuine().type == "sentence"` on configured segment tags (default `["abc"]`), preserving original order for exact dictionary words, `userdb` phrases, English words, dates, numbers, and punctuation.
- **Full Pinyin and 7 Double Pinyin layouts**: Auto-detects Full Pinyin, Natural/ZRM, Flypy, MSPY (`; = ing`), Sogou (`; = ing`), Intelligent ABC, Ziguang (`; = ing`), and Pinyin Jiajia (PYJJ), supporting `rime-ice`, `rime-frost`, `rime-wanxiang`, and custom schemas.
- **Local CPU inference and in-process reuse**: Runs quantized ONNX Runtime models with a 16-slot prefix KV-cache offline on CPU. Initializes asynchronously on a background thread and pins the native module in-process (Windows `GetModuleHandleExA(PIN)` / Unix `dlopen(..., RTLD_NODELETE)`) so Rime redeploys reuse the loaded engine.
- **Dynamic model introspection**: Reads ONNX graph geometry and `unified_vocab.json` at startup, supporting both 12-layer `ML-pumpkin-1103` (default, ~82 MB) and 8-layer `zhen-feather-preview-1121` (~38 MB) with the same native binary.
- **Bounded UIA context and privacy controls (Windows)**: A dedicated background MTA thread reads up to 64 active-paragraph characters before the caret via Windows UIA (`< 2 µs` hot path), bounded by `RangeFromChild(&focused)` (falling back to `DocumentRange`) and `U+FFFC` object separators. Focus changes immediately invalidate the frozen composition context. Password fields, protected processes, masked bullet strings (checked on both full text and the 96-char near-caret tail), password managers, remote-session clients, and local terminals fail closed, returning empty context and suspending feedback logging.

### Requirements & Limitations

- **Platform**:
  - **Windows 10 / 11 (x64)**: Official Weasel `>= 0.17.4` (`librime >= 1.17.0`), with prebuilt releases, automated installer, and full Windows UIA context support.
  - **Linux / macOS**: The Rust core (`librime_neural.so` / `librime_neural.dylib`) and Lua loader support POSIX single-provider Lua 5.4 symbol resolution and source builds (`cargo build --release --locked`), verified in CI on `ubuntu-latest`; cross-application surrounding-text reading is currently Windows UIA only.
- **Memory footprint**: The default 12-layer `ML-pumpkin-1103` model plus ONNX Runtime uses roughly 150–250 MB of RAM. For memory- or battery-constrained machines, use the 8-layer `zhen-feather` model (~38 MB).
- **Context window**: Short scoring context entering the ONNX KV-cache is capped at 128 characters; opt-in `enable_document_entities: true` extends homophone entity scanning to 1,024 preceding characters within the focused control. On 300+ character semantic disambiguation, accuracy still trails 1–2 GB GPU-hosted LLMs.

---

## Installation & Configuration

### Option 1: Autonomous Installation via AI Agent

Paste the prompt below into a shell-capable AI coding agent (Claude Code, Codex, Cursor, Antigravity, Gemini CLI, etc.). The agent reads the repository's setup guide and helper scripts to audit your Weasel environment, back up existing `*.custom.yaml` files, download and verify binaries and models, merge schema patches, deploy Weasel, and verify the logs:

```text
Follow https://raw.githubusercontent.com/Chth1z/rime-neural/main/skills/rime-neural-setup/SKILL.md to install and configure rime-neural for my Windows Weasel setup, then deploy and verify the installation.
```

### Option 2: Manual Installation

Ensure official [Weasel Nightly `0.17.4+`](https://github.com/rime/weasel/releases) (`librime >= 1.17.0`) is installed.

#### 1. Install DLL, Lua Filter, ONNX Runtime & Model

Run in PowerShell (`-Prebuilt` downloads the prebuilt release bundle without requiring Rust):

```powershell
# Default: 12-layer ML-pumpkin-1103 (~82 MB)
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt

# Optional: 8-layer zhen-feather (~38 MB)
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt -Model feather
```

The installer:
1. Downloads `rime-neural-*-win-x64.zip` from GitHub Releases and verifies it against the standalone `SHA256SUMS.txt` asset.
2. Downloads the selected Apache-2.0 ONNX model from `dd1000001000/zhiyi_models` and verifies SHA-256 on `.onnx`, `tokenizer.json`, and `unified_vocab.json`.
3. Downloads the pinned, SHA-512/SHA-256-verified official `onnxruntime.dll` (win-x64) from Microsoft NuGet (or uses an explicit `-OrtDllPath`).
4. Stops Weasel, backs up existing files and tracks created files for full rollback on error, installs files into the Rime user directory (honoring registry `RimeUserDir`, defaulting to `%APPDATA%\Rime\neural\`; non-ASCII user profiles can pass `-DataDir` to an ASCII path) and `lua\rime_neural.lua`, then restarts Weasel.

Omit `-Prebuilt` to compile from source (requires Rust 1.92+).

> **Recommended companion model**: Download [`wanxiang-lts-zh-hans.gram`](https://github.com/amzxyz/rime-wanxiang-data/releases/tag/LTS) into your Rime user directory and enable `grammar/language: wanxiang-lts-zh-hans`.

#### 2. Mount in Your Schema's `*.custom.yaml`

Templates for `rime_ice`, Flypy, MSPY, `rime_frost` ZRM, and `rime-wanxiang` are in [`examples/`](examples/). Minimal patch:

```yaml
patch:
  # Expose Top-8 sentence candidates from Rime's beam search
  translator/max_sentences: 8
  translator/sentence_cutoff_threshold: 0.3

  # Mount FIRST (before simplifier / OpenCC filters)
  engine/filters/@before 0: lua_filter@*rime_neural

  # Optional F4 toggle switch (enabled by default when omitted)
  switches/+:
    - name: neural_rerank
      states: [ 统计, 神经 ]
      reset: 1

  # All fields below are defaults and optional
  rime_neural:
    option_name: neural_rerank
    pinyin_layout: auto          # auto | full | zrm | flypy | mspy | sogou | abc | ziguang | jiajia
    max_candidates: 8
    fusion_weights: [1.0, -1.4, 0.05, 0.15, 0.0]
    p2c_verify_margin: 0.20      # <= 0 disables Pass-2 P2C synthesis
    enable_typo_recovery: true   # automatic 1-edit typo recovery (proven-dead spellings only)
    enable_uia_context: true
    enable_document_entities: false  # opt-in 1,024-char focus-clamped document entity scan (0 extra KV tokens)
    threads: 0                   # 0 = auto (clamped 4..8)
    log_feedback: true           # zero-plaintext numeric local log (5 MiB active + 1 rotated .1)
    sync_init: false             # async background load
```

Save and run **Weasel Deploy (重新部署)**.

> **`rime-wanxiang` Double Pinyin users**: Because `rime-wanxiang` retains `schema_id: wanxiang` when switching double pinyin via `speller/algebra`, set `pinyin_layout` explicitly (e.g. `flypy`, `zrm`, `mspy`).

#### 3. Optional: Enable Pre-Commit Local Span Repair & Literal Lock

To enable word-level pre-commit homophone repair (`Alt+r`) and verbatim ASCII literal locking (`Alt+l`), merge [`examples/rime_neural_repair.custom.yaml`](examples/rime_neural_repair.custom.yaml) into your schema's `*.custom.yaml`:

```yaml
patch:
  "engine/processors/@before 0": lua_processor@*rime_neural_repair
  rime_neural_repair:
    repair_key: "Alt+r"   # Set "" to disable homophone repair
    lock_key: "Alt+l"     # Set "" to disable literal lock
```

---

## Benchmarks

### Test Environment

- **Runtime & model**: Official Weasel `0.17.4` (`librime 1.17.0`), default 12-layer quantized `ML-pumpkin-1103` (`threads: 0`, auto-clamped `4..8`), `enable_user_dict: false`, UIA disabled; Intel Core Ultra 7 255HX.
- **Schema & dictionary baseline**: End-to-end benchmarks use [`Chth1z/rime-chth1z`](https://github.com/Chth1z/rime-chth1z) (upstream `gaboolic/rime-frost` dictionaries + `wanxiang-lts-zh-hans.gram`, `41448` and `GB18030-2022` rare-character tables commented out, `grammar/collocation_penalty: -15.5`). Same-environment neural-OFF baselines and a `-14` penalty control row are included below.

### 1. Major Rime Schemas (`gaboolic/rime-schema-compare/report/latest.md`)

Evaluated on `public_articles_2026-11-19.txt` with the same `wanxiang-lts-zh-hans.gram` as [`rime-schema-compare`](https://github.com/gaboolic/rime-schema-compare/blob/main/report/latest.md).

| Schema / Setup | Sentence Acc | Char Acc | Notes |
| :--- | :---: | :---: | :--- |
| `mingyuepinyin` (no model) | `63.32%` | `92.15%` | Upstream report (199 sents) |
| `luna_pinyin_with_octagram` (stock Luna + essay-bgw) | `66.83%` | `93.78%` | Upstream report |
| `rime_ice` (no model) | `69.35%` | `93.93%` | Upstream report |
| `wanxiang` (no model) | `69.35%` | `94.02%` | Upstream report |
| `rime_frost` (no model) | `70.85%` | `94.27%` | Upstream report |
| `rime_ice_with_gram` (+ Wanxiang `.gram`) | `74.37%` | `95.95%` | Upstream report |
| `rime_wanxiang_with_gram` (+ Wanxiang `.gram`) | `78.89%` | `96.84%` | Upstream report |
| `rime_wubi_sentens_wubi86_with_gram` (Wubi + Wanxiang) | `79.40%` | `96.99%` | Upstream report |
| `rime_frost_with_gram` (Frost + Wanxiang `.gram`) | `82.91%` | `97.38%` | Upstream report (`collocation_penalty: -14`) |
| Local control: Frost + Wanxiang `.gram` (neural OFF, `-14`) | `83.00%` | `97.39%` | 200 sents; matches upstream baseline |
| Local `rime-chth1z`: Frost + Wanxiang `.gram` (neural OFF, `-15.5`) | `80.50%` | `96.95%` | Same config with neural disabled |
| **Frost + Wanxiang `.gram` + `rime-neural v1.0.0` (Full, zero-ctx)** | **`89.50%`** | **`98.42%`** | Identical 179/200 on `-14` & `-15.5` (`+6.50%`) |
| **Frost + Wanxiang `.gram` + `rime-neural v1.0.0` (ZRM DP, zero-ctx)** | **`90.00%`** | **`98.52%`** | 180/200 |
| **Frost + Wanxiang `.gram` + `rime-neural v1.0.0` (Full / ZRM DP, paragraph ctx)** | **`91.00%`** | **`98.67%`** | Paragraph `commit_history` (182/200) |

### 2. Cross-Domain 246 Sentences vs. Commercial IMEs (`rime-schema-compare/report/other_latest.md`)

Evaluated on the 246-sentence corpus from [`other_latest.md`](https://github.com/gaboolic/rime-schema-compare/blob/main/report/other_latest.md) (commit `99da339a`: `news 36`, `novel 136`, `prose 38`, `tech 31`, `test 5`) with identical synonym normalization.

| IME / Setup | All 246 (Sent) | Char Acc | `news` | `novel` | `prose` | `tech` | `test` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| WeChat Pinyin (upstream Notepad black-box) | `51.63%` | `89.18%` | `61.11%` | `47.79%` | `50.00%` | `58.06%` | `60.0%` |
| Sogou Pinyin (upstream Notepad black-box) | `54.47%` | `77.02%` | `77.78%` | `45.59%` | `50.00%` | `74.19%` | `40.0%` |
| Microsoft Pinyin (upstream Notepad black-box) | `67.89%` | `92.71%` | `83.33%` | `63.24%` | `63.16%` | `74.19%` | `80.0%` |
| Shouxin Pinyin (upstream Notepad black-box) | `71.14%` | `94.48%` | `86.11%` | `66.18%` | `65.79%` | `80.65%` | `80.0%` |
| Upstream `rime_frost_with_gram` (2026-04-25) | `71.54%` | `95.45%` | `88.89%` | `67.65%` | `55.26%` | `83.87%` | `100%` |
| Local `rime-chth1z` stat only (Frost + Wanxiang, neural OFF) | `76.02%` | `95.85%` | `97.22%` | `69.12%` | `71.05%` | `83.87%` | `100%` |
| **Frost + Wanxiang `.gram` + `rime-neural v1.0.0` (Full, strict zero-ctx)** | **`81.30%`** | **`97.29%`** | **`97.22%`** | `77.21%` | `68.42%` | **`93.55%`** | **`100%`** |
| **Frost + Wanxiang `.gram` + `rime-neural v1.0.0` (ZRM DP, strict zero-ctx)** | **`81.71%`** | `97.37%` | **`97.22%`** | **`77.94%`** | `68.42%` | **`93.55%`** | **`100%`** |
| **Frost + Wanxiang `.gram` + `rime-neural v1.0.0` (ZRM DP, paragraph ctx)** | **`81.71%`** | **`97.51%`** | **`97.22%`** | `76.47%` | **`76.32%`** | `90.32%` | **`100%`** |

> Commercial IME figures are quoted from `rime-schema-compare/report/other_latest.md` (foreground Notepad GUI automation, subject to host state and default settings).

### 3. Open Neural Reranker Pool (`metasequoiaime/chinese-ime-lm` `sentences-ctx-v1`)

Evaluated by scoring the fixed MSIME candidate pool `eval/sentences-ctx-v1/pool.jsonl` from [`metasequoiaime/chinese-ime-lm`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md) (1,631 clauses from 2026-09~10 government documents and KdConv / CrossWOZ / RiSAWOZ dialogues) directly with `rime_neural.dll v1.0.0`:

| Scorer | Size | With Context (1,196) | No Context (435) | All 1,631 | Runtime |
| :--- | :---: | :---: | :---: | :---: | :--- |
| MSIME original rank | — | `70.1%` (838) | `67.4%` (293) | `69.34%` | — |
| `msime-4M` | 4.5 MB | `81.9%` (979) | `77.5%` (337) | `80.69%` | CPU 1T |
| `msime-25M` | 25.5 MB | `87.0%` (1041) | `86.0%` (374) | `86.76%` | CPU 1T |
| Qingjian `Hanzhang-Tongbian` | 46 MB | `92.1%` (1102) | `81.4%` (354) | `89.27%` | Apple GPU |
| Qingjian `Hanzhang-Zhiwei` | 56 MB | `91.2%` (1091) | `88.3%` (384) | `90.44%` | Apple GPU |
| Zhiyi `1.4V2-Long-L12 int8` | 84 MB | `93.8%` (1122) | `87.8%` (382) | `92.21%` | CPU 4T |
| Zhiyi `ML-pumpkin-1026 int8` | 92 MB | `94.6%` (1131) | `86.9%` (378) | `92.52%` | CPU 4T |
| **`rime-neural` (pure `ML-pumpkin-1103`, no prior)** | **82 MB** | **`95.15%`** (1138) | `87.13%` (379) | `93.01%` | CPU 4T |
| **`rime-neural` (`ML-pumpkin-1103` + Wanxiang `.gram`, no engine prior)** | **82 MB** | **`95.15%`** (1138) | **`88.28%`** (384) | **`93.32%`** | CPU 4T |
| **`rime-neural v1.0.0` default fusion** | **82 MB** | `94.98%` (1136) | **`88.51%`** (385) | `93.26%` | CPU 4T |
| *Ref: `Qwen2.5-0.5B-Instruct-4bit`* | *278 MB* | *`96.2%` (1150)* | *`85.5%` (372)* | *`93.32%`* | *Apple GPU (mlx)* |
| *Ref: `Qwen3-0.6B-8bit`* | *633 MB* | *`96.5%` (1154)* | *`78.9%` (343)* | *`91.78%`* | *Apple GPU (mlx)* |
| *Ref: `Qwen3.5-0.8B-8bit`* | *1.0 GB* | *`97.3%` (1164)* | *`87.4%` (380)* | *`94.67%`* | *Apple GPU (mlx)* |
| *Ref: `Qwen3.5-2B-4bit`* | *1.7 GB* | *`97.7%` (1169)* | *`89.7%` (390)* | *`95.59%`* | *Apple GPU (mlx)* |

### 4. 1-Edit Noisy-Typing Recovery (500 Real Sentences)

Evaluated in isolated official Weasel `0.17.4` on 500 sentences (`public_articles_2026-11-19.txt` + first 300 sentences of `zhihu_hotlist_2026-11-18.txt`) with deterministic 1-edit syllable-breaking typos evenly split across 4 classes (QWERTY neighbor, adjacent transposition, doubled key, dropped key; 125 each), paired with a 1,000-sentence clean control (including blind `zhihu_hold_500`):

| Setup & Split | Typo Recovery OFF (Sent / Char) | Typo Recovery `v2.0.0` (Sent / Char) | Delta |
| :--- | :---: | :---: | :---: |
| 1,000-sent clean control (ZRM DP) | `80.50%` (`805/1000`) | `80.50%` (`805/1000`) | `+0 / -0` (zero regression) |
| 1,000-sent clean control (Full Pinyin) | `80.00%` (`800/1000`) | `80.00%` (`800/1000`) | `+0 / -0` (zero regression) |
| 500-sent 1-edit noisy (ZRM DP, all 4 classes) | `0.00%` / `-0.91%` | **`52.80%` / `86.25%`** | **`+52.80 pp` / `+87.16 pp`** |
| 500-sent 1-edit noisy (ZRM DP, neighbor/swap/doubled, 375) | `0.00%` / `-8.13%` | **`66.13%` / `90.26%`** | **`+66.13 pp` / `+98.39 pp`** |
| 500-sent 1-edit noisy (Full Pinyin, all 4 classes) | `13.20%` / `48.12%` | **`54.20%` / `85.59%`** | **`+41.00 pp` / `+37.48 pp`** |

### 5. Keystroke Latency

Intel Core Ultra 7 255HX, isolated `rime.dll` benchmark:

| Mode | AC (1st-key / rest avg / peak) | Battery (1st-key / rest avg / peak) |
| :--- | :---: | :---: |
| Statistical baseline (ZRM DP + Wanxiang `.gram`) | `0.1 ms / 0.4 ms / 0.9 ms` | `0.4 ms / 1.6 ms / 5.2 ms` |
| **`rime-neural v2.0.0` (Double Pinyin, with typo + `rime_neural_repair`)** | **`0.1 ms / 1.7 ms / 10.0 ms`** | `0.4 ms / ~4.1 ms / ~22 ms` |
| **`rime-neural` (Full Pinyin)** | **`0.2 ms / 4.7 ms / 14.7 ms`** | `0.3 ms / ~9.2 ms / ~28 ms` |

---

## Architecture & Configuration Reference

```text
Official Rime Host (Windows Weasel / Linux / macOS, librime >= 1.17.0, Lua 5.4)
  ├── lua/rime_neural.lua            # Lua filter: cross-platform loader, layout resolution, sentence/tag gating, auto 1-edit typo recovery, focus-scoped context & feedback
  ├── lua/rime_neural_repair.lua     # Optional Lua processor: full-dict + 1-edit typo neural chained repair (Alt+r) & literal lock (Alt+l)
  └── neural/rime_neural.{dll,so,dylib} (Rust) # Dynamic Lua 5.4 C API (C-unwind), in-process pin
        ├── engine.rs                # ONNX introspection, 16-slot KV-cache, Pass-2 P2C & 1,024-char doc entity splice, per-schema .gram cache
        ├── typo.rs                  # Live-prefix automaton + Full Pinyin / 7 Double Pinyin 1-edit typo-tolerant lattice
        ├── gram.rs                  # Read-only single-pass Rime::Grammar/1.0 trie scorer (Win32 mmap / Unix immutable snapshot)
        ├── pinyin.rs                # 3-phase memoized Full Pinyin + DP splitter + 7 Double Pinyin layouts
        ├── uia.rs                   # Windows background MTA UIA (focus-change events + element-bounds clamp + 200ms timeout + denylist)
        └── feedback.rs              # Zero-plaintext numeric local commit log (5 MiB rotation + reparse-point / O_NOFOLLOW guard)
```

Implementation details are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

| Key (`rime_neural/...`) | Default | Description |
| :--- | :--- | :--- |
| `option_name` | `neural_rerank` | Rime boolean switch name; active by default if not declared in `switches` |
| `pinyin_layout` | `auto` | `auto` / `full` / `zrm` / `flypy` / `mspy` / `sogou` / `abc` / `ziguang` / `jiajia` (`pyjj`) |
| `tags` | `["abc"]` | Allowed Rime segment tags for reranking |
| `max_candidates` | `8` | Max leading `sentence` candidates sent to the reranker (clamped `2..64`) |
| `fusion_weights` | `[1.0, -1.4, 0.05, 0.15, 0.0]` | 5D linear weights `[logit_lp, ln(rank+1), gram_delta, char_lp_delta, logit_lp*has_ctx]` (0-based `rank`) |
| `p2c_verify_margin` | `0.20` | Pass-1 margin to trigger KV-cached Pass-2 P2C check (`<= 0` disables) |
| `p2c_top1_logit_diff` | `2.0` | Min Pass-2 logit lead to promote P2C to #1 |
| `p2c_top1_min_gram_diff` | `-2.0` | `.gram` per-char score floor for #1 P2C promotion (#2 insertion uses `value - 1.5`) |
| `p2c_top2_logit_diff` | `0.5` | Min Pass-2 logit lead to insert P2C at #2 |
| `enable_typo_recovery` | `true` | Automatic 1-edit typo recovery on proven-dead spellings (zero overhead on clean/live-prefix input) |
| `enable_uia_context` | `true` | Non-blocking background Windows UIA context (per-schema) |
| `enable_document_entities` | `false` | Opt-in 1,024-char focus-clamped document entity scan (requires `enable_uia_context: true`, zero extra KV tokens) |
| `threads` | `0` | ONNX Runtime CPU threads (`0` = auto `4..8`) |
| `min_input_len` | `4` (DP) / `6` (Full) | Min pinyin key count before reranking (clamped `>= 1`) |
| `data_dir` | `""` (auto) | Optional explicit `neural\` directory path (for non-ASCII Windows profiles; use single quotes or `/` in YAML) |
| `model_file` | `""` (auto) | Optional explicit `.onnx` filename under `neural\` |
| `t2s_config` | `""` (auto `tw2sp` / `t2s`) | Optional explicit OpenCC config for Traditional-to-Simplified context normalization |
| `log_feedback` | `true` | Zero-plaintext numeric local commit log for optional personal tuning |
| `sync_init` | `false` | `false` = async background load; `true` = blocking load (for benchmarks) |

## Privacy & Security

See [`SECURITY.md`](SECURITY.md) for the threat model and vulnerability reporting procedure.

- **Offline inference**: No network requests at runtime; downloads occur only during explicit installation with cryptographic hash verification for binaries, ONNX Runtime, and models.
- **Fail-closed sensitive context blocking**: Returns empty context and suspends feedback logging when:
  - the focused control reports `IsPassword == true`, denies access to `IsPassword`, or displays masked bullet/asterisk content (`••••••` / `******`);
  - the foreground or focused process is protected (`OpenProcess` returns `ERROR_ACCESS_DENIED`) or its PID cannot be resolved;
  - the process matches the denylist: password managers and authenticators (`KeePass*`, `KeePassXC`, `1Password`, `Bitwarden`, `Enpass`, `LastPass`, `Dashlane`, `RoboForm`, `NordPass`, `ProtonPass`, `Keeper`, `PasswordSafe`, `Authy`, `WinAuth`), credential/UAC dialogs (`CredentialUIBroker`, `consent.exe`, `LockApp`), remote desktop/SSH clients (`mstsc.exe`, `msrdc.exe`, `PuTTY`, `Xshell`, `MobaXterm*`, `FinalShell`), and local terminals (`WindowsTerminal`, `conhost`, `cmd`, `powershell`, `pwsh`, `mintty`, `alacritty`, `wezterm-gui`, `kitty`).
- **Zero-plaintext local feedback**: `%APPDATA%\Rime\neural\feedback.jsonl` stores only numeric model features (`logits`, `gram_deltas`, `char_lps`) and candidate index outcomes (`shown_top1_idx`, `chosen_idx`, `has_ctx`, `has_gram`), never persisting user text, keystrokes, candidates, or screen context. The logger rejects reparse points and rotates at `5 MiB`, retaining one `.1` backup. A temporary sharing lock allows the active file to reach `10 MiB`. Set `log_feedback: false` to disable.
- **Optional weight tuning**: After collecting 100+ local multi-candidate commits, run `python scripts\tune-fusion.py` (Python 3 + `numpy`; supports `--schema`, `--model`, `--has-gram`, and `--scrub-legacy-inplace`). The tuner evaluates three-decimal weights on five contiguous held-out blocks. A recommendation requires lower weighted log-loss under a one-sided test adjusted across the seven prior precisions (Bonferroni, nominal 5%, Hájek SE), non-decreasing Top-1 accuracy, and valid weights.

## Troubleshooting

- **No effect after deploy**: Inspect `%TEMP%\rime.weasel\*WARNING*` and recent `.INFO.` logs; verify `%APPDATA%\Rime\neural\` (or custom `data_dir`) contains `rime_neural.dll` (check File Version properties), `onnxruntime.dll`, a single `.onnx` model, `tokenizer.json`, and `unified_vocab.json`, plus `%APPDATA%\Rime\lua\rime_neural.lua`; if `switches` defines `neural_rerank`, confirm it is enabled in the `F4` menu.
- **Incompatible Weasel version**: Older builds (e.g. `0.16.x`) do not export Lua 5.4 C API symbols from `rime.dll`; the filter passes through until Weasel is upgraded to `0.17.4+`.
- **Uninstall**: Remove the `rime_neural` entries from your `*.custom.yaml`, stop Weasel, delete `lua\rime_neural.lua` and `neural\` (and your custom `-DataDir` if used), and redeploy.

## License & Credits

- Source code is licensed under [Apache-2.0](LICENSE).
- Supported ONNX models (`ML-pumpkin-1103`, `zhen-feather-preview-1121`) are published by [dd1000001000/zhiyi_models](https://github.com/dd1000001000/zhiyi_models) (Apache-2.0) and downloaded on demand with SHA-256 verification.
- Recommended companion `.gram` model: [amzxyz/rime-wanxiang-data](https://github.com/amzxyz/rime-wanxiang-data) (`wanxiang-lts-zh-hans.gram`). End-to-end baselines and evaluation protocols credit [gaboolic/rime-frost](https://github.com/gaboolic/rime-frost), [iDvel/rime-ice](https://github.com/iDvel/rime-ice), [gaboolic/rime-schema-compare](https://github.com/gaboolic/rime-schema-compare), and [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm).
- Built for [rime/weasel](https://github.com/rime/weasel) and [rime/librime](https://github.com/rime/librime).
