<div align="center">

# rime-neural

**零魔改、进程内运行的 Windows Rime / 小狼毫（Weasel）端侧 ONNX 神经整句重排插件**

[![CI & Release](https://github.com/Chth1z/rime-neural/actions/workflows/ci-release.yml/badge.svg)](https://github.com/Chth1z/rime-neural/actions/workflows/ci-release.yml)
[![Release](https://img.shields.io/github/v/release/Chth1z/rime-neural?label=release)](https://github.com/Chth1z/rime-neural/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64%20(Weasel%20%E2%89%A5%200.17.4)-0078D6.svg)](https://github.com/rime/weasel)

[English](README.md) | **简体中文**

</div>

---

## 项目简介

400 MB 级统计 N-gram 语法模型（如万象 `.gram`）已经让 Rime 拥有不错的整句能力，但仍受限于 2–5 字局部滑动窗口，无法利用更长距离的语义关系、跨句上文语境，也难以对同音整句做全局联合判别。

**`rime-neural`** 以 **Lua 5.4 ＋ Rust 原生扩展模块** 的形式直接接入**官方未修改的小狼毫（Weasel，`librime >= 1.17.0`）**：

- **零魔改官方客户端**：通过 `GetModuleHandleA("rime.dll")` 动态解析小狼毫导出的 Lua 5.4 C API，并用 `GetModuleHandleExA(PIN)` 将 DLL 钉驻宿主进程——无需自行重编 `rime.dll`，无需常驻外部服务，重新部署不崩溃且 `0 ms` 复用已预热引擎，后台异步加载 `0 ms` 阻塞启动。
- **纯本地 CPU 毫秒级响应**：基于 ONNX Runtime int8/w4 量化模型 ＋ 16 槽 LRU 前缀 KV-cache，无需联网、不唤醒笔记本独显（避免 D3cold 唤醒卡顿与高功耗），插电双拼每键均值 `~2.0 ms`（电池约 `~4.1 ms`）。
- **全拼 ＋ 7 大主流双拼开箱即用**：自动识别全拼与 **自然码（ZRM）、小鹤（Flypy）、微软（MSPY，含 `; = ing`）、搜狗（Sogou，含 `; = ing`）、智能 ABC、紫光（含 `; = ing`）、拼音加加（Jiajia / PYJJ）**。对照官方 `rime-ice` / `rime-frost` 全部 3,008 组 2 键派生码表穷举验证 **0 失配**，通用适配雾凇、白霜、万象与各类自建方案。
- **动态模型结构自省**：启动时直接从 ONNX 图和 `unified_vocab.json` 自动识别层数、注意力头数及 `choice_compact` 格式，同一份 `rime_neural.dll` 零重编同时支持 12 层高精度 **`ML-pumpkin-1103`**（默认）与 8 层轻量 **`zhen-feather-preview-1121`**。
- **零额外内存 `.gram` 融合 ＋ Pass-2 校验 P2C**：通过 Win32 `CreateFileMappingW` 与 `rime.dll` 共享页缓存读取 `.gram` 语法模型（`0 MB` 额外物理内存，`~0.01 ms`/句），自动沿用方案实际惩罚参数；并对约 7% 高潜力 OOV 候选复用前缀 KV-cache 做 Pass-2 跨编码器 ＋ `.gram` 双重校验合成。
- **非阻塞隐私加固 UIA 跨应用上文**：独立后台 STA 线程读取光标前最多 64 字（按键热路径 `< 2 µs`、首键异步预热）；通过 `RangeFromChild(focused)` 与 `U+FFFC` 边界钳制杜绝 Chromium/Electron 泄漏周边工具栏文字；按 `HWND` 隔离防跨应用串味；硬屏蔽密码框与常见密码管理器。
- **严格保护精确词库**：仅对 `cand:get_genuine().type == "sentence"` 的整句候选重排，绝不干扰词库精确词、用户词库、英文混输、日期、数字与标点。

## 系统架构

```text
官方小狼毫 (WeaselServer.exe / rime.dll, librime >= 1.17.0, Lua 5.4)
  ├── lua/rime_neural.lua                # 通用滤镜：自动识别方案、仅重排整句、首键预热 UIA、原词反馈抓取
  └── neural/rime_neural.dll (Rust)      # 动态绑定 rime.dll Lua C API + 进程钉驻防重部署卸载
        ├── engine.rs                    # ONNX 动态模型自省、选择性输出裁剪、16 槽 KV-cache、256 槽 LRU、Pass-2 P2C
        ├── gram.rs                      # Win32 CreateFileMappingW 共享内存映射 Rime::Grammar/1.0（零堆分配）
        ├── pinyin.rs                    # 记忆化 O(N*L) 全拼对齐 + 7 套双拼（3,008/3,008 零失配）
        ├── uia.rs                       # 非阻塞后台 STA UIA 读取（RangeFromChild 钳制 + 密码框硬屏蔽）
        └── feedback.rs                  # 本地脱敏无偏选词日志（5 MiB 自动轮转）
```

详细实现机制见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 运行环境

- **系统**：Windows 10 / 11（x64）
- **输入法**：官方 [小狼毫（Weasel）](https://github.com/rime/weasel/releases)，要求 `librime >= 1.17.0`（如 Weasel Nightly `0.17.4+`）
- **可选（推荐）**：搭配 Rime `.gram` 语法模型（如 [`wanxiang-lts-zh-hans.gram`](https://github.com/amzxyz/rime-wanxiang-data/releases/tag/LTS)）；未安装 `.gram` 时自动回退为纯神经重排

## 快速开始

### 第一步：一键安装 DLL、Lua 滤镜、ONNX Runtime 与模型

在 PowerShell 中执行（加 `-Prebuilt` 时**无需安装 Rust**）：

```powershell
# 默认：12 层 ML-pumpkin-1103（中文整句精度最高）
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt

# 可选：8 层 zhen-feather（更轻量，适合低功耗 / 老旧 CPU）
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt -Model feather
```

安装脚本会自动完成：
1. 从本仓库 GitHub Releases 下载官方预编译 `rime-neural-*-win-x64.zip` 并校验 `SHA256SUMS.txt`；
2. 从 `dd1000001000/zhiyi_models` 下载所选 Apache-2.0 ONNX 模型并校验 SHA-256；
3. 自动复用本机或从官方 Microsoft NuGet 拉取 `onnxruntime.dll`（win-x64）；
4. 安全停止小狼毫，将文件安装到 `%APPDATA%\Rime\neural\` 与 `%APPDATA%\Rime\lua\rime_neural.lua` 后重启小狼毫。

若本机已安装 Rust（1.80+）希望自行从源码编译：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

### 第二步：在方案的 `*.custom.yaml` 中挂载滤镜

仓库 [`examples/`](examples/) 已内置雾凇全拼、小鹤双拼、微软双拼、白霜自然码双拼、万象拼音的即用补丁。最简配置如下：

```yaml
patch:
  # 1. 放宽 Rime 已有整句 beam 输出阈值，向神经重排提供 Top-8 高质量候选
  translator/max_sentences: 8
  translator/sentence_cutoff_threshold: 0.3

  # 2. 在 filters 最前面挂载 rime_neural（务必位于简繁转换 / OpenCC 之前）
  engine/filters/@before 0: lua_filter@*rime_neural

  # 3. 可选：在 F4 菜单中添加开关（默认开启）
  switches/+:
    - name: neural_rerank
      states: [ 统计, 神经 ]
      reset: 1

  # 4. 重排参数（均为可选，以下为默认值）
  rime_neural:
    option_name: neural_rerank
    pinyin_layout: auto          # auto | full | zrm | flypy | mspy | sogou | abc | ziguang | jiajia
    max_candidates: 8
    fusion_weights: [1.0, -1.4, 0.05, 0.15, 0.0]
    p2c_verify_margin: 0.20      # <= 0 可关闭 Pass-2 P2C 合成
    p2c_top1_logit_diff: 2.0
    p2c_top1_min_gram_diff: -2.0
    p2c_top2_logit_diff: 0.5
    enable_uia_context: true
    threads: 0                   # 0 = 自动适配 CPU 核数（限幅 4..8）
    log_feedback: true           # 本地日志：%APPDATA%\Rime\neural\feedback.jsonl（5 MiB 上限）
    sync_init: false             # 后台异步加载，0 ms 启动延迟
```

保存后点击小狼毫 **「重新部署」** 即可生效。若移除 DLL/模型或在 `F4` 中关闭 `neural_rerank`，滤镜会自动零开销透传。

> **万象拼音（`rime-wanxiang`）双拼用户提示**：万象切换双拼时 `schema_id` 通常仍保持为 `wanxiang`，请将 `pinyin_layout` 显式设为对应的 `flypy` / `zrm` / `mspy` 等。

## 配置项速查

| 配置键（`rime_neural/...`） | 默认值 | 说明 |
| :--- | :--- | :--- |
| `option_name` | `neural_rerank` | 控制神经重排开关的 Rime 布尔选项名 |
| `pinyin_layout` | `auto` | `auto` / `full` / `zrm` / `flypy` / `mspy` / `sogou` / `abc` / `ziguang` / `jiajia`（`pyjj`） |
| `max_candidates` | `8` | 送入神经重排的头部 `sentence` 候选上限 |
| `fusion_weights` | `[1.0, -1.4, 0.05, 0.15, 0.0]` | 5 维线性先验 `[logit_lp, ln(rank+1), gram_delta, char_lp, logit_lp*has_ctx]` |
| `p2c_verify_margin` | `0.20` | 触发 KV 缓存 Pass-2 P2C 校验的 Pass-1 `char_logits` 裕度（`<= 0` 关闭） |
| `p2c_top1_logit_diff` | `2.0` | Pass-2 将 P2C 提升至首选所需的最小 logit 领先值 |
| `p2c_top1_min_gram_diff` | `-2.0` | Pass-2 提升至首选时 `.gram` 逐字分差安全底线 |
| `p2c_top2_logit_diff` | `0.5` | Pass-2 将 P2C 安全插入次选所需的最小 logit 领先值 |
| `enable_uia_context` | `true` | 非阻塞后台 Windows UIA 光标前文（支持按方案独立开关） |
| `threads` | `0` | ONNX Runtime CPU 线程数（`0` 自动探测并限幅 `4..8`） |
| `min_input_len` | 双拼 `4` / 全拼 `6` | 触发神经重排的最小有效拼音按键数 |
| `model_file` | `""`（自动探测） | 可选：显式指定 `%APPDATA%\Rime\neural\` 下的 ONNX 文件名 |
| `log_feedback` | `true` | 无偏记录脱敏后的多候选上屏事件供本地 MAP 调参 |
| `sync_init` | `false` | `false` 后台异步预热（`0 ms` 启动）；`true` 同步加载（评测脚本用） |

## 实测评测与横向对比

以下数据均在官方未修改的小狼毫 `0.17.4`（`librime 1.17.0`、严格关闭 `enable_user_dict` 隔离测试，沿用 [`gaboolic/rime-schema-compare`](https://github.com/gaboolic/rime-schema-compare) 同义词归一化与 Levenshtein 字级口径）于 Intel Core Ultra 7 255HX 实测获得。

### 1. 与主流 Rime 方案横向对比（公开语料 `public_articles_2026-11-19.txt`，全拼）

与 [`rime-schema-compare/report/latest.md`](https://github.com/gaboolic/rime-schema-compare/blob/main/report/latest.md) 采用完全相同的公开语料文件（SHA-256 `1a8edab3...`）与同版 `wanxiang-lts-zh-hans.gram`（SHA-256 `e9ddd501...`）：

| 方案 / 引擎 | 整句正确率 | 文字正确率 | 相对纯 N-gram 榜首 |
| :--- | :---: | :---: | :---: |
| `mingyuepinyin`（明月拼音，无模型） | `63.32%` | `92.15%` | `-19.59%` |
| `luna_pinyin_with_octagram`（原版朙月 + 八股文） | `66.83%` | `93.78%` | `-16.08%` |
| `rime_ice`（雾凇拼音，无模型） | `69.35%` | `93.93%` | `-13.56%` |
| `wanxiang`（万象拼音，无模型） | `69.35%` | `94.02%` | `-13.56%` |
| `rime_frost`（白霜拼音，无模型） | `70.85%` | `94.27%` | `-12.06%` |
| `rime_ice_with_gram`（雾凇 + 万象 400 MB） | `74.37%` | `95.95%` | `-8.54%` |
| `rime_wanxiang_with_gram`（万象 + 万象 400 MB） | `78.89%` | `96.84%` | `-4.02%` |
| `rime_wubi_sentens_wubi86_with_gram`（五笔整句 + 万象） | `79.40%` | `96.99%` | `-3.51%` |
| `rime_frost_with_gram`（白霜 + 万象 400 MB，纯 N-gram 榜首） | `82.91%` | `97.38%` | 基准 |
| **`rime_frost` + `rime-neural v1.0.0`（严格单句零上下文）** | **`89.50%`** | **`98.42%`** | **`+6.59%`** |
| **`rime_frost` + `rime-neural v1.0.0`（连续 `commit_history`）** | **`91.00%`** | **`98.72%`** | **`+8.09%`** |

### 2. 多神经重排模型横向对比（1,600 句双拼，同套 Rime Top-8 候选）

| 重排模型 | 模型体积 | 1,600 句总计（整句 / 文字） | 科技 200 | 知乎盲测验证 1,000 |
| :--- | :---: | :---: | :---: | :---: |
| 纯统计基线（白霜 + 万象 372 MB `.gram`） | — | `75.12%` / `95.51%` | `81.00%` | `75.80%` |
| `msime-4M`（神经语言模型，离线最优） | 4.5 MB | `76.12%`（`+1.00%`） | — | — |
| `msime-25M`（神经语言模型，离线最优） | 25.5 MB | `78.81%`（`+3.69%`） | — | — |
| `知意 1.4V2-Long-L12`（无 KV-cache，实跑 `commit_history`） | 84 MB | `81.31%` / `96.83%` | `90.50%` | `80.00%` |
| `zhen-feather-preview-1121`（8 层轻量 KV-cache，实跑） | 38 MB | — | `88.50%` | — |
| **`rime-neural v1.0.0`（`ML-pumpkin-1103`，实跑 `commit_history`）** | **82 MB** | **`82.19%` / `97.05%`** | **`91.00%`** | **`81.40%`** |
| **`rime-neural v1.0.0`（带标点上文，模拟真实 UIA 编辑器）** | **82 MB** | **`85.19%`** | **`92.00%`** | **`84.60%`** |

### 3. 防过拟合跨领域盲测（840 句，5 个完全未触碰过的领域）

参数从未在以下 5 个语料上调优，检验真实泛化能力（新闻 / 小说 / 散文 / 旧科技 / 2026-08-30 知乎）：

| 配置 | 840 句总计（整句 / 文字） | 新闻 (152) | 小说 (40) | 散文 (211) | 旧科技 (137) | 知乎 08-30 (300) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 纯统计基线（万象 `.gram`） | `60.48%` / `90.94%` | `70.4%` | `65.0%` | `37.0%` | `66.4%` | `68.7%` |
| **`rime-neural v1.0.0`（`[1.0, -1.4, 0.05, 0.15, 0.0]`）** | **`68.57%` / `93.52%`**（`+8.09%`） | **`78.3%`** | **`70.0%`** | **`46.4%`** | **`75.9%`** | **`75.7%`** |

### 4. 按键延迟实测（Ultra 7 255HX，隔离 `rime.dll` 基准）

| 模式 | 插电（首键 / 后续键均值 / 长句峰值） | 电池（首键 / 后续键均值 / 长句峰值） |
| :--- | :---: | :---: |
| 纯统计基线（自然码双拼） | `0.1 ms / 0.4 ms / 0.9 ms` | `0.4 ms / 1.6 ms / 5.2 ms` |
| **`rime-neural v1.0.0`（双拼）** | **`0.1 ms / 2.0 ms / 11.5 ms`** | **`0.4 ms / ~4.1 ms / ~22 ms`** |
| **`rime-neural v1.0.0`（全拼）** | **`0.2 ms / 4.7 ms / 14.7 ms`** | **`0.3 ms / ~9.2 ms / ~28 ms`** |

## 隐私与安全边界

- **100% 本地离线推理**：运行期间零网络请求；仅在手动执行安装脚本时下载并做 SHA-256 校验。
- **密码场景硬屏蔽**：控件报告 `IsPassword == true`，或前台窗口 / 焦点控件属于常见密码管理器与系统凭据界面（`KeePass`、`KeePassXC`、`1Password`、`Bitwarden`、`Enpass`、`LastPass`、`Dashlane`、`RoboForm`、`NordPass`、`ProtonPass`、`CredentialUIBroker`、`consent.exe`、`LockApp`、`mstsc.exe`）时，立即返回空上下文并暂停记录反馈。
- **防跨控件与跨应用串味**：UIA 读取范围严格钳制在 `RangeFromChild(&focused)` 与 `U+FFFC` 之内，切换前台窗口（`HWND`）立即隔离上一个应用的旧 `commit_history`。
- **脱敏且限容的本地反馈**：`feedback.jsonl` 自动将 4 位以上数字串、邮箱、URL 与 12 位以上 ASCII Token 脱敏为 `<REDACTED>`，上文截断 64 字，文件上限 5 MiB 自动轮转；设为 `log_feedback: false` 即可彻底关闭。

## 可选：个人融合权重离线校准

日常使用积累一定量本地记录（`%APPDATA%\Rime\neural\feedback.jsonl`）后，可运行：

```powershell
python scripts\tune-fusion.py
```

脚本会在通用跨领域先验 `[1.0, -1.4, 0.05, 0.15, 0.0]` 附近执行 **Newton-Raphson MAP 岭回归 ＋ 5 折交叉验证加权对数损失（NLL）＋ 成对 95% 置信区间检验（`z >= 1.645`，Kish 有效样本量 `N_eff`）**，且要求样本 `>= 100` 条、Top-1 准确率不下降才会给出新权重（25 轮零假设盲测误伤率 `0/25`，真实偏好偏移检出率 `88%–100%`）。

## 常见问题

- **重新部署后没有生效**：检查 `%TEMP%\rime.weasel\*WARNING*` 日志，确认 `%APPDATA%\Rime\neural\` 下包含 `rime_neural.dll`、`onnxruntime.dll`、ONNX 模型、`tokenizer.json`、`unified_vocab.json`，且 `%APPDATA%\Rime\lua\rime_neural.lua` 已就位。
- **小狼毫版本要求**：需要 `librime >= 1.17.0` 的官方小狼毫（导出 Lua 5.4 C API 符号）；老版本缺少导出符号时会自动无害透传回纯统计模式。
- **如何关闭 UIA 上下文或本地日志**：在方案 `*.custom.yaml` 的 `rime_neural:` 下将 `enable_uia_context` / `log_feedback` 设为 `false` 后重新部署即可。

## 协议与致谢

- **`rime-neural`** 源码按 [Apache-2.0](LICENSE) 协议开源。
- 默认兼容的端侧重排模型（`ML-pumpkin-1103`、`zhen-feather-preview-1121`）来自 [dd1000001000/zhiyi_models](https://github.com/dd1000001000/zhiyi_models)（Apache-2.0），按需下载并做 SHA-256 校验，不随源码入库。
- `.gram` 共享内存映射兼容 [`Rime::Grammar/1.0`](https://github.com/lotem/librime-octagram) 格式（如 [amzxyz/rime-wanxiang-data](https://github.com/amzxyz/rime-wanxiang-data)）。
- 面向 [rime/weasel](https://github.com/rime/weasel) 与 [rime/librime](https://github.com/rime/librime) 生态构建。
