<div align="center">

# rime-neural

**Windows Rime / 小狼毫（Weasel）本地 ONNX 神经整句重排插件**

[![CI & Release](https://github.com/Chth1z/rime-neural/actions/workflows/ci-release.yml/badge.svg)](https://github.com/Chth1z/rime-neural/actions/workflows/ci-release.yml)
[![Release](https://img.shields.io/github/v/release/Chth1z/rime-neural?label=release)](https://github.com/Chth1z/rime-neural/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64%20(Weasel%20%E2%89%A5%200.17.4)-0078D6.svg)](https://github.com/rime/weasel)

**简体中文** | [English](README.en.md)

</div>

---

## 简介

`rime-neural` 是面向 Windows 官方小狼毫（Weasel `>= 0.17.4`，`librime >= 1.17.0`）的进程内 ONNX 神经整句重排插件，由 Lua 5.4 滤镜和 Rust 原生模块组成，直接挂载在现有 Rime 输入方案中运行。

### 特性

- **与 `.gram` 语言模型协同**：支持单独运行，也支持搭配中文统计语法模型（推荐 [`wanxiang-lts-zh-hans.gram`](https://github.com/amzxyz/rime-wanxiang-data/releases/tag/LTS)）。Rime 借助 `.gram` 生成 Top-8 整句候选，插件按路径共享只读 `.gram` 映射，按方案应用惩罚参数，联合融合神经模型分数、`.gram` 搭配分与候选位置先验，并对高置信度未登录整句执行 KV 缓存 Pass-2 校验合成。
- **仅作用于整句候选**：仅重排配置分段标签（默认 `abc`）下 `cand:get_genuine().type == "sentence"` 的整句候选，词典精确词条、用户词典、英文混输、日期、数字与标点保持原序。
- **全拼与 7 套双拼**：自动识别全拼以及自然码（ZRM）、小鹤（Flypy）、微软（MSPY，含 `; = ing`）、搜狗（Sogou，含 `; = ing`）、智能 ABC、紫光（含 `; = ing`）、拼音加加（PYJJ），适配雾凇、白霜、万象及自定义方案。
- **本地 CPU 推理与进程内复用**：基于 ONNX Runtime 量化模型与 16 槽前缀 KV-cache 在 CPU 离线运行；启动阶段异步加载模型，模块通过 `GetModuleHandleExA(PIN)` 钉驻小狼毫进程，重新部署时直接复用已加载实例。
- **多模型自动识别**：启动时读取 ONNX 图结构与 `unified_vocab.json`，同一份 `rime_neural.dll` 支持 12 层 `ML-pumpkin-1103`（默认，约 82 MB）与 8 层 `zhen-feather-preview-1121`（约 38 MB）。
- **跨应用前文读取与隐私保护**：独立后台 MTA 线程通过 Windows UIA 读取光标前最多 64 字（按键热路径 `< 2 µs`），读取范围严格钳制在当前焦点控件边界（`RangeFromChild` / `DocumentRange`）与 `U+FFFC` 对象边界之内；每次编码会话冻结上文快照，焦点或窗口切换时立即失效。已确认安全的非 `TextPattern` 控件仅在相同焦点周期（`(HWND, focus_epoch)`）内沿用 Rime 会话历史与已确认前缀。遇到密码控件、受保护进程、密码管理器、远程会话客户端及本地终端时 fail-closed 返回空上文并停止记录日志。

### 运行要求与限制

- **平台**：Windows 10 / 11 (x64)，官方小狼毫 `>= 0.17.4`（`librime >= 1.17.0`）。暂不支持 macOS（鼠须管）与 Linux。
- **内存占用**：默认 12 层模型 `ML-pumpkin-1103` 连同 ONNX Runtime 约占用 150–250 MB 内存；对内存或续航敏感的设备可选用 8 层模型 `zhen-feather`（权重约 38 MB）。
- **上下文窗口**：前文窗口上限 128 字，在 300 字以上的超长前文场景中，准确率低于运行在 GPU 上的 1–2 GB 通用大语言模型。

---

## 安装与配置

### 方式一：由 AI Agent 自动安装

将以下指令发送给支持终端操作的 AI Agent（如 Claude Code、Codex、Cursor、Antigravity、Gemini CLI 等），Agent 会读取仓库内的安装指引与辅助脚本，依次完成环境检测、配置备份、下载校验、`*.custom.yaml` 合并、部署及日志验证：

```text
请阅读 https://raw.githubusercontent.com/Chth1z/rime-neural/main/skills/rime-neural-setup/SKILL.md，按其中的步骤为本机的小狼毫安装并配置 rime-neural，完成后执行部署并验证结果。
```

### 方式二：手动安装

确认本机已安装官方 [小狼毫 Nightly `0.17.4+`](https://github.com/rime/weasel/releases)（`librime >= 1.17.0`）。

#### 1. 安装插件、ONNX Runtime 与模型

在 PowerShell 中执行（加 `-Prebuilt` 下载预编译包，无需 Rust 工具链）：

```powershell
# 默认：12 层 ML-pumpkin-1103（约 82 MB）
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt

# 可选：8 层 zhen-feather（约 38 MB）
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prebuilt -Model feather
```

安装脚本会依次完成：
1. 从 GitHub Releases 下载预编译 `rime-neural-*-win-x64.zip` 并校验独立资产 `SHA256SUMS.txt`；
2. 从 `dd1000001000/zhiyi_models` 下载所选 Apache-2.0 ONNX 模型并全量校验 `.onnx`、`tokenizer.json` 与 `unified_vocab.json` 的 SHA-256；
3. 从 Microsoft NuGet 下载锁定版本并校验 SHA-512 / SHA-256 的官方 `onnxruntime.dll`（win-x64，也支持通过 `-OrtDllPath` 指定本机 DLL）；
4. 停止小狼毫并备份待覆盖文件，将组件安装至 Rime 用户目录（自动读取注册表 `RimeUserDir`，默认 `%APPDATA%\Rime\neural\`；非 ASCII 用户名可通过 `-DataDir` 指定 ASCII 路径）与 `lua\rime_neural.lua`，失败时完整回滚，完成后重启小狼毫。

从源码编译（需 Rust 1.92+）时，省略 `-Prebuilt` 直接执行 `scripts\install.ps1`。

> **推荐配套语言模型**：从 [amzxyz/rime-wanxiang-data (LTS)](https://github.com/amzxyz/rime-wanxiang-data/releases/tag/LTS) 下载 `wanxiang-lts-zh-hans.gram` 放入 Rime 用户目录，并在方案中启用 `grammar/language: wanxiang-lts-zh-hans`。

#### 2. 在方案的 `*.custom.yaml` 中挂载滤镜

仓库 [`examples/`](examples/) 提供了雾凇全拼、小鹤双拼、微软双拼、白霜自然码双拼与万象拼音的补丁模板。最小配置如下：

```yaml
patch:
  # 放宽 Rime 整句 beam 输出数量，向神经重排提供 Top-8 候选
  translator/max_sentences: 8
  translator/sentence_cutoff_threshold: 0.3

  # 挂载于 filters 首位（位于简繁转换 / OpenCC 之前）
  engine/filters/@before 0: lua_filter@*rime_neural

  # 可选：在 F4 菜单中添加「统计 / 神经」切换开关（省略时默认开启）
  switches/+:
    - name: neural_rerank
      states: [ 统计, 神经 ]
      reset: 1

  # 以下均为默认值，可按需省略或覆写
  rime_neural:
    option_name: neural_rerank
    pinyin_layout: auto          # auto | full | zrm | flypy | mspy | sogou | abc | ziguang | jiajia
    max_candidates: 8
    fusion_weights: [1.0, -1.4, 0.05, 0.15, 0.0]
    p2c_verify_margin: 0.20      # <= 0 关闭 Pass-2 P2C 合成
    enable_uia_context: true
    threads: 0                   # 0 = 自动适配 CPU 核数（限幅 4..8）
    log_feedback: true           # 零明文纯数值本地日志（单文件上限 5 MiB + 1 份 .1 轮转）
    sync_init: false             # 异步加载
```

保存后执行小狼毫 **「重新部署」**。

> **万象拼音（`rime-wanxiang`）双拼用户**：万象切换双拼时 `schema_id` 通常保持为 `wanxiang`，`auto` 会识别为全拼，需将 `pinyin_layout` 显式设为对应的双拼布局（如 `flypy`、`zrm`、`mspy`）。

---

## 评测数据

### 测试环境

- **运行环境与模型**：官方小狼毫 `0.17.4`（`librime 1.17.0`），默认 12 层量化模型 `ML-pumpkin-1103`（`threads: 0`，自动限幅 `4..8` 线程），测试期间关闭用户词库（`enable_user_dict: false`）与 UIA 读屏；硬件为 Intel Core Ultra 7 255HX。
- **方案与词库基线**：端到端评测配置见 [`Chth1z/rime-chth1z`](https://github.com/Chth1z/rime-chth1z)（上游 `gaboolic/rime-frost` 词库 + `wanxiang-lts-zh-hans.gram`，注释 `41448` 与 `GB18030-2022` 生僻字表，`grammar/collocation_penalty: -15.5`）。下表同时列出同环境关闭 `rime-neural` 的统计基线及 `-14` 惩罚参数对照。

### 1. Rime 主流方案对比（`gaboolic/rime-schema-compare/report/latest.md`）

语料为 [`rime-schema-compare`](https://github.com/gaboolic/rime-schema-compare/blob/main/report/latest.md) 的 `public_articles_2026-11-19.txt`（阮一峰科技爱好者周刊正文）与同版 `wanxiang-lts-zh-hans.gram`。

| 方案 / 组合 | 整句正确率 | 文字正确率 | 说明 |
| :--- | :---: | :---: | :--- |
| `mingyuepinyin`（明月，无模型） | `63.32%` | `92.15%` | 上游报告（199 句） |
| `luna_pinyin_with_octagram`（朙月 + 八股文） | `66.83%` | `93.78%` | 上游报告 |
| `rime_ice`（雾凇，无模型） | `69.35%` | `93.93%` | 上游报告 |
| `wanxiang`（万象，无模型） | `69.35%` | `94.02%` | 上游报告 |
| `rime_frost`（白霜，无模型） | `70.85%` | `94.27%` | 上游报告 |
| `rime_ice_with_gram`（雾凇 + 万象 `.gram`） | `74.37%` | `95.95%` | 上游报告 |
| `rime_wanxiang_with_gram`（万象 + 万象 `.gram`） | `78.89%` | `96.84%` | 上游报告 |
| `rime_wubi_sentens_wubi86_with_gram`（五笔整句 + 万象） | `79.40%` | `96.99%` | 上游报告 |
| `rime_frost_with_gram`（白霜 + 万象 `.gram`） | `82.91%` | `97.38%` | 上游报告（`collocation_penalty: -14`） |
| 本机对照：白霜 + 万象 `.gram`（关 `rime-neural`，`-14`） | `83.00%` | `97.39%` | 200 句，复现上游统计基线 |
| 本机 `rime-chth1z`：白霜 + 万象 `.gram`（关 `rime-neural`，`-15.5`） | `80.50%` | `96.95%` | 同配置关闭神经重排 |
| **白霜 + 万象 `.gram` + `rime-neural v1.0.0`（全拼·严格零上文）** | **`89.50%`** | **`98.42%`** | `-14` 与 `-15.5` 均为 179/200（`+6.50%`） |
| **白霜 + 万象 `.gram` + `rime-neural v1.0.0`（双拼·严格零上文）** | **`90.00%`** | **`98.52%`** | 自然码双拼 180/200 |
| **白霜 + 万象 `.gram` + `rime-neural v1.0.0`（全拼 / 双拼·段落上文）** | **`91.00%`** | **`98.67%`** | 按段落保留 `commit_history`（182/200） |

### 2. 跨领域 246 句商业输入法参考（`rime-schema-compare/report/other_latest.md`）

语料为 [`other_latest.md`](https://github.com/gaboolic/rime-schema-compare/blob/main/report/other_latest.md) 对应的 246 句（commit `99da339a`：`news 36`、`novel 136`、`prose 38`、`tech 31`、`test 5`），同义词归一化规则与原报告一致。

| 输入法 / 方案 | 总整句 (246句) | 总字正确率 | `news` | `novel` | `prose` | `tech` | `test` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 微信输入法（原报告 Notepad 黑盒） | `51.63%` | `89.18%` | `61.11%` | `47.79%` | `50.00%` | `58.06%` | `60.0%` |
| 搜狗拼音（原报告 Notepad 黑盒） | `54.47%` | `77.02%` | `77.78%` | `45.59%` | `50.00%` | `74.19%` | `40.0%` |
| 微软拼音（原报告 Notepad 黑盒） | `67.89%` | `92.71%` | `83.33%` | `63.24%` | `63.16%` | `74.19%` | `80.0%` |
| 手心输入法（原报告 Notepad 黑盒） | `71.14%` | `94.48%` | `86.11%` | `66.18%` | `65.79%` | `80.65%` | `80.0%` |
| 原报告 `rime_frost_with_gram`（2026-04-25） | `71.54%` | `95.45%` | `88.89%` | `67.65%` | `55.26%` | `83.87%` | `100%` |
| 本机 `rime-chth1z` 纯统计（白霜 + 万象，关 neural） | `76.02%` | `95.85%` | `97.22%` | `69.12%` | `71.05%` | `83.87%` | `100%` |
| **白霜 + 万象 `.gram` + `rime-neural v1.0.0`（全拼·严格零上文）** | **`81.30%`** | **`97.29%`** | **`97.22%`** | `77.21%` | `68.42%` | **`93.55%`** | **`100%`** |
| **白霜 + 万象 `.gram` + `rime-neural v1.0.0`（双拼·严格零上文）** | **`81.71%`** | `97.37%` | **`97.22%`** | **`77.94%`** | `68.42%` | **`93.55%`** | **`100%`** |
| **白霜 + 万象 `.gram` + `rime-neural v1.0.0`（双拼·段落上文）** | **`81.71%`** | **`97.51%`** | **`97.22%`** | `76.47%` | **`76.32%`** | `90.32%` | **`100%`** |

> 商业输入法数据引自 `rime-schema-compare/report/other_latest.md`（Notepad 前台自动化黑盒采集，受系统环境、默认设置与联网状态影响）。

### 3. 开源神经重排器同池对比（`metasequoiaime/chinese-ime-lm` `sentences-ctx-v1`）

使用正式版 `rime_neural.dll v1.0.0` 直接在 [`metasequoiaime/chinese-ime-lm`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md) 的公开固定候选池 `eval/sentences-ctx-v1/pool.jsonl`（2026 年 9–10 月政府公文与 KdConv / CrossWOZ / RiSAWOZ 对话，共 1,631 条整句，候选由 MSIME 生成）上逐条打分：

| 打分器 | 权重体积 | 有上文 (1,196) | 无上文 (435) | 总计 (1,631) | 运行环境 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| MSIME 引擎原序 | — | `70.1%` (838) | `67.4%` (293) | `69.34%` | — |
| `msime-4M` | 4.5 MB | `81.9%` (979) | `77.5%` (337) | `80.69%` | CPU 1T |
| `msime-25M` | 25.5 MB | `87.0%` (1041) | `86.0%` (374) | `86.76%` | CPU 1T |
| 青简 `含章·通变` | 46 MB | `92.1%` (1102) | `81.4%` (354) | `89.27%` | Apple GPU |
| 青简 `含章·知微` | 56 MB | `91.2%` (1091) | `88.3%` (384) | `90.44%` | Apple GPU |
| 知意 `1.4V2-Long-L12 int8` | 84 MB | `93.8%` (1122) | `87.8%` (382) | `92.21%` | CPU 4T |
| 知意 `ML-pumpkin-1026 int8` | 92 MB | `94.6%` (1131) | `86.9%` (378) | `92.52%` | CPU 4T |
| **`rime-neural`（纯 `ML-pumpkin-1103`，无先验）** | **82 MB** | **`95.15%`** (1138) | `87.13%` (379) | `93.01%` | CPU 4T |
| **`rime-neural`（`ML-pumpkin-1103` + 万象 `.gram`，无引擎先验）** | **82 MB** | **`95.15%`** (1138) | **`88.28%`** (384) | **`93.32%`** | CPU 4T |
| **`rime-neural v1.0.0` 默认全融合** | **82 MB** | `94.98%` (1136) | **`88.51%`** (385) | `93.26%` | CPU 4T |
| *参考：`Qwen2.5-0.5B-Instruct-4bit`* | *278 MB* | *`96.2%` (1150)* | *`85.5%` (372)* | *`93.32%`* | *Apple GPU (mlx)* |
| *参考：`Qwen3-0.6B-8bit`* | *633 MB* | *`96.5%` (1154)* | *`78.9%` (343)* | *`91.78%`* | *Apple GPU (mlx)* |
| *参考：`Qwen3.5-0.8B-8bit`* | *1.0 GB* | *`97.3%` (1164)* | *`87.4%` (380)* | *`94.67%`* | *Apple GPU (mlx)* |
| *参考：`Qwen3.5-2B-4bit`* | *1.7 GB* | *`97.7%` (1169)* | *`89.7%` (390)* | *`95.59%`* | *Apple GPU (mlx)* |

### 4. 按键延迟

Intel Core Ultra 7 255HX，隔离 `rime.dll` 基准实测：

| 模式 | 插电（首键 / 后续键均值 / 长句峰值） | 电池（首键 / 后续键均值 / 长句峰值） |
| :--- | :---: | :---: |
| 纯统计（自然码双拼 + 万象 `.gram`） | `0.1 ms / 0.4 ms / 0.9 ms` | `0.4 ms / 1.6 ms / 5.2 ms` |
| **`rime-neural v1.0.0`（双拼）** | **`0.1 ms / 2.0 ms / 11.5 ms`** | **`0.4 ms / ~4.1 ms / ~22 ms`** |
| **`rime-neural v1.0.0`（全拼）** | **`0.2 ms / 4.7 ms / 14.7 ms`** | **`0.3 ms / ~9.2 ms / ~28 ms`** |

---

## 架构与配置项

```text
官方小狼毫 (WeaselServer.exe / rime.dll, librime >= 1.17.0, Lua 5.4)
  ├── lua/rime_neural.lua            # Lua 滤镜：布局识别、整句/标签门控、会话上文冻结、纯数值反馈
  └── neural/rime_neural.dll (Rust)  # 动态解析 Lua 5.4 C API（C-unwind），进程内钉驻，内嵌 PE 版本信息
        ├── engine.rs                # ONNX 结构自省、选择性输出裁剪、16 槽 KV-cache、Pass-2 P2C、按方案 .gram 缓存
        ├── gram.rs                  # Win32 共享内存只读映射 Rime::Grammar/1.0（兼容 BMP/非 BMP UTF-8）
        ├── pinyin.rs                # 3 阶段记忆化全拼对齐 + DP 切分 + 7 套双拼码表
        ├── uia.rs                   # 后台 MTA UIA（焦点变更事件 + 控件边界钳制 + 200ms 超时 + 密码/终端屏蔽）
        └── feedback.rs              # 零明文纯数值本地日志（5 MiB 自动轮转 + 重解析点防护）
```

实现细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

| 配置键（`rime_neural/...`） | 默认值 | 说明 |
| :--- | :--- | :--- |
| `option_name` | `neural_rerank` | 控制重排开关的 Rime 布尔选项名；未在方案 `switches` 中声明时默认启用 |
| `pinyin_layout` | `auto` | `auto` / `full` / `zrm` / `flypy` / `mspy` / `sogou` / `abc` / `ziguang` / `jiajia`（`pyjj`） |
| `tags` | `["abc"]` | 允许参与重排的 Rime 分段标签白名单 |
| `max_candidates` | `8` | 参与重排的头部 `sentence` 候选数量上限（限幅 `2..64`） |
| `fusion_weights` | `[1.0, -1.4, 0.05, 0.15, 0.0]` | 5 维线性融合权重 `[logit_lp, ln(rank+1), gram_delta, char_lp_delta, logit_lp*has_ctx]`（`rank` 从 0 起始） |
| `p2c_verify_margin` | `0.20` | 触发 KV 缓存 Pass-2 P2C 校验的裕度阈值（`<= 0` 关闭） |
| `p2c_top1_logit_diff` | `2.0` | Pass-2 候选升至首选所需的最小 logit 领先值 |
| `p2c_top1_min_gram_diff` | `-2.0` | Pass-2 升至首选的 `.gram` 逐字分差底线（插入次选底线为该值减 `1.5`） |
| `p2c_top2_logit_diff` | `0.5` | Pass-2 候选插入次选所需的最小 logit 领先值 |
| `enable_uia_context` | `true` | 非阻塞后台 Windows UIA 前文读取（按方案独立生效） |
| `threads` | `0` | ONNX Runtime CPU 线程数（`0` 自动适配并限幅 `4..8`） |
| `min_input_len` | 双拼 `4` / 全拼 `6` | 触发神经重排的最小有效拼音按键数（下限 `1`） |
| `data_dir` | `""`（自动探测） | 可选：显式指定 `neural\` 数据目录（非 ASCII Windows 用户名可指向纯 ASCII 路径，YAML 中建议用单引号或 `/`） |
| `model_file` | `""`（自动探测） | 可选：显式指定 `neural\` 下的 `.onnx` 文件名 |
| `t2s_config` | `""`（自动 `tw2sp` / `t2s`） | 可选：显式指定繁体上文转简体的 OpenCC 配置文件（如 `t2s.json`） |
| `log_feedback` | `true` | 零明文纯数值本地日志开关（供可选的个人权重校准脚本使用） |
| `sync_init` | `false` | `false` 后台异步加载；`true` 同步加载（离线评测脚本用） |

## 隐私与安全边界

完整的威胁模型与安全漏洞报告方式见 [`SECURITY.md`](SECURITY.md)。

- **本地离线推理**：推理阶段无网络请求；下载行为仅发生在执行安装脚本时，且对二进制、ONNX Runtime 与模型均执行加密哈希校验。
- **敏感场景 fail-closed 屏蔽**：以下情形均返回空上文并暂停反馈记录：
  - 焦点控件报告 `IsPassword == true`，或访问 `IsPassword` 属性时返回访问拒绝，或内容呈掩码圆点 / 星号（`••••••` / `******`）；
  - 前台或焦点进程为受保护进程（`OpenProcess` 返回 `ERROR_ACCESS_DENIED`）或 PID 无法解析；
  - 命中敏感进程名单：密码管理器与验证器（`KeePass*`、`KeePassXC`、`1Password`、`Bitwarden`、`Enpass`、`LastPass`、`Dashlane`、`RoboForm`、`NordPass`、`ProtonPass`、`Keeper`、`PasswordSafe`、`Authy`、`WinAuth`）、系统凭据 / UAC 界面（`CredentialUIBroker`、`consent.exe`、`LockApp`）、远程桌面与 SSH 客户端（`mstsc.exe`、`msrdc.exe`、`PuTTY`、`Xshell`、`MobaXterm*`、`FinalShell`）以及本地终端（`WindowsTerminal`、`conhost`、`cmd`、`powershell`、`pwsh`、`mintty`、`alacritty`、`wezterm-gui`、`kitty`）。
- **零明文本地日志**：`%APPDATA%\Rime\neural\feedback.jsonl` 仅记录模型数值特征（`logits`、`gram_deltas`、`char_lps`）与候选索引结果（`shown_top1_idx`、`chosen_idx`、`has_ctx`、`has_gram`），不写入用户输入文本、按键串、候选词或屏幕上下文；写入端拒绝重解析点，在 `5 MiB` 时轮转并保留一份 `.1` 备份；遇到临时共享锁时，活动文件可增长至 `10 MiB`；设置 `log_feedback: false` 可关闭。
- **可选个人权重校准**：积累 100 条以上本地选词记录后，可使用 Python 3 + `numpy` 运行 `python scripts\tune-fusion.py`（支持 `--schema`、`--model`、`--has-gram` 筛选及 `--scrub-legacy-inplace` 清理历史格式）；调参器用五个连续留出块评估保留三位小数的权重。仅当加权对数损失下降并通过针对七组先验精度的 Bonferroni 单侧检验（名义显著性水平 5%，Hájek 稳健标准误）、Top-1 准确率不下降且权重满足约束时，才输出建议。

## 常见问题

- **部署后未生效**：检查 `%TEMP%\rime.weasel\*WARNING*` 与近期 `.INFO.` 日志；确认 `%APPDATA%\Rime\neural\`（或自定义 `data_dir`）下包含 `rime_neural.dll`（可在文件属性中核对版本号）、`onnxruntime.dll`、单个 `.onnx` 模型、`tokenizer.json`、`unified_vocab.json`，且 `%APPDATA%\Rime\lua\rime_neural.lua` 存在；若方案配置了 `switches`，确认 `F4` 菜单中 `neural_rerank` 处于开启状态。
- **小狼毫版本不兼容**：旧版小狼毫（如 `0.16.x`）的 `rime.dll` 未导出 Lua 5.4 C API 符号，滤镜会自动透传至统计模式，升级至小狼毫 `0.17.4+` 即可启用。
- **卸载**：移除 `*.custom.yaml` 中的 `rime_neural` 相关配置，退出小狼毫后删除 `lua\rime_neural.lua` 与 `neural\` 目录（若使用了自定义 `-DataDir`，一并删除该目录），重新部署即可。

## 协议与致谢

- 源码基于 [Apache-2.0](LICENSE) 协议开源。
- 默认支持的 ONNX 重排模型（`ML-pumpkin-1103`、`zhen-feather-preview-1121`）来自 [dd1000001000/zhiyi_models](https://github.com/dd1000001000/zhiyi_models)（Apache-2.0），按需下载并校验 SHA-256。
- 推荐配套的中文语法模型来自 [amzxyz/rime-wanxiang-data](https://github.com/amzxyz/rime-wanxiang-data)（`wanxiang-lts-zh-hans.gram`）；端到端评测基线基于 [gaboolic/rime-frost](https://github.com/gaboolic/rime-frost) 与 [iDvel/rime-ice](https://github.com/iDvel/rime-ice)；评测口径与语料参考 [gaboolic/rime-schema-compare](https://github.com/gaboolic/rime-schema-compare) 及 [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm)。
- 面向 [rime/weasel](https://github.com/rime/weasel) 与 [rime/librime](https://github.com/rime/librime) 生态构建。
