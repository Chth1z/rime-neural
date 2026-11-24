# Rime Schema Patching Reference (`*.custom.yaml`)

Use this reference when editing or creating `<schema_id>.custom.yaml` inside `<RimeUserDir>`.

## Core Rules (Never Violate)

1. **Patch `<schema_id>.custom.yaml`, not `<schema_id>.schema.yaml`** — unless the user explicitly maintains their own schema files in git and asks you to edit the schema directly.
2. **Back up before modifying**: Before editing any existing `*.custom.yaml`, save `<file>.bak.<YYYYMMDD-HHMMSS>`.
3. **Preserve existing keys under `patch:`**: Read the entire existing `*.custom.yaml` first and merge new keys into the single top-level `patch:` mapping. Never add a second `patch:` block (YAML silently overwrites the first one).
4. **Filter ordering is critical**: `lua_filter@*rime_neural` MUST run **before** any `simplifier` / OpenCC filter (e.g. `s2t`, `s2twp`, `t2s`) and before `uniquifier`. Placing it at position `0` (`engine/filters/@before 0`) is safe for almost all schemas (`rime-ice`, `rime-frost`, `rime-wanxiang`, standard double pinyin).
   - If the user's `*.custom.yaml` ALREADY replaces the entire `engine/filters` list, insert `lua_filter@*rime_neural` as the first entry of that list instead of using `engine/filters/@before 0`.
   - Never mount `lua_filter@*rime_neural` twice.
5. **Switches merging rule**:
   - If the user's `*.custom.yaml` does NOT already define `switches:`, use `switches/+:` to append `neural_rerank`.
   - If the user's `*.custom.yaml` ALREADY defines a full `switches:` list, append the `neural_rerank` entry to that list instead of adding `switches/+:`.
6. **Never touch**: `user.yaml`, `*.userdb/`, `custom_phrase.txt`, `weasel.custom.yaml`, or synced dictionaries (`cn_dicts/`, `cn_dicts_cell/`, `en_dicts/`).

---

## Standard Minimal Patch

Works for `rime_ice`, `rime_frost`, `double_pinyin`, `double_pinyin_flypy`, `double_pinyin_mspy`, `double_pinyin_sogou`, `double_pinyin_abc`, `double_pinyin_ziguang`, `double_pinyin_pyjj`:

```yaml
patch:
  # Expose Top-8 sentence beam candidates so the reranker has enough candidates to choose from
  translator/max_sentences: 8
  translator/sentence_cutoff_threshold: 0.3

  # Mount FIRST (before simplifier / OpenCC)
  engine/filters/@before 0: lua_filter@*rime_neural

  # Optional F4 toggle switch (defaults to ON)
  switches/+:
    - name: neural_rerank
      states: [ 统计, 神经 ]
      reset: 1

  rime_neural:
    option_name: neural_rerank
    pinyin_layout: auto
    max_candidates: 8
    fusion_weights: [1.0, -1.4, 0.05, 0.15, 0.0]
    p2c_verify_margin: 0.20
    p2c_top1_logit_diff: 2.0
    p2c_top1_min_gram_diff: -2.0
    p2c_top2_logit_diff: 0.5
    enable_uia_context: true
    threads: 0
    log_feedback: true
    sync_init: false
```

---

## `pinyin_layout` Mapping Table

| Schema / Layout | `pinyin_layout` value | Notes |
| :--- | :--- | :--- |
| Full Pinyin (`rime_ice`, `rime_frost`, `luna_pinyin*`) | `auto` (resolves to `full`) | Safe to keep `auto` |
| Natural / 自然码 (`double_pinyin`, `rime_frost_double_pinyin`) | `auto` (resolves to `zrm`) | Safe to keep `auto` |
| Xiaohe / 小鹤 (`double_pinyin_flypy`) | `auto` (resolves to `flypy`) | Safe to keep `auto` |
| Microsoft / 微软 (`double_pinyin_mspy`) | `auto` (resolves to `mspy`) | Supports `; = ing` automatically |
| Sogou / 搜狗 (`double_pinyin_sogou`) | `auto` (resolves to `sogou`) | Supports `; = ing` automatically |
| Intelligent ABC / 智能 ABC (`double_pinyin_abc`) | `auto` (resolves to `abc`) | Safe to keep `auto` |
| Ziguang / 紫光 (`double_pinyin_ziguang`) | `auto` (resolves to `ziguang`) | Supports `; = ing` automatically |
| Pinyin Jiajia / 拼音加加 (`double_pinyin_pyjj`) | `auto` (resolves to `jiajia`) | Alias `pyjj` also accepted |
| **`rime-wanxiang` (`wanxiang` / `wanxiang_pro`)** | **Set explicitly!** | `wanxiang` keeps `schema_id: wanxiang` even when double pinyin is enabled via `speller/algebra`. Inspect `wanxiang.custom.yaml` / `wanxiang.schema.yaml` and set `full`, `zrm`, `flypy`, `mspy`, `sogou`, `abc`, `ziguang`, or `jiajia` explicitly! |
| Custom `schema_id` (e.g. `my_flypy`, `zrm_custom`) | **Set explicitly!** | Set explicitly if the `schema_id` does not contain a standard suffix |

---

## Optional: Pairing with Wanxiang `.gram` (`wanxiang-lts-zh-hans.gram`)

`rime-neural` works standalone without any `.gram` file, **but pairing it with `wanxiang-lts-zh-hans.gram` produces significantly higher accuracy** (Rime generates a much stronger Top-8 candidate beam, and `rime-neural` memory-maps the `.gram` read-only with `0 MB` extra RAM to fuse `.gram` features + Pass-2 P2C verification).

If the user opted to install `wanxiang-lts-zh-hans.gram` into `<RimeUserDir>\wanxiang-lts-zh-hans.gram` AND their schema does not already configure `grammar/language`:

```yaml
  # Optional: enable Wanxiang statistical grammar alongside rime-neural
  grammar/language: wanxiang-lts-zh-hans
  grammar/collocation_max_length: 5
  grammar/collocation_min_length: 2
  grammar/collocation_penalty: -14
  grammar/non_collocation_penalty: -4
  translator/contextual_suggestions: true
  translator/max_homographs: 2
```

> Note: `rime-wanxiang` schemas already configure `grammar:` upstream — do NOT overwrite `rime-wanxiang`'s existing `grammar:` block unless the user asks.

---

## Optional: Lightweight Model (`zhen-feather`)

Each ONNX model ships with its own paired `tokenizer.json` and `unified_vocab.json`, so keep only one model active in `<RimeUserDir>\neural\`. Running `scripts\install.ps1 -Prebuilt -Model feather` (or `-Model pumpkin`) automatically installs the matching vocab/tokenizer and removes conflicting `.onnx` files; `model_file` can stay omitted for auto-detection.

To point a schema at a specific model filename inside an existing paired directory, set the single dotted key under `patch:` (never add a second `rime_neural:` block):

```yaml
  rime_neural/model_file: laya.one.w4.onnx
```
