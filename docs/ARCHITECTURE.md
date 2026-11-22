# Architecture & Design Notes (`rime-neural` v1.0.0)

## 1. Zero-Patch In-Process Execution Model

Official Weasel (`librime >= 1.17.0`, Lua 5.4.8) exports the Lua 5.4 C API directly from `rime.dll` and permits `package.loadlib`.

- **`src/lua_api.rs`**: Dynamically resolves the 18 required `lua_*` symbols from `GetModuleHandleA("rime.dll")` with zero import `.lib` dependency. Pins `rime_neural.dll` in the host process via `GetModuleHandleExA(PIN | FROM_ADDRESS)` so Rime redeploys (`lua_close` -> `FreeLibrary`) never unload active ONNX Runtime worker pools or the UIA STA thread, and subsequent redeploys reuse the warm singleton in `0 ms`.
- **`lua/rime_neural.lua`**: Calls `init_async(...)` so `WeaselServer.exe` startup incurs `0 ms` delay while ONNX Runtime loads in a background thread. Before readiness (or if DLL/model is absent, or `neural_rerank` is toggled off), candidates pass through untouched.

## 2. Inference Pipeline (`src/engine.rs`)

1. **Strict `sentence`-Only Gating**: Only contiguous leading candidates with `cand:get_genuine().type == "sentence"` and matching `(start, _end)` are reranked. Exact dictionary phrases (`phrase`), userdb phrases (`user_phrase`), English words, dates, numbers, and punctuation are never touched.
2. **Dynamic Model Architecture Introspection**:
   - Inspects `past_enc_k` and `past_choice_k` shapes directly from `session.inputs()` (`[enc_layers, 1, heads, p, head_dim]` and `[choice_layers, 1, heads, p, head_dim]`), and checks `choice_compact` in `unified_vocab.json` (switching between `[MASK] + encode(" " + c)` for 12L `ML-pumpkin-1103` and no-prepend compact markers for 8L `zhen-feather`).
   - Pre-builds 3 selective `ort::RunOptions` via `OutputSelector::no_default()`:
     - Context KV pass: fetches only `present_enc_k/v` + `present_choice_k/v`
     - Pass-1 keystroke: fetches only `logits` + `char_logits` (avoiding ~3 MB/keystroke unused KV output copies)
     - Pass-2 P2C verification: fetches only `logits`
3. **16-Slot Prefix KV-Cache + 256-Entry Evaluation LRU**: True LRU promotion on both caches, with empty context pinned in the KV cache.
4. **Zero-Extra-RAM `.gram` Fusion (`src/gram.rs`)**: Memory-maps `Rime::Grammar/1.0` (e.g., `wanxiang-lts-zh-hans.gram`) read-only via Win32 `CreateFileMappingW`, sharing the OS page cache with `rime.dll` (`0 MB` extra physical RAM, `~0.01 ms`/sentence, zero per-char heap allocations). Reads `grammar/collocation_penalty` and `grammar/non_collocation_penalty` directly from the active schema.
5. **5-Feature Cross-Domain Linear Fusion**:
   $$s_i = w_0 \cdot \log\operatorname{softmax}(\text{logit})_i + w_1 \cdot \ln(i + 1) + w_2 \cdot \Delta_{\text{gram}, i} + w_3 \cdot \Delta_{\text{char\_lp}, i} + w_4 \cdot \mathbb{1}_{\text{ctx}} \cdot \log\operatorname{softmax}(\text{logit})_i$$
   Default OOD-audited prior: `[1.0, -1.4, 0.05, 0.15, 0.0]` (`w_ctx_lp = 0.0` explicitly disables the interaction term that showed slight domain overfitting on unseen prose/news).
6. **KV-Cached Pass-2 Verified P2C Synthesis**: Decodes `char_logits` from Pass-1 (`0 ms` extra forward, zero `HashMap` allocations via sorted `syl_to_cids` binary search). Only when the synthesized string is novel (not in Rime's top-K) and beats Rime's best `char_logits` margin by `verify_margin` (~7% of sentences), runs a lightweight 3-candidate Pass-2 check reusing the warm prefix KV-cache and `.gram` floor (`top1_logit_diff > 2.0` and `gram_delta > -2.0` for #1; `top2_logit_diff > 0.5` for #2).

## 3. Universal Pinyin Recovery (`src/pinyin.rs`)

- **7 Double Pinyin layouts**: Natural (ZRM), Flypy (小鹤), MSPY (微软, `; = ing`), Sogou (搜狗, `; = ing`), ABC (智能ABC), Ziguang (紫光, `; = ing`), Jiajia/PYJJ (拼音加加). Exhaustively verified against all 3,008 derived 2-key codes in official `rime-ice` / `rime-frost` schemas (100% match, including self-leading and `o`-prefixed zero-initial aliases).
- **Full Pinyin**: Memoized `O(N * L)` alignment against the leading candidate's readings (no exponential backtracking worst case), respecting explicit `'` and space syllable separators, with safe zeroing of `char_lps` when candidates have differing character lengths (`先` vs `西安`).

## 4. Non-Blocking, Privacy-Hardened UIA Context (`src/uia.rs`)

- All COM / UIA work runs on one dedicated background STA thread; keystroke hot path never blocks (`< 2 µs`).
- Pre-warmed asynchronously on keystrokes 1–2 so context is already fresh when keystroke 4+ triggers reranking.
- Clamps `IUIAutomationTextPattern` ranges strictly inside `RangeFromChild(&focused)` and breaks on `U+FFFC` (Object Replacement Character) so Chromium/Electron apps never bleed surrounding UI labels/toolbars into IME context.
- Synchronously tracks foreground `HWND` to prevent cross-app `commit_history` bleed after switching windows.
- Hard-blocks password fields (`CurrentIsPassword == true`) and password managers / credential UIs (`KeePass`, `KeePassXC`, `1Password`, `Bitwarden`, `Enpass`, `LastPass`, `Dashlane`, `RoboForm`, `NordPass`, `ProtonPass`, `CredentialUIBroker`, `consent.exe`, `LockApp`, `mstsc.exe`) on both foreground HWND and focused element PID — forcing empty context and suppressing feedback logging.

## 5. Unbiased Feedback & Personal MAP Tuning (`src/feedback.rs` + `scripts/tune-fusion.py`)

- Captures pre-OpenCC genuine Simplified candidate text (so Taiwan Traditional `s2twp` wording changes still match cleanly), redacts digits (>= 4 chars), emails, URLs, and long ASCII tokens to `<REDACTED>`, truncates context to 64 chars, and caps local storage at 5 MiB with automatic rotation.
- Logs 100% of multi-candidate commits (`sample_weight = 1.0`) to eliminate selection bias.
- `scripts/tune-fusion.py` runs exact damped Newton-Raphson MAP shrinkage around `W_PRIOR`, picks regularization `lambda` by 5-fold out-of-fold weighted log-loss (NLL), and gates updates on `N >= 100` plus a one-sided paired 95% CI test (`z >= 1.645`, Kish effective `N_eff`) and non-decreasing Top-1 accuracy.
