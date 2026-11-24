mod engine;
mod feedback;
mod gram;
mod lua_api;
mod pinyin;
mod uia;

use std::ffi::c_int;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use engine::{FusionWeights, GramRequest, InitConfig, P2cConfig};
use lua_api::{
    LUA_TBOOLEAN, LUA_TSTRING, LUA_TTABLE, LuaApi, LuaInteger, LuaState, check_lua54_abi, get_api,
    pop, push_str, rawset_field, read_str,
};
use pinyin::PinyinLayout;
use uia::UiaContext;

const MAX_LUA_ARRAY_LEN: usize = 256;

unsafe fn read_str_array(api: &LuaApi, l: LuaState, idx: c_int) -> Vec<String> {
    // SAFETY: Caller passes a valid `lua_State*`. Checks `LUA_TTABLE` before `lua_rawlen`/`lua_rawgeti`.
    unsafe {
        if (api.lua_type)(l, idx) != LUA_TTABLE {
            return Vec::new();
        }
        let n = usize::try_from((api.rawlen)(l, idx))
            .unwrap_or(0)
            .min(MAX_LUA_ARRAY_LEN);
        let mut out = Vec::with_capacity(n);
        for i in 1..=(n as LuaInteger) {
            (api.rawgeti)(l, idx, i);
            // Preserve 1:1 positional indexing with Lua's candidate table even if an entry is non-string.
            out.push(read_str(api, l, -1).unwrap_or_default());
            pop(api, l, 1);
        }
        out
    }
}

unsafe fn read_f32_array(api: &LuaApi, l: LuaState, idx: c_int) -> Vec<f32> {
    // SAFETY: Caller passes a valid `lua_State*`. Checks `LUA_TTABLE` before `lua_rawlen`/`lua_rawgeti`.
    unsafe {
        if (api.lua_type)(l, idx) != LUA_TTABLE {
            return Vec::new();
        }
        let n = usize::try_from((api.rawlen)(l, idx))
            .unwrap_or(0)
            .min(MAX_LUA_ARRAY_LEN);
        let mut out = Vec::with_capacity(n);
        for i in 1..=(n as LuaInteger) {
            (api.rawgeti)(l, idx, i);
            let mut isnum: c_int = 0;
            let v = (api.tonumberx)(l, -1, &mut isnum);
            out.push(if isnum != 0 && v.is_finite() {
                v as f32
            } else {
                0.0
            });
            pop(api, l, 1);
        }
        out
    }
}

unsafe fn push_f32_array(api: &LuaApi, l: LuaState, vals: &[f32]) {
    // SAFETY: Caller passes a valid `lua_State*` with stack headroom.
    unsafe {
        (api.createtable)(l, vals.len() as c_int, 0);
        for (i, &v) in vals.iter().enumerate() {
            (api.pushnumber)(l, if v.is_finite() { v as f64 } else { 0.0 });
            (api.rawseti)(l, -2, (i + 1) as LuaInteger);
        }
    }
}

unsafe fn read_weights(api: &LuaApi, l: LuaState, idx: c_int) -> FusionWeights {
    let mut w = FusionWeights::default();
    // SAFETY: Caller passes a valid `lua_State*`.
    unsafe {
        if (api.lua_type)(l, idx) == LUA_TTABLE {
            let vals = read_f32_array(api, l, idx);
            if !vals.is_empty() && vals[0].is_finite() {
                w.w_logit = vals[0] as f64;
            }
            if vals.len() >= 2 && vals[1].is_finite() {
                w.w_rank = vals[1] as f64;
            }
            if vals.len() >= 3 && vals[2].is_finite() {
                w.w_gram = vals[2] as f64;
            }
            if vals.len() >= 4 && vals[3].is_finite() {
                w.w_char_lp = vals[3] as f64;
            }
            if vals.len() >= 5 && vals[4].is_finite() {
                w.w_ctx_lp = vals[4] as f64;
            }
            return w;
        }

        let mut isnum: c_int = 0;
        let rp = (api.tonumberx)(l, idx, &mut isnum);
        if isnum != 0 && rp.is_finite() {
            w.w_logit = 1.0;
            w.w_rank = -rp.abs();
            w.w_gram = 0.0;
            w.w_char_lp = 0.0;
            w.w_ctx_lp = 0.0;
        }
    }
    w
}

unsafe fn read_table_f32(
    api: &LuaApi,
    l: LuaState,
    idx: c_int,
    field: &str,
    default_val: f32,
) -> f32 {
    let tbl_idx = if idx < 0 { idx - 1 } else { idx };
    // SAFETY: Slot `idx` was checked to be `LUA_TTABLE`; `lua_rawget` avoids metamethod dispatch.
    unsafe {
        push_str(api, l, field);
        (api.rawget)(l, tbl_idx);
        let mut isnum: c_int = 0;
        let v = (api.tonumberx)(l, -1, &mut isnum);
        pop(api, l, 1);
        if isnum != 0 && v.is_finite() {
            v as f32
        } else {
            default_val
        }
    }
}

unsafe fn read_p2c_cfg(api: &LuaApi, l: LuaState, idx: c_int) -> P2cConfig {
    let mut cfg = P2cConfig::default();
    // SAFETY: Caller passes a valid `lua_State*`.
    unsafe {
        if (api.lua_type)(l, idx) == LUA_TTABLE {
            cfg.verify_margin = read_table_f32(api, l, idx, "verify_margin", cfg.verify_margin);
            cfg.top1_logit_diff =
                read_table_f32(api, l, idx, "top1_logit_diff", cfg.top1_logit_diff);
            cfg.top1_min_gram_diff =
                read_table_f32(api, l, idx, "top1_min_gram_diff", cfg.top1_min_gram_diff);
            cfg.top2_logit_diff =
                read_table_f32(api, l, idx, "top2_logit_diff", cfg.top2_logit_diff);
            cfg.legacy_top1_margin =
                read_table_f32(api, l, idx, "legacy_top1_margin", cfg.legacy_top1_margin);
            cfg.legacy_top2_margin =
                read_table_f32(api, l, idx, "legacy_top2_margin", cfg.legacy_top2_margin);
            return cfg;
        }
        let mut isnum: c_int = 0;
        let t1 = (api.tonumberx)(l, idx, &mut isnum);
        if isnum != 0 && t1.is_finite() {
            cfg.legacy_top1_margin = t1 as f32;
        }
        isnum = 0;
        let t2 = (api.tonumberx)(l, idx + 1, &mut isnum);
        if isnum != 0 && t2.is_finite() {
            cfg.legacy_top2_margin = t2 as f32;
        }
    }
    cfg
}

unsafe fn read_layout(api: &LuaApi, l: LuaState, idx: c_int) -> PinyinLayout {
    // SAFETY: Caller passes a valid `lua_State*`.
    unsafe {
        if (api.lua_type)(l, idx) == LUA_TSTRING
            && let Some(s) = read_str(api, l, idx)
        {
            return PinyinLayout::from_name_or_schema(&s, "");
        }
        if (api.toboolean)(l, idx) != 0 {
            PinyinLayout::Zrm
        } else {
            PinyinLayout::Full
        }
    }
}

unsafe fn read_init_config(api: &LuaApi, l: LuaState) -> Result<InitConfig, String> {
    // SAFETY: Caller passes a valid `lua_State*`.
    unsafe {
        let model_dir = read_str(api, l, 1).ok_or("missing arg 1: model_dir")?;
        let ort_dll = read_str(api, l, 2).ok_or("missing arg 2: ort_dll")?;
        let mut isnum: c_int = 0;
        let threads_raw = (api.tointegerx)(l, 3, &mut isnum);
        let threads = if isnum != 0 && threads_raw >= 0 {
            usize::try_from(threads_raw).unwrap_or(16)
        } else {
            0 // auto-detect
        };
        let gram_path = read_str(api, l, 4)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        isnum = 0;
        let col_raw = (api.tonumberx)(l, 5, &mut isnum);
        let gram_collocation_penalty = if isnum != 0 && col_raw.is_finite() {
            col_raw
        } else {
            -15.5
        };
        isnum = 0;
        let non_col_raw = (api.tonumberx)(l, 6, &mut isnum);
        let gram_non_collocation_penalty = if isnum != 0 && non_col_raw.is_finite() {
            non_col_raw
        } else {
            -4.0
        };
        let model_file = read_str(api, l, 7).filter(|s| !s.is_empty());

        Ok(InitConfig {
            model_dir: PathBuf::from(model_dir),
            ort_dll: PathBuf::from(ort_dll),
            gram_path,
            gram_collocation_penalty,
            gram_non_collocation_penalty,
            gram_max_length: gram::DEFAULT_COLLOCATION_MAX_LENGTH,
            model_file,
            threads,
        })
    }
}

/// init(model_dir, ort_dll, threads, [gram_path], [col_penalty], [non_col_penalty], [model_file])
unsafe extern "C" fn lua_init(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        // SAFETY: `l` is valid for the duration of this Lua C call.
        let cfg = unsafe { read_init_config(api, l) }?;
        engine::init_engine(&cfg)
    }));
    // SAFETY: Push at most 2 values to Lua stack; check stack headroom first.
    unsafe {
        if (api.checkstack)(l, 4) == 0 {
            return 0;
        }
        match res {
            Ok(Ok(())) => {
                (api.pushboolean)(l, 1);
                (api.pushnil)(l);
                2
            }
            Ok(Err(e)) => {
                (api.pushboolean)(l, 0);
                push_str(api, l, &e);
                2
            }
            Err(_) => {
                (api.pushboolean)(l, 0);
                push_str(api, l, "panic in rime_neural.init");
                2
            }
        }
    }
}

/// init_async(model_dir, ort_dll, threads, [gram_path], [col_penalty], [non_col_penalty], [model_file])
unsafe extern "C" fn lua_init_async(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        // SAFETY: `l` is valid for this C call.
        let cfg = unsafe { read_init_config(api, l) }?;
        engine::init_engine_async(cfg);
        Ok(())
    }));
    // SAFETY: Push 1 boolean return value onto `l`.
    unsafe {
        (api.pushboolean)(l, if matches!(res, Ok(Ok(()))) { 1 } else { 0 });
    }
    1
}

unsafe extern "C" fn lua_is_ready(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let ready = catch_unwind(AssertUnwindSafe(engine::is_ready)).unwrap_or(false);
    // SAFETY: Push 1 boolean return value onto `l`.
    unsafe {
        (api.pushboolean)(l, if ready { 1 } else { 0 });
    }
    1
}

/// resolve_layout(layout_opt, schema_id) -> (layout_name, is_dp, uses_semicolon)
unsafe extern "C" fn lua_resolve_layout(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let layout = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `l` is valid for this C call.
        let (opt, schema) = unsafe {
            (
                read_str(api, l, 1).unwrap_or_default(),
                read_str(api, l, 2).unwrap_or_default(),
            )
        };
        PinyinLayout::from_name_or_schema(&opt, &schema)
    }))
    .unwrap_or(PinyinLayout::Full);
    // SAFETY: Push 3 return values onto `l`.
    unsafe {
        if (api.checkstack)(l, 4) == 0 {
            return 0;
        }
        push_str(api, l, layout.as_str());
        (api.pushboolean)(l, if layout.is_double_pinyin() { 1 } else { 0 });
        (api.pushboolean)(l, if layout.uses_semicolon() { 1 } else { 0 });
    }
    3
}

/// set_uia_enabled(bool)
unsafe extern "C" fn lua_set_uia_enabled(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let ok = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `l` is valid for this C call.
        let enable = unsafe { (api.toboolean)(l, 1) != 0 };
        uia::set_enabled(enable);
    }))
    .is_ok();
    // SAFETY: Push 1 boolean onto `l`.
    unsafe {
        (api.pushboolean)(l, if ok { 1 } else { 0 });
    }
    1
}

/// query_uia_context(live_preedit, raw_input, [cand_preview])
///   -> (status_str, text_str, hwnd_int, preedit_stripped_bool)
unsafe extern "C" fn lua_query_uia_context(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `l` is valid for this C call.
        let (preedit, raw_input, cand_preview) = unsafe {
            (
                read_str(api, l, 1).unwrap_or_default(),
                read_str(api, l, 2).unwrap_or_default(),
                read_str(api, l, 3).unwrap_or_default(),
            )
        };
        uia::query_context(&preedit, &raw_input, &cand_preview)
    }));
    // SAFETY: Push 4 return values onto `l` after ensuring stack capacity.
    unsafe {
        if (api.checkstack)(l, 6) == 0 {
            return 0;
        }
        match res {
            Ok((UiaContext::Blocked, hwnd)) => {
                push_str(api, l, "blocked");
                push_str(api, l, "");
                (api.pushinteger)(l, hwnd as LuaInteger);
                (api.pushboolean)(l, 0);
                4
            }
            Ok((
                UiaContext::Text {
                    text,
                    preedit_stripped,
                },
                hwnd,
            )) => {
                push_str(api, l, "text");
                push_str(api, l, &text);
                (api.pushinteger)(l, hwnd as LuaInteger);
                (api.pushboolean)(l, if preedit_stripped { 1 } else { 0 });
                4
            }
            Ok((UiaContext::WindowChanged, hwnd)) => {
                push_str(api, l, "window_changed");
                push_str(api, l, "");
                (api.pushinteger)(l, hwnd as LuaInteger);
                (api.pushboolean)(l, 0);
                4
            }
            Ok((UiaContext::FallbackSameWindow, hwnd)) => {
                push_str(api, l, "fallback");
                push_str(api, l, "");
                (api.pushinteger)(l, hwnd as LuaInteger);
                (api.pushboolean)(l, 0);
                4
            }
            Err(_) => {
                push_str(api, l, "fallback");
                push_str(api, l, "");
                (api.pushinteger)(l, uia::foreground_hwnd() as LuaInteger);
                (api.pushboolean)(l, 0);
                4
            }
        }
    }
}

/// kick_uia_refresh() -> nil (non-blocking background refresh hint)
unsafe extern "C" fn lua_kick_uia_refresh(_l: LuaState) -> c_int {
    let _ = catch_unwind(AssertUnwindSafe(uia::kick_refresh));
    0
}

/// kick_uia_after_commit() -> nil (post-commit background refresh after brief settle delay)
unsafe extern "C" fn lua_kick_uia_after_commit(_l: LuaState) -> c_int {
    let _ = catch_unwind(AssertUnwindSafe(uia::kick_after_commit));
    0
}

/// rerank(context, raw_input, layout_str_or_bool, cands_tbl, weights_tbl, [p2c_cfg_tbl])
unsafe extern "C" fn lua_rerank(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(
        || -> Result<engine::RerankResult, String> {
            if !engine::is_ready() {
                return Err("not ready".to_string());
            }
            // SAFETY: `l` is valid for this C call.
            let (ctx, raw_input, layout, cands, weights, p2c_cfg) = unsafe {
                (
                    read_str(api, l, 1).unwrap_or_default(),
                    read_str(api, l, 2).unwrap_or_default(),
                    read_layout(api, l, 3),
                    read_str_array(api, l, 4),
                    read_weights(api, l, 5),
                    read_p2c_cfg(api, l, 6),
                )
            };

            engine::rerank(
                &ctx,
                &raw_input,
                layout,
                &cands,
                weights,
                p2c_cfg,
                &GramRequest::Default,
            )
        },
    ));

    // SAFETY: Build return table using `rawset_field` (no metamethod dispatch) with checked stack.
    unsafe {
        if (api.checkstack)(l, 8) == 0 {
            return 0;
        }
        match res {
            Ok(Ok(rr)) => {
                (api.createtable)(l, 0, 6);

                let order = rr.order;
                rawset_field(api, l, "order", |api, l| {
                    (api.createtable)(l, order.len() as c_int, 0);
                    for (i, idx) in order.iter().enumerate() {
                        (api.pushinteger)(l, *idx as LuaInteger);
                        (api.rawseti)(l, -2, (i + 1) as LuaInteger);
                    }
                });
                drop(order);

                let p2c_text = rr.p2c_text;
                rawset_field(api, l, "p2c_text", |api, l| {
                    if let Some(ref txt) = p2c_text {
                        push_str(api, l, txt);
                    } else {
                        (api.pushnil)(l);
                    }
                });
                drop(p2c_text);

                let p2c_pos = rr.p2c_pos;
                rawset_field(api, l, "p2c_pos", |api, l| {
                    (api.pushinteger)(l, p2c_pos as LuaInteger);
                });

                let logits = rr.logits;
                rawset_field(api, l, "logits", |api, l| push_f32_array(api, l, &logits));
                drop(logits);

                let gram_deltas = rr.gram_deltas;
                rawset_field(api, l, "gram_deltas", |api, l| {
                    push_f32_array(api, l, &gram_deltas);
                });
                drop(gram_deltas);

                let char_lps = rr.char_lps;
                rawset_field(api, l, "char_lps", |api, l| {
                    push_f32_array(api, l, &char_lps)
                });
                drop(char_lps);

                1
            }
            _ => {
                (api.pushnil)(l);
                1
            }
        }
    }
}

/// Zero-plaintext feedback logger.
/// Supports both:
/// - Numeric v2 signature (11 args):
///   `log_feedback(schema, has_ctx_bool, logits_tbl, gram_tbl, char_tbl, shown_top1_idx, chosen_idx, p2c_shown, p2c_chosen, chosen_rank, [log_path])`
/// - Legacy 13-arg signature (for older Lua scripts): converts to zero-plaintext v2 in memory before writing.
unsafe extern "C" fn lua_log_feedback(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let (model_name, has_gram) = engine::active_feedback_meta();
        // SAFETY: `l` is valid for this C call.
        unsafe {
            let nargs = (api.gettop)(l);
            let schema = read_str(api, l, 1).unwrap_or_default();
            let is_v2 = (api.lua_type)(l, 2) == LUA_TBOOLEAN
                || (nargs <= 11 && (api.lua_type)(l, 3) == LUA_TTABLE);
            if is_v2 {
                let has_ctx = (api.toboolean)(l, 2) != 0;
                let logits = read_f32_array(api, l, 3);
                let gram_deltas = read_f32_array(api, l, 4);
                let char_lps = read_f32_array(api, l, 5);

                let mut isnum: c_int = 0;
                let shown_raw = (api.tointegerx)(l, 6, &mut isnum);
                let shown_top1_idx = if isnum != 0 {
                    i32::try_from(shown_raw).unwrap_or(-1)
                } else {
                    -1
                };
                isnum = 0;
                let chosen_raw = (api.tointegerx)(l, 7, &mut isnum);
                let chosen_idx = if isnum != 0 {
                    i32::try_from(chosen_raw).unwrap_or(-1)
                } else {
                    -1
                };
                let p2c_shown = (api.toboolean)(l, 8) != 0;
                let p2c_chosen = (api.toboolean)(l, 9) != 0;
                isnum = 0;
                let rank_raw = (api.tointegerx)(l, 10, &mut isnum);
                let chosen_rank = if isnum != 0 {
                    i32::try_from(rank_raw).unwrap_or(0)
                } else {
                    0
                };
                let log_path = match read_str(api, l, 11).filter(|s| !s.is_empty()) {
                    Some(p) => PathBuf::from(p),
                    None => engine::model_dir_path()
                        .ok_or("engine not ready")?
                        .join("feedback.jsonl"),
                };
                // Reverted = neural promoted another candidate (or P2C `-2`) to #1, and user chose Rime #1 (`0`).
                let reverted = (shown_top1_idx > 0 || shown_top1_idx == -2) && chosen_idx == 0;
                let rec = feedback::FeedbackRecord::new_numeric(
                    &schema,
                    &model_name,
                    has_gram,
                    has_ctx,
                    &logits,
                    &gram_deltas,
                    &char_lps,
                    shown_top1_idx,
                    chosen_idx,
                    p2c_shown,
                    p2c_chosen,
                    chosen_rank,
                    reverted,
                );
                return feedback::append_feedback(&log_path, &rec);
            }

            // Legacy 13-arg compatibility: derive purely numeric indices in memory; never persist raw text
            let log_path = match read_str(api, l, 13).filter(|s| !s.is_empty()) {
                Some(p) => PathBuf::from(p),
                None => engine::model_dir_path()
                    .ok_or("engine not ready")?
                    .join("feedback.jsonl"),
            };
            let ctx = read_str(api, l, 3).unwrap_or_default();
            let cands = read_str_array(api, l, 4);
            let logits = read_f32_array(api, l, 5);
            let gram_deltas = read_f32_array(api, l, 6);
            let char_lps = read_f32_array(api, l, 7);
            let p2c_cand = read_str(api, l, 8).filter(|s| !s.is_empty());
            let shown_top1 = read_str(api, l, 9).unwrap_or_default();
            let chosen_text = read_str(api, l, 10).unwrap_or_default();
            let mut isnum: c_int = 0;
            let rank_raw = (api.tointegerx)(l, 11, &mut isnum);
            let chosen_rank = if isnum != 0 {
                i32::try_from(rank_raw).unwrap_or(0)
            } else {
                0
            };

            let has_ctx = ctx.trim().chars().count() >= 2;
            let p2c_shown = p2c_cand.is_some();
            let chosen_idx = cands
                .iter()
                .position(|c| c == &chosen_text)
                .map(|i| i as i32)
                .unwrap_or(-1);
            let p2c_chosen = p2c_cand
                .as_deref()
                .is_some_and(|p| p == chosen_text && chosen_idx < 0);
            let shown_top1_idx = if p2c_cand.as_deref().is_some_and(|p| p == shown_top1)
                && !cands.iter().any(|c| c == &shown_top1)
            {
                -2
            } else {
                cands
                    .iter()
                    .position(|c| c == &shown_top1)
                    .map(|i| i as i32)
                    .unwrap_or(-1)
            };
            let reverted = (shown_top1_idx > 0 || shown_top1_idx == -2) && chosen_idx == 0;

            let rec = feedback::FeedbackRecord::new_numeric(
                &schema,
                &model_name,
                has_gram,
                has_ctx,
                &logits,
                &gram_deltas,
                &char_lps,
                shown_top1_idx,
                chosen_idx,
                p2c_shown,
                p2c_chosen,
                chosen_rank,
                reverted,
            );
            feedback::append_feedback(&log_path, &rec)
        }
    }));
    // SAFETY: Push 1 boolean onto `l`.
    unsafe {
        (api.pushboolean)(l, if matches!(res, Ok(Ok(()))) { 1 } else { 0 });
    }
    1
}

unsafe extern "C" fn lua_stats(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| {
        let st = engine::get_stats();
        let uia_on = uia::is_enabled();
        let uia_alive = uia::is_worker_alive();
        (st, uia_on, uia_alive)
    }));
    // SAFETY: Populate stats table via `rawset_field` after checking stack capacity.
    unsafe {
        if (api.checkstack)(l, 8) == 0 {
            return 0;
        }
        let Ok((st, uia_on, uia_alive)) = res else {
            (api.pushnil)(l);
            return 1;
        };
        (api.createtable)(l, 0, 14);
        rawset_field(api, l, "ready", |api, l| {
            (api.pushboolean)(l, if st.ready { 1 } else { 0 })
        });
        rawset_field(api, l, "calls", |api, l| {
            (api.pushinteger)(l, st.calls as LuaInteger)
        });
        rawset_field(api, l, "cache_hits", |api, l| {
            (api.pushinteger)(l, st.cache_hits as LuaInteger)
        });
        rawset_field(api, l, "kv_hits", |api, l| {
            (api.pushinteger)(l, st.kv_hits as LuaInteger)
        });
        rawset_field(api, l, "p2c_pass2_calls", |api, l| {
            (api.pushinteger)(l, st.p2c_pass2_calls as LuaInteger)
        });
        rawset_field(api, l, "avg_ms", |api, l| (api.pushnumber)(l, st.avg_ms));
        rawset_field(api, l, "fused_kv", |api, l| {
            (api.pushboolean)(l, if st.fused_kv { 1 } else { 0 })
        });
        rawset_field(api, l, "has_gram", |api, l| {
            (api.pushboolean)(l, if st.has_gram { 1 } else { 0 })
        });
        rawset_field(api, l, "uia_enabled", |api, l| {
            (api.pushboolean)(l, if uia_on { 1 } else { 0 })
        });
        rawset_field(api, l, "uia_worker_alive", |api, l| {
            (api.pushboolean)(l, if uia_alive { 1 } else { 0 })
        });
        rawset_field(api, l, "enc_layers", |api, l| {
            (api.pushinteger)(l, st.enc_layers as LuaInteger)
        });
        rawset_field(api, l, "choice_compact", |api, l| {
            (api.pushboolean)(l, if st.choice_compact { 1 } else { 0 })
        });
        let model_name = st.model_name;
        rawset_field(api, l, "model_name", |api, l| push_str(api, l, &model_name));
        drop(model_name);
        let last_error = st.last_error;
        rawset_field(api, l, "last_error", |api, l| {
            if let Some(ref err) = last_error {
                push_str(api, l, err);
            } else {
                (api.pushnil)(l);
            }
        });
        drop(last_error);
        1
    }
}

/// Lua 5.4 C module entry point invoked via `package.loadlib(..., "luaopen_rime_neural")`.
///
/// # Safety
/// Must only be called by the host Lua 5.4 VM (`librime-lua` / `rime.dll`) with a valid `lua_State*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_rime_neural(l: LuaState) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Pin module before spawning any threads or returning exported closures; refuse
        // to load if pinning fails so `lua_close()` can never unload active ORT/UIA threads.
        let pinned =
            unsafe { lua_api::pin_self_module(luaopen_rime_neural as *const std::ffi::c_void) };
        if !pinned {
            return 0;
        }
        // SAFETY: Resolve Lua 5.4 C API from host `rime.dll` and verify ABI compatibility.
        let Ok(api) = (unsafe { get_api() }) else {
            return 0;
        };
        // SAFETY: `l` is a valid `lua_State*` passed by `package.loadlib`.
        unsafe {
            if !check_lua54_abi(api, l) || (api.checkstack)(l, 6) == 0 {
                return 0;
            }
            (api.createtable)(l, 0, 11);

            macro_rules! export_fn {
                ($name:literal, $func:ident) => {
                    rawset_field(api, l, $name, |api, l| {
                        (api.pushcclosure)(l, $func, 0);
                    });
                };
            }

            export_fn!("init", lua_init);
            export_fn!("init_async", lua_init_async);
            export_fn!("is_ready", lua_is_ready);
            export_fn!("resolve_layout", lua_resolve_layout);
            export_fn!("set_uia_enabled", lua_set_uia_enabled);
            export_fn!("query_uia_context", lua_query_uia_context);
            export_fn!("kick_uia_refresh", lua_kick_uia_refresh);
            export_fn!("kick_uia_after_commit", lua_kick_uia_after_commit);
            export_fn!("rerank", lua_rerank);
            export_fn!("log_feedback", lua_log_feedback);
            export_fn!("stats", lua_stats);
        }
        1
    }))
    .unwrap_or(0)
}
