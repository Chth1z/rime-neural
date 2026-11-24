mod engine;
mod feedback;
mod gram;
mod lua_api;
mod pinyin;
mod typo;
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
const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

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
            // Non-numbers convert to zero. FeedbackRecord sanitizes non-finite f32 features.
            out.push((api.tonumberx)(l, -1, std::ptr::null_mut()) as f32);
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

/// Read up to 5 fusion weights directly as `f64` via `rawgeti` + `tonumberx`, preserving full
/// `f64` precision and keeping `FusionWeights::default()` on `nil`, `NaN`, or non-numeric entries.
unsafe fn read_weights(api: &LuaApi, l: LuaState, idx: c_int) -> FusionWeights {
    let mut w = FusionWeights::default();
    // SAFETY: Caller passes a valid `lua_State*`.
    unsafe {
        if (api.lua_type)(l, idx) == LUA_TTABLE {
            let fields: [&mut f64; 5] = [
                &mut w.w_logit,
                &mut w.w_rank,
                &mut w.w_gram,
                &mut w.w_char_lp,
                &mut w.w_ctx_lp,
            ];
            for (i, slot) in fields.into_iter().enumerate() {
                (api.rawgeti)(l, idx, (i + 1) as LuaInteger);
                let mut isnum: c_int = 0;
                let v = (api.tonumberx)(l, -1, &mut isnum);
                pop(api, l, 1);
                if isnum != 0 && v.is_finite() {
                    *slot = v;
                }
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

unsafe fn read_table_f64(
    api: &LuaApi,
    l: LuaState,
    idx: c_int,
    field: &str,
    default_val: f64,
) -> f64 {
    let tbl_idx = if idx < 0 { idx - 1 } else { idx };
    // SAFETY: Slot `idx` was checked to be `LUA_TTABLE`; `lua_rawget` avoids metamethod dispatch.
    unsafe {
        push_str(api, l, field);
        (api.rawget)(l, tbl_idx);
        let mut isnum: c_int = 0;
        let v = (api.tonumberx)(l, -1, &mut isnum);
        pop(api, l, 1);
        if isnum != 0 && v.is_finite() {
            v
        } else {
            default_val
        }
    }
}

unsafe fn read_table_f32(
    api: &LuaApi,
    l: LuaState,
    idx: c_int,
    field: &str,
    default_val: f32,
) -> f32 {
    // SAFETY: Delegates to `read_table_f64` with the same preconditions.
    let v = unsafe { read_table_f64(api, l, idx, field, default_val as f64) } as f32;
    if v.is_finite() { v } else { default_val }
}

unsafe fn read_table_path(api: &LuaApi, l: LuaState, idx: c_int, field: &str) -> Option<PathBuf> {
    let tbl_idx = if idx < 0 { idx - 1 } else { idx };
    // SAFETY: Slot `idx` was checked to be `LUA_TTABLE`; `lua_rawget` avoids metamethod dispatch.
    unsafe {
        push_str(api, l, field);
        (api.rawget)(l, tbl_idx);
        let res = read_str(api, l, -1);
        pop(api, l, 1);
        res.filter(|s| !s.is_empty()).map(PathBuf::from)
    }
}

unsafe fn read_table_bool(api: &LuaApi, l: LuaState, idx: c_int, field: &str) -> Option<bool> {
    let tbl_idx = if idx < 0 { idx - 1 } else { idx };
    // SAFETY: Slot `idx` was checked to be `LUA_TTABLE`; `lua_rawget` avoids metamethod dispatch.
    unsafe {
        push_str(api, l, field);
        (api.rawget)(l, tbl_idx);
        let ty = (api.lua_type)(l, -1);
        let res = if ty == LUA_TBOOLEAN {
            Some((api.toboolean)(l, -1) != 0)
        } else {
            None
        };
        pop(api, l, 1);
        res
    }
}

unsafe fn read_table_usize(
    api: &LuaApi,
    l: LuaState,
    idx: c_int,
    field: &str,
    default_val: usize,
) -> usize {
    let tbl_idx = if idx < 0 { idx - 1 } else { idx };
    // SAFETY: Slot `idx` was checked to be `LUA_TTABLE`; `lua_rawget` avoids metamethod dispatch.
    unsafe {
        push_str(api, l, field);
        (api.rawget)(l, tbl_idx);
        let mut isnum: c_int = 0;
        let v = (api.tointegerx)(l, -1, &mut isnum);
        pop(api, l, 1);
        if isnum != 0 && v > 0 {
            usize::try_from(v).unwrap_or(default_val)
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
        let t1 = (api.tonumberx)(l, idx, &mut isnum) as f32;
        if isnum != 0 && t1.is_finite() {
            cfg.legacy_top1_margin = t1;
        }
        isnum = 0;
        let t2 = (api.tonumberx)(l, idx + 1, &mut isnum) as f32;
        if isnum != 0 && t2.is_finite() {
            cfg.legacy_top2_margin = t2;
        }
    }
    cfg
}

/// Resolve a `.gram` path from either a string or a list of candidate paths (`{ user, shared }`),
/// picking the first existing file via UTF-16 `Path::is_file()`.
fn pick_existing_gram_path(candidates: &[String]) -> Option<PathBuf> {
    let mut first_non_empty: Option<PathBuf> = None;
    for s in candidates {
        if s.is_empty() {
            continue;
        }
        let p = PathBuf::from(s);
        if p.is_file() {
            return Some(p);
        }
        if first_non_empty.is_none() {
            first_non_empty = Some(p);
        }
    }
    first_non_empty
}

unsafe fn read_gram_req(api: &LuaApi, l: LuaState, idx: c_int) -> GramRequest {
    // SAFETY: Caller passes a valid `lua_State*`.
    unsafe {
        if (api.lua_type)(l, idx) != LUA_TTABLE {
            return GramRequest::Default;
        }
        if read_table_bool(api, l, idx, "enabled") == Some(false) {
            return GramRequest::Disabled;
        }
        let paths = [
            read_table_path(api, l, idx, "path"),
            read_table_path(api, l, idx, "fallback_path"),
        ];
        let col = read_table_f64(
            api,
            l,
            idx,
            "col_penalty",
            gram::DEFAULT_COLLOCATION_PENALTY,
        );
        let non_col = read_table_f64(
            api,
            l,
            idx,
            "non_col_penalty",
            gram::DEFAULT_NON_COLLOCATION_PENALTY,
        );
        let max_len = read_table_usize(
            api,
            l,
            idx,
            "max_length",
            gram::DEFAULT_COLLOCATION_MAX_LENGTH,
        );
        GramRequest::custom(paths, col, non_col, max_len)
    }
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
        let gram_path = if (api.lua_type)(l, 4) == LUA_TTABLE {
            let candidates = read_str_array(api, l, 4);
            pick_existing_gram_path(&candidates)
        } else {
            read_str(api, l, 4)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
        };
        isnum = 0;
        let col_raw = (api.tonumberx)(l, 5, &mut isnum);
        let gram_collocation_penalty = if isnum != 0 && col_raw.is_finite() {
            col_raw
        } else {
            gram::DEFAULT_COLLOCATION_PENALTY
        };
        isnum = 0;
        let non_col_raw = (api.tonumberx)(l, 6, &mut isnum);
        let gram_non_collocation_penalty = if isnum != 0 && non_col_raw.is_finite() {
            non_col_raw
        } else {
            gram::DEFAULT_NON_COLLOCATION_PENALTY
        };
        let model_file = read_str(api, l, 7).filter(|s| !s.is_empty());
        isnum = 0;
        let max_len_raw = (api.tointegerx)(l, 8, &mut isnum);
        let gram_max_length = if isnum != 0 && max_len_raw > 0 {
            usize::try_from(max_len_raw).unwrap_or(gram::DEFAULT_COLLOCATION_MAX_LENGTH)
        } else {
            gram::DEFAULT_COLLOCATION_MAX_LENGTH
        };

        Ok(InitConfig {
            model_dir: PathBuf::from(model_dir),
            ort_dll: PathBuf::from(ort_dll),
            gram_path,
            gram_collocation_penalty,
            gram_non_collocation_penalty,
            gram_max_length,
            model_file,
            threads,
        })
    }
}

/// init(model_dir, ort_dll, threads, [gram_path_or_tbl], [col_penalty], [non_col_penalty], [model_file], [gram_max_length])
unsafe extern "C-unwind" fn lua_init(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // Read Lua arguments outside `catch_unwind` so Lua stack/GC calls never unwind through it.
    // SAFETY: `l` is valid for the duration of this Lua C call.
    let cfg_res = unsafe { read_init_config(api, l) };
    let res = match cfg_res {
        Ok(cfg) => catch_unwind(AssertUnwindSafe(|| engine::init_engine(&cfg))),
        Err(e) => Ok(Err(e)),
    };
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

/// init_async(model_dir, ort_dll, threads, [gram_path_or_tbl], [col_penalty], [non_col_penalty], [model_file], [gram_max_length])
unsafe extern "C-unwind" fn lua_init_async(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: `l` is valid for this C call.
    let cfg_res = unsafe { read_init_config(api, l) };
    let res = match cfg_res {
        Ok(cfg) => catch_unwind(AssertUnwindSafe(|| {
            engine::init_engine_async(cfg);
            Ok(())
        })),
        Err(e) => Ok(Err(e)),
    };
    // SAFETY: Push `(ok, [err])` onto `l`.
    unsafe {
        if (api.checkstack)(l, 4) == 0 {
            return 0;
        }
        match res {
            Ok(Ok(())) => {
                (api.pushboolean)(l, 1);
                1
            }
            Ok(Err(e)) => {
                (api.pushboolean)(l, 0);
                push_str(api, l, &e);
                2
            }
            Err(_) => {
                (api.pushboolean)(l, 0);
                push_str(api, l, "panic in rime_neural.init_async");
                2
            }
        }
    }
}

unsafe extern "C-unwind" fn lua_is_ready(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    let ready = catch_unwind(AssertUnwindSafe(engine::is_ready)).unwrap_or(false);
    // SAFETY: Push 1 boolean return value onto `l`.
    unsafe {
        if (api.checkstack)(l, 2) == 0 {
            return 0;
        }
        (api.pushboolean)(l, if ready { 1 } else { 0 });
    }
    1
}

/// resolve_layout(layout_opt, schema_id) -> (layout_name, is_dp, uses_semicolon)
unsafe extern "C-unwind" fn lua_resolve_layout(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: `l` is valid for this C call.
    let (opt, schema) = unsafe {
        (
            read_str(api, l, 1).unwrap_or_default(),
            read_str(api, l, 2).unwrap_or_default(),
        )
    };
    let layout = catch_unwind(AssertUnwindSafe(|| {
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
unsafe extern "C-unwind" fn lua_set_uia_enabled(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: `l` is valid for this C call.
    let enable = unsafe { (api.toboolean)(l, 1) != 0 };
    let ok = catch_unwind(AssertUnwindSafe(|| {
        uia::set_enabled(enable);
    }))
    .is_ok();
    // SAFETY: Push 1 boolean onto `l`.
    unsafe {
        if (api.checkstack)(l, 2) == 0 {
            return 0;
        }
        (api.pushboolean)(l, if ok { 1 } else { 0 });
    }
    1
}

/// set_doc_entities_enabled(bool)
unsafe extern "C-unwind" fn lua_set_doc_entities_enabled(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: `l` is valid for this C call.
    let enable = unsafe { (api.toboolean)(l, 1) != 0 };
    let ok = catch_unwind(AssertUnwindSafe(|| {
        uia::set_doc_entities_enabled(enable);
    }))
    .is_ok();
    // SAFETY: Push 1 boolean onto `l`.
    unsafe {
        if (api.checkstack)(l, 2) == 0 {
            return 0;
        }
        (api.pushboolean)(l, if ok { 1 } else { 0 });
    }
    1
}

/// query_uia_context(live_preedit, raw_input, [cand_preview])
///   -> (status_str, text_str, hwnd_int, preedit_stripped_bool, focus_epoch, entity_text_str)
unsafe extern "C-unwind" fn lua_query_uia_context(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: `l` is valid for this C call.
    let (preedit, raw_input, cand_preview) = unsafe {
        (
            read_str(api, l, 1).unwrap_or_default(),
            read_str(api, l, 2).unwrap_or_default(),
            read_str(api, l, 3).unwrap_or_default(),
        )
    };
    let res = catch_unwind(AssertUnwindSafe(|| {
        uia::query_context(&preedit, &raw_input, &cand_preview)
    }));
    let (context, hwnd, epoch) =
        res.unwrap_or_else(|_| (UiaContext::Blocked, uia::foreground_hwnd(), 0));
    let (status, text, stripped, entity_text) = match context {
        UiaContext::Blocked => ("blocked", String::new(), false, String::new()),
        UiaContext::Text {
            text,
            preedit_stripped,
            entity_text,
        } => ("text", text, preedit_stripped, entity_text),
        UiaContext::WindowChanged => ("window_changed", String::new(), false, String::new()),
        UiaContext::FallbackSameWindow => ("fallback", String::new(), false, String::new()),
        UiaContext::Unavailable => ("unavailable", String::new(), false, String::new()),
    };
    // SAFETY: Push 6 return values onto `l` after ensuring stack capacity.
    unsafe {
        if (api.checkstack)(l, 7) == 0 {
            return 0;
        }
        push_str(api, l, status);
        push_str(api, l, &text);
        (api.pushinteger)(l, hwnd as LuaInteger);
        (api.pushboolean)(l, stripped.into());
        (api.pushinteger)(l, epoch as LuaInteger);
        push_str(api, l, &entity_text);
        6
    }
}

/// kick_uia_refresh() -> nil (non-blocking background refresh hint)
unsafe extern "C-unwind" fn lua_kick_uia_refresh(_l: LuaState) -> c_int {
    let _ = catch_unwind(AssertUnwindSafe(uia::kick_refresh));
    0
}

/// kick_uia_after_commit() -> nil (post-commit background refresh after brief settle delay)
unsafe extern "C-unwind" fn lua_kick_uia_after_commit(_l: LuaState) -> c_int {
    let _ = catch_unwind(AssertUnwindSafe(uia::kick_after_commit));
    0
}

/// rerank(context, raw_input, layout_str_or_bool, cands_tbl, weights_tbl, [p2c_cfg_tbl], [gram_cfg_tbl], [entity_ctx_str])
unsafe extern "C-unwind" fn lua_rerank(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // Extract all Lua arguments BEFORE `catch_unwind` (table lookups invoke `pushlstring`).
    // SAFETY: `l` is valid for this C call; ensure stack headroom for table field lookups.
    let args = unsafe {
        if (api.checkstack)(l, 8) == 0 {
            return 0;
        }
        (
            read_str(api, l, 1).unwrap_or_default(),
            read_str(api, l, 2).unwrap_or_default(),
            read_layout(api, l, 3),
            read_str_array(api, l, 4),
            read_weights(api, l, 5),
            read_p2c_cfg(api, l, 6),
            read_gram_req(api, l, 7),
            read_str(api, l, 8).unwrap_or_default(),
        )
    };
    let (ctx, raw_input, layout, cands, weights, p2c_cfg, gram_req, entity_ctx) = args;
    let res = catch_unwind(AssertUnwindSafe(
        || -> Result<engine::RerankResult, String> {
            if !engine::is_ready() {
                return Err("not ready".to_string());
            }
            engine::rerank(
                &ctx,
                &entity_ctx,
                &raw_input,
                layout,
                &cands,
                weights,
                p2c_cfg,
                &gram_req,
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
                (api.createtable)(l, 0, 8);

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

                rawset_field(api, l, "has_ctx", |api, l| {
                    (api.pushboolean)(l, if rr.has_ctx { 1 } else { 0 })
                });
                rawset_field(api, l, "has_gram", |api, l| {
                    (api.pushboolean)(l, if rr.has_gram { 1 } else { 0 })
                });

                1
            }
            Ok(Err(e)) => {
                (api.pushnil)(l);
                push_str(api, l, &e);
                2
            }
            Err(_) => {
                (api.pushnil)(l);
                push_str(api, l, "panic in rime_neural.rerank");
                2
            }
        }
    }
}

enum ParsedFeedbackArgs {
    V2 {
        schema: String,
        has_ctx: bool,
        logits: Vec<f32>,
        gram_deltas: Vec<f32>,
        char_lps: Vec<f32>,
        shown_top1_idx: i32,
        chosen_idx: i32,
        p2c_shown: bool,
        p2c_chosen: bool,
        chosen_rank: i32,
        log_path: Option<PathBuf>,
        has_gram_override: Option<bool>,
    },
    Legacy {
        schema: String,
        ctx: String,
        cands: Vec<String>,
        logits: Vec<f32>,
        gram_deltas: Vec<f32>,
        char_lps: Vec<f32>,
        p2c_cand: Option<String>,
        shown_top1: String,
        chosen_text: String,
        chosen_rank: i32,
        log_path: Option<PathBuf>,
    },
}

unsafe fn read_feedback_args(api: &LuaApi, l: LuaState) -> ParsedFeedbackArgs {
    // SAFETY: Caller passes a valid `lua_State*`.
    unsafe {
        let nargs = (api.gettop)(l);
        let schema = read_str(api, l, 1).unwrap_or_default();
        let arg2_ty = (api.lua_type)(l, 2);
        let arg3_ty = (api.lua_type)(l, 3);
        let is_v2 = arg2_ty == LUA_TBOOLEAN
            || (nargs <= 12 && (arg3_ty == LUA_TTABLE || arg2_ty != LUA_TSTRING));
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
            let log_path = read_str(api, l, 11)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from);
            let has_gram_override = if nargs >= 12 && (api.lua_type)(l, 12) == LUA_TBOOLEAN {
                Some((api.toboolean)(l, 12) != 0)
            } else {
                None
            };
            return ParsedFeedbackArgs::V2 {
                schema,
                has_ctx,
                logits,
                gram_deltas,
                char_lps,
                shown_top1_idx,
                chosen_idx,
                p2c_shown,
                p2c_chosen,
                chosen_rank,
                log_path,
                has_gram_override,
            };
        }

        let log_path = read_str(api, l, 13)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
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
        ParsedFeedbackArgs::Legacy {
            schema,
            ctx,
            cands,
            logits,
            gram_deltas,
            char_lps,
            p2c_cand,
            shown_top1,
            chosen_text,
            chosen_rank,
            log_path,
        }
    }
}

/// Zero-plaintext feedback logger.
/// Supports both:
/// - Numeric v2 signature (11–12 args):
///   `log_feedback(schema, has_ctx_bool, logits_tbl, gram_tbl, char_tbl, shown_top1_idx, chosen_idx, p2c_shown, p2c_chosen, chosen_rank, [log_path], [has_gram_bool])`
/// - Legacy 13-arg signature (for older Lua scripts): converts to zero-plaintext v2 in memory before writing.
unsafe extern "C-unwind" fn lua_log_feedback(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: Extract Lua arguments outside `catch_unwind`.
    let parsed = unsafe { read_feedback_args(api, l) };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let (model_name, engine_has_gram) = engine::active_feedback_meta();
        match parsed {
            ParsedFeedbackArgs::V2 {
                schema,
                has_ctx,
                logits,
                gram_deltas,
                char_lps,
                shown_top1_idx,
                chosen_idx,
                p2c_shown,
                p2c_chosen,
                chosen_rank,
                log_path,
                has_gram_override,
            } => {
                let log_path = match log_path {
                    Some(p) => p,
                    None => engine::model_dir_path()
                        .ok_or("engine not ready")?
                        .join("feedback.jsonl"),
                };
                let has_gram = has_gram_override.unwrap_or(engine_has_gram);
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
            ParsedFeedbackArgs::Legacy {
                schema,
                ctx,
                cands,
                logits,
                gram_deltas,
                char_lps,
                p2c_cand,
                shown_top1,
                chosen_text,
                chosen_rank,
                log_path,
            } => {
                let log_path = match log_path {
                    Some(p) => p,
                    None => engine::model_dir_path()
                        .ok_or("engine not ready")?
                        .join("feedback.jsonl"),
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
                    engine_has_gram,
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
        }
    }));
    // SAFETY: Push `(ok, [err])` onto `l`.
    unsafe {
        if (api.checkstack)(l, 4) == 0 {
            return 0;
        }
        match res {
            Ok(Ok(())) => {
                (api.pushboolean)(l, 1);
                1
            }
            Ok(Err(e)) => {
                (api.pushboolean)(l, 0);
                push_str(api, l, &e);
                2
            }
            Err(_) => {
                (api.pushboolean)(l, 0);
                push_str(api, l, "panic in rime_neural.log_feedback");
                2
            }
        }
    }
}

/// suggest_typo_inputs(raw_input, layout_str_or_bool, [mode_str], [max_results])
///   -> array of { input = "...", cost = f64, kind = "..." }
unsafe extern "C-unwind" fn lua_suggest_typo_inputs(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: Extract arguments outside `catch_unwind`.
    let (raw_input, layout, span_mode, max_results) = unsafe {
        let raw = read_str(api, l, 1).unwrap_or_default();
        let layout = read_layout(api, l, 2);
        let mode = read_str(api, l, 3).unwrap_or_default();
        let mut isnum: c_int = 0;
        let max_n = (api.tointegerx)(l, 4, &mut isnum);
        let cap = if isnum != 0 && max_n > 0 {
            (max_n as usize).min(32)
        } else {
            6
        };
        (raw, layout, mode == "span", cap)
    };
    let res = catch_unwind(AssertUnwindSafe(|| {
        engine::suggest_typo_inputs(&raw_input, layout, span_mode, max_results)
    }));
    // SAFETY: Push Lua array of tables using `rawset_field`.
    unsafe {
        if (api.checkstack)(l, 8) == 0 {
            return 0;
        }
        match res {
            Ok(Ok(items)) => {
                (api.createtable)(l, items.len() as c_int, 0);
                for (i, item) in items.iter().enumerate() {
                    (api.createtable)(l, 0, 3);
                    rawset_field(api, l, "input", |api, l| push_str(api, l, &item.input));
                    rawset_field(api, l, "cost", |api, l| {
                        (api.pushnumber)(l, item.edit_cost as f64)
                    });
                    rawset_field(api, l, "kind", |api, l| push_str(api, l, item.kind));
                    (api.rawseti)(l, -2, (i + 1) as LuaInteger);
                }
                1
            }
            _ => {
                (api.createtable)(l, 0, 0);
                1
            }
        }
    }
}

/// score_gram(cands_tbl, [gram_cfg_tbl]) -> (scores_tbl, has_gram_bool)
unsafe extern "C-unwind" fn lua_score_gram(l: LuaState) -> c_int {
    // SAFETY: Invoked by Lua 5.4 host with valid `lua_State*`.
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: Extract arguments outside `catch_unwind`.
    let (cands, gram_req) = unsafe {
        if (api.checkstack)(l, 6) == 0 {
            return 0;
        }
        (read_str_array(api, l, 1), read_gram_req(api, l, 2))
    };
    let res = catch_unwind(AssertUnwindSafe(|| {
        engine::score_gram_sentences(&cands, &gram_req)
    }));
    // SAFETY: Push `(scores_tbl, has_gram_bool)`.
    unsafe {
        if (api.checkstack)(l, 4) == 0 {
            return 0;
        }
        match res {
            Ok(Ok((scores, has_gram))) => {
                push_f32_array(api, l, &scores);
                (api.pushboolean)(l, if has_gram { 1 } else { 0 });
                2
            }
            _ => {
                (api.createtable)(l, 0, 0);
                (api.pushboolean)(l, 0);
                2
            }
        }
    }
}

unsafe extern "C-unwind" fn lua_stats(l: LuaState) -> c_int {
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
        (api.createtable)(l, 0, 15);
        rawset_field(api, l, "version", |api, l| push_str(api, l, PKG_VERSION));
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
pub unsafe extern "C-unwind" fn luaopen_rime_neural(l: LuaState) -> c_int {
    // Resolve Lua 5.4 host symbols & verify ABI BEFORE pinning this DLL permanently in memory.
    // SAFETY: Dynamic lookup against loaded host modules (`rime.dll` / `librime.dll` / `lua54.dll`).
    let Ok(api) = (unsafe { get_api() }) else {
        return 0;
    };
    // SAFETY: `l` is a valid `lua_State*` passed by `package.loadlib`.
    unsafe {
        if !check_lua54_abi(api, l) || (api.checkstack)(l, 8) == 0 {
            return 0;
        }
        // Pin module before returning exported closures so `lua_close()` during Rime redeploys
        // can never unload active ORT or UIA worker threads.
        let pinned = lua_api::pin_self_module(luaopen_rime_neural as *const std::ffi::c_void);
        if !pinned {
            push_str(api, l, "failed to pin rime_neural.dll in host process");
            return 1;
        }

        (api.createtable)(l, 0, 15);

        macro_rules! export_fn {
            ($name:literal, $func:ident) => {
                rawset_field(api, l, $name, |api, l| {
                    (api.pushcclosure)(l, $func, 0);
                });
            };
        }

        rawset_field(api, l, "version", |api, l| push_str(api, l, PKG_VERSION));
        export_fn!("init", lua_init);
        export_fn!("init_async", lua_init_async);
        export_fn!("is_ready", lua_is_ready);
        export_fn!("resolve_layout", lua_resolve_layout);
        export_fn!("set_uia_enabled", lua_set_uia_enabled);
        export_fn!("set_doc_entities_enabled", lua_set_doc_entities_enabled);
        export_fn!("query_uia_context", lua_query_uia_context);
        export_fn!("kick_uia_refresh", lua_kick_uia_refresh);
        export_fn!("kick_uia_after_commit", lua_kick_uia_after_commit);
        export_fn!("rerank", lua_rerank);
        export_fn!("suggest_typo_inputs", lua_suggest_typo_inputs);
        export_fn!("score_gram", lua_score_gram);
        export_fn!("log_feedback", lua_log_feedback);
        export_fn!("stats", lua_stats);
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::c_char;
    use std::sync::Mutex;

    #[derive(Clone)]
    enum MockVal {
        Nil,
        Bool(bool),
        Num(f64),
        Int(i64),
        Str(String),
        Arr(Vec<MockVal>),
        Map(HashMap<String, MockVal>),
    }

    struct MockLua {
        stack: Vec<MockVal>,
    }

    static MOCK_LOCK: Mutex<()> = Mutex::new(());

    impl MockLua {
        fn resolve(&self, idx: c_int) -> usize {
            if idx < 0 {
                (self.stack.len() as isize + idx as isize) as usize
            } else {
                (idx - 1) as usize
            }
        }
    }

    unsafe extern "C-unwind" fn m_version(_: LuaState) -> lua_api::LuaNumber {
        504.0
    }
    unsafe extern "C-unwind" fn m_checkstack(_: LuaState, _: c_int) -> c_int {
        1
    }
    unsafe extern "C-unwind" fn m_gettop(l: LuaState) -> c_int {
        // SAFETY: `l` points to a test-owned `MockLua`.
        unsafe { (*(l as *mut MockLua)).stack.len() as c_int }
    }
    unsafe extern "C-unwind" fn m_createtable(_: LuaState, _: c_int, _: c_int) {}
    unsafe extern "C-unwind" fn m_pushcclosure(_: LuaState, _: lua_api::LuaCFunction, _: c_int) {}
    unsafe extern "C-unwind" fn m_rawget(l: LuaState, idx: c_int) -> c_int {
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &mut *(l as *mut MockLua) };
        let t_idx = st.resolve(idx);
        let key = match st.stack.pop() {
            Some(MockVal::Str(s)) => s,
            _ => String::new(),
        };
        let val = match st.stack.get(t_idx) {
            Some(MockVal::Map(m)) => m.get(&key).cloned().unwrap_or(MockVal::Nil),
            _ => MockVal::Nil,
        };
        st.stack.push(val);
        0
    }
    unsafe extern "C-unwind" fn m_rawset(_: LuaState, _: c_int) {}
    unsafe extern "C-unwind" fn m_type(l: LuaState, idx: c_int) -> c_int {
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &*(l as *const MockLua) };
        let pos = st.resolve(idx);
        match st.stack.get(pos) {
            Some(MockVal::Bool(_)) => LUA_TBOOLEAN,
            Some(MockVal::Str(_)) => LUA_TSTRING,
            Some(MockVal::Arr(_)) | Some(MockVal::Map(_)) => LUA_TTABLE,
            Some(MockVal::Num(_)) | Some(MockVal::Int(_)) => 3,
            _ => 0,
        }
    }
    unsafe extern "C-unwind" fn m_tolstring(
        l: LuaState,
        idx: c_int,
        out_len: *mut usize,
    ) -> *const c_char {
        // SAFETY: `l` points to a test-owned `MockLua`; string remains alive on stack.
        let st = unsafe { &*(l as *const MockLua) };
        let pos = st.resolve(idx);
        if let Some(MockVal::Str(s)) = st.stack.get(pos) {
            if !out_len.is_null() {
                // SAFETY: Caller passes valid out pointer.
                unsafe {
                    *out_len = s.len();
                }
            }
            s.as_ptr() as *const c_char
        } else {
            std::ptr::null()
        }
    }
    unsafe extern "C-unwind" fn m_pushlstring(
        l: LuaState,
        s: *const c_char,
        len: usize,
    ) -> *const c_char {
        // SAFETY: `s` points to `len` readable bytes passed from Rust `push_str`.
        let bytes = unsafe { std::slice::from_raw_parts(s as *const u8, len) };
        let owned = String::from_utf8_lossy(bytes).into_owned();
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &mut *(l as *mut MockLua) };
        st.stack.push(MockVal::Str(owned));
        std::ptr::null()
    }
    unsafe extern "C-unwind" fn m_pushboolean(_: LuaState, _: c_int) {}
    unsafe extern "C-unwind" fn m_pushinteger(_: LuaState, _: LuaInteger) {}
    unsafe extern "C-unwind" fn m_pushnumber(_: LuaState, _: lua_api::LuaNumber) {}
    unsafe extern "C-unwind" fn m_pushnil(_: LuaState) {}
    unsafe extern "C-unwind" fn m_toboolean(l: LuaState, idx: c_int) -> c_int {
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &*(l as *const MockLua) };
        match st.stack.get(st.resolve(idx)) {
            Some(MockVal::Bool(false)) | Some(MockVal::Nil) | None => 0,
            _ => 1,
        }
    }
    unsafe extern "C-unwind" fn m_tointegerx(
        l: LuaState,
        idx: c_int,
        isnum: *mut c_int,
    ) -> LuaInteger {
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &*(l as *const MockLua) };
        let (ok, val) = match st.stack.get(st.resolve(idx)) {
            Some(MockVal::Int(i)) => (1, *i),
            Some(MockVal::Num(f)) if f.is_finite() && f.fract() == 0.0 => (1, *f as i64),
            _ => (0, 0),
        };
        if !isnum.is_null() {
            // SAFETY: Caller passes valid out pointer.
            unsafe {
                *isnum = ok;
            }
        }
        val
    }
    unsafe extern "C-unwind" fn m_tonumberx(
        l: LuaState,
        idx: c_int,
        isnum: *mut c_int,
    ) -> lua_api::LuaNumber {
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &*(l as *const MockLua) };
        let (ok, val) = match st.stack.get(st.resolve(idx)) {
            Some(MockVal::Num(f)) => (1, *f),
            Some(MockVal::Int(i)) => (1, *i as f64),
            _ => (0, 0.0),
        };
        if !isnum.is_null() {
            // SAFETY: Caller passes valid out pointer.
            unsafe {
                *isnum = ok;
            }
        }
        val
    }
    unsafe extern "C-unwind" fn m_rawlen(l: LuaState, idx: c_int) -> u64 {
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &*(l as *const MockLua) };
        match st.stack.get(st.resolve(idx)) {
            Some(MockVal::Arr(v)) => v.len() as u64,
            _ => 0,
        }
    }
    unsafe extern "C-unwind" fn m_rawgeti(l: LuaState, idx: c_int, n: LuaInteger) -> c_int {
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &mut *(l as *mut MockLua) };
        let pos = st.resolve(idx);
        let val = match st.stack.get(pos) {
            Some(MockVal::Arr(v)) if n >= 1 => {
                v.get((n - 1) as usize).cloned().unwrap_or(MockVal::Nil)
            }
            _ => MockVal::Nil,
        };
        st.stack.push(val);
        0
    }
    unsafe extern "C-unwind" fn m_rawseti(_: LuaState, _: c_int, _: LuaInteger) {}
    unsafe extern "C-unwind" fn m_settop(l: LuaState, idx: c_int) {
        // SAFETY: `l` points to a test-owned `MockLua`.
        let st = unsafe { &mut *(l as *mut MockLua) };
        let new_len = if idx < 0 {
            (st.stack.len() as isize + idx as isize + 1).max(0) as usize
        } else {
            idx as usize
        };
        st.stack.truncate(new_len);
    }

    fn mock_lua_api() -> LuaApi {
        LuaApi {
            version: m_version,
            checkstack: m_checkstack,
            gettop: m_gettop,
            createtable: m_createtable,
            pushcclosure: m_pushcclosure,
            rawget: m_rawget,
            rawset: m_rawset,
            lua_type: m_type,
            tolstring: m_tolstring,
            pushlstring: m_pushlstring,
            pushboolean: m_pushboolean,
            pushinteger: m_pushinteger,
            pushnumber: m_pushnumber,
            pushnil: m_pushnil,
            toboolean: m_toboolean,
            tointegerx: m_tointegerx,
            tonumberx: m_tonumberx,
            rawlen: m_rawlen,
            rawgeti: m_rawgeti,
            rawseti: m_rawseti,
            settop: m_settop,
        }
    }

    #[test]
    fn test_read_weights_preserves_f64_precision_and_defaults_on_nil_or_nan() {
        let _g = MOCK_LOCK.lock().unwrap();
        let api = mock_lua_api();
        let mut lua = MockLua {
            stack: vec![MockVal::Arr(vec![
                MockVal::Num(1.0),
                MockVal::Nil,
                MockVal::Num(0.05),
                MockVal::Num(f64::NAN),
                MockVal::Num(0.25),
            ])],
        };
        let ptr = (&mut lua as *mut MockLua) as LuaState;
        // SAFETY: `ptr` points to live `MockLua`.
        let w = unsafe { read_weights(&api, ptr, 1) };
        assert_eq!(w.w_logit, 1.0);
        assert_eq!(w.w_rank, FusionWeights::default().w_rank);
        assert_eq!(w.w_gram, 0.05); // exact f64 (no f32 drift)
        assert_eq!(w.w_char_lp, FusionWeights::default().w_char_lp);
        assert_eq!(w.w_ctx_lp, 0.25);
        assert_eq!(lua.stack.len(), 1);
    }

    #[test]
    fn p2c_thresholds_use_defaults_on_f32_overflow() {
        let api = mock_lua_api();
        let mut lua = MockLua {
            stack: vec![
                MockVal::Map(HashMap::from([(
                    "verify_margin".into(),
                    MockVal::Num(1e200),
                )])),
                MockVal::Num(-1e200),
                MockVal::Num(1.25),
            ],
        };
        let ptr = (&mut lua as *mut MockLua) as LuaState;
        // SAFETY: `ptr` points to live `MockLua`.
        unsafe {
            assert_eq!(
                read_p2c_cfg(&api, ptr, 1).verify_margin,
                P2cConfig::default().verify_margin
            );
            let legacy = read_p2c_cfg(&api, ptr, 2);
            assert_eq!(
                legacy.legacy_top1_margin,
                P2cConfig::default().legacy_top1_margin
            );
            assert_eq!(legacy.legacy_top2_margin, 1.25);
        }
    }

    #[test]
    fn test_read_gram_req_handles_disabled_and_custom_tables() {
        let _g = MOCK_LOCK.lock().unwrap();
        let api = mock_lua_api();
        let mut disabled_map = HashMap::new();
        disabled_map.insert("enabled".to_string(), MockVal::Bool(false));
        let mut custom_map = HashMap::new();
        custom_map.insert(
            "path".to_string(),
            MockVal::Str("C:\\fake.gram".to_string()),
        );
        custom_map.insert(
            "fallback_path".to_string(),
            MockVal::Str("C:\\shared.gram".to_string()),
        );
        custom_map.insert("col_penalty".to_string(), MockVal::Num(-14.0));
        custom_map.insert("non_col_penalty".to_string(), MockVal::Num(-4.0));
        custom_map.insert("max_length".to_string(), MockVal::Int(7));

        let mut lua = MockLua {
            stack: vec![
                MockVal::Nil,
                MockVal::Map(disabled_map),
                MockVal::Map(custom_map),
            ],
        };
        let ptr = (&mut lua as *mut MockLua) as LuaState;
        // SAFETY: `ptr` points to live `MockLua`.
        unsafe {
            assert_eq!(read_gram_req(&api, ptr, 1), GramRequest::Default);
            assert_eq!(read_gram_req(&api, ptr, 2), GramRequest::Disabled);
            assert_eq!(
                read_gram_req(&api, ptr, 3),
                GramRequest::custom(
                    [
                        Some(PathBuf::from("C:\\fake.gram")),
                        Some(PathBuf::from("C:\\shared.gram"))
                    ],
                    -14.0,
                    -4.0,
                    7
                )
            );
            assert_eq!(lua.stack.len(), 3);
        }
    }
}
