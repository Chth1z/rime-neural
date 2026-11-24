mod engine;
mod feedback;
mod gram;
mod lua_api;
mod pinyin;
mod uia;

use std::ffi::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use engine::{FusionWeights, InitConfig, P2cConfig};
use lua_api::{get_api, pop, push_str, read_str, LuaInteger, LuaState};
use pinyin::PinyinLayout;
use uia::UiaContext;

const LUA_TTABLE: c_int = 5;
const LUA_TSTRING: c_int = 4;

unsafe fn read_str_array(api: &lua_api::LuaApi, l: LuaState, idx: c_int) -> Vec<String> {
    if (api.lua_type)(l, idx) != LUA_TTABLE {
        return Vec::new();
    }
    let n = usize::try_from((api.rawlen)(l, idx)).unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for i in 1..=(n as LuaInteger) {
        (api.rawgeti)(l, idx, i);
        if let Some(s) = read_str(api, l, -1) {
            out.push(s);
        }
        pop(api, l, 1);
    }
    out
}

unsafe fn read_f32_array(api: &lua_api::LuaApi, l: LuaState, idx: c_int) -> Vec<f32> {
    if (api.lua_type)(l, idx) != LUA_TTABLE {
        return Vec::new();
    }
    let n = usize::try_from((api.rawlen)(l, idx)).unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for i in 1..=(n as LuaInteger) {
        (api.rawgeti)(l, idx, i);
        let mut isnum: c_int = 0;
        let v = (api.tonumberx)(l, -1, &mut isnum);
        out.push(if isnum != 0 { v as f32 } else { 0.0 });
        pop(api, l, 1);
    }
    out
}

unsafe fn push_f32_array(api: &lua_api::LuaApi, l: LuaState, vals: &[f32]) {
    (api.createtable)(l, vals.len() as c_int, 0);
    for (i, &v) in vals.iter().enumerate() {
        (api.pushnumber)(l, v as f64);
        (api.rawseti)(l, -2, (i + 1) as LuaInteger);
    }
}

unsafe fn read_weights(api: &lua_api::LuaApi, l: LuaState, idx: c_int) -> FusionWeights {
    let mut w = FusionWeights::default();
    if (api.lua_type)(l, idx) == LUA_TTABLE {
        let vals = read_f32_array(api, l, idx);
        if !vals.is_empty() {
            w.w_logit = vals[0] as f64;
        }
        if vals.len() >= 2 {
            w.w_rank = vals[1] as f64;
        }
        if vals.len() >= 3 {
            w.w_gram = vals[2] as f64;
        }
        if vals.len() >= 4 {
            w.w_char_lp = vals[3] as f64;
        }
        if vals.len() >= 5 {
            w.w_ctx_lp = vals[4] as f64;
        }
        return w;
    }

    let mut isnum: c_int = 0;
    let rp = (api.tonumberx)(l, idx, &mut isnum);
    if isnum != 0 {
        w.w_logit = 1.0;
        w.w_rank = -rp.abs();
        w.w_gram = 0.0;
        w.w_char_lp = 0.0;
        w.w_ctx_lp = 0.0;
    }
    w
}

unsafe fn read_table_f32(
    api: &lua_api::LuaApi,
    l: LuaState,
    idx: c_int,
    field: &str,
    default_val: f32,
) -> f32 {
    let tbl_idx = if idx < 0 { idx - 1 } else { idx };
    push_str(api, l, field);
    (api.rawget)(l, tbl_idx);
    let mut isnum: c_int = 0;
    let v = (api.tonumberx)(l, -1, &mut isnum);
    pop(api, l, 1);
    if isnum != 0 {
        v as f32
    } else {
        default_val
    }
}

unsafe fn read_p2c_cfg(api: &lua_api::LuaApi, l: LuaState, idx: c_int) -> P2cConfig {
    let mut cfg = P2cConfig::default();
    if (api.lua_type)(l, idx) == LUA_TTABLE {
        cfg.verify_margin = read_table_f32(api, l, idx, "verify_margin", cfg.verify_margin);
        cfg.top1_logit_diff = read_table_f32(api, l, idx, "top1_logit_diff", cfg.top1_logit_diff);
        cfg.top1_min_gram_diff =
            read_table_f32(api, l, idx, "top1_min_gram_diff", cfg.top1_min_gram_diff);
        cfg.top2_logit_diff = read_table_f32(api, l, idx, "top2_logit_diff", cfg.top2_logit_diff);
        cfg.legacy_top1_margin =
            read_table_f32(api, l, idx, "legacy_top1_margin", cfg.legacy_top1_margin);
        cfg.legacy_top2_margin =
            read_table_f32(api, l, idx, "legacy_top2_margin", cfg.legacy_top2_margin);
        return cfg;
    }
    let mut isnum: c_int = 0;
    let t1 = (api.tonumberx)(l, idx, &mut isnum);
    if isnum != 0 {
        cfg.legacy_top1_margin = t1 as f32;
    }
    isnum = 0;
    let t2 = (api.tonumberx)(l, idx + 1, &mut isnum);
    if isnum != 0 {
        cfg.legacy_top2_margin = t2 as f32;
    }
    cfg
}

unsafe fn read_layout(api: &lua_api::LuaApi, l: LuaState, idx: c_int) -> PinyinLayout {
    if (api.lua_type)(l, idx) == LUA_TSTRING {
        if let Some(s) = read_str(api, l, idx) {
            return PinyinLayout::from_name_or_schema(&s, "");
        }
    }
    if (api.toboolean)(l, idx) != 0 {
        PinyinLayout::Zrm
    } else {
        PinyinLayout::Full
    }
}

unsafe fn read_init_config(api: &lua_api::LuaApi, l: LuaState) -> Result<InitConfig, String> {
    let model_dir = read_str(api, l, 1).ok_or("missing arg 1: model_dir")?;
    let ort_dll = read_str(api, l, 2).ok_or("missing arg 2: ort_dll")?;
    let mut isnum: c_int = 0;
    let threads_raw = (api.tointegerx)(l, 3, &mut isnum);
    let threads = if isnum != 0 && threads_raw >= 0 {
        threads_raw as usize
    } else {
        0 // auto-detect
    };
    let gram_path = read_str(api, l, 4)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    isnum = 0;
    let col_raw = (api.tonumberx)(l, 5, &mut isnum);
    let gram_collocation_penalty = if isnum != 0 { col_raw } else { -15.5 };
    isnum = 0;
    let non_col_raw = (api.tonumberx)(l, 6, &mut isnum);
    let gram_non_collocation_penalty = if isnum != 0 { non_col_raw } else { -4.0 };
    let model_file = read_str(api, l, 7).filter(|s| !s.is_empty());

    Ok(InitConfig {
        model_dir: PathBuf::from(model_dir),
        ort_dll: PathBuf::from(ort_dll),
        gram_path,
        gram_collocation_penalty,
        gram_non_collocation_penalty,
        model_file,
        threads,
    })
}

/// init(model_dir, ort_dll, threads, [gram_path], [col_penalty], [non_col_penalty], [model_file])
unsafe extern "C" fn lua_init(l: LuaState) -> c_int {
    let Ok(api) = get_api() else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let cfg = read_init_config(api, l)?;
        engine::init_engine(&cfg)
    }));
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

/// init_async(model_dir, ort_dll, threads, [gram_path], [col_penalty], [non_col_penalty], [model_file])
unsafe extern "C" fn lua_init_async(l: LuaState) -> c_int {
    let Ok(api) = get_api() else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let cfg = read_init_config(api, l)?;
        engine::init_engine_async(cfg);
        Ok(())
    }));
    (api.pushboolean)(l, if matches!(res, Ok(Ok(()))) { 1 } else { 0 });
    1
}

unsafe extern "C" fn lua_is_ready(l: LuaState) -> c_int {
    let Ok(api) = get_api() else {
        return 0;
    };
    (api.pushboolean)(l, if engine::is_ready() { 1 } else { 0 });
    1
}

/// resolve_layout(layout_opt, schema_id) -> (layout_name, is_dp, uses_semicolon)
unsafe extern "C" fn lua_resolve_layout(l: LuaState) -> c_int {
    let Ok(api) = get_api() else {
        return 0;
    };
    let layout = catch_unwind(AssertUnwindSafe(|| {
        let opt = read_str(api, l, 1).unwrap_or_default();
        let schema = read_str(api, l, 2).unwrap_or_default();
        PinyinLayout::from_name_or_schema(&opt, &schema)
    }))
    .unwrap_or(PinyinLayout::Full);
    push_str(api, l, layout.as_str());
    (api.pushboolean)(l, if layout.is_double_pinyin() { 1 } else { 0 });
    (api.pushboolean)(l, if layout.uses_semicolon() { 1 } else { 0 });
    3
}

/// set_uia_enabled(bool)
unsafe extern "C" fn lua_set_uia_enabled(l: LuaState) -> c_int {
    let Ok(api) = get_api() else {
        return 0;
    };
    let ok = catch_unwind(AssertUnwindSafe(|| {
        let enable = (api.toboolean)(l, 1) != 0;
        uia::set_enabled(enable);
    }))
    .is_ok();
    (api.pushboolean)(l, if ok { 1 } else { 0 });
    1
}

/// query_uia_context(live_preedit, raw_input, [cand_preview]) -> (status_str, text_str, hwnd_int)
unsafe extern "C" fn lua_query_uia_context(l: LuaState) -> c_int {
    let Ok(api) = get_api() else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| {
        let preedit = read_str(api, l, 1).unwrap_or_default();
        let raw_input = read_str(api, l, 2).unwrap_or_default();
        let cand_preview = read_str(api, l, 3).unwrap_or_default();
        let ctx = uia::query_context(&preedit, &raw_input, &cand_preview);
        let hwnd = uia::foreground_hwnd();
        (ctx, hwnd)
    }));
    match res {
        Ok((UiaContext::Blocked, hwnd)) => {
            push_str(api, l, "blocked");
            push_str(api, l, "");
            (api.pushinteger)(l, hwnd as LuaInteger);
            3
        }
        Ok((UiaContext::Text(txt), hwnd)) => {
            push_str(api, l, "text");
            push_str(api, l, &txt);
            (api.pushinteger)(l, hwnd as LuaInteger);
            3
        }
        Ok((UiaContext::WindowChanged, hwnd)) => {
            push_str(api, l, "window_changed");
            push_str(api, l, "");
            (api.pushinteger)(l, hwnd as LuaInteger);
            3
        }
        _ => {
            push_str(api, l, "fallback");
            push_str(api, l, "");
            (api.pushinteger)(l, uia::foreground_hwnd() as LuaInteger);
            3
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
    let Ok(api) = get_api() else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(
        || -> Result<engine::RerankResult, String> {
            if !engine::is_ready() {
                return Err("not ready".to_string());
            }
            let ctx = read_str(api, l, 1).unwrap_or_default();
            let raw_input = read_str(api, l, 2).unwrap_or_default();
            let layout = read_layout(api, l, 3);
            let cands = read_str_array(api, l, 4);
            let weights = read_weights(api, l, 5);
            let p2c_cfg = read_p2c_cfg(api, l, 6);

            engine::rerank(&ctx, &raw_input, layout, &cands, weights, p2c_cfg)
        },
    ));

    match res {
        Ok(Ok(rr)) => {
            (api.createtable)(l, 0, 6);

            (api.createtable)(l, rr.order.len() as c_int, 0);
            for (i, idx) in rr.order.into_iter().enumerate() {
                (api.pushinteger)(l, idx as LuaInteger);
                (api.rawseti)(l, -2, (i + 1) as LuaInteger);
            }
            (api.setfield)(l, -2, c"order".as_ptr());

            if let Some(ref txt) = rr.p2c_text {
                push_str(api, l, txt);
            } else {
                (api.pushnil)(l);
            }
            (api.setfield)(l, -2, c"p2c_text".as_ptr());

            (api.pushinteger)(l, rr.p2c_pos as LuaInteger);
            (api.setfield)(l, -2, c"p2c_pos".as_ptr());

            push_f32_array(api, l, &rr.logits);
            (api.setfield)(l, -2, c"logits".as_ptr());

            push_f32_array(api, l, &rr.gram_deltas);
            (api.setfield)(l, -2, c"gram_deltas".as_ptr());

            push_f32_array(api, l, &rr.char_lps);
            (api.setfield)(l, -2, c"char_lps".as_ptr());

            1
        }
        _ => {
            (api.pushnil)(l);
            1
        }
    }
}

/// log_feedback(schema, raw_input, context, cands_tbl, logits_tbl, gram_tbl, char_tbl, p2c_cand_or_nil, shown_top1, chosen_text, chosen_rank, reverted, [log_path])
unsafe extern "C" fn lua_log_feedback(l: LuaState) -> c_int {
    let Ok(api) = get_api() else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let log_path = match read_str(api, l, 13).filter(|s| !s.is_empty()) {
            Some(p) => PathBuf::from(p),
            None => engine::model_dir_path()
                .ok_or("engine not ready")?
                .join("feedback.jsonl"),
        };
        let schema = read_str(api, l, 1).unwrap_or_default();
        let raw_input = read_str(api, l, 2).unwrap_or_default();
        let ctx = read_str(api, l, 3).unwrap_or_default();
        let cands = read_str_array(api, l, 4);
        let logits = read_f32_array(api, l, 5);
        let gram_deltas = read_f32_array(api, l, 6);
        let char_lps = read_f32_array(api, l, 7);
        let p2c_cand = read_str(api, l, 8).filter(|s| !s.is_empty());
        let shown_top1 = read_str(api, l, 9).unwrap_or_default();
        let chosen_text = read_str(api, l, 10).unwrap_or_default();
        let mut isnum: c_int = 0;
        let chosen_rank = (api.tointegerx)(l, 11, &mut isnum) as i32;
        let reverted = (api.toboolean)(l, 12) != 0;

        let rec = feedback::FeedbackRecord {
            ts: feedback::now_unix_secs(),
            sample_weight: 1.0,
            schema: &schema,
            raw_input: &raw_input,
            context: feedback::sanitize_for_log(&ctx),
            cands: &cands,
            logits: &logits,
            gram_deltas: &gram_deltas,
            char_lps: &char_lps,
            p2c_cand: p2c_cand.as_deref(),
            shown_top1: &shown_top1,
            chosen_text: &chosen_text,
            chosen_rank,
            reverted,
        };
        feedback::append_feedback(&log_path, &rec)
    }));
    (api.pushboolean)(l, if matches!(res, Ok(Ok(()))) { 1 } else { 0 });
    1
}

unsafe extern "C" fn lua_stats(l: LuaState) -> c_int {
    let Ok(api) = get_api() else {
        return 0;
    };
    let res = catch_unwind(AssertUnwindSafe(|| {
        let st = engine::get_stats();
        let uia_on = uia::is_enabled();
        (st, uia_on)
    }));
    let Ok((st, uia_on)) = res else {
        (api.pushnil)(l);
        return 1;
    };
    (api.createtable)(l, 0, 13);
    (api.pushboolean)(l, if st.ready { 1 } else { 0 });
    (api.setfield)(l, -2, c"ready".as_ptr());
    (api.pushinteger)(l, st.calls as LuaInteger);
    (api.setfield)(l, -2, c"calls".as_ptr());
    (api.pushinteger)(l, st.cache_hits as LuaInteger);
    (api.setfield)(l, -2, c"cache_hits".as_ptr());
    (api.pushinteger)(l, st.kv_hits as LuaInteger);
    (api.setfield)(l, -2, c"kv_hits".as_ptr());
    (api.pushinteger)(l, st.p2c_pass2_calls as LuaInteger);
    (api.setfield)(l, -2, c"p2c_pass2_calls".as_ptr());
    (api.pushnumber)(l, st.avg_ms);
    (api.setfield)(l, -2, c"avg_ms".as_ptr());
    (api.pushboolean)(l, if st.fused_kv { 1 } else { 0 });
    (api.setfield)(l, -2, c"fused_kv".as_ptr());
    (api.pushboolean)(l, if st.has_gram { 1 } else { 0 });
    (api.setfield)(l, -2, c"has_gram".as_ptr());
    (api.pushboolean)(l, if uia_on { 1 } else { 0 });
    (api.setfield)(l, -2, c"uia_enabled".as_ptr());
    (api.pushinteger)(l, st.enc_layers as LuaInteger);
    (api.setfield)(l, -2, c"enc_layers".as_ptr());
    (api.pushboolean)(l, if st.choice_compact { 1 } else { 0 });
    (api.setfield)(l, -2, c"choice_compact".as_ptr());
    push_str(api, l, &st.model_name);
    (api.setfield)(l, -2, c"model_name".as_ptr());
    if let Some(ref err) = st.last_error {
        push_str(api, l, err);
    } else {
        (api.pushnil)(l);
    }
    (api.setfield)(l, -2, c"last_error".as_ptr());
    1
}

/// Lua 5.4 C module entry point invoked via `package.loadlib(..., "luaopen_rime_neural")`.
///
/// # Safety
/// Must only be called by the host Lua 5.4 VM (`librime-lua` / `rime.dll`) with a valid `lua_State*`.
#[no_mangle]
pub unsafe extern "C" fn luaopen_rime_neural(l: LuaState) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        lua_api::pin_self_module(luaopen_rime_neural as *const std::ffi::c_void);
        let Ok(api) = get_api() else {
            return 0;
        };
        (api.createtable)(l, 0, 11);

        macro_rules! export_fn {
            ($name:literal, $func:ident) => {
                (api.pushcclosure)(l, $func, 0);
                (api.setfield)(l, -2, concat!($name, "\0").as_ptr() as *const c_char);
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
        1
    }))
    .unwrap_or(0)
}
