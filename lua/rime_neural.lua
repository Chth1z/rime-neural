-- rime-neural (v1.0.0): Universal ONNX neural sentence reranker filter for Rime / Weasel.
-- Loads `neural/rime_neural.dll` in-process via official Weasel's exported Lua 5.4 C API.
-- Features:
--   * Zero custom rime.dll required; works on official unmodified Weasel (librime >= 1.17.0).
--   * Universal layout auto-detection: Full Pinyin + 7 Double Pinyin layouts
--     (ZRM, Flypy, MSPY, Sogou, ABC, Ziguang, Jiajia/PYJJ) across rime-ice / rime-frost / rime-wanxiang.
--   * Non-blocking background initialization (`init_async`) -> 0 ms Weasel startup delay.
--   * Dynamic KV-cache ONNX architecture introspection (supports 12L ML-pumpkin-1103, 8L zhen-feather, etc.)
--     + selective ORT output fetching + direct mmap `.gram` score fusion + KV-cached Pass-2 verified P2C.
--   * Non-blocking background Windows UIA preceding-text context (clamped to focused control,
--     keystroke-1 pre-warm, window-isolated against cross-app bleed, hard password blocking).
--   * Strict dictionary protection: only reranks `sentence` candidates, never overrides exact
--     dictionary/userdb hits (`phrase`, `user_phrase`, `table`, `completion`, etc.).
--   * Traditional-to-Simplified context normalization (`t2s.json`) + pre-OpenCC genuine commit capture.
--   * Optional privacy-first unbiased local feedback logger (`neural/feedback.jsonl`) for MAP tuning.
--   * Graceful zero-cost pass-through if DLL/model is absent or `neural_rerank` option is off.

local M = {}

local native_mod = nil
local native_loaded = false
local native_attempted = false

local function join_path(a, b)
    if a:sub(-1) == "\\" or a:sub(-1) == "/" then
        return a .. b
    end
    return a .. "\\" .. b
end

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end

local function has_any_model(dir, model_file)
    if model_file and #model_file > 0 then
        return file_exists(join_path(dir, model_file))
    end
    return file_exists(join_path(dir, "laya.one.fused.int8g.onnx"))
        or file_exists(join_path(dir, "laya.one.int8g.onnx"))
        or file_exists(join_path(dir, "laya.one.w4.onnx"))
        or file_exists(join_path(dir, "laya.int8g.onnx"))
end

local function resolve_data_dir(user_dir, model_file)
    local primary = join_path(user_dir, "neural")
    if file_exists(join_path(primary, "rime_neural.dll")) and has_any_model(primary, model_file) then
        return primary
    end
    -- Graceful fallback for users migrating from earlier `zhiyi/` folder layout
    local legacy = join_path(user_dir, "zhiyi")
    if file_exists(join_path(legacy, "rime_neural.dll")) and has_any_model(legacy, model_file) then
        return legacy
    end
    return primary
end

local function ensure_native(user_dir, data_dir, threads, gram_file, col_pen, non_col_pen, model_file, sync_init)
    local ort_path = join_path(data_dir, "onnxruntime.dll")
    local gram_path = ""
    if gram_file and #gram_file > 0 then
        local gp = join_path(user_dir, gram_file)
        if file_exists(gp) then
            gram_path = gp
        end
    end

    if native_attempted then
        if native_loaded and sync_init and native_mod and not native_mod.is_ready() then
            native_mod.init(data_dir, ort_path, threads, gram_path, col_pen, non_col_pen, model_file or "")
        end
        return native_loaded
    end
    native_attempted = true

    local dll_path = join_path(data_dir, "rime_neural.dll")
    if not (file_exists(dll_path) and file_exists(ort_path) and has_any_model(data_dir, model_file)) then
        return false
    end

    local loader, err = package.loadlib(dll_path, "luaopen_rime_neural")
    if not loader then
        if log and log.warning then
            log.warning("rime_neural: loadlib failed: " .. tostring(err))
        end
        return false
    end

    local ok_mod, mod = pcall(loader)
    if not ok_mod or type(mod) ~= "table" or type(mod.rerank) ~= "function" then
        if log and log.warning then
            log.warning("rime_neural: luaopen_rime_neural failed: " .. tostring(mod))
        end
        return false
    end

    if sync_init then
        local ok_init, init_err = mod.init(data_dir, ort_path, threads, gram_path, col_pen, non_col_pen, model_file or "")
        if not ok_init then
            if log and log.warning then
                log.warning("rime_neural: init failed: " .. tostring(init_err))
            end
            return false
        end
    else
        mod.init_async(data_dir, ort_path, threads, gram_path, col_pen, non_col_pen, model_file or "")
    end

    native_mod = mod
    native_loaded = true
    return true
end

local function collect_composition_prefix(context, segment_start)
    if segment_start <= 0 or not context.composition then
        return ""
    end
    local comp = context.composition
    if type(comp.toSegmentation) ~= "function" then
        return ""
    end
    local segs = comp:toSegmentation()
    if not segs then
        return ""
    end
    local out = {}
    for i = 0, segs.size - 1 do
        local seg = segs:get_at(i)
        if seg and seg._end <= segment_start then
            local cand = seg:get_selected_candidate()
            if cand then
                local g = cand:get_genuine()
                local txt = (g and g.text and #g.text > 0) and g.text or cand.text
                if txt and #txt > 0 then
                    out[#out + 1] = txt
                end
            end
        end
    end
    return table.concat(out)
end

local function append_commit_history(parts, context)
    local hist = context.commit_history
    if not hist then
        return
    end
    if type(hist.to_table) == "function" then
        local tbl = hist:to_table()
        if type(tbl) == "table" and #tbl > 0 then
            local start_i = math.max(1, #tbl - 5)
            for i = start_i, #tbl do
                local rec = tbl[i]
                if rec and rec.text and #rec.text > 0 then
                    parts[#parts + 1] = rec.text
                end
            end
        end
    elseif type(hist.latest_text) == "function" then
        local lt = hist:latest_text()
        if lt and #lt > 0 then
            parts[#parts + 1] = lt
        end
    elseif type(hist.back) == "function" then
        local rec = hist:back()
        if rec and rec.text and #rec.text > 0 then
            parts[#parts + 1] = rec.text
        end
    end
end

--- Resolve preceding context with strict cross-app isolation & password protection.
--- Returns `(normalized_ctx_str, is_password_blocked)`.
local function get_preceding_context(context, segment_start, first_preedit, raw_seg, env)
    local parts = {}
    local comp_prefix = collect_composition_prefix(context, segment_start)

    if env.enable_uia and native_mod and type(native_mod.query_uia_context) == "function" then
        local full_preedit = first_preedit or ""
        local ok_p, pre = pcall(context.get_preedit, context)
        if ok_p and pre and pre.text and #pre.text > #full_preedit then
            full_preedit = pre.text
        end
        local status, uia_text = native_mod.query_uia_context(full_preedit, raw_seg or "")
        if status == "blocked" then
            return "", true
        elseif status == "text" then
            if uia_text and #uia_text > 0 then
                parts[#parts + 1] = uia_text
                -- Avoid doubling confirmed multi-segment prefix if inline preedit already echoed it
                if #comp_prefix > 0 and uia_text:sub(-#comp_prefix) ~= comp_prefix then
                    parts[#parts + 1] = comp_prefix
                end
            elseif #comp_prefix > 0 then
                parts[#parts + 1] = comp_prefix
            end
        elseif status == "window_changed" then
            if #comp_prefix > 0 then
                parts[#parts + 1] = comp_prefix
            end
        else
            append_commit_history(parts, context)
            if #comp_prefix > 0 then
                parts[#parts + 1] = comp_prefix
            end
        end
    else
        append_commit_history(parts, context)
        if #comp_prefix > 0 then
            parts[#parts + 1] = comp_prefix
        end
    end

    if #parts == 0 then
        return "", false
    end
    local text = table.concat(parts)
    if #text > 384 then
        local cut = #text - 384 + 1
        while cut <= #text and (string.byte(text, cut) >= 0x80 and string.byte(text, cut) < 0xC0) do
            cut = cut + 1
        end
        text = string.sub(text, cut)
    end
    if env.t2s then
        local ok, conv = pcall(env.t2s.convert, env.t2s, text)
        if ok and conv then
            text = conv
        end
    end
    return text, false
end

--- Extract the pre-OpenCC genuine Simplified committed text when available so `s2twp` phrasing
--- differences (e.g. `程序优化` -> `程式最佳化`) still match `snap.shown` accurately.
local function extract_genuine_commit(ctx, env)
    if ctx.composition and type(ctx.composition.toSegmentation) == "function" then
        local ok_s, segs = pcall(ctx.composition.toSegmentation, ctx.composition)
        if ok_s and segs and segs.size and segs.size > 0 then
            local parts = {}
            local all_ok = true
            for i = 0, segs.size - 1 do
                local seg = segs:get_at(i)
                local cand = seg and seg:get_selected_candidate()
                local g = cand and cand:get_genuine()
                local txt = (g and g.text and #g.text > 0) and g.text or (cand and cand.text)
                if txt and #txt > 0 then
                    parts[#parts + 1] = txt
                else
                    all_ok = false
                    break
                end
            end
            if all_ok and #parts > 0 then
                return table.concat(parts)
            end
        end
    end

    local committed = ctx:get_commit_text() or ""
    if env.t2s and #committed > 0 then
        local ok_c, conv = pcall(env.t2s.convert, env.t2s, committed)
        if ok_c and conv then
            return conv
        end
    end
    return committed
end

local function detect_namespace(cfg)
    for _, ns in ipairs({ "rime_neural", "neural_rerank", "zhiyi_rerank" }) do
        if cfg:get_string(ns .. "/option_name")
            or cfg:get_string(ns .. "/pinyin_layout")
            or cfg:get_list(ns .. "/fusion_weights")
            or cfg:get_int(ns .. "/max_candidates")
            or cfg:get_bool(ns .. "/sync_init") ~= nil
            or cfg:get_bool(ns .. "/enable_uia_context") ~= nil
        then
            return ns
        end
    end
    return "rime_neural"
end

local function load_fusion_weights(cfg, ns)
    local defaults = { 1.0, -1.4, 0.05, 0.15, 0.0 }
    local list = cfg:get_list(ns .. "/fusion_weights")
    if list and list.size >= 5 then
        local out = {}
        for i = 0, 4 do
            local v = list:get_value_at(i)
            local num = v and v:get_double()
            out[i + 1] = num or defaults[i + 1]
        end
        return out
    end
    return defaults
end

local function load_p2c_cfg(cfg, ns)
    return {
        verify_margin = cfg:get_double(ns .. "/p2c_verify_margin") or 0.20,
        top1_logit_diff = cfg:get_double(ns .. "/p2c_top1_logit_diff") or 2.0,
        top1_min_gram_diff = cfg:get_double(ns .. "/p2c_top1_min_gram_diff") or -2.0,
        top2_logit_diff = cfg:get_double(ns .. "/p2c_top2_logit_diff") or 0.5,
        legacy_top1_margin = cfg:get_double(ns .. "/p2c_top1_threshold") or 1.0,
        legacy_top2_margin = cfg:get_double(ns .. "/p2c_top2_threshold") or 0.45,
    }
end

function M.init(env)
    local cfg = env.engine.schema.config
    local ns = detect_namespace(cfg)
    env.schema_id = env.engine.schema.schema_id
    env.option_name = cfg:get_string(ns .. "/option_name") or "neural_rerank"
    env.max_candidates = cfg:get_int(ns .. "/max_candidates") or 8
    env.fusion_weights = load_fusion_weights(cfg, ns)
    env.p2c_cfg = load_p2c_cfg(cfg, ns)
    env.threads = cfg:get_int(ns .. "/threads") or 0
    env.model_file = cfg:get_string(ns .. "/model_file") or ""
    env.sync_init = (cfg:get_bool(ns .. "/sync_init") == true)

    local user_dir = rime_api.get_user_data_dir()
    local ud_low = user_dir:lower()
    local in_temp = (ud_low:find("\\temp\\", 1, true) ~= nil) or (ud_low:find("/temp/", 1, true) ~= nil)

    local data_dir = resolve_data_dir(user_dir, env.model_file)
    local allow_temp_log = (cfg:get_bool(ns .. "/log_feedback_in_temp") == true)
    env.log_feedback = (cfg:get_bool(ns .. "/log_feedback") == true) and (not in_temp or allow_temp_log)
    env.feedback_path = join_path(data_dir, "feedback.jsonl")

    local uia_opt = cfg:get_bool(ns .. "/enable_uia_context")
    if uia_opt == nil then
        uia_opt = true
    end
    local allow_temp_uia = (cfg:get_bool(ns .. "/enable_uia_in_temp") == true)
    env.enable_uia = uia_opt and (not in_temp or allow_temp_uia)

    env.last_rerank = nil

    local gram_lang = cfg:get_string("grammar/language")
    local gram_file = (gram_lang and #gram_lang > 0) and (gram_lang .. ".gram") or ""
    local col_pen = cfg:get_double("grammar/collocation_penalty") or -15.5
    local non_col_pen = cfg:get_double("grammar/non_collocation_penalty") or -4.0

    env.ready = ensure_native(
        user_dir,
        data_dir,
        env.threads,
        gram_file,
        col_pen,
        non_col_pen,
        env.model_file,
        env.sync_init
    )

    -- Resolve pinyin layout universally (auto-detects Full + 7 Double Pinyin schemes)
    local layout_opt = cfg:get_string(ns .. "/pinyin_layout") or "auto"
    if layout_opt == "auto" then
        local dp_legacy = cfg:get_bool(ns .. "/double_pinyin")
        if dp_legacy == false then
            layout_opt = "full"
        end
    end
    if env.ready and native_mod and type(native_mod.resolve_layout) == "function" then
        env.layout, env.is_double_pinyin, env.uses_semicolon = native_mod.resolve_layout(layout_opt, env.schema_id)
    else
        env.is_double_pinyin = (env.schema_id:find("double_pinyin", 1, true) ~= nil)
        env.layout = env.is_double_pinyin and "zrm" or "full"
        env.uses_semicolon = false
    end

    env.min_input_len = cfg:get_int(ns .. "/min_input_len") or (env.is_double_pinyin and 4 or 6)

    if env.ready and env.enable_uia and native_mod and type(native_mod.set_uia_enabled) == "function" then
        native_mod.set_uia_enabled(true)
    end

    local ok_occ, occ = pcall(Opencc, "t2s.json")
    env.t2s = (ok_occ and occ) or nil

    if (env.log_feedback or env.enable_uia) and env.engine.context and env.engine.context.commit_notifier then
        env.commit_conn = env.engine.context.commit_notifier:connect(function(ctx)
            if env.enable_uia and native_mod and type(native_mod.kick_uia_refresh) == "function" then
                native_mod.kick_uia_refresh()
            end

            if not env.log_feedback then
                return
            end
            local snap = env.last_rerank
            env.last_rerank = nil
            if not snap or snap.blocked or not native_mod or type(native_mod.log_feedback) ~= "function" then
                return
            end
            if ctx:get_option("ascii_mode") then
                return
            end

            local norm_commit = extract_genuine_commit(ctx, env)
            if #norm_commit < 6 then
                return
            end

            local chosen_rank = 0
            for i = 1, #snap.shown do
                if snap.shown[i] == norm_commit then
                    chosen_rank = i
                    break
                end
            end
            if chosen_rank == 0 then
                local top1 = snap.shown[1] or ""
                if math.abs(#norm_commit - #top1) > 6 then
                    return
                end
            end

            -- Log 100% of multi-candidate commits (unbiased stream -> zero selection bias for MAP tuning;
            -- disk usage is strictly bounded by feedback.rs 5 MiB auto-rotation ~15,000 sentences).
            native_mod.log_feedback(
                env.schema_id,
                snap.raw_seg,
                snap.pre_ctx,
                snap.cands,
                snap.logits,
                snap.gram_deltas,
                snap.char_lps,
                snap.p2c_text or "",
                snap.shown[1] or "",
                norm_commit,
                chosen_rank,
                false,
                env.feedback_path
            )
        end)
    end
end

function M.func(input, env)
    local ctx = env.engine.context
    if not env.ready or not native_mod or not ctx:get_option(env.option_name) or not native_mod.is_ready() then
        for cand in input:iter() do
            yield(cand)
        end
        return
    end

    local cur_input = ctx.input or ""
    -- Pre-warm UIA snapshot asynchronously on early keystrokes (< 1 µs) so context is already
    -- fresh when keystroke 4+ triggers neural reranking (even after window switch or idle pause).
    if env.enable_uia and #cur_input <= 2 and type(native_mod.kick_uia_refresh) == "function" then
        native_mod.kick_uia_refresh()
    end

    local function clear_stale_snapshot()
        if env.last_rerank and cur_input ~= env.last_rerank.full_input then
            env.last_rerank = nil
        end
    end

    local max_k = env.max_candidates
    local is_first = true
    local passthrough = false
    local flushed = false
    local seg_start = 0
    local seg_end = 0
    local raw_seg = ""
    local first_preedit = ""
    local sent_cands = nil
    local sent_texts = nil
    local seen_texts = nil

    local function flush_sentences()
        flushed = true
        if not sent_cands then
            return
        end
        if #sent_cands >= 2 then
            local pre_ctx, is_blocked = get_preceding_context(ctx, seg_start, first_preedit, raw_seg, env)
            local res = native_mod.rerank(
                pre_ctx,
                raw_seg,
                env.layout,
                sent_texts,
                env.fusion_weights,
                env.p2c_cfg
            )
            if type(res) == "table" and type(res.order) == "table" and #res.order == #sent_cands then
                local p2c_cand = nil
                if type(res.p2c_text) == "string" and #res.p2c_text > 0 and (res.p2c_pos == 1 or res.p2c_pos == 2) then
                    p2c_cand = Candidate("sentence", seg_start, seg_end, res.p2c_text, "")
                    p2c_cand.preedit = first_preedit
                end

                local emit_list = {}
                local shown_texts = {}
                if p2c_cand and res.p2c_pos == 1 then
                    emit_list[#emit_list + 1] = p2c_cand
                    shown_texts[#shown_texts + 1] = res.p2c_text
                end
                for i = 1, #res.order do
                    local idx = res.order[i]
                    local c = sent_cands[idx]
                    if c then
                        emit_list[#emit_list + 1] = c
                        shown_texts[#shown_texts + 1] = c.text
                        if #emit_list == 1 and p2c_cand and res.p2c_pos == 2 then
                            emit_list[#emit_list + 1] = p2c_cand
                            shown_texts[#shown_texts + 1] = res.p2c_text
                        end
                    end
                end

                -- Record snapshot BEFORE yielding (Rime Menu is lazy and pauses coroutine after page 1)
                if env.log_feedback then
                    env.last_rerank = {
                        full_input = cur_input,
                        raw_seg = raw_seg,
                        pre_ctx = pre_ctx,
                        cands = sent_texts,
                        logits = res.logits or {},
                        gram_deltas = res.gram_deltas or {},
                        char_lps = res.char_lps or {},
                        p2c_text = res.p2c_text,
                        shown = shown_texts,
                        blocked = is_blocked,
                    }
                end

                for i = 1, #emit_list do
                    yield(emit_list[i])
                end
                return
            end
        else
            clear_stale_snapshot()
        end
        for i = 1, #sent_cands do
            yield(sent_cands[i])
        end
    end

    for cand in input:iter() do
        if passthrough or flushed then
            yield(cand)
        elseif is_first then
            is_first = false
            local g0 = cand:get_genuine()
            if not g0 or g0.type ~= "sentence" then
                clear_stale_snapshot()
                passthrough = true
                yield(cand)
            else
                seg_start = cand.start
                seg_end = cand._end
                first_preedit = cand.preedit or ""
                raw_seg = string.sub(cur_input, seg_start + 1, seg_end)
                local key_cnt = 0
                local has_non_pinyin = false
                local allow_semi = env.uses_semicolon
                for i = 1, #raw_seg do
                    local b = string.byte(raw_seg, i)
                    if b >= 97 and b <= 122 or (allow_semi and b == 59) then
                        key_cnt = key_cnt + 1
                    elseif (b >= 65 and b <= 90) or (b >= 48 and b <= 57) then
                        -- Uppercase ASCII or digits (e.g. mixed `用AI` / `5G网络` bridge) would shift DP 2-key pairs
                        has_non_pinyin = true
                        key_cnt = key_cnt + 1
                    end
                end
                if (env.is_double_pinyin and has_non_pinyin)
                    or key_cnt < env.min_input_len
                    or (env.is_double_pinyin and (key_cnt % 2) == 1)
                then
                    clear_stale_snapshot()
                    passthrough = true
                    yield(cand)
                else
                    sent_cands = { cand }
                    sent_texts = { cand.text }
                    seen_texts = { [cand.text] = true }
                end
            end
        else
            local g = cand:get_genuine()
            if #sent_cands < max_k
                and g and g.type == "sentence"
                and cand.start == seg_start and cand._end == seg_end
            then
                if not seen_texts[cand.text] then
                    seen_texts[cand.text] = true
                    sent_cands[#sent_cands + 1] = cand
                    sent_texts[#sent_texts + 1] = cand.text
                end
            else
                flush_sentences()
                yield(cand)
            end
        end
    end

    if not passthrough and not flushed and sent_cands then
        flush_sentences()
    end
end

function M.fini(env)
    if env.commit_conn then
        pcall(function() env.commit_conn:disconnect() end)
        env.commit_conn = nil
    end
    env.last_rerank = nil
    env.t2s = nil
end

return M
