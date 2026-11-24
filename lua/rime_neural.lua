-- rime-neural (v1.1.0): Universal ONNX neural sentence reranker filter for Rime / Weasel.
-- Loads `neural/rime_neural.dll` in-process via official Weasel's exported Lua 5.4 C API.
-- Features:
--   * Zero custom rime.dll required; works on official unmodified Weasel (librime >= 1.17.0).
--   * Universal layout auto-detection: Full Pinyin + 7 Double Pinyin layouts
--     (ZRM, Flypy, MSPY, Sogou, ABC, Ziguang, Jiajia/PYJJ) across rime-ice / rime-frost / rime-wanxiang.
--   * Non-blocking background initialization (`init_async`) -> 0 ms Weasel startup delay.
--   * Dynamic KV-cache ONNX architecture introspection (supports 12L ML-pumpkin-1103, 8L zhen-feather, etc.)
--     + selective ORT output fetching + per-schema isolated mmap `.gram` fusion + KV-cached Pass-2 P2C.
--   * Non-blocking background Windows UIA preceding-text context (frozen per composition,
--     clamped to focused control, focus-change invalidation, fail-closed password/terminal blocking).
--   * Strict dictionary & tag protection: only reranks `sentence` candidates on matching segments (`abc`),
--     never overrides exact dictionary/userdb hits (`phrase`, `user_phrase`, `table`, `completion`, etc.).
--   * Traditional-to-Simplified context normalization (`tw2sp.json` / `t2s.json`) + pre-OpenCC commit capture.
--   * Zero-plaintext numeric local feedback logger (`neural/feedback.jsonl`) for unbiased MAP tuning.
--   * Graceful zero-cost pass-through if DLL/model is absent or the configured toggle option is off.

local M = {}

local native_mod = nil
local native_loaded = false
local failed_dll_paths = {}
local attached_gram_keys = {}
local logged_init_error = false

-- Module-level OpenCC converter cache (shared across schemas; `tw2sp` loaded lazily on demand)
local shared_t2s = nil
local shared_t2s_tried = false
local shared_tw2sp = nil
local shared_tw2sp_tried = false

local function get_shared_t2s()
    if not shared_t2s_tried then
        shared_t2s_tried = true
        local ok, occ = pcall(Opencc, "t2s.json")
        shared_t2s = (ok and occ) or nil
    end
    return shared_t2s
end

local function get_shared_tw2sp()
    if not shared_tw2sp_tried then
        shared_tw2sp_tried = true
        local ok, occ = pcall(Opencc, "tw2sp.json")
        shared_tw2sp = (ok and occ) or nil
    end
    return shared_tw2sp
end

--- Normalize forward slashes to Windows backslashes so `package.loadlib` (`LoadLibraryExA` with
--- `LOAD_WITH_ALTERED_SEARCH_PATH`) always receives canonical Windows paths.
local function normalize_win_path(p)
    if type(p) ~= "string" then
        return ""
    end
    local s = p:gsub("/", "\\")
    while #s > 3 and s:sub(-1) == "\\" do
        s = s:sub(1, -2)
    end
    return s
end

local function join_path(a, b)
    local base = normalize_win_path(a)
    if base:sub(-1) == "\\" then
        return base .. b
    end
    return base .. "\\" .. b
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

local function has_non_ascii(s)
    return type(s) == "string" and s:find("[\128-\255]") ~= nil
end

local function is_pure_ascii(s)
    return type(s) == "string" and #s > 0 and s:find("[\128-\255]") == nil
end

local function utf8_char_count(s)
    if type(s) ~= "string" or #s == 0 then
        return 0
    end
    if utf8 and type(utf8.len) == "function" then
        local n = utf8.len(s)
        if n then
            return n
        end
    end
    local count = 0
    for i = 1, #s do
        local b = string.byte(s, i)
        if b < 0x80 or b >= 0xC0 then
            count = count + 1
        end
    end
    return count
end

-- Trim ASCII whitespace, NBSP U+00A0 (`\194\160`), and CJK ideographic space U+3000 (`\227\128\128`)
-- so Lua fallback matches Rust `str::trim()`.
local function trim_ws(s)
    local cur = s
    while true do
        local next_s = cur:gsub("^[%s]+", ""):gsub("[%s]+$", "")
        if #next_s >= 2 and next_s:sub(1, 2) == "\194\160" then
            next_s = next_s:sub(3)
        end
        if #next_s >= 2 and next_s:sub(-2) == "\194\160" then
            next_s = next_s:sub(1, -3)
        end
        if #next_s >= 3 and next_s:sub(1, 3) == "\227\128\128" then
            next_s = next_s:sub(4)
        end
        if #next_s >= 3 and next_s:sub(-3) == "\227\128\128" then
            next_s = next_s:sub(1, -4)
        end
        if next_s == cur then
            return cur
        end
        cur = next_s
    end
end

local function get_shared_data_dir()
    if rime_api and type(rime_api.get_shared_data_dir) == "function" then
        local ok_s, shared_dir = pcall(rime_api.get_shared_data_dir)
        if ok_s and type(shared_dir) == "string" and #shared_dir > 0 then
            return normalize_win_path(shared_dir)
        end
    end
    return ""
end

local function is_temp_user_dir(user_dir)
    local ud = normalize_win_path(user_dir):lower()
    if ud:find("\\appdata\\local\\temp\\", 1, true) ~= nil
        or ud:find("\\windows\\temp\\", 1, true) ~= nil
    then
        return true
    end
    for _, env_var in ipairs({ "TEMP", "TMP" }) do
        local tv = os.getenv(env_var)
        if type(tv) == "string" and #tv > 0 then
            local norm_tv = normalize_win_path(tv):lower()
            if ud == norm_tv or ud:sub(1, #norm_tv + 1) == (norm_tv .. "\\") then
                return true
            end
        end
    end
    return false
end

local function resolve_data_dir(user_dir, shared_dir, model_file, custom_data_dir)
    if custom_data_dir and #custom_data_dir > 0 then
        return normalize_win_path(custom_data_dir)
    end
    local primary = join_path(user_dir, "neural")
    if file_exists(join_path(primary, "rime_neural.dll")) and has_any_model(primary, model_file) then
        return primary
    end
    -- Fallback to shared_data_dir\neural (typically ASCII Program Files path for non-ASCII usernames)
    if shared_dir and #shared_dir > 0 then
        local shared_neural = join_path(shared_dir, "neural")
        if file_exists(join_path(shared_neural, "rime_neural.dll"))
            and has_any_model(shared_neural, model_file)
        then
            return shared_neural
        end
    end
    -- Graceful fallback for users migrating from earlier `zhiyi/` folder layout
    local legacy = join_path(user_dir, "zhiyi")
    if file_exists(join_path(legacy, "rime_neural.dll")) and has_any_model(legacy, model_file) then
        return legacy
    end
    return primary
end

local function gram_attach_key(gram_paths, col_pen, non_col_pen, gram_max_len)
    if type(gram_paths) ~= "table" or #gram_paths == 0 then
        return ""
    end
    return table.concat(gram_paths, "|")
        .. ":" .. tostring(col_pen)
        .. ":" .. tostring(non_col_pen)
        .. ":" .. tostring(gram_max_len)
end

local function ensure_native(
    data_dir,
    threads,
    gram_paths,
    col_pen,
    non_col_pen,
    gram_max_len,
    model_file,
    sync_init,
    explicit_data_dir
)
    local ort_path = join_path(data_dir, "onnxruntime.dll")
    -- Pass candidate `.gram` paths (`{ user_dir\x.gram, shared_dir\x.gram }`) to Rust so UTF-16
    -- `Path::is_file()` resolves them without Lua ANSI `io.open` failures.
    local gram_arg = (type(gram_paths) == "table" and #gram_paths > 0) and gram_paths or ""
    local gkey = gram_attach_key(gram_paths, col_pen, non_col_pen, gram_max_len)

    if native_loaded and native_mod then
        if sync_init and not native_mod.is_ready() then
            local ok_init, init_err = native_mod.init(
                data_dir, ort_path, threads, gram_arg, col_pen, non_col_pen, model_file or "", gram_max_len
            )
            if not ok_init and log and log.warning then
                log.warning("rime_neural: init failed: " .. tostring(init_err))
            end
            if ok_init == true and #gkey > 0 then
                attached_gram_keys[gkey] = true
            end
            return ok_init == true
        elseif not native_mod.is_ready() and type(native_mod.stats) == "function" then
            local st = native_mod.stats()
            if type(st) == "table" and type(st.last_error) == "string" and #st.last_error > 0 then
                logged_init_error = false
                native_mod.init_async(
                    data_dir, ort_path, threads, gram_arg, col_pen, non_col_pen, model_file or "", gram_max_len
                )
            end
        elseif native_mod.is_ready() and #gkey > 0 and not attached_gram_keys[gkey] then
            attached_gram_keys[gkey] = true
            native_mod.init_async(
                data_dir, ort_path, threads, gram_arg, col_pen, non_col_pen, model_file or "", gram_max_len
            )
        end
        return true
    end

    local dll_path = join_path(data_dir, "rime_neural.dll")
    if failed_dll_paths[dll_path] then
        return false
    end

    if not file_exists(dll_path) then
        failed_dll_paths[dll_path] = true
        if log and log.warning then
            if has_non_ascii(dll_path) then
                log.warning(
                    "rime_neural: cannot access non-ASCII path via Lua ANSI io.open/loadlib ("
                        .. dll_path
                        .. "); configure rime_neural/data_dir to an ASCII path or place neural/ under Weasel shared data_dir"
                )
            elseif explicit_data_dir and #explicit_data_dir > 0 then
                log.warning("rime_neural: configured rime_neural/data_dir missing DLL: " .. dll_path)
            end
        end
        return false
    end

    local loader, err = package.loadlib(dll_path, "luaopen_rime_neural")
    if not loader then
        failed_dll_paths[dll_path] = true
        if log and log.warning then
            log.warning("rime_neural: loadlib failed: " .. tostring(err))
        end
        return false
    end

    local ok_mod, mod = pcall(loader)
    if not ok_mod or type(mod) ~= "table" or type(mod.rerank) ~= "function" then
        failed_dll_paths[dll_path] = true
        if log and log.warning then
            log.warning("rime_neural: luaopen_rime_neural failed: " .. tostring(mod))
        end
        return false
    end

    -- Retain loaded native module even if initial `sync_init` fails, so another schema with valid
    -- config (e.g. valid `model_file`) can retry without re-calling `loadlib`.
    native_mod = mod
    native_loaded = true

    if sync_init then
        local ok_init, init_err = mod.init(
            data_dir, ort_path, threads, gram_arg, col_pen, non_col_pen, model_file or "", gram_max_len
        )
        if not ok_init then
            if log and log.warning then
                log.warning("rime_neural: init failed: " .. tostring(init_err))
            end
            return false
        end
        if #gkey > 0 then
            attached_gram_keys[gkey] = true
        end
    else
        mod.init_async(
            data_dir, ort_path, threads, gram_arg, col_pen, non_col_pen, model_file or "", gram_max_len
        )
        if #gkey > 0 then
            attached_gram_keys[gkey] = true
        end
    end

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
    local ok_s, segs = pcall(comp.toSegmentation, comp)
    if not ok_s or not segs then
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

local function utf8_tail_bytes(text, max_bytes)
    if #text <= max_bytes then
        return text
    end
    local cut = #text - max_bytes + 1
    while cut <= #text and (string.byte(text, cut) >= 0x80 and string.byte(text, cut) < 0xC0) do
        cut = cut + 1
    end
    return string.sub(text, cut)
end

local TRAD_OPTIONS = {
    "traditionalization",
    "zh_trad",
    "s2t",
    "s2tw",
    "s2twp",
    "s2hk",
    "zh_tw",
    "zh_hk",
    "zh_hant",
}

local function is_traditional_mode(ctx)
    for i = 1, #TRAD_OPTIONS do
        if ctx:get_option(TRAD_OPTIONS[i]) == true then
            return true
        end
    end
    return false
end

--- Normalize preceding text or fallback commit text to Mainland Simplified for the neural model.
--- Skips OpenCC FFI on pure-ASCII text, uses a 1-entry cache per `env`, and avoids double
--- conversion when `prefer_tw2sp` is active.
local function normalize_to_simplified(text, prefer_tw2sp, env)
    if not text or #text == 0 or is_pure_ascii(text) then
        return text or ""
    end
    local cache = env.norm_cache
    if cache and cache.text == text and cache.prefer == prefer_tw2sp then
        return cache.out
    end

    local result = text
    if env.t2s_override then
        local ok, conv = pcall(env.t2s_override.convert, env.t2s_override, text)
        result = (ok and conv) or text
    elseif prefer_tw2sp then
        local tw2sp = get_shared_tw2sp()
        if tw2sp then
            local ok, conv = pcall(tw2sp.convert, tw2sp, text)
            if ok and conv then
                result = conv
            end
        else
            local t2s = get_shared_t2s()
            if t2s then
                local ok, conv = pcall(t2s.convert, t2s, text)
                if ok and conv then
                    result = conv
                end
            end
        end
    else
        local t2s = get_shared_t2s()
        if t2s then
            local ok, conv = pcall(t2s.convert, t2s, text)
            if ok and conv then
                result = conv
            end
        end
        if result ~= text then
            local tw2sp = get_shared_tw2sp()
            if tw2sp then
                local ok, conv = pcall(tw2sp.convert, tw2sp, text)
                if ok and conv then
                    result = conv
                end
            end
        end
    end

    env.norm_cache = { text = text, prefer = prefer_tw2sp, out = result }
    return result
end

local function preedit_before_caret(context, fallback_preedit)
    local out = fallback_preedit or ""
    local ok_p, pre = pcall(context.get_preedit, context)
    if ok_p and pre and type(pre.text) == "string" and #pre.text > 0 then
        local txt = pre.text
        local cp = pre.caret_pos
        if type(cp) == "number" and cp > 0 and cp < #txt then
            txt = txt:sub(1, cp)
        end
        if #txt >= #out then
            out = txt
        end
    end
    return out
end

--- Resolve preceding context with strict cross-app isolation, password protection, and per-
--- composition freezing (preventing 1-keystroke lagged inline-preedit pollution and KV-cache
--- thrashing under Weasel's default `inline_preedit: true`).
--- Returns `(normalized_ctx_str, is_password_blocked, comp_prefix)`.
local function get_preceding_context(context, segment_start, first_preedit, raw_seg, first_cand_text, env)
    local comp_prefix = collect_composition_prefix(context, segment_start)
    local is_trad = is_traditional_mode(context)
    local base_ctx = ""

    if env.enable_uia and native_mod and type(native_mod.query_uia_context) == "function" then
        local full_preedit = preedit_before_caret(context, first_preedit)
        local frozen = env.comp_ctx
        -- Only pass `cand_preview` on the initial capture of a composition so repeated words
        -- mid-composition are never falsely stripped.
        local preview_arg = (frozen == nil) and (first_cand_text or "") or ""
        local status, uia_text, hwnd, preedit_stripped = native_mod.query_uia_context(
            full_preedit, raw_seg or "", preview_arg
        )
        if status == "blocked" then
            env.comp_ctx = nil
            return "", true, comp_prefix
        end

        if frozen and frozen.hwnd == hwnd and status ~= "window_changed" then
            base_ctx = frozen.base_ctx
        elseif status == "text" then
            if uia_text and #uia_text > 0 then
                base_ctx = normalize_to_simplified(utf8_tail_bytes(uia_text, 384), is_trad, env)
                -- Under Traditional mode + `preedit_type: preview`, Weasel echoes the Traditional
                -- candidate inline; only strip its Simplified equivalent on initial capture when
                -- Rust did not already strip a pinyin/preview suffix and the candidate is >= 2 chars.
                if not preedit_stripped
                    and is_trad
                    and first_cand_text
                    and utf8_char_count(first_cand_text) >= 2
                    and #base_ctx >= #first_cand_text
                    and base_ctx:sub(-#first_cand_text) == first_cand_text
                then
                    base_ctx = trim_ws(base_ctx:sub(1, #base_ctx - #first_cand_text))
                end
            end
            env.comp_ctx = { hwnd = hwnd, base_ctx = base_ctx }
        elseif status == "window_changed" then
            base_ctx = ""
            env.comp_ctx = nil
        else
            local hist_parts = {}
            append_commit_history(hist_parts, context)
            if #hist_parts > 0 then
                base_ctx = normalize_to_simplified(utf8_tail_bytes(table.concat(hist_parts), 384), is_trad, env)
            end
            env.comp_ctx = { hwnd = hwnd, base_ctx = base_ctx }
        end
    else
        local hist_parts = {}
        append_commit_history(hist_parts, context)
        if #hist_parts > 0 then
            base_ctx = normalize_to_simplified(utf8_tail_bytes(table.concat(hist_parts), 384), is_trad, env)
        end
    end

    if #base_ctx == 0 then
        return utf8_tail_bytes(comp_prefix, 384), false, comp_prefix
    end
    -- Deduplicate confirmed multi-segment prefix after Simplified normalization so Traditional UIA
    -- text (`我們`) and Simplified `comp_prefix` (`我们`) never double up.
    if #comp_prefix > 0 and base_ctx:sub(-#comp_prefix) ~= comp_prefix then
        return utf8_tail_bytes(base_ctx .. comp_prefix, 384), false, comp_prefix
    end
    return utf8_tail_bytes(base_ctx, 384), false, comp_prefix
end

--- Extract the pre-OpenCC genuine Simplified committed text strictly for the reranked segment
--- range `[seg_start, seg_end)` when available. Preserves accurate matching even when the user
--- commits a sentence by pressing trailing punctuation (`,`, `.`, `?`, `!`, `:`, `;`).
local function extract_genuine_commit(ctx, env, seg_start, seg_end)
    if ctx.composition and type(ctx.composition.toSegmentation) == "function" then
        local ok_s, segs = pcall(ctx.composition.toSegmentation, ctx.composition)
        if ok_s and segs and segs.size and segs.size > 0 then
            local parts = {}
            local all_ok = true
            local matched_range = false
            for i = 0, segs.size - 1 do
                local seg = segs:get_at(i)
                if seg then
                    local in_range = (seg_start == nil or seg_end == nil)
                        or (seg.start >= seg_start and seg._end <= seg_end)
                    if in_range then
                        matched_range = true
                        local cand = seg:get_selected_candidate()
                        local g = cand and cand:get_genuine()
                        local txt = (g and g.text and #g.text > 0) and g.text or (cand and cand.text)
                        if txt and #txt > 0 then
                            parts[#parts + 1] = txt
                        else
                            all_ok = false
                            break
                        end
                    end
                end
            end
            if matched_range and all_ok and #parts > 0 then
                return table.concat(parts), true
            end
        end
    end

    local committed = ctx:get_commit_text() or ""
    if #committed > 0 then
        return normalize_to_simplified(committed, is_traditional_mode(ctx), env), false
    end
    return committed, false
end

local function detect_namespace(cfg)
    for _, ns in ipairs({ "rime_neural", "neural_rerank", "zhiyi_rerank" }) do
        if type(cfg.get_item) == "function" and cfg:get_item(ns) ~= nil then
            return ns
        end
        if cfg:get_string(ns .. "/option_name")
            or cfg:get_string(ns .. "/pinyin_layout")
            or cfg:get_string(ns .. "/data_dir")
            or cfg:get_string(ns .. "/model_file")
            or cfg:get_list(ns .. "/fusion_weights")
            or cfg:get_int(ns .. "/max_candidates")
            or cfg:get_bool(ns .. "/sync_init") ~= nil
            or cfg:get_bool(ns .. "/enable_uia_context") ~= nil
            or cfg:get_bool(ns .. "/log_feedback") ~= nil
        then
            return ns
        end
    end
    return "rime_neural"
end

local function is_option_declared_in_switches(cfg, option_name)
    if type(option_name) ~= "string" or #option_name == 0 then
        return false
    end
    local switches = cfg:get_list("switches")
    if not switches or not switches.size or switches.size <= 0 then
        return false
    end
    for i = 0, switches.size - 1 do
        local name = cfg:get_string("switches/@" .. i .. "/name")
        if name == option_name then
            return true
        end
        local opts = cfg:get_list("switches/@" .. i .. "/options")
        if opts and opts.size and opts.size > 0 then
            for j = 0, opts.size - 1 do
                local v = opts:get_value_at(j)
                if v and v:get_string() == option_name then
                    return true
                end
            end
        end
    end
    return false
end

local function is_rerank_enabled(ctx, env)
    if not env.require_option then
        return true
    end
    return ctx:get_option(env.option_name) == true
end

local function load_fusion_weights(cfg, ns)
    local defaults = { 1.0, -1.4, 0.05, 0.15, 0.0 }
    local list = cfg:get_list(ns .. "/fusion_weights")
    if list and list.size and list.size >= 1 then
        local out = { defaults[1], defaults[2], defaults[3], defaults[4], defaults[5] }
        local n = math.min(5, list.size)
        for i = 0, n - 1 do
            local v = list:get_value_at(i)
            local num = v and v:get_double()
            if num ~= nil then
                out[i + 1] = num
            end
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

local function load_allowed_tags(cfg, ns)
    local list = cfg:get_list(ns .. "/tags")
    if list and list.size and list.size > 0 then
        local out = {}
        local count = 0
        for i = 0, list.size - 1 do
            local v = list:get_value_at(i)
            local s = v and v:get_string()
            if s and #s > 0 then
                out[s] = true
                count = count + 1
            end
        end
        if count > 0 then
            return out
        end
    end
    return { abc = true }
end

local function segment_matches_tags(ctx, allowed_tags)
    local comp = ctx.composition
    if not comp or type(comp.back) ~= "function" then
        return true, nil
    end
    local ok_b, seg = pcall(comp.back, comp)
    if not ok_b or not seg then
        return true, nil
    end
    if type(seg.has_tag) ~= "function" then
        return true, seg
    end
    for tag, _ in pairs(allowed_tags) do
        if seg:has_tag(tag) then
            return true, seg
        end
    end
    return false, seg
end

function M.init(env)
    local cfg = env.engine.schema.config
    local ns = detect_namespace(cfg)
    env.schema_id = env.engine.schema.schema_id

    local opt_raw = cfg:get_string(ns .. "/option_name")
    if opt_raw == nil then
        env.option_name = "neural_rerank"
    else
        env.option_name = opt_raw
    end
    -- If the configured option is omitted from `switches:` (or explicitly empty), treat neural
    -- reranking as enabled by default so the documented optional F4 toggle switch is truly optional.
    env.require_option = (#env.option_name > 0) and is_option_declared_in_switches(cfg, env.option_name)

    local max_cands = cfg:get_int(ns .. "/max_candidates") or 8
    if max_cands < 2 then
        max_cands = 2
    elseif max_cands > 32 then
        max_cands = 32
    end
    env.max_candidates = max_cands
    env.fusion_weights = load_fusion_weights(cfg, ns)
    env.p2c_cfg = load_p2c_cfg(cfg, ns)
    env.tags = load_allowed_tags(cfg, ns)
    env.threads = cfg:get_int(ns .. "/threads") or 0
    env.model_file = cfg:get_string(ns .. "/model_file") or ""
    env.sync_init = (cfg:get_bool(ns .. "/sync_init") == true)

    local user_dir = normalize_win_path(rime_api.get_user_data_dir())
    local shared_dir = get_shared_data_dir()
    local in_temp = is_temp_user_dir(user_dir)

    local custom_data_dir = cfg:get_string(ns .. "/data_dir") or ""
    local data_dir = resolve_data_dir(user_dir, shared_dir, env.model_file, custom_data_dir)
    env.data_dir = data_dir
    env.user_dir = user_dir
    env.shared_dir = shared_dir

    local allow_temp_log = (cfg:get_bool(ns .. "/log_feedback_in_temp") == true)
    local fb_opt = cfg:get_bool(ns .. "/log_feedback")
    if fb_opt == nil then
        fb_opt = true
    end
    env.log_feedback = fb_opt and (not in_temp or allow_temp_log)
    -- Always store feedback under writable `user_dir\neural\feedback.jsonl` (Rust uses UTF-16
    -- `CreateFileW`, so non-ASCII user profiles work without writing into read-only Program Files).
    env.feedback_path = join_path(join_path(user_dir, "neural"), "feedback.jsonl")

    local uia_opt = cfg:get_bool(ns .. "/enable_uia_context")
    if uia_opt == nil then
        uia_opt = true
    end
    local allow_temp_uia = (cfg:get_bool(ns .. "/enable_uia_in_temp") == true)
    env.enable_uia = uia_opt and (not in_temp or allow_temp_uia)
    env.uia_activated = false

    env.last_rerank = nil
    env.comp_ctx = nil
    env.norm_cache = nil

    local gram_lang = cfg:get_string("grammar/language")
    local gram_file = (gram_lang and #gram_lang > 0) and (gram_lang .. ".gram") or ""
    local col_pen = cfg:get_double("grammar/collocation_penalty") or -15.5
    local non_col_pen = cfg:get_double("grammar/non_collocation_penalty") or -4.0
    local gram_max_len = cfg:get_int("grammar/collocation_max_length") or 5
    if gram_max_len < 2 then
        gram_max_len = 5
    elseif gram_max_len > 8 then
        gram_max_len = 8
    end

    local gram_paths = {}
    if #gram_file > 0 then
        gram_paths[#gram_paths + 1] = join_path(user_dir, gram_file)
        if #shared_dir > 0 and shared_dir:lower() ~= user_dir:lower() then
            gram_paths[#gram_paths + 1] = join_path(shared_dir, gram_file)
        end
    end
    env.gram_paths = gram_paths
    env.col_pen = col_pen
    env.non_col_pen = non_col_pen
    env.gram_max_len = gram_max_len
    env.gram_key = gram_attach_key(gram_paths, col_pen, non_col_pen, gram_max_len)
    if #gram_paths > 0 then
        env.gram_req = {
            path = gram_paths[1],
            fallback_path = gram_paths[2] or "",
            col_penalty = col_pen,
            non_col_penalty = non_col_pen,
            max_length = gram_max_len,
        }
    else
        env.gram_req = { enabled = false }
    end

    env.ready = ensure_native(
        data_dir,
        env.threads,
        gram_paths,
        col_pen,
        non_col_pen,
        gram_max_len,
        env.model_file,
        env.sync_init,
        custom_data_dir
    )

    -- Resolve pinyin layout universally (auto-detects Full + 7 Double Pinyin schemes)
    local layout_opt = cfg:get_string(ns .. "/pinyin_layout")
    if not layout_opt or #layout_opt == 0 then
        layout_opt = "auto"
    end
    if layout_opt == "auto" then
        local dp_legacy = cfg:get_bool(ns .. "/double_pinyin")
        if dp_legacy == false then
            layout_opt = "full"
        elseif dp_legacy == true then
            layout_opt = "zrm"
        end
    end
    if env.ready and native_mod and type(native_mod.resolve_layout) == "function" then
        env.layout, env.is_double_pinyin, env.uses_semicolon = native_mod.resolve_layout(layout_opt, env.schema_id)
    else
        env.is_double_pinyin = (env.schema_id:find("double_pinyin", 1, true) ~= nil)
        env.layout = env.is_double_pinyin and "zrm" or "full"
        env.uses_semicolon = false
    end

    local min_len = cfg:get_int(ns .. "/min_input_len") or (env.is_double_pinyin and 4 or 6)
    if min_len < 2 then
        min_len = 2
    end
    env.min_input_len = min_len

    local t2s_override = cfg:get_string(ns .. "/t2s_config") or ""
    if #t2s_override > 0 then
        local ok_ov, occ_ov = pcall(Opencc, t2s_override)
        env.t2s_override = (ok_ov and occ_ov) or nil
    else
        env.t2s_override = nil
    end

    if env.ready and (env.log_feedback or env.enable_uia)
        and env.engine.context and env.engine.context.commit_notifier
    then
        env.commit_conn = env.engine.context.commit_notifier:connect(function(ctx)
            -- Always unfreeze per-composition UIA snapshot on commit so next sentence reads fresh text
            env.comp_ctx = nil

            local rerank_on = is_rerank_enabled(ctx, env)
            if env.enable_uia
                and rerank_on
                and not ctx:get_option("ascii_mode")
                and native_mod
                and native_mod.is_ready()
            then
                if type(native_mod.kick_uia_after_commit) == "function" then
                    native_mod.kick_uia_after_commit()
                elseif type(native_mod.kick_uia_refresh) == "function" then
                    native_mod.kick_uia_refresh()
                end
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

            local norm_commit, from_range = extract_genuine_commit(ctx, env, snap.seg_start, snap.seg_end)
            if not from_range
                and snap.comp_prefix and #snap.comp_prefix > 0
                and norm_commit:sub(1, #snap.comp_prefix) == snap.comp_prefix
            then
                norm_commit = norm_commit:sub(#snap.comp_prefix + 1)
            end
            if #norm_commit < 6 or is_pure_ascii(norm_commit) then
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

            -- Derive purely numeric indices in memory; never persist user text, keystrokes, or UIA context.
            local chosen_idx = -1
            for i = 1, #snap.cands do
                if snap.cands[i] == norm_commit then
                    chosen_idx = i - 1
                    break
                end
            end
            local p2c_shown = (type(snap.p2c_text) == "string" and #snap.p2c_text > 0)
            local p2c_chosen = p2c_shown and (chosen_idx < 0) and (norm_commit == snap.p2c_text)
            local shown_top1_idx = -1
            local top1_text = snap.shown[1] or ""
            if p2c_shown and top1_text == snap.p2c_text then
                shown_top1_idx = -2
                for i = 1, #snap.cands do
                    if snap.cands[i] == top1_text then
                        shown_top1_idx = i - 1
                        break
                    end
                end
            else
                for i = 1, #snap.cands do
                    if snap.cands[i] == top1_text then
                        shown_top1_idx = i - 1
                        break
                    end
                end
            end

            -- Log 100% of multi-candidate commits as zero-plaintext numeric features
            -- (bounded by feedback.rs 5 MiB auto-rotation).
            native_mod.log_feedback(
                env.schema_id,
                snap.has_ctx,
                snap.logits,
                snap.gram_deltas,
                snap.char_lps,
                shown_top1_idx,
                chosen_idx,
                p2c_shown,
                p2c_chosen,
                chosen_rank,
                env.feedback_path,
                snap.has_gram
            )
        end)
    end
end

function M.func(input, env)
    local ctx = env.engine.context
    if not env.ready or not native_mod or not is_rerank_enabled(ctx, env) or not native_mod.is_ready() then
        env.last_rerank = nil
        env.comp_ctx = nil
        if native_mod and not logged_init_error and type(native_mod.is_ready) == "function"
            and not native_mod.is_ready() and type(native_mod.stats) == "function"
        then
            local st = native_mod.stats()
            if type(st) == "table" and type(st.last_error) == "string" and #st.last_error > 0 then
                logged_init_error = true
                if log and log.warning then
                    log.warning("rime_neural: async init failed: " .. st.last_error)
                end
            end
        end
        for cand in input:iter() do
            yield(cand)
        end
        return
    end

    -- Lazily enable background UIA worker only after ONNX engine is ready and option is active
    if env.enable_uia and not env.uia_activated and type(native_mod.set_uia_enabled) == "function" then
        env.uia_activated = true
        native_mod.set_uia_enabled(true)
    end

    -- One-shot lazy `.gram` attachment once async init finishes (covers multi-schema startup races)
    if #env.gram_key > 0 and not attached_gram_keys[env.gram_key] then
        attached_gram_keys[env.gram_key] = true
        local ort_path = join_path(env.data_dir, "onnxruntime.dll")
        native_mod.init_async(
            env.data_dir,
            ort_path,
            env.threads,
            env.gram_paths,
            env.col_pen,
            env.non_col_pen,
            env.model_file or "",
            env.gram_max_len
        )
    end

    local cur_input = ctx.input or ""
    if #cur_input <= 1 then
        -- Brand-new composition: unfreeze previous composition's UIA snapshot and pre-warm UIA
        env.comp_ctx = nil
    end
    -- Pre-warm UIA snapshot asynchronously on early keystrokes (< 1 µs) so context is already
    -- fresh when keystroke 4+ triggers neural reranking (even after window switch or idle pause).
    if env.enable_uia and #cur_input <= 2 and type(native_mod.kick_uia_refresh) == "function" then
        native_mod.kick_uia_refresh()
    end

    -- Clear snapshot on every non-rerank transition EXCEPT when a trailing punctuation segment
    -- (`,`, `.`, `?`, `!`, `:`, `;`) immediately follows the reranked sentence right before auto-commit.
    local function maybe_clear_snapshot(is_trailing_punct, cand_start)
        local snap = env.last_rerank
        if not snap then
            return
        end
        if is_trailing_punct
            and cand_start == snap.seg_end
            and #cur_input > #snap.full_input
            and cur_input:sub(1, #snap.full_input) == snap.full_input
        then
            return
        end
        env.last_rerank = nil
    end

    -- Respect configured segment tags (default `abc`) so reverse-lookup / radical / stroke
    -- `script_translator` segments are never sent to the Mandarin pinyin reranker.
    local tags_ok, active_seg = segment_matches_tags(ctx, env.tags)
    if not tags_ok then
        local is_punct = active_seg ~= nil
            and type(active_seg.has_tag) == "function"
            and active_seg:has_tag("punct")
        maybe_clear_snapshot(is_punct, active_seg and active_seg.start or -1)
        for cand in input:iter() do
            yield(cand)
        end
        return
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
            local pre_ctx, is_blocked, comp_prefix = get_preceding_context(
                ctx,
                seg_start,
                first_preedit,
                raw_seg,
                sent_texts[1] or "",
                env
            )
            local res = native_mod.rerank(
                pre_ctx,
                raw_seg,
                env.layout,
                sent_texts,
                env.fusion_weights,
                env.p2c_cfg,
                env.gram_req
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
                        shown_texts[#shown_texts + 1] = sent_texts[idx]
                        if #emit_list == 1 and p2c_cand and res.p2c_pos == 2 then
                            emit_list[#emit_list + 1] = p2c_cand
                            shown_texts[#shown_texts + 1] = res.p2c_text
                        end
                    end
                end

                local has_ctx_flag = res.has_ctx
                if type(has_ctx_flag) ~= "boolean" then
                    has_ctx_flag = (utf8_char_count(trim_ws(pre_ctx)) >= 2)
                end

                -- Record snapshot BEFORE yielding (Rime Menu is lazy and pauses coroutine after page 1)
                if env.log_feedback then
                    env.last_rerank = {
                        full_input = cur_input,
                        seg_start = seg_start,
                        seg_end = seg_end,
                        comp_prefix = comp_prefix,
                        has_ctx = has_ctx_flag,
                        has_gram = (res.has_gram == true),
                        cands = sent_texts,
                        logits = res.logits or {},
                        gram_deltas = res.gram_deltas or {},
                        char_lps = res.char_lps or {},
                        p2c_text = (p2c_cand and res.p2c_text) or nil,
                        shown = shown_texts,
                        blocked = is_blocked,
                    }
                end

                for i = 1, #emit_list do
                    yield(emit_list[i])
                end
                return
            else
                env.last_rerank = nil
            end
        else
            env.last_rerank = nil
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
                maybe_clear_snapshot(g0 ~= nil and g0.type == "punct", cand.start)
                passthrough = true
                yield(cand)
            else
                seg_start = cand.start
                seg_end = cand._end
                first_preedit = cand.preedit or ""
                raw_seg = string.sub(cur_input, seg_start + 1, seg_end)
                local key_cnt = 0
                local tail_len = 0
                local has_non_pinyin = false
                local allow_semi = env.uses_semicolon
                for i = 1, #raw_seg do
                    local b = string.byte(raw_seg, i)
                    if (b >= 97 and b <= 122) or (allow_semi and b == 59) then
                        key_cnt = key_cnt + 1
                        tail_len = tail_len + 1
                    elseif b == 39 or b == 32 then
                        -- Explicit syllable delimiter (`'` or space) starts a new chunk
                        tail_len = 0
                    else
                        -- Uppercase ASCII, digits, or auxiliary-code delimiters (`/`, `;`, `-`)
                        has_non_pinyin = true
                        key_cnt = key_cnt + 1
                        tail_len = tail_len + 1
                    end
                end
                if has_non_pinyin
                    or key_cnt < env.min_input_len
                    or (env.is_double_pinyin and (tail_len % 2) == 1)
                then
                    env.last_rerank = nil
                    passthrough = true
                    yield(cand)
                else
                    local gtxt = (g0.text and #g0.text > 0) and g0.text or cand.text
                    sent_cands = { cand }
                    sent_texts = { gtxt }
                    seen_texts = { [gtxt] = true }
                end
            end
        else
            local g = cand:get_genuine()
            if #sent_cands < max_k
                and g and g.type == "sentence"
                and cand.start == seg_start and cand._end == seg_end
            then
                local gtxt = (g.text and #g.text > 0) and g.text or cand.text
                if not seen_texts[gtxt] then
                    seen_texts[gtxt] = true
                    sent_cands[#sent_cands + 1] = cand
                    sent_texts[#sent_texts + 1] = gtxt
                    if #sent_cands == max_k then
                        -- Flush immediately once pool reaches `max_k` instead of eagerly pulling candidate `max_k + 1`
                        flush_sentences()
                    end
                end
            else
                flush_sentences()
                yield(cand)
            end
        end
    end

    if is_first then
        env.last_rerank = nil
        env.comp_ctx = nil
    elseif not passthrough and not flushed and sent_cands then
        flush_sentences()
    end
end

function M.fini(env)
    if env.commit_conn then
        pcall(function() env.commit_conn:disconnect() end)
        env.commit_conn = nil
    end
    env.last_rerank = nil
    env.comp_ctx = nil
    env.norm_cache = nil
    env.t2s_override = nil
end

return M
