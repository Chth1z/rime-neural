-- rime-neural (v2.0.0): Universal ONNX neural sentence reranker filter for Rime / Weasel / Unix.
-- Loads native `rime_neural` shared library in-process via host Rime's exported Lua 5.4 C API.
-- Features:
--   * Zero custom librime required; works on official unmodified Weasel (librime >= 1.17.0) and Unix/macOS hosts.
--   * Universal layout auto-detection: Full Pinyin + 7 Double Pinyin layouts
--     (ZRM, Flypy, MSPY, Sogou, ABC, Ziguang, Jiajia/PYJJ) across rime-ice / rime-frost / rime-wanxiang.
--   * Non-blocking background initialization (`init_async`) -> 0 ms startup delay.
--   * Dynamic KV-cache ONNX architecture introspection (supports 12L ML-pumpkin-1103, 8L zhen-feather, etc.)
--     + selective ORT output fetching + per-schema isolated mmap `.gram` fusion + KV-cached Pass-2 P2C.
--   * Non-blocking background Windows UIA preceding-text context (frozen per composition,
--     clamped to focused control, focus-change invalidation, fail-closed password/terminal blocking),
--     plus opt-in 1,024-char focus-clamped document entity lexicon with zero KV prefix overhead.
--   * Automatic 1-edit typo-tolerant lattice recovery for proven-dead spellings (Full Pinyin & 7 Double Pinyin
--     layouts), strictly gated by live-prefix protection for zero clean-typing regression.
--   * Strict dictionary & tag protection: only reranks `sentence` candidates on matching segments (`abc`),
--     never overrides exact dictionary/userdb hits (`phrase`, `user_phrase`, `table`, `completion`, etc.).
--   * Traditional-to-Simplified context normalization (`tw2sp.json` / `t2s.json`) + pre-OpenCC commit capture.
--   * Zero-plaintext numeric local feedback logger (`neural/feedback.jsonl`) for unbiased MAP tuning.
--   * Graceful zero-cost pass-through if native library/model is absent or the configured toggle option is off.

local M = {}

local native_mod = nil
local native_loaded = false
local failed_dll_paths = {}
local attached_gram_keys = {}
local logged_init_error = false
local active_doc_entities_flag = nil

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

local IS_WINDOWS = (package.config and package.config:sub(1, 1) == "\\") or false
local PATH_SEP = IS_WINDOWS and "\\" or "/"

--- Normalize path separators to the host OS convention (`\` on Windows, `/` on Unix).
local function normalize_os_path(p)
    if type(p) ~= "string" then
        return ""
    end
    local s = IS_WINDOWS and p:gsub("/", "\\") or p:gsub("\\", "/")
    local min_len = IS_WINDOWS and 3 or 1
    while #s > min_len and s:sub(-1) == PATH_SEP do
        s = s:sub(1, -2)
    end
    return s
end

local function join_path(a, b)
    local base = normalize_os_path(a)
    if base:sub(-1) == PATH_SEP then
        return base .. b
    end
    return base .. PATH_SEP .. b
end

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end

local function resolve_native_lib_path(dir)
    if IS_WINDOWS then
        return join_path(dir, "rime_neural.dll")
    end
    for _, name in ipairs({ "librime_neural.dylib", "librime_neural.so", "rime_neural.so" }) do
        local p = join_path(dir, name)
        if file_exists(p) then
            return p
        end
    end
    return join_path(dir, "librime_neural.so")
end

local function resolve_ort_lib_path(dir)
    if IS_WINDOWS then
        return join_path(dir, "onnxruntime.dll")
    end
    for _, name in ipairs({ "libonnxruntime.dylib", "libonnxruntime.so", "libonnxruntime.so.1" }) do
        local p = join_path(dir, name)
        if file_exists(p) then
            return p
        end
    end
    return join_path(dir, "libonnxruntime.so")
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
            return normalize_os_path(shared_dir)
        end
    end
    return ""
end

local function is_temp_user_dir(user_dir)
    local ud = normalize_os_path(user_dir):lower()
    if IS_WINDOWS then
        if ud:find("\\appdata\\local\\temp\\", 1, true) ~= nil
            or ud:find("\\windows\\temp\\", 1, true) ~= nil
        then
            return true
        end
    elseif ud:sub(1, 5) == "/tmp/" or ud:sub(1, 9) == "/var/tmp/" then
        return true
    end
    for _, env_var in ipairs({ "TEMP", "TMP", "TMPDIR" }) do
        local tv = os.getenv(env_var)
        if type(tv) == "string" and #tv > 0 then
            local norm_tv = normalize_os_path(tv):lower()
            if ud == norm_tv or ud:sub(1, #norm_tv + 1) == (norm_tv .. PATH_SEP) then
                return true
            end
        end
    end
    return false
end

local function resolve_data_dir(user_dir, shared_dir, model_file, custom_data_dir)
    if custom_data_dir and #custom_data_dir > 0 then
        return normalize_os_path(custom_data_dir)
    end
    local primary = join_path(user_dir, "neural")
    if file_exists(resolve_native_lib_path(primary)) and has_any_model(primary, model_file) then
        return primary
    end
    -- Fallback to shared_data_dir/neural (typically ASCII Program Files or /usr/share path)
    if shared_dir and #shared_dir > 0 then
        local shared_neural = join_path(shared_dir, "neural")
        if file_exists(resolve_native_lib_path(shared_neural))
            and has_any_model(shared_neural, model_file)
        then
            return shared_neural
        end
    end
    -- Graceful fallback for users migrating from earlier `zhiyi/` folder layout
    local legacy = join_path(user_dir, "zhiyi")
    if file_exists(resolve_native_lib_path(legacy)) and has_any_model(legacy, model_file) then
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
    local ort_path = resolve_ort_lib_path(data_dir)
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

    local dll_path = resolve_native_lib_path(data_dir)
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
    package.loaded["_rime_neural_native"] = mod

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

local function append_commit_history(parts, context, max_entries)
    local limit = math.max(0, math.min(6, max_entries or 6))
    if limit == 0 then
        return
    end
    local hist = context.commit_history
    if not hist then
        return
    end
    if type(hist.to_table) == "function" then
        local tbl = hist:to_table()
        if type(tbl) == "table" and #tbl > 0 then
            local start_i = math.max(1, #tbl - limit + 1)
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
--- Convert optional document entity text to Simplified without polluting `env.norm_cache`.
local function normalize_doc_entities_once(text, is_trad, env)
    if not text or #text == 0 or is_pure_ascii(text) then
        return text or ""
    end
    local conv_obj = env.t2s_override
    if not conv_obj then
        conv_obj = is_trad and (get_shared_tw2sp() or get_shared_t2s()) or get_shared_t2s()
    end
    if conv_obj then
        local ok, conv = pcall(conv_obj.convert, conv_obj, text)
        if ok and conv then
            return conv
        end
    end
    return text
end

--- Returns `(model_context, allow_feedback, entity_context)`.
local function get_preceding_context(context, segment_start, first_preedit, raw_seg, first_cand_text, env)
    local comp_prefix = collect_composition_prefix(context, segment_start)
    local base_ctx = ""
    local entity_ctx = ""

    if env.enable_uia then
        if type(native_mod.query_uia_context) ~= "function" then
            return utf8_tail_bytes(comp_prefix, 384), false, ""
        end
        local full_preedit = preedit_before_caret(context, first_preedit)
        local frozen = env.comp_ctx
        local status, uia_text, hwnd, preedit_stripped, focus_epoch, doc_entities =
            native_mod.query_uia_context(full_preedit, raw_seg, first_cand_text)
        if status ~= "text" and status ~= "fallback" then
            env.comp_ctx = nil
            env.last_commit_focus = nil
            return status == "window_changed" and utf8_tail_bytes(comp_prefix, 384) or "", false, ""
        elseif frozen and frozen.base_ctx
            and frozen.hwnd == hwnd and frozen.focus_epoch == focus_epoch
        then
            base_ctx = frozen.base_ctx
            entity_ctx = frozen.entity_ctx or ""
        elseif status == "text" then
            local is_trad = is_traditional_mode(context)
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
            if env.enable_doc_entities and type(doc_entities) == "string" and #doc_entities > 0 then
                entity_ctx = normalize_doc_entities_once(utf8_tail_bytes(doc_entities, 3072), is_trad, env)
            end
            env.comp_ctx = {
                hwnd = hwnd,
                focus_epoch = focus_epoch,
                base_ctx = base_ctx,
                entity_ctx = entity_ctx,
            }
        else
            -- Non-TextPattern classified control (e.g. WeChat, QQ, IDE canvas): reuse Rime's
            -- session commit history only while (hwnd, focus_epoch) matches the last commit in
            -- this same control, preventing cross-control bleed on focus moves.
            local prev_commit = env.last_commit_focus
            if prev_commit and prev_commit.hwnd == hwnd and prev_commit.focus_epoch == focus_epoch
                and (prev_commit.count or 0) > 0
            then
                local hist_parts = {}
                append_commit_history(hist_parts, context, prev_commit.count)
                if #hist_parts > 0 then
                    local history = utf8_tail_bytes(table.concat(hist_parts), 384)
                    base_ctx = normalize_to_simplified(history, is_traditional_mode(context), env)
                end
                env.comp_ctx = { hwnd = hwnd, focus_epoch = focus_epoch, base_ctx = base_ctx }
            else
                env.comp_ctx = { hwnd = hwnd, focus_epoch = focus_epoch }
            end
        end
    else
        local hist_parts = {}
        append_commit_history(hist_parts, context)
        if #hist_parts > 0 then
            local history = utf8_tail_bytes(table.concat(hist_parts), 384)
            base_ctx = normalize_to_simplified(history, is_traditional_mode(context), env)
        end
    end

    if #base_ctx == 0 then
        return utf8_tail_bytes(comp_prefix, 384), true, entity_ctx
    end
    -- Deduplicate confirmed multi-segment prefix after Simplified normalization so Traditional UIA
    -- text (`我們`) and Simplified `comp_prefix` (`我们`) never double up.
    if #comp_prefix > 0 and base_ctx:sub(-#comp_prefix) ~= comp_prefix then
        return utf8_tail_bytes(base_ctx .. comp_prefix, 384), true, entity_ctx
    end
    return utf8_tail_bytes(base_ctx, 384), true, entity_ctx
end

-- OpenCC followed by uniquifier can merge distinct source sentences into one visible choice.
local function menu_source_text(cand, snap)
    if not cand or cand.start ~= snap.seg_start or cand._end ~= snap.seg_end then
        return nil
    end
    if type(cand.type) == "string" and cand.type:sub(1, 12) == "rime_neural_" then
        return nil
    end
    local sources = cand:get_genuines()
    if not sources or #sources == 0 then
        return nil
    end
    local text = sources[1].text
    for i = 2, #sources do
        if sources[i].text ~= text then
            return nil
        end
    end
    return text
end

-- The commit notifier runs before Rime clears the composition. Read the actual menu after
-- downstream filtering (including OpenCC and uniquifier), matching the selected input range.
local function committed_choice(ctx, snap)
    if not ctx.input or ctx.input:sub(1, snap.seg_end) ~= snap.input_prefix then
        return
    end
    local segs = ctx.composition and ctx.composition:toSegmentation()
    if not segs then
        return
    end
    for i = 0, segs.size - 1 do
        local seg = segs:get_at(i)
        if seg and seg.start == snap.seg_start then
            local cand = seg:get_selected_candidate()
            local chosen = menu_source_text(cand, snap)
            if not chosen then
                return
            end
            local top1 = menu_source_text(seg:get_candidate_at(0), snap)
            local p2c_shown = false
            if snap.p2c_text then
                p2c_shown = chosen == snap.p2c_text or top1 == snap.p2c_text
                if not p2c_shown and seg.menu then
                    -- Only inspect entries Rime has already prepared for this menu.
                    for j = 1, seg.menu:candidate_count() - 1 do
                        if menu_source_text(seg:get_candidate_at(j), snap) == snap.p2c_text then
                            p2c_shown = true
                            break
                        end
                    end
                end
            end
            return chosen, top1, seg.selected_index + 1, p2c_shown
        end
    end
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

local function clear_unless_trailing_punct(env, is_punct, start)
    local snap = env.last_rerank
    if snap and is_punct and start == snap.seg_end
        and env.engine.context.input:sub(1, start) == snap.input_prefix
    then
        return
    end
    env.last_rerank = nil
end

-- Lazy read-only dictionary translator used strictly on proven-dead typo inputs (zero cost on
-- clean typing, zero userdict learning or secondary-correction side effects).
local function get_typo_translator(env)
    if env.typo_tr_tried then
        return env.typo_tr
    end
    env.typo_tr_tried = true
    if not Component or type(Component.Translator) ~= "function" then
        return nil
    end
    local cfg = env.engine and env.engine.schema and env.engine.schema.config
    if cfg then
        local ok_ns, tr_ns = pcall(function()
            local dict = cfg:get_string("translator/dictionary") or ""
            if #dict == 0 then
                return nil
            end
            local ns = "rime_neural_lookup"
            cfg:set_string(ns .. "/dictionary", dict)
            local prism = cfg:get_string("translator/prism") or ""
            if #prism > 0 then
                cfg:set_string(ns .. "/prism", prism)
            end
            local packs = cfg:get_list("translator/packs")
            if packs and packs.size and packs.size > 0 then
                for i = 0, packs.size - 1 do
                    local pv = packs:get_value_at(i)
                    local ps = pv and pv:get_string()
                    if ps and #ps > 0 then
                        cfg:set_string(ns .. "/packs/@" .. i, ps)
                    end
                end
            end
            local ctx_sug = cfg:get_bool("translator/contextual_suggestions")
            if ctx_sug ~= nil then
                cfg:set_bool(ns .. "/contextual_suggestions", ctx_sug)
            end
            local max_hp = cfg:get_int("translator/max_homophones")
            cfg:set_int(ns .. "/max_homophones", (max_hp and max_hp > 1) and max_hp or 8)
            local max_hg = cfg:get_int("translator/max_homographs")
            if max_hg and max_hg > 0 then
                cfg:set_int(ns .. "/max_homographs", max_hg)
            end
            local max_st = cfg:get_int("translator/max_sentences")
            cfg:set_int(ns .. "/max_sentences", (max_st and max_st > 1) and max_st or 6)
            local cutoff = cfg:get_double("translator/sentence_cutoff_threshold")
            if cutoff ~= nil then
                cfg:set_double(ns .. "/sentence_cutoff_threshold", cutoff)
            end
            cfg:set_bool(ns .. "/enable_user_dict", false)
            cfg:set_bool(ns .. "/enable_sentence", true)
            cfg:set_bool(ns .. "/enable_completion", false)
            cfg:set_bool(ns .. "/enable_encoder", false)
            cfg:set_bool(ns .. "/enable_correction", false)
            return Component.Translator(env.engine, ns, "script_translator")
        end)
        if ok_ns and tr_ns then
            env.typo_tr = tr_ns
        end
    end
    return env.typo_tr
end

--- Fast-path check for automatic 1-edit typo suggestions on a raw segment.
--- Returns `nil` in ~1 µs on any clean or live-prefix input with zero UIA or translator calls.
local function find_broken_typo_fixes(env, raw_seg)
    if not env.enable_typo_recovery
        or type(native_mod.suggest_typo_inputs) ~= "function"
        or type(native_mod.score_gram) ~= "function"
    then
        return nil
    end
    if type(raw_seg) ~= "string" or #raw_seg < env.min_input_len or #raw_seg > 96 then
        return nil
    end
    local max_fixes = (env.is_double_pinyin and (#raw_seg % 2) == 1) and 24 or 12
    local fixes = native_mod.suggest_typo_inputs(raw_seg, env.layout, "broken", max_fixes)
    if type(fixes) ~= "table" or #fixes == 0 then
        return nil
    end
    return fixes
end

--- Automatic 1-edit typo recovery for proven-dead raw inputs (never fires on clean or live-prefix
--- typing). Returns a synthesized `Candidate` over `[seg_start, seg_end]` if a repaired spelling
--- passes `.gram` fluency + neural verification, or `nil` otherwise.
local function try_typo_recovery(
    env,
    fixes,
    raw_seg,
    seg_start,
    seg_end,
    first_preedit,
    orig_best_text,
    orig_is_full_coverage,
    pre_ctx,
    entity_ctx
)
    if type(fixes) ~= "table" or #fixes == 0 then
        return nil
    end
    local tr = get_typo_translator(env)
    if not tr then
        return nil
    end

    -- Query the schema's read-only translator for each 1-edit repaired spelling and collect
    -- full-span Chinese hits that individually pass `.gram` fluency gating.
    local best_fix = nil
    local best_fix_cands = nil
    local best_fix_score = -1e9
    local orig_g = env.non_col_pen or -4.0
    local n_orig_chars = math.max(4, math.min(16, utf8_char_count(orig_best_text or "")))
    local trunc_floor = (env.non_col_pen or -4.0) + 0.5

    if orig_is_full_coverage and type(orig_best_text) == "string" and #orig_best_text > 0 then
        local og, has_gram = native_mod.score_gram({ orig_best_text }, env.gram_req)
        if not has_gram then
            return nil
        end
        if type(og) == "table" and type(og[1]) == "number" then
            orig_g = og[1]
        end
    end

    local function cand_passes_fluency(g_val, fcost)
        if orig_is_full_coverage then
            local gain = g_val - orig_g
            local total_gain = gain * n_orig_chars
            local min_per_char = (orig_g < 1.0) and 0.30 or 0.35
            local min_total = (orig_g < 1.0) and 3.5 or (5.0 + 2.5 * fcost)
            return gain > min_per_char and total_gain > min_total
        end
        return g_val > trunc_floor
    end

    for i = 1, #fixes do
        local fix = fixes[i]
        local finp = fix and fix.input
        local fcost = (fix and fix.cost) or 1.0
        if type(finp) == "string" and #finp >= 2 then
            local qseg = Segment(0, #finp)
            qseg.tags = { abc = true }
            local ok_q, xlat = pcall(tr.query, tr, finp, qseg)
            if ok_q and xlat then
                local fcands, fseen = {}, {}
                local scanned = 0
                for c in xlat:iter() do
                    scanned = scanned + 1
                    local g = c:get_genuine()
                    if g and g.start == 0 and g._end == #finp
                        and (g.type == "sentence" or g.type == "phrase" or g.type == "user_phrase")
                    then
                        local txt = (g.text and #g.text > 0) and g.text or c.text
                        if txt and #txt > 0 and not is_pure_ascii(txt) and not fseen[txt] then
                            fseen[txt] = true
                            fcands[#fcands + 1] = txt
                            if #fcands >= 6 then
                                break
                            end
                        end
                    end
                    if scanned >= 48 then
                        break
                    end
                end
                if #fcands > 0 then
                    local gscores, has_gram = native_mod.score_gram(fcands, env.gram_req)
                    if not has_gram then
                        return nil
                    end
                    local eligible = {}
                    local top_g = -1e9
                    if type(gscores) == "table" then
                        for k = 1, #fcands do
                            local v = gscores[k]
                            if type(v) == "number" and cand_passes_fluency(v, fcost) then
                                eligible[#eligible + 1] = fcands[k]
                                if v > top_g then
                                    top_g = v
                                end
                            end
                        end
                    end
                    if #eligible > 0 then
                        local adj = top_g - 0.6 * fcost
                        if adj > best_fix_score then
                            best_fix_score = adj
                            best_fix = { input = finp, cost = fcost, top_g = top_g }
                            best_fix_cands = eligible
                        end
                    end
                end
            end
        end
    end

    if not best_fix or not best_fix_cands or #best_fix_cands == 0 then
        return nil
    end

    -- When the winning repaired spelling has multiple fluency-verified candidate sentences, use
    -- the 12-layer neural reranker to select the semantically best one in context.
    local winner = best_fix_cands[1]
    if #best_fix_cands >= 2 then
        local res = native_mod.rerank(
            pre_ctx or "",
            best_fix.input,
            env.layout,
            best_fix_cands,
            env.fusion_weights,
            { verify_margin = 0.0 },
            env.gram_req,
            entity_ctx or ""
        )
        if type(res) == "table" and type(res.order) == "table" and res.order[1] then
            winner = best_fix_cands[res.order[1]] or winner
        end
    end

    if winner == orig_best_text then
        return nil
    end
    local cand = Candidate("rime_neural_typo", seg_start, seg_end, winner, "")
    cand.preedit = (first_preedit and #first_preedit > 0) and first_preedit or raw_seg
    return cand
end

-- librime calls this for the segment whose menu it is building. composition:back() may
-- already be a later punctuation or reverse-lookup segment.
function M.tags_match(seg, env)
    for tag in pairs(env.tags) do
        if seg:has_tag(tag) then
            env.matched_seg_start = seg.start
            env.matched_seg_end = seg._end
            return true
        end
    end
    clear_unless_trailing_punct(env, seg:has_tag("punct"), seg.start)
    return false
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

    local user_dir = normalize_os_path(rime_api.get_user_data_dir())
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
    -- Always store feedback under writable `user_dir/neural/feedback.jsonl`.
    env.feedback_path = join_path(join_path(user_dir, "neural"), "feedback.jsonl")

    local uia_opt = cfg:get_bool(ns .. "/enable_uia_context")
    if uia_opt == nil then
        uia_opt = true
    end
    local allow_temp_uia = (cfg:get_bool(ns .. "/enable_uia_in_temp") == true)
    env.enable_uia = IS_WINDOWS and uia_opt and (not in_temp or allow_temp_uia)
    env.enable_doc_entities = (cfg:get_bool(ns .. "/enable_document_entities") == true) and env.enable_uia
    env.uia_activated = false

    local typo_opt = cfg:get_bool(ns .. "/enable_typo_recovery")
    if typo_opt == nil then
        typo_opt = true
    end
    env.enable_typo_recovery = typo_opt
    env.typo_tr = nil
    env.typo_tr_tried = false
    env.matched_seg_start = nil
    env.matched_seg_end = nil

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
    local layouts_map = package.loaded["_rime_neural_layouts"]
    if type(layouts_map) ~= "table" then
        layouts_map = {}
        package.loaded["_rime_neural_layouts"] = layouts_map
    end
    layouts_map[env.schema_id] = env.layout

    local gram_reqs_map = package.loaded["_rime_neural_gram_reqs"]
    if type(gram_reqs_map) ~= "table" then
        gram_reqs_map = {}
        package.loaded["_rime_neural_gram_reqs"] = gram_reqs_map
    end
    gram_reqs_map[env.schema_id] = env.gram_req

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
        -- Clear() also runs on cancellation, when there is no commit or new segment to filter.
        if env.engine.context.update_notifier then
            env.update_conn = env.engine.context.update_notifier:connect(function(ctx)
                if (env.comp_ctx or env.last_rerank) and not ctx:is_composing() then
                    env.comp_ctx = nil
                    env.last_rerank = nil
                    package.loaded["_rime_neural_last_ctx"] = nil
                end
            end)
        end
        env.commit_conn = env.engine.context.commit_notifier:connect(function(ctx)
            -- Always unfreeze per-composition UIA snapshot on commit so next sentence reads fresh text
            local prev_focus = env.comp_ctx
            env.comp_ctx = nil
            package.loaded["_rime_neural_last_ctx"] = nil

            local rerank_on = is_rerank_enabled(ctx, env)
            local uia_ok = env.enable_uia
                and rerank_on
                and not ctx:get_option("ascii_mode")
                and native_mod
                and native_mod.is_ready()

            -- Sample live focus once so both same-control fallback history and feedback verification
            -- share a single non-blocking check.
            local live_status, live_hwnd, live_epoch = nil, 0, 0
            if env.enable_uia and native_mod and type(native_mod.query_uia_context) == "function"
                and (prev_focus or (env.last_rerank and env.last_rerank.focus))
            then
                live_status, _, live_hwnd, _, live_epoch = native_mod.query_uia_context("", "", "")
            end
            local ctext = ctx:get_commit_text() or ""
            if prev_focus and #ctext > 0 and (live_status == "text" or live_status == "fallback")
                and live_hwnd == prev_focus.hwnd and live_epoch == prev_focus.focus_epoch
            then
                local prev_cf = env.last_commit_focus
                local cnt = (prev_cf and prev_cf.hwnd == live_hwnd and prev_cf.focus_epoch == live_epoch)
                    and math.min(6, (prev_cf.count or 0) + 1) or 1
                env.last_commit_focus = { hwnd = live_hwnd, focus_epoch = live_epoch, count = cnt }
            else
                env.last_commit_focus = nil
            end

            if uia_ok then
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
            if not snap or not rerank_on or not native_mod or type(native_mod.log_feedback) ~= "function" then
                return
            end
            if ctx:get_option("ascii_mode") then
                return
            end

            if snap.focus then
                if (live_status ~= "text" and live_status ~= "fallback")
                    or live_hwnd ~= snap.focus.hwnd or live_epoch ~= snap.focus.focus_epoch
                then
                    return
                end
            end
            local chosen, top1, chosen_rank, p2c_shown = committed_choice(ctx, snap)
            if not chosen then
                return
            end

            -- Derive purely numeric indices in memory; never persist user text, keystrokes, or UIA context.
            local chosen_idx = -1
            local shown_top1_idx = (p2c_shown and top1 == snap.p2c_text) and -2 or -1
            for i = 1, #snap.cands do
                if snap.cands[i] == chosen then
                    chosen_idx = i - 1
                end
                if snap.cands[i] == top1 then
                    shown_top1_idx = i - 1
                end
            end
            local p2c_chosen = chosen_idx < 0 and chosen == snap.p2c_text

            -- Log every eligible multi-candidate commit as zero-plaintext numeric features
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
        if active_doc_entities_flag == true and native_mod
            and type(native_mod.set_doc_entities_enabled) == "function"
        then
            active_doc_entities_flag = false
            native_mod.set_doc_entities_enabled(false)
        end
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

    -- Lazily enable background UIA worker only after ONNX engine is ready and option is active.
    -- Always sync `doc_entities` flag (including opt-out when switching to a non-UIA schema).
    local want_doc = (env.enable_uia and env.enable_doc_entities) == true
    if active_doc_entities_flag ~= want_doc and type(native_mod.set_doc_entities_enabled) == "function" then
        active_doc_entities_flag = want_doc
        native_mod.set_doc_entities_enabled(want_doc)
    end
    if env.enable_uia and not env.uia_activated and type(native_mod.set_uia_enabled) == "function" then
        env.uia_activated = true
        native_mod.set_uia_enabled(true)
    end

    -- One-shot lazy `.gram` attachment once async init finishes (covers multi-schema startup races)
    if #env.gram_key > 0 and not attached_gram_keys[env.gram_key] then
        attached_gram_keys[env.gram_key] = true
        local ort_path = resolve_ort_lib_path(env.data_dir)
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
    -- Pre-warm UIA snapshot asynchronously on early keystrokes (< 1 µs) so context is already
    -- fresh when keystroke 4+ triggers neural reranking (even after window switch or idle pause).
    if env.enable_uia and #cur_input <= 2 and type(native_mod.kick_uia_refresh) == "function" then
        native_mod.kick_uia_refresh()
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
            local pre_ctx, allow_feedback, entity_ctx = get_preceding_context(
                ctx,
                seg_start,
                first_preedit,
                raw_seg,
                sent_texts[1] or "",
                env
            )
            package.loaded["_rime_neural_last_ctx"] = {
                engine_id = tostring(env.engine),
                schema_id = env.schema_id,
                input_prefix = cur_input:sub(1, seg_end),
                pre_ctx = pre_ctx,
                hwnd = env.comp_ctx and env.comp_ctx.hwnd or 0,
                focus_epoch = env.comp_ctx and env.comp_ctx.focus_epoch or 0,
            }
            local res = native_mod.rerank(
                pre_ctx,
                raw_seg,
                env.layout,
                sent_texts,
                env.fusion_weights,
                env.p2c_cfg,
                env.gram_req,
                entity_ctx
            )
            if type(res) == "table" and type(res.order) == "table" and #res.order == #sent_cands then
                local top1_text = (res.p2c_pos == 1 and res.p2c_text) or sent_texts[res.order[1]] or sent_texts[1]
                -- Check automatic 1-edit typo recovery on proven-dead spellings (returns nil in ~1 µs on clean input).
                local fixes = find_broken_typo_fixes(env, raw_seg)
                local typo_cand = fixes and try_typo_recovery(
                    env,
                    fixes,
                    raw_seg,
                    seg_start,
                    seg_end,
                    first_preedit,
                    top1_text,
                    true,
                    pre_ctx,
                    entity_ctx
                )
                if typo_cand then
                    env.last_rerank = nil
                    yield(typo_cand)
                    for _, idx in ipairs(res.order) do
                        yield(sent_cands[idx])
                    end
                    return
                end

                local p2c_cand = nil
                if type(res.p2c_text) == "string" and #res.p2c_text > 0 and (res.p2c_pos == 1 or res.p2c_pos == 2) then
                    p2c_cand = Candidate("sentence", seg_start, seg_end, res.p2c_text, "")
                    p2c_cand.preedit = first_preedit
                end

                -- Record snapshot BEFORE yielding (Rime Menu is lazy and pauses coroutine after page 1)
                if env.log_feedback and allow_feedback then
                    local has_ctx_flag = res.has_ctx
                    if type(has_ctx_flag) ~= "boolean" then
                        has_ctx_flag = (utf8_char_count(trim_ws(pre_ctx)) >= 2)
                    end
                    env.last_rerank = {
                        input_prefix = cur_input:sub(1, seg_end),
                        seg_start = seg_start,
                        seg_end = seg_end,
                        focus = env.comp_ctx,
                        has_ctx = has_ctx_flag,
                        has_gram = (res.has_gram == true),
                        cands = sent_texts,
                        logits = res.logits or {},
                        gram_deltas = res.gram_deltas or {},
                        char_lps = res.char_lps or {},
                        p2c_text = (p2c_cand and res.p2c_text) or nil,
                    }
                else
                    env.last_rerank = nil
                end

                for i, idx in ipairs(res.order) do
                    if p2c_cand and i == res.p2c_pos then
                        yield(p2c_cand)
                    end
                    yield(sent_cands[idx])
                end
                return
            else
                env.last_rerank = nil
            end
        else
            env.last_rerank = nil
            -- Even a single full-span `sentence` candidate can be a Jianpin garbage parse of a typo.
            local fixes = (#sent_cands == 1) and find_broken_typo_fixes(env, raw_seg) or nil
            if fixes then
                local pre_ctx, _, entity_ctx = get_preceding_context(
                    ctx, seg_start, first_preedit, raw_seg, sent_texts[1] or "", env
                )
                local typo_cand = try_typo_recovery(
                    env, fixes, raw_seg, seg_start, seg_end, first_preedit, sent_texts[1] or "", true, pre_ctx, entity_ctx
                )
                if typo_cand then
                    yield(typo_cand)
                end
            end
        end
        for i = 1, #sent_cands do
            yield(sent_cands[i])
        end
    end

    local m_start = env.matched_seg_start
    local m_end = env.matched_seg_end
    env.matched_seg_start = nil
    env.matched_seg_end = nil

    for cand in input:iter() do
        if passthrough or flushed then
            yield(cand)
        elseif is_first then
            is_first = false
            local g0 = cand:get_genuine()
            -- If Rime truncated an unparseable segment (e.g. broken Double Pinyin pair `jntmtmqovfbuco`
            -- -> `今天天` covering `[0..6)` of `[0..14)`), check fast-path typo eligibility BEFORE
            -- touching UIA or translator. If no typo recovery fires (e.g. clean 9-key Double Pinyin
            -- live prefix with 8-key sentences), fall straight through to normal sentence reranking.
            local handled_trunc_typo = false
            if env.enable_typo_recovery
                and m_start and m_end and m_end > m_start
                and cand.start == m_start and cand._end < m_end
            then
                local raw_full = string.sub(cur_input, m_start + 1, m_end)
                local fixes = find_broken_typo_fixes(env, raw_full)
                if fixes then
                    local pre_ctx, _, entity_ctx = get_preceding_context(
                        ctx, m_start, cand.preedit or "", raw_full, cand.text or "", env
                    )
                    local typo_cand = try_typo_recovery(
                        env, fixes, raw_full, m_start, m_end, cand.preedit or "", cand.text or "", false, pre_ctx, entity_ctx
                    )
                    if typo_cand then
                        env.last_rerank = nil
                        passthrough = true
                        handled_trunc_typo = true
                        yield(typo_cand)
                        yield(cand)
                    end
                end
            end
            if handled_trunc_typo then
                -- Already emitted recovered candidate followed by native partial candidate.
            elseif not g0 or g0.type ~= "sentence" then
                clear_unless_trailing_punct(env, g0 ~= nil and g0.type == "punct", cand.start)
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
        -- Empty native stream on a matched `abc` segment (e.g. broken Double Pinyin pair at index 0)
        if env.enable_typo_recovery and m_start and m_end and m_end > m_start then
            local raw_full = string.sub(cur_input, m_start + 1, m_end)
            local fixes = find_broken_typo_fixes(env, raw_full)
            if fixes then
                local pre_ctx, _, entity_ctx = get_preceding_context(
                    ctx, m_start, raw_full, raw_full, "", env
                )
                local typo_cand = try_typo_recovery(
                    env, fixes, raw_full, m_start, m_end, raw_full, "", false, pre_ctx, entity_ctx
                )
                if typo_cand then
                    yield(typo_cand)
                end
            end
        end
    elseif not passthrough and not flushed and sent_cands then
        flush_sentences()
    end
end

function M.fini(env)
    if env.update_conn then
        pcall(function() env.update_conn:disconnect() end)
        env.update_conn = nil
    end
    if env.commit_conn then
        pcall(function() env.commit_conn:disconnect() end)
        env.commit_conn = nil
    end
    if active_doc_entities_flag == true and native_mod
        and type(native_mod.set_doc_entities_enabled) == "function"
    then
        active_doc_entities_flag = false
        native_mod.set_doc_entities_enabled(false)
    end
    local cached = package.loaded["_rime_neural_last_ctx"]
    if type(cached) == "table" and cached.engine_id == tostring(env.engine) then
        package.loaded["_rime_neural_last_ctx"] = nil
    end
    env.typo_tr = nil
    env.typo_tr_tried = false
    env.last_rerank = nil
    env.comp_ctx = nil
    env.last_commit_focus = nil
    env.norm_cache = nil
    env.t2s_override = nil
end

return M
