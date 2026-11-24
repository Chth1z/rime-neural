-- rime_neural_repair (v2.0.0): Opt-in pre-commit full-dictionary neural span repair and
-- literal-lock processor for Rime / Weasel (librime >= 1.17.0, Lua 5.4).
--
-- Mount at the front of `engine/processors` (opt-in):
--   patch:
--     "engine/processors/@before 0": lua_processor@*rime_neural_repair
--
-- Features:
--   * Constant-time idle path: normal typing pays at most two `KeyEvent:eq` comparisons.
--   * Pre-commit full-dictionary + 1-edit typo + neural-ranked span repair (`Alt+r` by default):
--     steps across word boundaries (`Left`/`Right`) without moving the physical caret, queries
--     the active schema's full dictionary plus 1-edit typo neighbors via a read-only
--     `script_translator`, ranks alternatives in full-sentence context with the shared
--     `rime_neural` engine, supports chained multi-word repairs pre-commit, and reverts cleanly
--     (`Escape` / `Alt+r`) to the exact pre-repair menu and selection.
--   * Pre-commit literal lock (`Alt+l` by default): seals verbatim ASCII identifiers
--     (`GPUv2`, `OAuth2`, `v1.2.1`, `rime-neural`, `CI/CD`, etc.) as a native confirmed `raw`
--     segment inside a Chinese composition; close with `Alt+l`, `Space`, or `Escape`.

local P = {}
local ACCEPT, NOOP = 1, 2
local LITERAL_CHAR_PAT = "^[A-Za-z0-9._/%-+]$"
local LITERAL_TOKEN_PAT = "^[A-Za-z0-9._/%-+]+$"

local function parse_key_option(cfg, path, default_repr)
    local repr = default_repr
    if cfg then
        local ok, val = pcall(function()
            return cfg:get_string(path)
        end)
        if ok and type(val) == "string" then
            repr = val
        end
    end
    if repr == "" or repr == "none" or repr == "disabled" then
        return nil
    end
    local ok, ev = pcall(KeyEvent, repr)
    if ok and ev and ev.keycode ~= 0 then
        return ev
    end
    local fb_ok, fb_ev = pcall(KeyEvent, default_repr)
    return (fb_ok and fb_ev and fb_ev.keycode ~= 0 and fb_ev) or nil
end

local function is_pure_ascii(s)
    return type(s) == "string" and #s > 0 and s:find("[\128-\255]") == nil
end

local function tail_segment(ctx)
    local comp = ctx.composition
    if comp and not comp:empty() then
        return comp:back()
    end
    return nil
end

local function get_shared_native()
    local m = package.loaded["_rime_neural_native"]
    if type(m) == "table" and type(m.rerank) == "function" then
        return m
    end
    return nil
end

local function get_schema_layout(e)
    local schema_id = e.engine and e.engine.schema and e.engine.schema.schema_id or ""
    local map = package.loaded["_rime_neural_layouts"]
    if type(map) == "table" and type(map[schema_id]) == "string" and #map[schema_id] > 0 then
        return map[schema_id]
    end
    local native = get_shared_native()
    if native and type(native.resolve_layout) == "function" then
        local layout = native.resolve_layout("auto", schema_id)
        if type(layout) == "string" and #layout > 0 then
            return layout
        end
    end
    if schema_id:find("double_pinyin", 1, true) ~= nil then
        return "zrm"
    end
    return "full"
end

local function get_schema_gram_req(e)
    local schema_id = e.engine and e.engine.schema and e.engine.schema.schema_id or ""
    local map = package.loaded["_rime_neural_gram_reqs"]
    if type(map) == "table" and type(map[schema_id]) == "table" then
        return map[schema_id]
    end
    return nil
end

--- Lazy read-only `script_translator` for full-dictionary span queries (zero userdict learning or
--- secondary-correction side effects; allocated only on the first `Alt+r` press).
local function get_lookup_translator(e)
    if e.lookup_tr_tried then
        return e.lookup_tr
    end
    e.lookup_tr_tried = true
    if not Component or type(Component.Translator) ~= "function" then
        return nil
    end
    local cfg = e.engine and e.engine.schema and e.engine.schema.config
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
            return Component.Translator(e.engine, ns, "script_translator")
        end)
        if ok_ns and tr_ns then
            e.lookup_tr = tr_ns
        end
    end
    return e.lookup_tr
end

local FALLBACK_TRAD_OPTS = {
    { "s2twp", "s2twp.json" },
    { "s2tw", "s2tw.json" },
    { "s2hk", "s2hk.json" },
    { "s2t", "s2t.json" },
    { "traditionalization", "s2t.json" },
    { "zh_trad", "s2t.json" },
    { "zh_tw", "s2twp.json" },
    { "zh_hk", "s2hk.json" },
    { "zh_hant", "s2t.json" },
}

--- Convert a Simplified draft or hint string through the schema's active output OpenCC converter
--- when Traditional output mode is toggled on (keeps source text intact for neural/.gram scoring).
local function format_output_text(e, ctx, text)
    if type(text) ~= "string" or #text == 0 or is_pure_ascii(text) then
        return text or ""
    end
    local cfg = e.engine and e.engine.schema and e.engine.schema.config
    local opencc_cfg = nil
    if cfg then
        local opt = cfg:get_string("simplifier/option_name")
        if not opt or #opt == 0 then
            opt = "traditionalization"
        end
        if ctx:get_option(opt) == true then
            opencc_cfg = cfg:get_string("simplifier/opencc_config")
            if not opencc_cfg or #opencc_cfg == 0 then
                opencc_cfg = "s2t.json"
            end
        end
    end
    if not opencc_cfg then
        for _, pair in ipairs(FALLBACK_TRAD_OPTS) do
            if ctx:get_option(pair[1]) == true then
                opencc_cfg = pair[2]
                break
            end
        end
    end
    if not opencc_cfg then
        return text
    end
    if e.out_opencc_cfg ~= opencc_cfg then
        e.out_opencc_cfg = opencc_cfg
        local ok, occ = pcall(Opencc, opencc_cfg)
        e.out_opencc = (ok and occ) or nil
    end
    if e.out_opencc then
        local ok_c, conv = pcall(e.out_opencc.convert, e.out_opencc, text)
        if ok_c and type(conv) == "string" and #conv > 0 then
            return conv
        end
    end
    return text
end

local function split_edge_delims(raw)
    local pre, core, post = raw:match("^([' ]*)(.-)([' ]*)$")
    if not core then
        return "", raw, ""
    end
    return pre or "", core, post or ""
end

local function clone_words(words)
    local out = {}
    for i = 1, #words do
        local w = words[i]
        out[i] = {
            lo = w.lo,
            hi = w.hi,
            sep_before = w.sep_before,
            typed_raw = w.typed_raw,
            lookup_raw = w.lookup_raw,
            sep_after = w.sep_after,
            source_text = w.source_text,
            draft_text = w.draft_text,
            code = w.code,
        }
    end
    return out
end

local function join_draft_text(words)
    local t = {}
    for i = 1, #words do
        t[i] = words[i].draft_text
    end
    return table.concat(t)
end

local function join_lookup_raw(words, override_idx, override_raw)
    local t = {}
    for i = 1, #words do
        local w = words[i]
        local core = (i == override_idx and override_raw) or w.lookup_raw
        t[i] = w.sep_before .. core .. w.sep_after
    end
    return table.concat(t)
end

--- Extract exact word boundaries, raw spelling slices, and dictionary codes from a genuine
--- `Sentence` (or standalone `Phrase` / `user_phrase`) candidate.
local function extract_sentence_words(cand, full_input)
    if not cand then
        return nil
    end
    local g = cand:get_genuine()
    if not g then
        return nil
    end
    local s = g:to_sentence()
    if s then
        local entries, widths = s.entrys, s.word_lengths
        if not entries or not widths or #entries == 0 or #entries ~= #widths then
            return nil
        end

        local parts, texts, pos = {}, {}, cand.start
        for i, entry in ipairs(entries) do
            local n = widths[i]
            if not n or n <= 0 or not entry or entry.text == "" then
                return nil
            end
            local raw_slice = full_input and full_input:sub(pos + 1, pos + n) or ""
            local sep_b, core, sep_a = split_edge_delims(raw_slice)
            local code_str = (entry.code and entry.code:print()) or ""
            parts[i] = {
                lo = pos,
                hi = pos + n,
                sep_before = sep_b,
                typed_raw = core,
                lookup_raw = core,
                sep_after = sep_a,
                source_text = entry.text,
                draft_text = entry.text,
                code = code_str,
            }
            texts[i] = entry.text
            pos = pos + n
        end
        if pos == cand._end and table.concat(texts) == g.text then
            return parts, s.lang_name
        end
        return nil
    end

    if (g.type == "phrase" or g.type == "user_phrase")
        and cand._end > cand.start
        and g.text and #g.text > 0 and not is_pure_ascii(g.text)
    then
        local raw_slice = full_input and full_input:sub(cand.start + 1, cand._end) or ""
        local sep_b, core, sep_a = split_edge_delims(raw_slice)
        if #core >= 2 then
            return {
                {
                    lo = cand.start,
                    hi = cand._end,
                    sep_before = sep_b,
                    typed_raw = core,
                    lookup_raw = core,
                    sep_after = sep_a,
                    source_text = g.text,
                    draft_text = g.text,
                    code = "",
                },
            }, ""
        end
    end
    return nil
end

--- Lightweight preceding context for neural sentence ranking during `Alt+r`.
--- Validates session identity and live focus epoch before reusing the filter's authorized context,
--- and falls back strictly to confirmed segments inside the current composition (never unscoped
--- `commit_history`, which could leak text across controls or into password fields).
local function get_repair_context(e, ctx, seg_start, seg_end)
    local schema_id = e.engine and e.engine.schema and e.engine.schema.schema_id or ""
    local cached = package.loaded["_rime_neural_last_ctx"]
    if type(cached) == "table"
        and cached.engine_id == tostring(e.engine)
        and cached.schema_id == schema_id
        and cached.input_prefix == (ctx.input or ""):sub(1, seg_end)
        and type(cached.pre_ctx) == "string"
    then
        local focus_ok = true
        if cached.focus_epoch and cached.focus_epoch ~= 0 then
            local native = get_shared_native()
            if native and type(native.query_uia_context) == "function" then
                local st, _, hwnd, _, epoch = native.query_uia_context("", "", "")
                if (st ~= "text" and st ~= "fallback")
                    or hwnd ~= cached.hwnd
                    or epoch ~= cached.focus_epoch
                then
                    focus_ok = false
                end
            else
                focus_ok = false
            end
        end
        if focus_ok then
            return cached.pre_ctx
        end
    end
    local parts = {}
    if seg_start > 0 and ctx.composition and type(ctx.composition.toSegmentation) == "function" then
        local ok_s, segs = pcall(ctx.composition.toSegmentation, ctx.composition)
        if ok_s and segs then
            for i = 0, segs.size - 1 do
                local g = segs:get_at(i)
                if g and g._end <= seg_start then
                    local c = g:get_selected_candidate()
                    local gen = c and c:get_genuine()
                    local txt = (gen and gen.text and #gen.text > 0 and gen.text) or (c and c.text) or ""
                    if #txt > 0 then
                        parts[#parts + 1] = txt
                    end
                end
            end
        end
    end
    return table.concat(parts)
end

--- Query the read-only `script_translator` for exact-span Chinese matches of `raw_span`.
local function query_span_words(tr, raw_span, max_scan, max_collect)
    local out = {}
    if not tr or type(raw_span) ~= "string" or #raw_span == 0 then
        return out
    end
    local qseg = Segment(0, #raw_span)
    qseg.tags = { abc = true }
    local ok_q, xlat = pcall(tr.query, tr, raw_span, qseg)
    if not ok_q or not xlat then
        return out
    end
    local seen = {}
    local scanned = 0
    for c in xlat:iter() do
        scanned = scanned + 1
        local g = c:get_genuine()
        if g and g.start == 0 and g._end == #raw_span
            and (g.type == "phrase" or g.type == "user_phrase" or g.type == "sentence")
        then
            local txt = (g.text and #g.text > 0) and g.text or c.text
            if txt and #txt > 0 and not is_pure_ascii(txt) and not seen[txt] then
                seen[txt] = true
                out[#out + 1] = txt
                if #out >= max_collect then
                    break
                end
            end
        end
        if scanned >= max_scan then
            break
        end
    end
    return out
end

--- Collect and rank full-dictionary + 1-edit typo alternatives for `s.words[s.target]`.
local function collect_alternatives(e, ctx, s)
    local base, target = s.words, s.target
    local w = base[target]
    if not w then
        return {}
    end
    local cur_text = w.draft_text

    local left_parts, right_parts = {}, {}
    for i = 1, target - 1 do
        left_parts[#left_parts + 1] = base[i].draft_text
    end
    for i = target + 1, #base do
        right_parts[#right_parts + 1] = base[i].draft_text
    end
    local left_draft = table.concat(left_parts)
    local right_draft = table.concat(right_parts)
    local cur_full = left_draft .. cur_text .. right_draft

    local exact_rows = {}
    local seen = { [cur_text] = true }
    local next_rank = 0

    -- Tier A: Native menu homophones (only when unchained so `menu_index` maps 1:1 to `orig_menu`).
    if s.edits == 0 and s.orig_menu and w.code ~= "" then
        for index = 0, 31 do
            local c = s.orig_menu:get_candidate_at(index)
            if not c then
                break
            end
            local other, lang = extract_sentence_words(c, s.input)
            if other and lang == s.lang and #other == #base then
                local matches = true
                for i, ow in ipairs(other) do
                    local bw = base[i]
                    if ow.lo ~= bw.lo or ow.hi ~= bw.hi or ow.code ~= bw.code
                        or (i ~= target and ow.source_text ~= bw.draft_text)
                    then
                        matches = false
                        break
                    end
                end
                local text = other[target].source_text
                if matches and not seen[text] then
                    seen[text] = true
                    next_rank = next_rank + 1
                    exact_rows[#exact_rows + 1] = {
                        text = text,
                        lookup_raw = w.typed_raw,
                        menu_index = index,
                        is_typo = false,
                        edit_cost = 0.0,
                        dict_rank = next_rank,
                    }
                    if #exact_rows >= 24 then
                        break
                    end
                end
            end
        end
    end

    -- Tier B: Full schema dictionary matches for the word's exact spelling (and current lookup spelling).
    local tr = get_lookup_translator(e)
    if tr and #w.typed_raw > 0 then
        local spans = { w.typed_raw }
        if #w.lookup_raw > 0 and w.lookup_raw ~= w.typed_raw then
            spans[#spans + 1] = w.lookup_raw
        end
        for _, sp in ipairs(spans) do
            local hits = query_span_words(tr, sp, 48, 24)
            for _, txt in ipairs(hits) do
                if not seen[txt] then
                    seen[txt] = true
                    next_rank = next_rank + 1
                    exact_rows[#exact_rows + 1] = {
                        text = txt,
                        lookup_raw = sp,
                        menu_index = nil,
                        is_typo = false,
                        edit_cost = 0.0,
                        dict_rank = next_rank,
                    }
                    if #exact_rows >= 24 then
                        break
                    end
                end
            end
            if #exact_rows >= 24 then
                break
            end
        end
    end

    -- Tier C: 1-edit span typo alternatives (e.g. ZRM `gsye -> gsyr -> 公园`, Full `jintan -> jintian`).
    local native = get_shared_native()
    local layout = get_schema_layout(e)
    local gram_req = get_schema_gram_req(e)
    local typo_groups = {}
    if tr and native and type(native.suggest_typo_inputs) == "function"
        and #w.typed_raw >= 2 and #w.typed_raw <= 24
    then
        local fixes = native.suggest_typo_inputs(w.typed_raw, layout, "span", 20)
        if type(fixes) == "table" then
            for _, fix in ipairs(fixes) do
                local finp = fix and fix.input
                local fcost = (fix and fix.cost) or 1.0
                if type(finp) == "string" and #finp >= 2 and finp ~= w.typed_raw then
                    local hits = query_span_words(tr, finp, 24, 4)
                    local g_rows = {}
                    for _, txt in ipairs(hits) do
                        if not seen[txt] then
                            seen[txt] = true
                            next_rank = next_rank + 1
                            g_rows[#g_rows + 1] = {
                                text = txt,
                                lookup_raw = finp,
                                menu_index = nil,
                                is_typo = true,
                                edit_cost = fcost,
                                dict_rank = next_rank,
                            }
                        end
                    end
                    if #g_rows > 0 then
                        typo_groups[#typo_groups + 1] = {
                            input = finp,
                            cost = fcost,
                            rows = g_rows,
                        }
                    end
                end
            end
        end
    end

    -- If native neural engine is unavailable, return exact-spelling dictionary rows + typo rows cleanly.
    if not native or type(native.is_ready) ~= "function" or not native.is_ready() then
        local flat = {}
        for _, r in ipairs(exact_rows) do
            flat[#flat + 1] = r
            if #flat >= 9 then
                return flat
            end
        end
        for _, g in ipairs(typo_groups) do
            for _, r in ipairs(g.rows) do
                flat[#flat + 1] = r
                if #flat >= 9 then
                    return flat
                end
            end
        end
        return flat
    end

    -- Tier D: Score in full-sentence context with `.gram` + 12-layer neural cross-encoder.
    local pre_ctx = get_repair_context(e, ctx, s.seg_start, s.seg_end)
    local has_gram_fn = (type(native.score_gram) == "function")
    local has_active_gram = false
    local cur_gram = -12.0
    local best_exact_gram = -15.0

    -- 1. Pre-score current draft + all exact-spelling sentences with `.gram` (microseconds).
    if has_gram_fn then
        local batch = { cur_full }
        for i = 1, #exact_rows do
            batch[i + 1] = left_draft .. exact_rows[i].text .. right_draft
        end
        local gs, hg = native.score_gram(batch, gram_req)
        has_active_gram = (hg == true)
        if has_active_gram and type(gs) == "table" then
            if type(gs[1]) == "number" then
                cur_gram = gs[1]
            end
            for i = 1, #exact_rows do
                local v = type(gs[i + 1]) == "number" and gs[i + 1] or -15.0
                exact_rows[i].gram = v
                if v > best_exact_gram then
                    best_exact_gram = v
                end
            end
        end
    end

    -- Narrow exact-spelling pool to top 7 finalists before the single neural forward pass.
    local exact_finalists = exact_rows
    if #exact_rows > 7 then
        local sorted = {}
        for i = 1, #exact_rows do
            sorted[i] = exact_rows[i]
        end
        table.sort(sorted, function(a, b)
            local sa = (a.gram or -15.0) + (a.menu_index ~= nil and 0.6 or 0.0)
            local sb = (b.gram or -15.0) + (b.menu_index ~= nil and 0.6 or 0.0)
            if sa ~= sb then
                return sa > sb
            end
            return a.dict_rank < b.dict_rank
        end)
        exact_finalists = {}
        for i = 1, 7 do
            exact_finalists[i] = sorted[i]
        end
    end

    -- One 12-layer neural pass over exact-spelling homophones under the true spelling.
    if #exact_finalists >= 1 then
        local cands = { cur_full }
        for i = 1, #exact_finalists do
            cands[i + 1] = left_draft .. exact_finalists[i].text .. right_draft
        end
        local eff_raw = join_lookup_raw(base, target, w.typed_raw)
        local res = native.rerank(
            pre_ctx,
            eff_raw,
            layout,
            cands,
            { 1.0, 0.0, 0.05, 0.0, 0.0 },
            { verify_margin = 0.0 },
            gram_req,
            ""
        )
        if type(res) == "table" and type(res.logits) == "table" then
            for i = 1, #exact_finalists do
                local lg = type(res.logits[i + 1]) == "number" and res.logits[i + 1] or -10.0
                local gd = (type(res.gram_deltas) == "table" and type(res.gram_deltas[i + 1]) == "number")
                    and res.gram_deltas[i + 1] or 0.0
                exact_finalists[i].score = lg + 0.05 * gd
            end
            table.sort(exact_finalists, function(a, b)
                if a.score ~= b.score then
                    return a.score > b.score
                end
                return a.dict_rank < b.dict_rank
            end)
        end
    end

    -- 2. Score 1-edit typo groups via `.gram`; neutrally rerank the winning typo spelling group
    --    under its OWN corrected spelling so pinyin conditioning matches (`gsye -> gsyr -> 公园`).
    local promoted_typos = {}
    local tail_typos = {}
    if #typo_groups > 0 and has_active_gram then
        local flat_typo = {}
        local flat_sents = {}
        for _, g in ipairs(typo_groups) do
            for _, r in ipairs(g.rows) do
                flat_typo[#flat_typo + 1] = r
                flat_sents[#flat_sents + 1] = left_draft .. r.text .. right_draft
            end
        end
        local tgs = native.score_gram(flat_sents, gram_req)
        for i = 1, #flat_typo do
            flat_typo[i].gram = (type(tgs) == "table" and type(tgs[i]) == "number") and tgs[i] or -15.0
        end

        -- Find the highest-fluency typo spelling group.
        local best_g = nil
        local best_g_adj = -1e9
        for _, g in ipairs(typo_groups) do
            local top_v = -15.0
            for _, r in ipairs(g.rows) do
                if r.gram > top_v then
                    top_v = r.gram
                end
            end
            g.top_gram = top_v
            local adj = top_v - 1.1 * g.cost
            if adj > best_g_adj then
                best_g_adj = adj
                best_g = g
            end
        end

        -- Use 12-layer neural scoring inside the winning typo spelling group when >= 2 words compete.
        if best_g and #best_g.rows >= 2 and best_g.top_gram > cur_gram + 0.5 then
            local tcands = {}
            for i = 1, #best_g.rows do
                tcands[i] = left_draft .. best_g.rows[i].text .. right_draft
            end
            local rep_raw = join_lookup_raw(base, target, best_g.input)
            local tres = native.rerank(
                pre_ctx,
                rep_raw,
                layout,
                tcands,
                { 1.0, 0.0, 0.05, 0.0, 0.0 },
                { verify_margin = 0.0 },
                gram_req,
                ""
            )
            if type(tres) == "table" and type(tres.logits) == "table" then
                local max_lg = -1e9
                for i = 1, #best_g.rows do
                    local lg = type(tres.logits[i]) == "number" and tres.logits[i] or -10.0
                    best_g.rows[i].neural_logit = lg
                    if lg > max_lg then
                        max_lg = lg
                    end
                end
                for i = 1, #best_g.rows do
                    -- Relative neural penalty inside the same typo spelling ensures semantic winner
                    -- (`公园` over `公元`) leads cleanly.
                    best_g.rows[i].typo_rank_score = best_g.rows[i].gram + 0.25 * (best_g.rows[i].neural_logit - max_lg)
                end
                table.sort(best_g.rows, function(a, b)
                    if a.typo_rank_score ~= b.typo_rank_score then
                        return a.typo_rank_score > b.typo_rank_score
                    end
                    return a.dict_rank < b.dict_rank
                end)
            end
        end

        -- Promote typo candidates ahead of exact-spelling rows only when fluency decisively beats
        -- both the current draft and the best exact-spelling alternative.
        local promo_floor = math.max(cur_gram, best_exact_gram) + 0.25
        for _, r in ipairs(flat_typo) do
            local rscore = (r.typo_rank_score or r.gram) - 1.1 * r.edit_cost
            r.sort_adj = rscore
            if rscore > promo_floor then
                promoted_typos[#promoted_typos + 1] = r
            elseif r.gram > cur_gram - 2.0 then
                tail_typos[#tail_typos + 1] = r
            end
        end
        table.sort(promoted_typos, function(a, b)
            if a.sort_adj ~= b.sort_adj then
                return a.sort_adj > b.sort_adj
            end
            return a.dict_rank < b.dict_rank
        end)
        table.sort(tail_typos, function(a, b)
            if a.sort_adj ~= b.sort_adj then
                return a.sort_adj > b.sort_adj
            end
            return a.dict_rank < b.dict_rank
        end)
    end

    -- Assemble final top-9 list: decisively better typo fixes first, then neural-ranked exact
    -- homophones, then up to 2 fallback fluent typo neighbors if slots remain.
    local out = {}
    local max_promo = math.min(3, #promoted_typos)
    for i = 1, max_promo do
        out[#out + 1] = promoted_typos[i]
    end
    for i = 1, #exact_finalists do
        if #out >= 9 then
            break
        end
        out[#out + 1] = exact_finalists[i]
    end
    local added_tail = 0
    for i = 1, #tail_typos do
        if #out >= 9 or added_tail >= 2 then
            break
        end
        out[#out + 1] = tail_typos[i]
        added_tail = added_tail + 1
    end
    return out
end

--- Borrow the active segment's menu to show the focused word and numbered alternatives in the
--- candidate comment while preserving the current draft text for safe fallback commit.
local function show_repair_menu(e, ctx, s, seg)
    local rows = collect_alternatives(e, ctx, s)
    local cur_draft = join_draft_text(s.words)
    local hints = { "[" .. format_output_text(e, ctx, s.words[s.target].draft_text) .. "]" }
    for i, row in ipairs(rows) do
        hints[#hints + 1] = i .. " " .. format_output_text(e, ctx, row.text)
    end
    if #rows == 0 then
        hints[#hints + 1] = "(no alternate)"
    end

    local comment = table.concat(hints, "  ")
    local row
    if s.edits == 0 then
        row = ShadowCandidate(s.prev_cand, s.id, s.prev_cand.text, comment)
    else
        row = Candidate(s.id, s.seg_start, s.seg_end, format_output_text(e, ctx, cur_draft), comment)
        row.preedit = s.orig_preedit
    end

    local menu = Menu()
    menu:add_translation(Translation(function()
        yield(row)
    end))
    if menu:prepare(1) ~= 1 then
        return false
    end
    seg.menu = menu
    seg.selected_index = 0
    s.rows = rows
    return true
end

local function start_repair(e, ctx)
    local seg = tail_segment(ctx)
    if not seg or seg.status ~= "kGuess" or not seg:has_tag("abc") then
        return NOOP
    end
    local c = seg:get_selected_candidate()
    if not c or c.start ~= seg.start or c._end ~= seg._end then
        return NOOP
    end
    local pos = ctx.caret_pos
    if pos < c.start or pos > c._end then
        return NOOP
    end

    local draft = e.last_draft
    local state
    if draft
        and c.type == draft.id
        and ctx.input == draft.input
        and pos == draft.pos
        and c.start == draft.seg_start
        and c._end == draft.seg_end
    then
        -- Chained multi-word repair on an already-repaired pre-commit draft.
        e.serial = (e.serial or 0) + 1
        local cloned = clone_words(draft.words)
        local target = math.max(1, math.min(#cloned, draft.last_target or #cloned))
        state = {
            id = "rime_neural_repair_" .. e.serial,
            input = ctx.input,
            pos = pos,
            seg_start = draft.seg_start,
            seg_end = draft.seg_end,
            orig_source = draft.orig_source,
            orig_menu = draft.orig_menu,
            orig_index = draft.orig_index,
            orig_preedit = draft.orig_preedit,
            prev_cand = c,
            prev_menu = seg.menu,
            prev_index = seg.selected_index,
            words = cloned,
            lang = draft.lang,
            edits = draft.edits,
            target = target,
        }
    else
        local parts, lang = extract_sentence_words(c, ctx.input)
        if not parts then
            return NOOP
        end
        e.last_draft = nil
        -- Target the word containing the caret; at a boundary, prefer the word immediately to its left.
        local target = 1
        while target < #parts and pos > parts[target].hi do
            target = target + 1
        end
        e.serial = (e.serial or 0) + 1
        state = {
            id = "rime_neural_repair_" .. e.serial,
            input = ctx.input,
            pos = pos,
            seg_start = seg.start,
            seg_end = seg._end,
            orig_source = c,
            orig_menu = seg.menu,
            orig_index = seg.selected_index,
            orig_preedit = c.preedit or ctx.input:sub(seg.start + 1, seg._end),
            prev_cand = c,
            prev_menu = seg.menu,
            prev_index = seg.selected_index,
            words = parts,
            lang = lang,
            edits = 0,
            target = target,
        }
    end

    if not show_repair_menu(e, ctx, state, seg) then
        return NOOP
    end
    e.repair = state
    return ACCEPT
end

local function apply_repair_choice(e, ctx, s, seg, chosen)
    local w = s.words[s.target]
    w.draft_text = chosen.text
    w.lookup_raw = chosen.lookup_raw or w.typed_raw
    local full_draft = join_draft_text(s.words)

    -- Restoring the exact original sentence returns cleanly to the native menu.
    local orig_gen = s.orig_source and s.orig_source:get_genuine()
    local orig_text = (orig_gen and orig_gen.text) or (s.orig_source and s.orig_source.text) or ""
    if full_draft == orig_text then
        seg.menu = s.orig_menu
        seg.selected_index = s.orig_index
        e.last_draft = nil
        e.repair = nil
        return ACCEPT
    end

    -- Unchained selection matching an existing native menu sentence keeps the native `Sentence`.
    if s.edits == 0 and chosen.menu_index ~= nil then
        local mc = s.orig_menu:get_candidate_at(chosen.menu_index)
        local mcg = mc and mc:get_genuine()
        if mc and ((mcg and mcg.text == full_draft) or mc.text == full_draft) then
            seg.menu = s.orig_menu
            seg.selected_index = chosen.menu_index
            e.last_draft = nil
            e.repair = nil
            return ACCEPT
        end
    end

    -- Emit a clean pre-commit repaired candidate without wrapping `orig_source` with altered text,
    -- applying active OpenCC conversion (e.g. Traditional output) to the visible/committed string.
    e.serial = (e.serial or 0) + 1
    local draft_id = "rime_neural_draft_" .. e.serial
    local display_draft = format_output_text(e, ctx, full_draft)
    local cand = Candidate(draft_id, s.seg_start, s.seg_end, display_draft, "")
    cand.preedit = s.orig_preedit

    local menu = Menu()
    menu:add_translation(Translation(function()
        yield(cand)
    end))
    if menu:prepare(1) ~= 1 then
        seg.menu = s.prev_menu
        seg.selected_index = s.prev_index
        e.repair = nil
        return ACCEPT
    end

    seg.menu = menu
    seg.selected_index = 0
    e.last_draft = {
        id = draft_id,
        input = s.input,
        pos = s.pos,
        seg_start = s.seg_start,
        seg_end = s.seg_end,
        orig_source = s.orig_source,
        orig_menu = s.orig_menu,
        orig_index = s.orig_index,
        orig_preedit = s.orig_preedit,
        lang = s.lang,
        words = s.words,
        edits = s.edits + 1,
        last_target = s.target,
    }
    e.repair = nil
    return ACCEPT
end

local function handle_repair_key(e, ctx, key)
    local s = e.repair
    local seg = tail_segment(ctx)
    local c = seg and seg:get_selected_candidate()
    if not c or c.type ~= s.id or seg.status ~= "kGuess" then
        e.repair = nil
        return NOOP
    end
    if ctx.input ~= s.input or ctx.caret_pos ~= s.pos then
        seg.menu = s.prev_menu
        seg.selected_index = s.prev_index
        e.repair = nil
        return NOOP
    end

    local k = key:repr()
    if k == "Left" or k == "Right" then
        local old = s.target
        local step = (k == "Left") and -1 or 1
        s.target = math.max(1, math.min(#s.words, old + step))
        if s.target ~= old and not show_repair_menu(e, ctx, s, seg) then
            s.target = old
        end
        return ACCEPT
    end

    if (e.repair_key and key:eq(e.repair_key)) or k == "Escape" then
        seg.menu = s.prev_menu
        seg.selected_index = s.prev_index
        e.repair = nil
        return ACCEPT
    end

    -- Keep repair mode open across bare modifier presses (e.g. pressing Alt before Alt+r toggle).
    local ch = key.keycode
    if ch == 0xffe1 or ch == 0xffe2 or ch == 0xffe3 or ch == 0xffe4
        or ch == 0xffe9 or ch == 0xffea or ch == 0xffeb or ch == 0xffec
    then
        return ACCEPT
    end

    local n = (k == "space") and 1 or tonumber(k:match("^[1-9]$") or "")
    if n then
        if not s.rows[n] then
            return ACCEPT
        end
        return apply_repair_choice(e, ctx, s, seg, s.rows[n])
    end

    -- Any other key cleanly restores the pre-repair menu before passing through.
    seg.menu = s.prev_menu
    seg.selected_index = s.prev_index
    e.repair = nil
    return NOOP
end

local function finish_literal(e)
    if e.watch then
        pcall(function()
            e.watch:disconnect()
        end)
    end
    e.watch = nil
    e.literal = nil
end

local function seal_literal(ctx, l)
    local comp = ctx.composition
    local segs = comp:toSegmentation()
    segs:reset_length(l.keep)
    if #ctx.input > l.lo then
        local raw = Segment(l.lo, #ctx.input)
        raw.tags = { raw = true }
        raw.status = "kConfirmed"
        comp:push_back(raw)
    end
    segs:forward()
end

local function start_literal(e, ctx)
    local input = ctx.input
    if ctx:get_option("full_shape") or ctx.caret_pos ~= #input then
        return NOOP
    end

    local segs = ctx.composition:toSegmentation()
    local keep, lo = 0, 0
    while keep < segs.size do
        local g = segs:get_at(keep)
        if g.status ~= "kSelected" and g.status ~= "kConfirmed" then
            break
        end
        if g.start ~= lo or g._end <= lo then
            return NOOP
        end
        lo = g._end
        keep = keep + 1
    end

    if lo < #input then
        local g = tail_segment(ctx)
        if segs.size ~= keep + 1 or not g or g.status ~= "kGuess"
            or g.start ~= lo or g._end ~= #input
            or not (g:has_tag("abc") or g:has_tag("raw"))
        then
            return NOOP
        end
        local c = g:get_selected_candidate()
        local genuine = c and c:get_genuine()
        local is_confirmable_sent = genuine and (
            genuine.type == "sentence"
            or genuine.type == "rime_neural_typo"
            or (e.last_draft and c.type == e.last_draft.id)
        )
        if is_confirmable_sent then
            if c.start ~= g.start or c._end ~= g._end then
                return NOOP
            end
            -- Confirm the fully covering Chinese sentence in-place so the user can immediately
            -- type a verbatim ASCII token right after it in one unbroken composition.
            g.status = "kSelected"
            segs:forward()
            lo = #input
            keep = keep + 1
            e.last_draft = nil
        elseif not input:sub(lo + 1):match(LITERAL_TOKEN_PAT) then
            return NOOP
        end
    end

    local l = { lo = lo, keep = keep, writing = false }
    e.literal = l
    if lo < #input then
        l.writing = true
        seal_literal(ctx, l)
        l.writing = false
    end
    e.watch = ctx.update_notifier:connect(function()
        if not l.writing then
            finish_literal(e)
        end
    end)
    return ACCEPT
end

local function handle_literal_key(e, ctx, key)
    local l = e.literal
    local ch = key.keycode
    if ctx:get_option("full_shape") or ctx:get_option("ascii_mode") then
        finish_literal(e)
        return NOOP
    end
    local repr = key:repr()
    if (e.lock_key and key:eq(e.lock_key)) or repr == "space" or repr == "Escape" then
        finish_literal(e)
        return ACCEPT
    end
    -- Consume bare Shift/Alt presses while entering uppercase letters or the closing chord.
    if ch == 0xffe1 or ch == 0xffe2 or ch == 0xffe9 or ch == 0xffea then
        return ACCEPT
    end
    if repr == "BackSpace" then
        if #ctx.input > l.lo and ctx.caret_pos == #ctx.input then
            l.writing = true
            ctx:pop_input(1)
            seal_literal(ctx, l)
            l.writing = false
            return ACCEPT
        end
        finish_literal(e)
        return ACCEPT
    end
    if not key:ctrl() and not key:alt() and not key:super()
        and ch >= 0 and ch < 128 and string.char(ch):match(LITERAL_CHAR_PAT)
    then
        l.writing = true
        ctx:push_input(string.char(ch))
        seal_literal(ctx, l)
        l.writing = false
        return ACCEPT
    end
    finish_literal(e)
    return NOOP
end

function P.init(e)
    local cfg = e.engine and e.engine.schema and e.engine.schema.config
    e.repair_key = parse_key_option(cfg, "rime_neural_repair/repair_key", "Alt+r")
    e.lock_key = parse_key_option(cfg, "rime_neural_repair/lock_key", "Alt+l")
    e.repair = nil
    e.last_draft = nil
    e.literal = nil
    e.watch = nil
    e.lookup_tr = nil
    e.lookup_tr_tried = false
    e.serial = 0
end

function P.func(key, e)
    if key:release() then
        return NOOP
    end
    if e.repair then
        return handle_repair_key(e, e.engine.context, key)
    end
    if e.literal then
        return handle_literal_key(e, e.engine.context, key)
    end
    local is_repair = e.repair_key ~= nil and key:eq(e.repair_key)
    local is_lock = e.lock_key ~= nil and key:eq(e.lock_key)
    if not is_repair and not is_lock then
        return NOOP
    end

    local ctx = e.engine.context
    if ctx:get_option("ascii_mode") then
        return NOOP
    end
    if is_lock then
        return start_literal(e, ctx)
    end
    return start_repair(e, ctx)
end

function P.fini(e)
    finish_literal(e)
    e.repair = nil
    e.last_draft = nil
    e.lookup_tr = nil
    e.lookup_tr_tried = false
    e.out_opencc = nil
    e.out_opencc_cfg = nil
end

return P
