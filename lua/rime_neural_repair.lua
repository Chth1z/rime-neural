-- rime_neural_repair (v1.3.0): Opt-in pre-commit local span repair and literal-lock processor
-- for Rime / Weasel (librime >= 1.17.0, Lua 5.4).
--
-- Mount at the front of `engine/processors` (opt-in):
--   patch:
--     "engine/processors/@before 0": lua_processor@*rime_neural_repair
--
-- Features:
--   * Constant-time idle path: normal typing pays at most two `KeyEvent:eq` comparisons.
--   * Pre-commit local homophone repair (`Alt+r` by default): steps across dictionary word
--     boundaries (`Left`/`Right`) inside the active sentence candidate without moving the
--     physical caret, highlights a native homophone alternative (`1..9`/`Space`) pre-commit,
--     or reverts cleanly (`Escape` / `Alt+r`) to the exact original menu and selection.
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

local function tail_segment(ctx)
    local comp = ctx.composition
    if comp and not comp:empty() then
        return comp:back()
    end
    return nil
end

--- Extract exact word boundaries and dictionary codes from a genuine Rime `Sentence` candidate.
--- Returns `(parts, lang_name)` only when word widths and texts align 1:1 with the displayed text.
local function extract_sentence_words(cand)
    if not cand then
        return nil
    end
    local g = cand:get_genuine()
    local s = g and g:to_sentence()
    if not s then
        return nil
    end

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
        local code_str = (entry.code and entry.code:print()) or ""
        parts[i] = {
            lo = pos,
            hi = pos + n,
            text = entry.text,
            code = code_str,
        }
        texts[i] = entry.text
        pos = pos + n
    end
    if pos == cand._end and table.concat(texts) == cand.text then
        return parts, s.lang_name
    end
    return nil
end

--- Scan the original menu for genuine `Sentence` alternatives that match every word's byte span,
--- dictionary code, and surrounding text, differing only at `s.target`.
local function find_homophone_alternatives(s)
    local base, target = s.words, s.target
    local rows, seen = {}, {}
    if not base[target] or base[target].code == "" then
        return rows
    end

    for index = 0, 31 do
        local c = s.menu:get_candidate_at(index)
        if not c then
            break
        end
        local other, lang = extract_sentence_words(c)
        if other and lang == s.lang and #other == #base then
            local matches = true
            for i, w in ipairs(other) do
                local b = base[i]
                if w.lo ~= b.lo or w.hi ~= b.hi or w.code ~= b.code
                    or (i ~= target and w.text ~= b.text)
                then
                    matches = false
                    break
                end
            end
            local text = other[target].text
            if matches and text ~= base[target].text and not seen[text] then
                rows[#rows + 1] = { index = index, text = text }
                seen[text] = true
                if #rows == 9 then
                    break
                end
            end
        end
    end
    return rows
end

--- Borrow the active segment's menu to show the focused word and numbered alternatives in the
--- candidate comment while preserving the original sentence text for safe fallback commit.
local function show_repair_menu(s, seg)
    local rows = find_homophone_alternatives(s)
    local hints = { "[" .. s.words[s.target].text .. "]" }
    for i, row in ipairs(rows) do
        hints[#hints + 1] = i .. " " .. row.text
    end
    if #rows == 0 then
        hints[#hints + 1] = "(no alternate)"
    end

    local row = ShadowCandidate(s.source, s.id, s.source.text, table.concat(hints, "  "))
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
    local parts, lang = extract_sentence_words(c)
    local pos = ctx.caret_pos
    if not parts or pos < c.start or pos > c._end then
        return NOOP
    end

    -- Target the word containing the caret; at a boundary, prefer the word immediately to its left.
    local target = 1
    while target < #parts and pos > parts[target].hi do
        target = target + 1
    end
    e.serial = (e.serial or 0) + 1
    local state = {
        id = "rime_neural_repair_" .. e.serial,
        input = ctx.input,
        pos = pos,
        source = c,
        menu = seg.menu,
        index = seg.selected_index,
        words = parts,
        lang = lang,
        target = target,
    }
    if not show_repair_menu(state, seg) then
        return NOOP
    end
    e.repair = state
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
        seg.menu = s.menu
        seg.selected_index = s.index
        e.repair = nil
        return NOOP
    end

    local k = key:repr()
    if k == "Left" or k == "Right" then
        local old = s.target
        local step = (k == "Left") and -1 or 1
        s.target = math.max(1, math.min(#s.words, old + step))
        if s.target ~= old and not show_repair_menu(s, seg) then
            s.target = old
        end
        return ACCEPT
    end

    if (e.repair_key and key:eq(e.repair_key)) or k == "Escape" then
        seg.menu = s.menu
        seg.selected_index = s.index
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
        seg.menu = s.menu
        seg.selected_index = s.rows[n].index
        e.repair = nil
        return ACCEPT
    end

    -- Any other key cleanly restores the original menu before passing through.
    seg.menu = s.menu
    seg.selected_index = s.index
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
        if genuine and genuine.type == "sentence" then
            if c.start ~= g.start or c._end ~= g._end then
                return NOOP
            end
            -- Confirm the fully covering Chinese sentence in-place so the user can immediately
            -- type a verbatim ASCII token right after it in one unbroken composition.
            g.status = "kSelected"
            segs:forward()
            lo = #input
            keep = keep + 1
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
    e.literal = nil
    e.watch = nil
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
end

return P
