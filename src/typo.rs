//! Physical QWERTY 1-edit typo recovery for Full Pinyin and all 7 Double Pinyin layouts.
//!
//! Provides two complementary primitives:
//! 1. `suggest_broken_input_repairs`: automatic zero-regression recovery triggered only when a
//!    raw input string cannot decompose cleanly into valid Mandarin syllables under `layout`
//!    (e.g., `jintisn` -> `jintian`, `womne` -> `women`, `jntmtmqo` -> `jntmtmqi`, `jintiaan` -> `jintian`).
//! 2. `suggest_span_typo_repairs`: explicit word-span 1-edit neighbor/transposition/insertion/deletion
//!    generator for interactive pre-commit `Alt+r` repair (recovers valid-syllable slips such as
//!    ZRM `gsye` -> `gsyr` or Full Pinyin `jintan` -> `jintian`).

use std::collections::{HashMap, HashSet};

use crate::pinyin::{PinyinLayout, dp_pair_to_syllable};

/// A single repaired raw keystroke sequence ranked by physical edit cost (lower is closer).
#[derive(Debug, Clone, PartialEq)]
pub struct TypoSuggestion {
    pub input: String,
    pub edit_cost: f32,
    pub kind: &'static str,
}

/// Staggered physical QWERTY coordinates `(row, col_x4)` in quarter-key units for exact distance.
fn qwerty_pos(b: u8) -> Option<(i32, i32)> {
    Some(match b.to_ascii_lowercase() {
        b'q' => (0, 0),
        b'w' => (0, 4),
        b'e' => (0, 8),
        b'r' => (0, 12),
        b't' => (0, 16),
        b'y' => (0, 20),
        b'u' => (0, 24),
        b'i' => (0, 28),
        b'o' => (0, 32),
        b'p' => (0, 36),
        b'a' => (1, 1),
        b's' => (1, 5),
        b'd' => (1, 9),
        b'f' => (1, 13),
        b'g' => (1, 17),
        b'h' => (1, 21),
        b'j' => (1, 25),
        b'k' => (1, 29),
        b'l' => (1, 33),
        b';' => (1, 37),
        b'z' => (2, 3),
        b'x' => (2, 7),
        b'c' => (2, 11),
        b'v' => (2, 15),
        b'b' => (2, 19),
        b'n' => (2, 23),
        b'm' => (2, 27),
        _ => return None,
    })
}

/// Return physical QWERTY neighbors of `b` with `(neighbor_byte, substitution_cost)`.
fn qwerty_neighbors(b: u8, allow_semi: bool) -> Vec<(u8, f32)> {
    let Some((r0, c0)) = qwerty_pos(b) else {
        return Vec::new();
    };
    let mut keys: Vec<u8> = (b'a'..=b'z').collect();
    if allow_semi {
        keys.push(b';');
    }
    let mut out: Vec<(u8, f32)> = Vec::with_capacity(8);
    for k in keys {
        if k == b.to_ascii_lowercase() {
            continue;
        }
        if let Some((r1, c1)) = qwerty_pos(k) {
            let dr = (r1 - r0) as f32;
            let dc = ((c1 - c0) as f32) * 0.25;
            let dist = (dr * dr + dc * dc).sqrt();
            if dist <= 1.38 {
                let cost = if dist <= 1.12 { 1.00 } else { 1.18 };
                out.push((k, cost));
            }
        }
    }
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// Normalize Full Pinyin `u`/`v` user aliases (`lue`/`nue`/`[jqxy]v*`) to canonical vocab form.
/// Excludes vowelless interjections (`m`, `n`, `ng`, `hm`, `hng`) from continuous full-pinyin
/// segmentation so typos like `womne`, `wommen`, or `womengng` are never masked as `wo + m + ...`
/// or `wo + meng + ng`.
fn is_valid_full_syllable(s: &str, valid: &HashSet<String>) -> bool {
    if !s
        .bytes()
        .any(|b| matches!(b, b'a' | b'e' | b'i' | b'o' | b'u' | b'v'))
    {
        return false;
    }
    if valid.contains(s) {
        return true;
    }
    match s {
        "lue" => return valid.contains("lve"),
        "nue" => return valid.contains("nve"),
        _ => {}
    }
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && matches!(bytes[0], b'j' | b'q' | b'x' | b'y') && bytes[1] == b'v' {
        let mut alt = String::with_capacity(s.len());
        alt.push(bytes[0] as char);
        alt.push('u');
        alt.push_str(&s[2..]);
        return valid.contains(&alt);
    }
    false
}

/// Compute forward and backward exact full-syllable reachability bitmasks for an ASCII chunk.
/// `fwd[i]` is true iff `chunk[0..i]` splits cleanly into valid Mandarin syllables.
/// `bwd[i]` is true iff `chunk[i..n]` splits cleanly into valid Mandarin syllables.
fn full_syllable_reachability(chunk: &str, valid: &HashSet<String>) -> (Vec<bool>, Vec<bool>) {
    let n = chunk.len();
    let mut fwd = vec![false; n + 1];
    let mut bwd = vec![false; n + 1];
    fwd[0] = true;
    for i in 0..n {
        if !fwd[i] {
            continue;
        }
        let max_end = (i + 6).min(n);
        for end in (i + 1)..=max_end {
            if is_valid_full_syllable(&chunk[i..end], valid) {
                fwd[end] = true;
            }
        }
    }
    bwd[n] = true;
    for i in (0..n).rev() {
        let max_end = (i + 6).min(n);
        for end in (i + 1)..=max_end {
            if bwd[end] && is_valid_full_syllable(&chunk[i..end], valid) {
                bwd[i] = true;
                break;
            }
        }
    }
    (fwd, bwd)
}

/// Minimum number of valid syllables needed to cover `chunk` (`None` if unparseable).
fn full_syllable_count(chunk: &str, valid: &HashSet<String>) -> Option<usize> {
    let n = chunk.len();
    let mut dp = vec![usize::MAX; n + 1];
    dp[0] = 0;
    for i in 0..n {
        if dp[i] == usize::MAX {
            continue;
        }
        let max_end = (i + 6).min(n);
        for end in (i + 1)..=max_end {
            if is_valid_full_syllable(&chunk[i..end], valid) && dp[i] + 1 < dp[end] {
                dp[end] = dp[i] + 1;
            }
        }
    }
    (dp[n] != usize::MAX).then_some(dp[n])
}

/// True if `prefix` is a non-empty strict prefix of any valid Mandarin syllable (including `u`/`v` aliases).
fn is_valid_syllable_prefix(prefix: &str, valid: &HashSet<String>) -> bool {
    if prefix.is_empty() || prefix.len() > 5 {
        return false;
    }
    if valid.iter().any(|s| s.starts_with(prefix)) {
        return true;
    }
    if "lue".starts_with(prefix) && valid.contains("lve") {
        return true;
    }
    if "nue".starts_with(prefix) && valid.contains("nve") {
        return true;
    }
    let bytes = prefix.as_bytes();
    if bytes.len() >= 2 && matches!(bytes[0], b'j' | b'q' | b'x' | b'y') && bytes[1] == b'v' {
        let mut alt = String::with_capacity(prefix.len());
        alt.push(bytes[0] as char);
        alt.push('u');
        alt.push_str(&prefix[2..]);
        return valid.iter().any(|s| s.starts_with(&alt));
    }
    false
}

/// True if `chunk` is a legal in-progress typing prefix (either completely valid syllables, or
/// valid syllables followed by a legal unfinished trailing syllable / single Double Pinyin initial).
/// Prevents automatic typo recovery from firing mid-word while the user is still typing normally.
pub fn is_live_pinyin_prefix(
    raw_input: &str,
    layout: PinyinLayout,
    valid_syllables: &HashSet<String>,
) -> bool {
    if raw_input.is_empty() {
        return true;
    }
    let allow_semi = layout.uses_semicolon();
    if !raw_input
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b == b'\'' || b == b' ' || (allow_semi && b == b';'))
    {
        return false;
    }

    let chunks: Vec<&str> = raw_input
        .split(['\'', ' '])
        .filter(|c| !c.is_empty())
        .collect();
    if chunks.is_empty() {
        return true;
    }

    // All non-final chunks (before explicit separators) may be single-letter abbreviations or full syllables.
    for &chunk in &chunks[..chunks.len() - 1] {
        if layout == PinyinLayout::Full {
            if chunk.len() == 1 {
                if !is_valid_syllable_prefix(chunk, valid_syllables) {
                    return false;
                }
            } else if full_syllable_count(chunk, valid_syllables).is_none() {
                return false;
            }
        } else {
            let b = chunk.as_bytes();
            if b.len() == 1 {
                if !b[0].is_ascii_lowercase() {
                    return false;
                }
            } else if !b.len().is_multiple_of(2)
                || !b
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|p| dp_pair_to_syllable(layout, p[0], p[1], valid_syllables).is_some())
            {
                return false;
            }
        }
    }

    let last = *chunks.last().unwrap();
    if layout == PinyinLayout::Full {
        let (fwd, _) = full_syllable_reachability(last, valid_syllables);
        if fwd[last.len()] {
            return true;
        }
        // Check if any valid reachable split point leaves a legal syllable prefix at the tail.
        let start_lo = last.len().saturating_sub(5);
        for p in start_lo..last.len() {
            if fwd[p] && is_valid_syllable_prefix(&last[p..], valid_syllables) {
                return true;
            }
        }
        false
    } else {
        let b = last.as_bytes();
        let pairs_len = (b.len() / 2) * 2;
        let prefix_ok = b[..pairs_len]
            .as_chunks::<2>()
            .0
            .iter()
            .all(|p| dp_pair_to_syllable(layout, p[0], p[1], valid_syllables).is_some());
        if !prefix_ok {
            return false;
        }
        // Even length -> all pairs valid; odd length -> all preceding pairs valid + legal trailing key.
        b.len() == pairs_len || b[pairs_len].is_ascii_lowercase()
    }
}

/// Return true if `raw_input` splits cleanly into valid full syllables under `layout`.
pub fn is_clean_pinyin_input(
    raw_input: &str,
    layout: PinyinLayout,
    valid_syllables: &HashSet<String>,
) -> bool {
    if raw_input.is_empty() {
        return true;
    }
    let allow_semi = layout.uses_semicolon();
    let bytes = raw_input.as_bytes();
    if !bytes
        .iter()
        .all(|&b| b.is_ascii_lowercase() || b == b'\'' || b == b' ' || (allow_semi && b == b';'))
    {
        return false;
    }

    if layout == PinyinLayout::Full {
        raw_input
            .split(['\'', ' '])
            .filter(|c| !c.is_empty())
            .all(|chunk| full_syllable_count(chunk, valid_syllables).is_some())
    } else {
        raw_input
            .split(['\'', ' '])
            .filter(|c| !c.is_empty())
            .all(|chunk| {
                let b = chunk.as_bytes();
                b.len() >= 2
                    && b.len().is_multiple_of(2)
                    && b.as_chunks::<2>().0.iter().all(|pair| {
                        dp_pair_to_syllable(layout, pair[0], pair[1], valid_syllables).is_some()
                    })
            })
    }
}

fn push_best(
    best: &mut HashMap<String, (f32, &'static str, usize)>,
    cand: String,
    cost: f32,
    kind: &'static str,
    syl_count: usize,
) {
    best.entry(cand)
        .and_modify(|e| {
            if cost < e.0 || ((cost - e.0).abs() < 1e-6 && syl_count < e.2) {
                *e = (cost, kind, syl_count);
            }
        })
        .or_insert((cost, kind, syl_count));
}

fn finalize_suggestions(
    best: HashMap<String, (f32, &'static str, usize)>,
    original: &str,
    max_results: usize,
) -> Vec<TypoSuggestion> {
    let mut items: Vec<(String, f32, &'static str, usize)> = best
        .into_iter()
        .filter(|(s, _)| s != original)
        .map(|(s, (c, k, sc))| (s, c, k, sc))
        .collect();
    // Prefer lower edit cost, then fewer syllables (prefers `jintian` (2) over `jin ti an` (3)).
    items.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.0.cmp(&b.0))
    });
    items
        .into_iter()
        .take(max_results)
        .map(|(input, edit_cost, kind, _)| TypoSuggestion {
            input,
            edit_cost,
            kind,
        })
        .collect()
}

/// Generate 1-edit Full Pinyin repairs strictly within byte range `[edit_lo..edit_hi]` of `chunk`.
fn generate_full_chunk_edits(
    chunk: &str,
    edit_lo: usize,
    edit_hi: usize,
    valid: &HashSet<String>,
    best: &mut HashMap<String, (f32, &'static str, usize)>,
) {
    let bytes = chunk.as_bytes();
    let n = bytes.len();
    let lo = edit_lo.min(n);
    let hi = edit_hi.min(n);

    // 1. Adjacent transpositions (e.g. `womne` -> `women`, `jintain` -> `jintian`)
    for i in lo..hi.saturating_sub(1) {
        if bytes[i] == bytes[i + 1] {
            continue;
        }
        let mut buf = bytes.to_vec();
        buf.swap(i, i + 1);
        if let Ok(s) = String::from_utf8(buf)
            && let Some(sc) = full_syllable_count(&s, valid)
        {
            push_best(best, s, 0.85, "transpose", sc);
        }
    }

    // 2. Single-key deletions (dedup doubled key `jintiaan` -> `jintian` has lower cost than stray char)
    if n >= 3 {
        for i in lo..hi {
            let is_dup =
                (i > 0 && bytes[i] == bytes[i - 1]) || (i + 1 < n && bytes[i] == bytes[i + 1]);
            let mut buf = Vec::with_capacity(n - 1);
            buf.extend_from_slice(&bytes[..i]);
            buf.extend_from_slice(&bytes[i + 1..]);
            if let Ok(s) = String::from_utf8(buf)
                && let Some(sc) = full_syllable_count(&s, valid)
            {
                let cost = if is_dup { 0.90 } else { 1.25 };
                let kind = if is_dup { "dedup" } else { "delete" };
                push_best(best, s, cost, kind, sc);
            }
        }
    }

    // 3. Physical QWERTY neighbor substitutions (e.g. `jintisn` -> `jintian`)
    for i in lo..hi {
        for (nb, cost) in qwerty_neighbors(bytes[i], false) {
            let mut buf = bytes.to_vec();
            buf[i] = nb;
            if let Ok(s) = String::from_utf8(buf)
                && let Some(sc) = full_syllable_count(&s, valid)
            {
                push_best(best, s, cost, "neighbor", sc);
            }
        }
    }

    // 4. Dropped-key insertions (vowels & frequent finals/nasals: `a,e,i,o,u,n,g,h`)
    const INSERT_KEYS: &[u8] = b"aeioungh";
    for i in lo..=hi {
        for &ins in INSERT_KEYS {
            let mut buf = Vec::with_capacity(n + 1);
            buf.extend_from_slice(&bytes[..i]);
            buf.push(ins);
            buf.extend_from_slice(&bytes[i..]);
            if let Ok(s) = String::from_utf8(buf)
                && let Some(sc) = full_syllable_count(&s, valid)
            {
                push_best(best, s, 1.12, "insert", sc);
            }
        }
    }
}

/// Generate 1-edit Double Pinyin repairs for a single chunk (`raw` without spaces/apostrophes).
fn generate_dp_chunk_edits(
    chunk: &str,
    layout: PinyinLayout,
    target_pair_idx: Option<usize>,
    valid: &HashSet<String>,
    best: &mut HashMap<String, (f32, &'static str, usize)>,
) {
    let bytes = chunk.as_bytes();
    let n = bytes.len();
    let allow_semi = layout.uses_semicolon();

    let all_pairs_valid = |b: &[u8]| -> Option<usize> {
        if b.len() < 2 || !b.len().is_multiple_of(2) {
            return None;
        }
        for pair in b.as_chunks::<2>().0 {
            dp_pair_to_syllable(layout, pair[0], pair[1], valid)?;
        }
        Some(b.len() / 2)
    };

    if n.is_multiple_of(2) {
        let pair_count = n / 2;
        let (p_lo, p_hi) = match target_pair_idx {
            Some(p) => (p.saturating_sub(1), (p + 2).min(pair_count)),
            None => (0, pair_count),
        };
        let byte_lo = p_lo * 2;
        let byte_hi = p_hi * 2;

        // 1. Adjacent transpositions (within pair or across adjacent pair boundary)
        for i in byte_lo..byte_hi.saturating_sub(1) {
            if bytes[i] == bytes[i + 1] {
                continue;
            }
            let mut buf = bytes.to_vec();
            buf.swap(i, i + 1);
            if let Some(sc) = all_pairs_valid(&buf)
                && let Ok(s) = String::from_utf8(buf)
            {
                push_best(best, s, 0.85, "transpose", sc);
            }
        }

        // 2. QWERTY physical neighbor substitution at each position in target window
        for i in byte_lo..byte_hi {
            for (nb, cost) in qwerty_neighbors(bytes[i], allow_semi) {
                let mut buf = bytes.to_vec();
                buf[i] = nb;
                if let Some(sc) = all_pairs_valid(&buf)
                    && let Ok(s) = String::from_utf8(buf)
                {
                    push_best(best, s, cost, "neighbor", sc);
                }
            }
        }
    } else if n >= 3 {
        // Odd-length Double Pinyin chunk:
        // 1. Single-key deletions (doubled key `jntmmtmqi` -> `jntmtmqi` or stray extra key).
        for i in 0..n {
            let is_dup =
                (i > 0 && bytes[i] == bytes[i - 1]) || (i + 1 < n && bytes[i] == bytes[i + 1]);
            let mut buf = Vec::with_capacity(n - 1);
            buf.extend_from_slice(&bytes[..i]);
            buf.extend_from_slice(&bytes[i + 1..]);
            if let Some(sc) = all_pairs_valid(&buf)
                && let Ok(s) = String::from_utf8(buf)
            {
                let cost = if is_dup { 0.90 } else { 1.25 };
                let kind = if is_dup { "dedup" } else { "delete" };
                push_best(best, s, cost, kind, sc);
            }
        }

        // 2. Single-key insertions for dropped Double Pinyin key (e.g. ZRM `jntqivf` -> `jntmqivf`):
        //    locate the first invalid pair `p` and try inserting one layout-legal key at `2*p..=2*p+2`.
        let first_bad_pair = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .position(|pair| dp_pair_to_syllable(layout, pair[0], pair[1], valid).is_none())
            .unwrap_or(bytes.len() / 2);
        let ins_lo = (first_bad_pair * 2).saturating_sub(1);
        let ins_hi = (first_bad_pair * 2 + 2).min(n);
        let keys: &[u8] = if allow_semi {
            b"abcdefghijklmnopqrstuvwxyz;"
        } else {
            b"abcdefghijklmnopqrstuvwxyz"
        };
        for pos in ins_lo..=ins_hi {
            for &ch in keys {
                let mut buf = Vec::with_capacity(n + 1);
                buf.extend_from_slice(&bytes[..pos]);
                buf.push(ch);
                buf.extend_from_slice(&bytes[pos..]);
                if let Some(sc) = all_pairs_valid(&buf)
                    && let Ok(s) = String::from_utf8(buf)
                {
                    push_best(best, s, 1.12, "insert", sc);
                }
            }
        }
    }
}

/// Automatically suggest up to `max_results` 1-edit repaired raw inputs **only** when `raw_input`
/// fails strict valid-syllable parsing under `layout`. Returns empty immediately on clean input
/// (`~1 µs`), guaranteeing zero overhead and zero false triggers on clean typing.
pub fn suggest_broken_input_repairs(
    raw_input: &str,
    layout: PinyinLayout,
    valid_syllables: &HashSet<String>,
    max_results: usize,
) -> Vec<TypoSuggestion> {
    if max_results == 0 || raw_input.len() < 3 || raw_input.len() > 96 {
        return Vec::new();
    }
    let allow_semi = layout.uses_semicolon();
    if !raw_input
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b == b'\'' || b == b' ' || (allow_semi && b == b';'))
    {
        return Vec::new();
    }
    if is_live_pinyin_prefix(raw_input, layout, valid_syllables) {
        return Vec::new();
    }

    // Locate the unique broken chunk when separators exist; if >1 chunks are broken, bail out
    // (1-edit cannot fix two independent broken chunks cleanly).
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let bytes = raw_input.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] == b'\'' || bytes[i] == b' ') {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && bytes[i] != b'\'' && bytes[i] != b' ' {
            i += 1;
        }
        if i > start {
            chunks.push((start, i));
        }
    }

    let mut broken_chunks = Vec::new();
    for (idx, &(s, e)) in chunks.iter().enumerate() {
        let ch = &raw_input[s..e];
        let ok = is_clean_pinyin_input(ch, layout, valid_syllables)
            || (layout == PinyinLayout::Full
                && idx + 1 < chunks.len()
                && ch.len() <= 2
                && is_valid_syllable_prefix(ch, valid_syllables));
        if !ok {
            broken_chunks.push(idx);
        }
    }
    if broken_chunks.len() != 1 {
        return Vec::new();
    }
    let (cs, ce) = chunks[broken_chunks[0]];
    let chunk = &raw_input[cs..ce];
    let mut chunk_best: HashMap<String, (f32, &'static str, usize)> = HashMap::new();

    if layout == PinyinLayout::Full {
        let (fwd, bwd) = full_syllable_reachability(chunk, valid_syllables);
        let max_fwd = fwd.iter().rposition(|&ok| ok).unwrap_or(0);
        let min_bwd = bwd.iter().position(|&ok| ok).unwrap_or(chunk.len());
        // The single typo must overlap `[edit_lo..edit_hi]` around the forward/backward gap.
        let edit_lo = min_bwd.min(max_fwd).saturating_sub(4);
        let edit_hi = (max_fwd.max(min_bwd) + 4).min(chunk.len());
        generate_full_chunk_edits(chunk, edit_lo, edit_hi, valid_syllables, &mut chunk_best);
    } else {
        let b = chunk.as_bytes();
        let target_pair = if b.len().is_multiple_of(2) {
            let invalid_pairs: Vec<usize> = b
                .as_chunks::<2>()
                .0
                .iter()
                .enumerate()
                .filter(|(_, p)| dp_pair_to_syllable(layout, p[0], p[1], valid_syllables).is_none())
                .map(|(k, _)| k)
                .collect();
            if invalid_pairs.len() > 2 {
                return Vec::new();
            }
            invalid_pairs.first().copied()
        } else {
            None
        };
        generate_dp_chunk_edits(chunk, layout, target_pair, valid_syllables, &mut chunk_best);
    }

    let mut full_best: HashMap<String, (f32, &'static str, usize)> = HashMap::new();
    for (fixed_chunk, meta) in chunk_best {
        let mut rebuilt = String::with_capacity(raw_input.len() + 2);
        rebuilt.push_str(&raw_input[..cs]);
        rebuilt.push_str(&fixed_chunk);
        rebuilt.push_str(&raw_input[ce..]);
        full_best.insert(rebuilt, meta);
    }
    finalize_suggestions(full_best, raw_input, max_results)
}

/// Suggest valid 1-edit typo alternatives for an explicit short word span (used by `Alt+r` repair
/// even when the user's slip accidentally formed another valid Mandarin syllable, e.g. ZRM `gsye`
/// -> `gsyr` or Full Pinyin `jintan` -> `jintian`).
pub fn suggest_span_typo_repairs(
    raw_span: &str,
    layout: PinyinLayout,
    valid_syllables: &HashSet<String>,
    max_results: usize,
) -> Vec<TypoSuggestion> {
    if max_results == 0 || raw_span.len() < 2 || raw_span.len() > 18 {
        return Vec::new();
    }
    let allow_semi = layout.uses_semicolon();
    if !raw_span
        .bytes()
        .all(|b| b.is_ascii_lowercase() || (allow_semi && b == b';'))
    {
        return Vec::new();
    }

    let mut best: HashMap<String, (f32, &'static str, usize)> = HashMap::new();
    if layout == PinyinLayout::Full {
        generate_full_chunk_edits(raw_span, 0, raw_span.len(), valid_syllables, &mut best);
    } else {
        generate_dp_chunk_edits(raw_span, layout, None, valid_syllables, &mut best);
    }
    finalize_suggestions(best, raw_span, max_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_valid_syllables() -> HashSet<String> {
        [
            "jin", "tian", "tan", "qi", "zhen", "bu", "cuo", "wo", "men", "qu", "gong", "yuan",
            "ye", "ni", "hao", "shi", "jie", "shang", "wu", "kan", "dian", "ying", "ta", "de",
            "fang", "an", "fan", "gan", "a", "an", "e", "en", "o", "ou", "ti", "si", "su",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    #[test]
    fn test_clean_and_live_prefix_inputs_return_zero_broken_repairs() {
        let valid = sample_valid_syllables();
        assert!(is_clean_pinyin_input(
            "jintiantianqizhenbucuo",
            PinyinLayout::Full,
            &valid
        ));
        assert!(
            suggest_broken_input_repairs("jintiantianqizhenbucuo", PinyinLayout::Full, &valid, 6)
                .is_empty()
        );
        // Legal unfinished mid-word prefixes must NEVER trigger automatic typo recovery!
        assert!(is_live_pinyin_prefix("jint", PinyinLayout::Full, &valid));
        assert!(is_live_pinyin_prefix("jintia", PinyinLayout::Full, &valid));
        assert!(suggest_broken_input_repairs("jintia", PinyinLayout::Full, &valid, 6).is_empty());
        assert!(is_live_pinyin_prefix("jntmt", PinyinLayout::Zrm, &valid));
        assert!(suggest_broken_input_repairs("jntmt", PinyinLayout::Zrm, &valid, 6).is_empty());
        assert!(is_clean_pinyin_input(
            "jntmtmqivfbuco",
            PinyinLayout::Zrm,
            &valid
        ));
        assert!(
            suggest_broken_input_repairs("jntmtmqivfbuco", PinyinLayout::Zrm, &valid, 6).is_empty()
        );
    }

    #[test]
    fn test_full_pinyin_broken_repairs_cover_neighbor_transpose_dedup_insert() {
        let valid = sample_valid_syllables();
        // Adjacent QWERTY neighbor: `s` for `a` in `jintisn`
        let r1 = suggest_broken_input_repairs("jintisn", PinyinLayout::Full, &valid, 6);
        assert!(
            r1.iter().any(|x| x.input == "jintian"),
            "expected jintian in {:?}",
            r1
        );
        // Adjacent transposition: `womne` -> `women`
        let r2 = suggest_broken_input_repairs("womnequgongyuan", PinyinLayout::Full, &valid, 6);
        assert_eq!(r2[0].input, "womenqugongyuan");
        assert_eq!(r2[0].kind, "transpose");
        // Doubled key in the middle of a sentence: `jintianntianqi` -> `jintiantianqi` and `wommenqugongyuan`
        let r3 = suggest_broken_input_repairs("jintianntianqi", PinyinLayout::Full, &valid, 6);
        assert!(
            r3.iter()
                .any(|x| x.input == "jintiantianqi" && x.kind == "dedup"),
            "expected dedup jintiantianqi in {:?}",
            r3
        );
        let r3b = suggest_broken_input_repairs("wommenqugongyuan", PinyinLayout::Full, &valid, 6);
        assert!(
            r3b.iter()
                .any(|x| x.input == "womenqugongyuan" && x.kind == "dedup"),
            "expected dedup womenqugongyuan in {:?}",
            r3b
        );
        // Doubled vowel that happens to parse as `jin ti a an` is recovered via explicit span repair
        let r3c = suggest_span_typo_repairs("jintiaan", PinyinLayout::Full, &valid, 6);
        assert!(
            r3c.iter()
                .any(|x| x.input == "jintian" && x.kind == "dedup"),
            "expected span dedup jintian in {:?}",
            r3c
        );
        // Dropped vowel: `gngyuan` -> `gongyuan`
        let r4 = suggest_broken_input_repairs("gngyuan", PinyinLayout::Full, &valid, 6);
        assert!(
            r4.iter()
                .any(|x| x.input == "gongyuan" && x.kind == "insert"),
            "expected insert gongyuan in {:?}",
            r4
        );
        // Explicit abbreviation chunk + broken chunk: `w'jintisn` -> `w'jintian`
        let r5 = suggest_broken_input_repairs("w'jintisn", PinyinLayout::Full, &valid, 6);
        assert!(
            r5.iter().any(|x| x.input == "w'jintian"),
            "expected w'jintian in {:?}",
            r5
        );
    }

    #[test]
    fn test_double_pinyin_broken_and_span_repairs() {
        let valid = sample_valid_syllables();
        // Invalid ZRM pair `qo` (neighbor `i` -> `qi`) inside `jntmtmqovfbuco`
        let r1 = suggest_broken_input_repairs("jntmtmqovfbuco", PinyinLayout::Zrm, &valid, 6);
        assert!(
            r1.iter().any(|x| x.input == "jntmtmqivfbuco"),
            "expected jntmtmqivfbuco in {:?}",
            r1
        );
        // Doubled key in ZRM: `jntmmtmqi` -> `jntmtmqi`
        let r2 = suggest_broken_input_repairs("jntmmtmqi", PinyinLayout::Zrm, &valid, 6);
        assert!(
            r2.iter()
                .any(|x| x.input == "jntmtmqi" && x.kind == "dedup"),
            "expected dedup jntmtmqi in {:?}",
            r2
        );
        // Dropped key in ZRM: `jntqivf` -> `jntmqivf` (`m` dropped -> `tq` invalid pair)
        let r2b = suggest_broken_input_repairs("jntqivf", PinyinLayout::Zrm, &valid, 12);
        assert!(
            r2b.iter()
                .any(|x| x.input == "jntmqivf" && x.kind == "insert"),
            "expected insert jntmqivf in {:?}",
            r2b
        );
        // Valid-syllable span slip: ZRM `gsye` (`公夜`) -> `gsyr` (`公园`) via Alt+r span repair
        let r3 = suggest_span_typo_repairs("gsye", PinyinLayout::Zrm, &valid, 8);
        assert!(
            r3.iter().any(|x| x.input == "gsyr"),
            "expected gsyr in {:?}",
            r3
        );
        // Valid-syllable Full Pinyin span slip: `jintan` -> `jintian`
        let r4 = suggest_span_typo_repairs("jintan", PinyinLayout::Full, &valid, 8);
        assert!(
            r4.iter().any(|x| x.input == "jintian"),
            "expected jintian in {:?}",
            r4
        );
    }
}
