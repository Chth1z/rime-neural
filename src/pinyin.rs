use std::collections::{HashMap, HashSet};

/// Supported pinyin input layouts (Full Pinyin + 7 major Rime Double Pinyin schemes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinyinLayout {
    Full,
    Zrm,
    Flypy,
    Mspy,
    Sogou,
    Abc,
    Ziguang,
    Jiajia,
}

impl PinyinLayout {
    /// Parse an explicit layout name or auto-detect from a Rime `schema_id`.
    pub fn from_name_or_schema(layout_opt: &str, schema_id: &str) -> Self {
        match layout_opt.trim().to_ascii_lowercase().as_str() {
            "full" | "quanpin" | "pinyin" => return Self::Full,
            "zrm" | "natural" => return Self::Zrm,
            "flypy" | "xiaohe" | "xhup" => return Self::Flypy,
            "mspy" | "microsoft" | "ms" => return Self::Mspy,
            "sogou" => return Self::Sogou,
            "abc" | "znabc" => return Self::Abc,
            "ziguang" | "zg" => return Self::Ziguang,
            "jiajia" | "pyjj" | "jj" => return Self::Jiajia,
            _ => {}
        }

        let s = schema_id.to_ascii_lowercase();
        let has_token = |tok: &str| {
            s.split(|c: char| !c.is_ascii_alphanumeric())
                .any(|part| part == tok)
        };
        if s.contains("flypy") || s.contains("xiaohe") || s.contains("xhup") {
            Self::Flypy
        } else if s.contains("mspy") || s.contains("microsoft") || has_token("ms") {
            Self::Mspy
        } else if s.contains("sogou") {
            Self::Sogou
        } else if has_token("abc") || s.contains("znabc") {
            Self::Abc
        } else if s.contains("ziguang") || has_token("zg") {
            Self::Ziguang
        } else if s.contains("jiajia") || s.contains("pyjj") || has_token("jj") {
            Self::Jiajia
        } else if s.contains("double_pinyin") || has_token("zrm") {
            Self::Zrm
        } else {
            Self::Full
        }
    }

    pub fn is_double_pinyin(self) -> bool {
        self != Self::Full
    }

    /// True for layouts that map a final (`ing`) to the semicolon key `;`.
    pub fn uses_semicolon(self) -> bool {
        matches!(self, Self::Mspy | Self::Sogou | Self::Ziguang)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Zrm => "zrm",
            Self::Flypy => "flypy",
            Self::Mspy => "mspy",
            Self::Sogou => "sogou",
            Self::Abc => "abc",
            Self::Ziguang => "ziguang",
            Self::Jiajia => "jiajia",
        }
    }
}

struct DoublePinyinSpec {
    zh_key: u8,
    ch_key: u8,
    sh_key: u8,
    /// Zero-initial 2-key direct overrides checked first (covers canonical + derived Rime aliases).
    zero_pairs: &'static [(u8, u8, &'static str)],
    /// Key -> possible finals (ordered by specificity; disambiguated via valid Mandarin table).
    finals_fn: fn(u8) -> &'static [&'static str],
}

fn zrm_finals(k: u8) -> &'static [&'static str] {
    match k {
        b'a' => &["a"],
        b'e' => &["e"],
        b'i' => &["i"],
        b'u' => &["u"],
        b'o' => &["o", "uo"],
        b'v' => &["v", "ui"],
        b'q' => &["iu"],
        b'w' => &["ua", "ia"],
        b'r' => &["uan", "er"],
        b't' => &["ve", "ue"],
        b'y' => &["uai", "ing"],
        b'p' => &["un"],
        b's' => &["iong", "ong"],
        b'd' => &["uang", "iang"],
        b'f' => &["en"],
        b'g' => &["eng"],
        b'h' => &["ang"],
        b'j' => &["an"],
        b'k' => &["ao"],
        b'l' => &["ai"],
        b'z' => &["ei"],
        b'x' => &["ie"],
        b'c' => &["iao"],
        b'b' => &["ou"],
        b'n' => &["in"],
        b'm' => &["ian"],
        _ => &[],
    }
}

fn flypy_finals(k: u8) -> &'static [&'static str] {
    match k {
        b'a' => &["a"],
        b'e' => &["e"],
        b'i' => &["i"],
        b'u' => &["u"],
        b'o' => &["o", "uo"],
        b'v' => &["v", "ui"],
        b'q' => &["iu"],
        b'w' => &["ei"],
        b'r' => &["uan"],
        b't' => &["ve", "ue"],
        b'y' => &["un"],
        b'p' => &["ie"],
        b's' => &["iong", "ong"],
        b'd' => &["ai"],
        b'f' => &["en"],
        b'g' => &["eng"],
        b'h' => &["ang"],
        b'j' => &["an"],
        b'k' => &["uai", "ing"],
        b'l' => &["uang", "iang"],
        b'z' => &["ou"],
        b'x' => &["ua", "ia"],
        b'c' => &["ao"],
        b'b' => &["in"],
        b'n' => &["iao"],
        b'm' => &["ian"],
        _ => &[],
    }
}

fn mspy_finals(k: u8) -> &'static [&'static str] {
    match k {
        b'a' => &["a"],
        b'e' => &["e"],
        b'i' => &["i"],
        b'u' => &["u"],
        b'o' => &["o", "uo"],
        b'q' => &["iu"],
        b'w' => &["ua", "ia"],
        b'r' => &["uan", "er"],
        b't' => &["ve", "ue"],
        b'y' => &["uai", "v"],
        b'p' => &["un"],
        b's' => &["iong", "ong"],
        b'd' => &["uang", "iang"],
        b'f' => &["en"],
        b'g' => &["eng"],
        b'h' => &["ang"],
        b'j' => &["an"],
        b'k' => &["ao"],
        b'l' => &["ai"],
        b'z' => &["ei"],
        b'x' => &["ie"],
        b'c' => &["iao"],
        b'v' => &["ui", "ve", "ue"],
        b'b' => &["ou"],
        b'n' => &["in"],
        b'm' => &["ian"],
        b';' => &["ing"],
        _ => &[],
    }
}

fn sogou_finals(k: u8) -> &'static [&'static str] {
    match k {
        b'v' => &["ui"],
        other => mspy_finals(other),
    }
}

fn abc_finals(k: u8) -> &'static [&'static str] {
    match k {
        b'a' => &["a"],
        b'e' => &["e"],
        b'i' => &["i"],
        b'u' => &["u"],
        b'o' => &["o", "uo"],
        b'v' => &["v"],
        b'q' => &["ei"],
        b'w' => &["ian"],
        b'r' => &["iu", "er"],
        b't' => &["uang", "iang"],
        b'y' => &["ing"],
        b'p' => &["uan"],
        b's' => &["iong", "ong"],
        b'd' => &["ua", "ia"],
        b'f' => &["en"],
        b'g' => &["eng"],
        b'h' => &["ang"],
        b'j' => &["an"],
        b'k' => &["ao"],
        b'l' => &["ai"],
        b'z' => &["iao"],
        b'x' => &["ie"],
        b'c' => &["uai", "in"],
        b'b' => &["ou"],
        b'n' => &["un"],
        b'm' => &["ve", "ue", "ui"],
        _ => &[],
    }
}

fn ziguang_finals(k: u8) -> &'static [&'static str] {
    match k {
        b'a' => &["a"],
        b'e' => &["e"],
        b'i' => &["i"],
        b'u' => &["u"],
        b'o' => &["o", "uo"],
        b'v' => &["v"],
        b'q' => &["ao"],
        b'w' => &["en"],
        b'r' => &["an"],
        b't' => &["eng"],
        b'y' => &["uai", "in"],
        b'p' => &["ai"],
        b's' => &["ang"],
        b'd' => &["ie"],
        b'f' => &["ian"],
        b'g' => &["uang", "iang"],
        b'h' => &["iong", "ong"],
        b'j' => &["iu", "er"],
        b'k' => &["ei"],
        b'l' => &["uan"],
        b'z' => &["ou"],
        b'x' => &["ua", "ia"],
        b'b' => &["iao"],
        b'n' => &["ve", "ue", "ui"],
        b'm' => &["un"],
        b';' => &["ing"],
        _ => &[],
    }
}

fn jiajia_finals(k: u8) -> &'static [&'static str] {
    match k {
        b'a' => &["a"],
        b'e' => &["e"],
        b'i' => &["i"],
        b'u' => &["u"],
        b'o' => &["o", "uo"],
        b'q' => &["ing", "er"],
        b'w' => &["ei"],
        b'r' => &["en"],
        b't' => &["eng"],
        b'y' => &["iong", "ong"],
        b'p' => &["ou"],
        b's' => &["ai"],
        b'd' => &["ao"],
        b'f' => &["an"],
        b'g' => &["ang"],
        b'h' => &["uang", "iang"],
        b'j' => &["ian"],
        b'k' => &["iao"],
        b'l' => &["in"],
        b'z' => &["un"],
        b'x' => &["uai", "ve", "ue"],
        b'c' => &["uan"],
        b'v' => &["v", "ui"],
        b'b' => &["ua", "ia"],
        b'n' => &["iu"],
        b'm' => &["ie"],
        _ => &[],
    }
}

const ZERO_ZRM: &[(u8, u8, &str)] = &[
    // Canonical ZRM zero-initials
    (b'a', b'a', "a"),
    (b'o', b'o', "o"),
    (b'e', b'e', "e"),
    (b'a', b'i', "ai"),
    (b'a', b'n', "an"),
    (b'a', b'o', "ao"),
    (b'a', b'h', "ang"),
    (b'e', b'i', "ei"),
    (b'e', b'n', "en"),
    (b'e', b'r', "er"),
    (b'e', b'g', "eng"),
    (b'o', b'u', "ou"),
    // Self-leading final-key forms derived by rime-ice / rime-frost ZRM (`derive/^([aoe])([ioun])$/$1$1$2/`)
    (b'a', b'l', "ai"),
    (b'a', b'j', "an"),
    (b'a', b'k', "ao"),
    (b'e', b'z', "ei"),
    (b'e', b'f', "en"),
    (b'o', b'b', "ou"),
    // Optional o-prefixed zero-initial aliases
    (b'o', b'a', "a"),
    (b'o', b'e', "e"),
    (b'o', b'l', "ai"),
    (b'o', b'j', "an"),
    (b'o', b'k', "ao"),
    (b'o', b'h', "ang"),
    (b'o', b'z', "ei"),
    (b'o', b'f', "en"),
    (b'o', b'g', "eng"),
    (b'o', b'r', "er"),
];

const ZERO_FLYPY: &[(u8, u8, &str)] = &[
    // Canonical & derived rime-ice / rime-frost Flypy zero-initials
    (b'a', b'a', "a"),
    (b'o', b'o', "o"),
    (b'e', b'e', "e"),
    (b'a', b'i', "ai"),
    (b'a', b'd', "ai"),
    (b'a', b'n', "an"),
    (b'a', b'j', "an"),
    (b'a', b'o', "ao"),
    (b'a', b'c', "ao"),
    (b'a', b'h', "ang"),
    (b'e', b'i', "ei"),
    (b'e', b'w', "ei"),
    (b'e', b'n', "en"),
    (b'e', b'f', "en"),
    (b'e', b'r', "er"),
    (b'e', b'g', "eng"),
    (b'o', b'u', "ou"),
    (b'o', b'z', "ou"),
    // Optional o-prefixed aliases (used by some custom / wanxiang Flypy configs; zero conflict since `o` is not a consonant)
    (b'o', b'a', "a"),
    (b'o', b'e', "e"),
    (b'o', b'd', "ai"),
    (b'o', b'j', "an"),
    (b'o', b'c', "ao"),
    (b'o', b'h', "ang"),
    (b'o', b'w', "ei"),
    (b'o', b'f', "en"),
    (b'o', b'g', "eng"),
    (b'o', b'r', "er"),
];

const ZERO_MSPY_SOGOU: &[(u8, u8, &str)] = &[
    (b'o', b'a', "a"),
    (b'o', b'e', "e"),
    (b'o', b'o', "o"),
    (b'o', b'l', "ai"),
    (b'o', b'j', "an"),
    (b'o', b'k', "ao"),
    (b'o', b'h', "ang"),
    (b'o', b'z', "ei"),
    (b'o', b'f', "en"),
    (b'o', b'g', "eng"),
    (b'o', b'r', "er"),
    (b'o', b'b', "ou"),
    // Self-leading aliases from Rime MSPY/Sogou algebra
    (b'a', b'a', "a"),
    (b'e', b'e', "e"),
    (b'a', b'l', "ai"),
    (b'a', b'j', "an"),
    (b'a', b'k', "ao"),
    (b'a', b'h', "ang"),
    (b'e', b'z', "ei"),
    (b'e', b'f', "en"),
    (b'e', b'g', "eng"),
    (b'e', b'r', "er"),
    (b'a', b'i', "ai"),
    (b'a', b'n', "an"),
    (b'a', b'o', "ao"),
    (b'e', b'i', "ei"),
    (b'e', b'n', "en"),
    (b'o', b'u', "ou"),
];

const ZERO_ABC: &[(u8, u8, &str)] = &[
    (b'o', b'a', "a"),
    (b'o', b'e', "e"),
    (b'o', b'o', "o"),
    (b'o', b'l', "ai"),
    (b'o', b'j', "an"),
    (b'o', b'k', "ao"),
    (b'o', b'h', "ang"),
    (b'o', b'q', "ei"),
    (b'o', b'f', "en"),
    (b'o', b'g', "eng"),
    (b'o', b'r', "er"),
    (b'o', b'b', "ou"),
];

const ZERO_ZIGUANG: &[(u8, u8, &str)] = &[
    (b'o', b'a', "a"),
    (b'o', b'e', "e"),
    (b'o', b'o', "o"),
    (b'o', b'p', "ai"),
    (b'o', b'r', "an"),
    (b'o', b'q', "ao"),
    (b'o', b's', "ang"),
    (b'o', b'k', "ei"),
    (b'o', b'w', "en"),
    (b'o', b't', "eng"),
    (b'o', b'j', "er"),
    (b'o', b'z', "ou"),
];

const ZERO_JIAJIA: &[(u8, u8, &str)] = &[
    (b'o', b'a', "a"),
    (b'o', b'e', "e"),
    (b'o', b'o', "o"),
    (b'o', b's', "ai"),
    (b'o', b'f', "an"),
    (b'o', b'd', "ao"),
    (b'o', b'g', "ang"),
    (b'o', b'w', "ei"),
    (b'o', b'r', "en"),
    (b'o', b't', "eng"),
    (b'o', b'q', "er"),
    (b'o', b'p', "ou"),
    (b'a', b'a', "a"),
    (b'e', b'e', "e"),
    (b'a', b's', "ai"),
    (b'a', b'f', "an"),
    (b'a', b'd', "ao"),
    (b'a', b'g', "ang"),
    (b'e', b'w', "ei"),
    (b'e', b'r', "en"),
    (b'e', b't', "eng"),
    (b'e', b'q', "er"),
    (b'a', b'i', "ai"),
    (b'a', b'n', "an"),
    (b'a', b'o', "ao"),
    (b'e', b'i', "ei"),
    (b'e', b'n', "en"),
    (b'o', b'u', "ou"),
];

fn layout_spec(layout: PinyinLayout) -> Option<DoublePinyinSpec> {
    Some(match layout {
        PinyinLayout::Full => return None,
        PinyinLayout::Zrm => DoublePinyinSpec {
            zh_key: b'v',
            ch_key: b'i',
            sh_key: b'u',
            zero_pairs: ZERO_ZRM,
            finals_fn: zrm_finals,
        },
        PinyinLayout::Flypy => DoublePinyinSpec {
            zh_key: b'v',
            ch_key: b'i',
            sh_key: b'u',
            zero_pairs: ZERO_FLYPY,
            finals_fn: flypy_finals,
        },
        PinyinLayout::Mspy => DoublePinyinSpec {
            zh_key: b'v',
            ch_key: b'i',
            sh_key: b'u',
            zero_pairs: ZERO_MSPY_SOGOU,
            finals_fn: mspy_finals,
        },
        PinyinLayout::Sogou => DoublePinyinSpec {
            zh_key: b'v',
            ch_key: b'i',
            sh_key: b'u',
            zero_pairs: ZERO_MSPY_SOGOU,
            finals_fn: sogou_finals,
        },
        PinyinLayout::Abc => DoublePinyinSpec {
            zh_key: b'a',
            ch_key: b'e',
            sh_key: b'v',
            zero_pairs: ZERO_ABC,
            finals_fn: abc_finals,
        },
        PinyinLayout::Ziguang => DoublePinyinSpec {
            zh_key: b'u',
            ch_key: b'a',
            sh_key: b'i',
            zero_pairs: ZERO_ZIGUANG,
            finals_fn: ziguang_finals,
        },
        PinyinLayout::Jiajia => DoublePinyinSpec {
            zh_key: b'v',
            ch_key: b'u',
            sh_key: b'i',
            zero_pairs: ZERO_JIAJIA,
            finals_fn: jiajia_finals,
        },
    })
}

fn map_initial(spec: &DoublePinyinSpec, a: u8) -> Option<&'static str> {
    if a == spec.zh_key {
        return Some("zh");
    }
    if a == spec.ch_key {
        return Some("ch");
    }
    if a == spec.sh_key {
        return Some("sh");
    }
    match a {
        b'b' => Some("b"),
        b'p' => Some("p"),
        b'm' => Some("m"),
        b'f' => Some("f"),
        b'd' => Some("d"),
        b't' => Some("t"),
        b'n' => Some("n"),
        b'l' => Some("l"),
        b'g' => Some("g"),
        b'k' => Some("k"),
        b'h' => Some("h"),
        b'j' => Some("j"),
        b'q' => Some("q"),
        b'x' => Some("x"),
        b'r' => Some("r"),
        b'z' => Some("z"),
        b'c' => Some("c"),
        b's' => Some("s"),
        b'y' => Some("y"),
        b'w' => Some("w"),
        _ => None,
    }
}

/// Convert a 2-key Double Pinyin pair under `layout` into a standard full pinyin syllable.
/// Uses `valid_syllables` (from `unified_vocab.json`) to disambiguate shared final keys accurately.
pub fn dp_pair_to_syllable(
    layout: PinyinLayout,
    a: u8,
    b: u8,
    valid_syllables: &HashSet<String>,
) -> Option<String> {
    let spec = layout_spec(layout)?;
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();

    for &(za, zb, syl) in spec.zero_pairs {
        if a == za && b == zb {
            return Some(syl.to_string());
        }
    }

    let ini = map_initial(&spec, a)?;
    let candidates = (spec.finals_fn)(b);
    if candidates.is_empty() {
        return None;
    }

    for &fin in candidates {
        let norm_fin = match (ini, fin) {
            ("j" | "q" | "x" | "y", "v") => "u",
            ("j" | "q" | "x" | "y", "ve") => "ue",
            ("n" | "l", "v") => "v",
            ("n" | "l", "ue") => "ve",
            (_, "v") => continue,
            ("b" | "p" | "m" | "f" | "w" | "y", "uo") => continue,
            (i, "o") if !matches!(i, "b" | "p" | "m" | "f" | "w" | "y") => continue,
            (_, "er") => continue,
            _ => fin,
        };

        let mut cand = String::with_capacity(ini.len() + norm_fin.len());
        cand.push_str(ini);
        cand.push_str(norm_fin);

        if valid_syllables.is_empty() || valid_syllables.contains(&cand) {
            return Some(cand);
        }
    }

    None
}

/// Parse Double Pinyin input into model unit tags for any supported layout.
/// Respects explicit syllable separators (`'`, space) by pairing keys independently per chunk
/// so abbreviated single-key segments (e.g. `w'm'jntm`) do not scramble subsequent syllables.
pub fn double_pinyin_to_units(
    layout: PinyinLayout,
    raw_input: &str,
    valid_syllables: &HashSet<String>,
) -> Vec<(&'static str, String)> {
    let Some(spec) = layout_spec(layout) else {
        return Vec::new();
    };
    let allow_semi = layout.uses_semicolon();
    let mut units = Vec::with_capacity(raw_input.len().div_ceil(2));

    for chunk in raw_input.split([' ', '\'']) {
        let keys: Vec<u8> = chunk
            .bytes()
            .filter(|&b| b.is_ascii_alphabetic() || (allow_semi && b == b';'))
            .map(|b| b.to_ascii_lowercase())
            .collect();
        if keys.is_empty() {
            continue;
        }
        let mut i = 0;
        while i + 1 < keys.len() {
            if let Some(syl) = dp_pair_to_syllable(layout, keys[i], keys[i + 1], valid_syllables) {
                units.push(("f", syl));
            } else {
                // Keep 1:1 slot alignment by emitting a partial initial unit instead of dropping.
                let p = map_initial(&spec, keys[i])
                    .map(str::to_string)
                    .unwrap_or_else(|| (keys[i] as char).to_string());
                units.push(("p", p));
            }
            i += 2;
        }
        if i < keys.len() {
            let p = map_initial(&spec, keys[i])
                .map(str::to_string)
                .unwrap_or_else(|| (keys[i] as char).to_string());
            units.push(("p", p));
        }
    }
    units
}

/// Maps common Full Pinyin `u`/`v` user aliases (e.g. `lue`, `jve`, `xv`) to the canonical
/// syllable spellings used by the neural vocabulary (`lve`, `jue`, `xu`).
fn canonical_alias(s: &str) -> Option<&'static str> {
    match s {
        "lue" => Some("lve"),
        "nue" => Some("nve"),
        "jv" => Some("ju"),
        "qv" => Some("qu"),
        "xv" => Some("xu"),
        "yv" => Some("yu"),
        "jve" => Some("jue"),
        "qve" => Some("que"),
        "xve" => Some("xue"),
        "yve" => Some("yue"),
        "jvan" => Some("juan"),
        "qvan" => Some("quan"),
        "xvan" => Some("xuan"),
        "yvan" => Some("yuan"),
        "jvn" => Some("jun"),
        "qvn" => Some("qun"),
        "xvn" => Some("xun"),
        "yvn" => Some("yun"),
        _ => None,
    }
}

/// Inverse of `canonical_alias`: returns the alternate spelling a user may type for a canonical
/// reading so `walk()` can align `celue` -> `策略` (`lve`) without disturbing `ce lu e` (`侧录额`).
fn user_input_alias(canonical: &str) -> Option<&'static str> {
    match canonical {
        "lve" => Some("lue"),
        "nve" => Some("nue"),
        "ju" => Some("jv"),
        "qu" => Some("qv"),
        "xu" => Some("xv"),
        "yu" => Some("yv"),
        "jue" => Some("jve"),
        "que" => Some("qve"),
        "xue" => Some("xve"),
        "yue" => Some("yve"),
        "juan" => Some("jvan"),
        "quan" => Some("qvan"),
        "xuan" => Some("xvan"),
        "yuan" => Some("yvan"),
        "jun" => Some("jvn"),
        "qun" => Some("qvn"),
        "xun" => Some("xvn"),
        "yun" => Some("yvn"),
        _ => None,
    }
}

/// Returns `true` if `a` and `b` have the same length (>= 3) and differ by exactly one
/// adjacent-letter transposition (e.g. `yuo` vs `you`, `xaing` vs `xiang`, `hoa` vs `hao`).
#[inline]
fn is_adjacent_transposition(a: &[u8], b: &[u8]) -> bool {
    let n = a.len();
    if n != b.len() || n < 3 {
        return false;
    }
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    if i + 1 >= n {
        return false;
    }
    a[i] == b[i + 1] && a[i + 1] == b[i] && a[(i + 2)..] == b[(i + 2)..]
}

/// O(L) dynamic-programming syllable splitter used as fallback when candidate-aligned `walk()`
/// fails (e.g. OOV character or non-standard spelling). Minimizes `(unmatched_partial_chars, units)`
/// so `n + guo/ge/gao` boundaries (`nanguo -> [nan, guo]`, `fanguolai -> [fan, guo, lai]`) are
/// preserved instead of greedily swallowing `*ng`.
fn greedy_split_chunk(
    chunk: &str,
    valid_syllables: &HashSet<String>,
    out: &mut Vec<(&'static str, String)>,
) {
    let n = chunk.len();
    if n == 0 {
        return;
    }
    // `dp[pos]` = best `(partial_chars, unit_count)` to cover `chunk[pos..]`.
    let mut dp = vec![(usize::MAX, usize::MAX); n + 1];
    let mut choice: Vec<(usize, &'static str, &str)> = vec![(0, "p", ""); n];
    dp[n] = (0, 0);

    for pos in (0..n).rev() {
        let max_len = 6.min(n - pos);
        for len in (1..=max_len).rev() {
            let sub = &chunk[pos..pos + len];
            // Avoid classifying isolated non-standard syllabic nasals / stray vowels (`m`, `n`,
            // `ng`, `o`, `hm`, `hng`) as full syllables when splitting unrecognized fragments.
            if matches!(sub, "m" | "n" | "ng" | "hm" | "hng" | "o") && n > len {
                continue;
            }
            let matched_syl: Option<&str> = if valid_syllables.contains(sub) {
                Some(sub)
            } else if let Some(canon) = canonical_alias(sub)
                && valid_syllables.contains(canon)
            {
                Some(canon)
            } else {
                None
            };
            if let Some(syl) = matched_syl {
                let (next_p, next_u) = dp[pos + len];
                let cand = (next_p, next_u.saturating_add(1));
                if cand < dp[pos] {
                    dp[pos] = cand;
                    choice[pos] = (pos + len, "f", syl);
                }
            }
        }

        let rem = &chunk[pos..];
        if rem.starts_with("zh") || rem.starts_with("ch") || rem.starts_with("sh") {
            let (next_p, next_u) = dp[pos + 2];
            let cand = (next_p.saturating_add(2), next_u.saturating_add(1));
            if cand < dp[pos] {
                dp[pos] = cand;
                choice[pos] = (pos + 2, "p", &rem[..2]);
            }
        }
        {
            let (next_p, next_u) = dp[pos + 1];
            let cand = (next_p.saturating_add(1), next_u.saturating_add(1));
            if cand < dp[pos] {
                dp[pos] = cand;
                choice[pos] = (pos + 1, "p", &rem[..1]);
            }
        }
    }

    let mut pos = 0;
    while pos < n {
        let (next, kind, syl) = choice[pos];
        out.push((kind, syl.to_string()));
        pos = next;
    }
}

/// Full-pinyin syllable recovery aligned to the leading candidate's characters via `readings`.
/// Respects explicit syllable separators (`'`, space), normalizes `lue`/`nue`/`[jqxy]v*` aliases
/// to canonical vocabulary spellings, supports bounded adjacent-transposition fuzzy recovery
/// (matching common `fuzz/` typo-correction algebra), and uses O(N*L) memoization.
pub fn full_pinyin_to_units(
    raw_input: &str,
    lead_text: &str,
    readings: &HashMap<String, Vec<String>>,
    valid_syllables: &HashSet<String>,
) -> Vec<(&'static str, String)> {
    // Build contiguous alpha code + bitmask of explicit separator positions so `walk` never crosses `'`
    let mut code = String::with_capacity(raw_input.len());
    let mut must_break = vec![false; raw_input.len() + 1];
    let mut saw_sep = false;
    for b in raw_input.bytes() {
        if b.is_ascii_alphabetic() {
            if saw_sep && !code.is_empty() {
                must_break[code.len()] = true;
            }
            saw_sep = false;
            code.push(b.to_ascii_lowercase() as char);
        } else if b == b'\'' || b == b' ' {
            saw_sep = true;
        }
    }
    must_break.truncate(code.len() + 1);

    let chars: Vec<char> = lead_text.chars().collect();
    let stride = code.len() + 1;
    let mut failed = vec![false; (chars.len() + 1) * stride];
    let mut out: Vec<(&'static str, String)> = Vec::with_capacity(chars.len());

    #[allow(clippy::too_many_arguments)]
    fn walk(
        i: usize,
        pos: usize,
        allow_fuzzy: bool,
        chars: &[char],
        code: &str,
        must_break: &[bool],
        readings: &HashMap<String, Vec<String>>,
        failed: &mut [bool],
        stride: usize,
        out: &mut Vec<(&'static str, String)>,
    ) -> bool {
        if i == chars.len() {
            return pos == code.len();
        }
        let state_idx = i * stride + pos;
        if failed[state_idx] {
            return false;
        }
        let mut ch_buf = [0u8; 4];
        let key = chars[i].encode_utf8(&mut ch_buf);
        let Some(rlist) = readings.get(key) else {
            failed[state_idx] = true;
            return false;
        };
        let last = i + 1 == chars.len();
        // Phase 1: Try ALL exact full-syllable readings across `rlist` before any fuzzy or partial
        // prefix match, so earlier longer polyphone readings (e.g. `称: ["cheng", "chen"]`) never
        // steal an exact shorter reading (`duichen` -> `对称`) as a partial prefix.
        for reading in rlist {
            if !reading.is_ascii() {
                continue;
            }
            let forms = [Some(reading.as_str()), user_input_alias(reading.as_str())];
            for form in forms.into_iter().flatten() {
                let flen = form.len();
                if pos + flen <= code.len() && code[pos..].starts_with(form) {
                    let crosses_sep = ((pos + 1)..(pos + flen)).any(|p| must_break[p]);
                    if !crosses_sep {
                        out.push(("f", reading.clone()));
                        if walk(
                            i + 1,
                            pos + flen,
                            allow_fuzzy,
                            chars,
                            code,
                            must_break,
                            readings,
                            failed,
                            stride,
                            out,
                        ) {
                            return true;
                        }
                        out.pop();
                    }
                }
            }
        }

        // Phase 2: When fuzzy recovery is enabled, try adjacent-letter transposition across all
        // canonical readings and `u`/`v` user aliases (`celeu` -> `策略`, `xavn` -> `选择`).
        if allow_fuzzy {
            for reading in rlist {
                if !reading.is_ascii() {
                    continue;
                }
                let forms = [Some(reading.as_str()), user_input_alias(reading.as_str())];
                for form in forms.into_iter().flatten() {
                    let flen = form.len();
                    if flen >= 3 && pos + flen <= code.len() {
                        let seg = &code.as_bytes()[pos..pos + flen];
                        if is_adjacent_transposition(seg, form.as_bytes()) {
                            let crosses_sep = ((pos + 1)..(pos + flen)).any(|p| must_break[p]);
                            if !crosses_sep {
                                out.push(("f", reading.clone()));
                                if walk(
                                    i + 1,
                                    pos + flen,
                                    allow_fuzzy,
                                    chars,
                                    code,
                                    must_break,
                                    readings,
                                    failed,
                                    stride,
                                    out,
                                ) {
                                    return true;
                                }
                                out.pop();
                            }
                        }
                    }
                }
            }
        }

        // Phase 3: Partial initial / trailing-prefix abbreviations after all full readings fail.
        for reading in rlist {
            if !reading.is_ascii() {
                continue;
            }
            let forms = [Some(reading.as_str()), user_input_alias(reading.as_str())];
            let rlen = reading.len();
            let b = reading.as_bytes();
            for k in (1..rlen).rev() {
                let is_initial =
                    k == 1 || (k == 2 && b[1] == b'h' && matches!(b[0], b'z' | b'c' | b's'));
                if !last && !is_initial {
                    continue;
                }
                for form in forms.into_iter().flatten() {
                    if k <= form.len()
                        && pos + k <= code.len()
                        && code[pos..].starts_with(&form[..k])
                    {
                        let crosses_sep = ((pos + 1)..(pos + k)).any(|p| must_break[p]);
                        if !crosses_sep {
                            out.push(("p", reading[..k].to_string()));
                            if walk(
                                i + 1,
                                pos + k,
                                allow_fuzzy,
                                chars,
                                code,
                                must_break,
                                readings,
                                failed,
                                stride,
                                out,
                            ) {
                                return true;
                            }
                            out.pop();
                        }
                    }
                }
            }
        }
        failed[state_idx] = true;
        false
    }

    if walk(
        0,
        0,
        false,
        &chars,
        &code,
        &must_break,
        readings,
        &mut failed,
        stride,
        &mut out,
    ) {
        return out;
    }

    // Second pass: allow single adjacent-letter transposition per syllable (e.g. `yuo->you`, `xaing->xiang`)
    failed.fill(false);
    out.clear();
    if walk(
        0,
        0,
        true,
        &chars,
        &code,
        &must_break,
        readings,
        &mut failed,
        stride,
        &mut out,
    ) {
        return out;
    }

    // Fallback: greedy longest-syllable split respecting explicit `'`/space chunks
    let mut greedy = Vec::new();
    for part in raw_input.split([' ', '\'']) {
        let chunk: String = part
            .bytes()
            .filter(|b| b.is_ascii_alphabetic())
            .map(|b| b.to_ascii_lowercase() as char)
            .collect();
        if !chunk.is_empty() {
            greedy_split_chunk(&chunk, valid_syllables, &mut greedy);
        }
    }
    greedy
}

/// Dispatch helper converting raw input into syllable units under any `PinyinLayout`.
pub fn input_to_units(
    layout: PinyinLayout,
    raw_input: &str,
    lead_text: &str,
    readings: &HashMap<String, Vec<String>>,
    valid_syllables: &HashSet<String>,
) -> Vec<(&'static str, String)> {
    if layout.is_double_pinyin() {
        double_pinyin_to_units(layout, raw_input, valid_syllables)
    } else {
        full_pinyin_to_units(raw_input, lead_text, readings, valid_syllables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_valid_syllables() -> HashSet<String> {
        [
            "a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er", "zhu", "zhui",
            "zhuang", "zhuai", "chuan", "shuo", "shuang", "shui", "nv", "lv", "lve", "nve", "jue",
            "que", "xue", "yue", "ju", "qu", "xu", "yu", "suo", "bo", "po", "mo", "fo", "wo", "yo",
            "rua", "ting", "ming", "jing", "liang", "jiang", "qiang", "xiang", "kuai", "guai",
            "ni", "hao", "zhong", "guo", "ren", "min", "tian", "qi", "dian", "jia", "bian", "yi",
            "xiong", "qiong", "jiong", "long", "dong", "kong", "gong", "hong", "nong", "liu",
            "jiu", "qiu", "xiu", "diu", "niu", "miu", "bie", "pie", "mie", "die", "tie", "nie",
            "lie", "jie", "qie", "xie", "piao", "miao", "diao", "tiao", "niao", "liao", "jiao",
            "qiao", "xiao", "kun", "gun", "hun", "dun", "tun", "lun", "zun", "cun", "sun", "run",
            "zhun", "chun", "shun", "xi", "xian", "fang", "fan", "gan", "nai", "ce", "lu", "dai",
            "ding", "jin", "yao", "juan", "quan", "xuan", "yuan", "jun", "qun", "xun", "yun", "ze",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    #[test]
    fn test_all_seven_double_pinyin_layouts() {
        let valid = sample_valid_syllables();

        // 1. ZRM (including rime-ice / rime-frost self-leading zero-initial aliases aj/ak/al/ef/ez!)
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'v', b'u', &valid).as_deref(),
            Some("zhu")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'a', b'j', &valid).as_deref(),
            Some("an")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'a', b'k', &valid).as_deref(),
            Some("ao")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'a', b'l', &valid).as_deref(),
            Some("ai")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'e', b'f', &valid).as_deref(),
            Some("en")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'e', b'z', &valid).as_deref(),
            Some("ei")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'o', b'b', &valid).as_deref(),
            Some("ou")
        );
        assert_eq!(
            double_pinyin_to_units(PinyinLayout::Zrm, "ajnl", &valid),
            vec![("f", "an".to_string()), ("f", "nai".to_string())]
        );

        // 2. Flypy (both self-leading and o-leading aliases)
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Flypy, b't', b'k', &valid).as_deref(),
            Some("ting")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Flypy, b'a', b'd', &valid).as_deref(),
            Some("ai")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Flypy, b'o', b'j', &valid).as_deref(),
            Some("an")
        );

        // 3. MSPY (uses `;` for `ing`)
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Mspy, b't', b';', &valid).as_deref(),
            Some("ting")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Mspy, b'n', b'y', &valid).as_deref(),
            Some("nv")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Mspy, b'j', b'v', &valid).as_deref(),
            Some("jue")
        );

        // 4. Sogou
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Sogou, b'm', b';', &valid).as_deref(),
            Some("ming")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Sogou, b'l', b't', &valid).as_deref(),
            Some("lve")
        );

        // 5. ABC
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Abc, b'a', b's', &valid).as_deref(),
            Some("zhong")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Abc, b'o', b'h', &valid).as_deref(),
            Some("ang")
        );

        // 6. Ziguang (uses `;` for `ing`)
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Ziguang, b'u', b'h', &valid).as_deref(),
            Some("zhong")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Ziguang, b't', b';', &valid).as_deref(),
            Some("ting")
        );

        // 7. Jiajia (including `double_pinyin_pyjj` alias)
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Jiajia, b'v', b'y', &valid).as_deref(),
            Some("zhong")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Jiajia, b'e', b'r', &valid).as_deref(),
            Some("en")
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Jiajia, b'e', b'q', &valid).as_deref(),
            Some("er")
        );

        // Auto-detection from schema_id
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "double_pinyin_flypy"),
            PinyinLayout::Flypy
        );
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "double_pinyin_pyjj"),
            PinyinLayout::Jiajia
        );
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "rime_frost_double_pinyin"),
            PinyinLayout::Zrm
        );
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "rime_ice"),
            PinyinLayout::Full
        );
    }

    #[test]
    fn test_full_pinyin_apostrophe_and_memoized_worst_case() {
        let valid = sample_valid_syllables();
        let mut readings: HashMap<String, Vec<String>> = HashMap::new();
        readings.insert("西".into(), vec!["xi".into()]);
        readings.insert("安".into(), vec!["an".into()]);
        readings.insert("先".into(), vec!["xian".into()]);
        readings.insert("中".into(), vec!["zhong".into()]);
        readings.insert("国".into(), vec!["guo".into()]);

        // Explicit apostrophe must keep `xi'an` as two syllables, never merged into `xian`
        assert_eq!(
            full_pinyin_to_units("xi'an", "西安", &readings, &valid),
            vec![("f", "xi".to_string()), ("f", "an".to_string())]
        );

        // Adversarial 24-char sentence with mismatched tail: memoization ensures < 1 ms, no exponential blowup
        let mut poly: HashMap<String, Vec<String>> = HashMap::new();
        poly.insert(
            "中".into(),
            vec!["z".into(), "zh".into(), "zho".into(), "zhong".into()],
        );
        let lead = "中".repeat(24);
        let bad_code = "z".repeat(23) + "q"; // forces full search failure at last char
        let t0 = std::time::Instant::now();
        let _ = full_pinyin_to_units(&bad_code, &lead, &poly, &valid);
        assert!(
            t0.elapsed().as_millis() < 50,
            "memoized full_pinyin_to_units took too long: {:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn test_full_pinyin_uv_aliases() {
        let valid = sample_valid_syllables();
        let mut readings: HashMap<String, Vec<String>> = HashMap::new();
        readings.insert("策".into(), vec!["ce".into()]);
        readings.insert("略".into(), vec!["lve".into()]);
        readings.insert("侧".into(), vec!["ce".into()]);
        readings.insert("录".into(), vec!["lu".into()]);
        readings.insert("额".into(), vec!["e".into()]);
        readings.insert("虐".into(), vec!["nve".into()]);
        readings.insert("待".into(), vec!["dai".into()]);
        readings.insert("决".into(), vec!["jue".into()]);
        readings.insert("定".into(), vec!["ding".into()]);
        readings.insert("需".into(), vec!["xu".into()]);
        readings.insert("要".into(), vec!["yao".into()]);
        readings.insert("选".into(), vec!["xuan".into()]);
        readings.insert("择".into(), vec!["ze".into()]);

        // `celue` aligned to `策略` recovers canonical `lve` instead of falling back to `ce|lu|e`
        assert_eq!(
            full_pinyin_to_units("celue", "策略", &readings, &valid),
            vec![("f", "ce".to_string()), ("f", "lve".to_string())]
        );
        // Ambiguous 3-syllable reading `侧录额` still aligns cleanly to `ce|lu|e`
        assert_eq!(
            full_pinyin_to_units("celue", "侧录额", &readings, &valid),
            vec![
                ("f", "ce".to_string()),
                ("f", "lu".to_string()),
                ("f", "e".to_string()),
            ]
        );
        assert_eq!(
            full_pinyin_to_units("nuedai", "虐待", &readings, &valid),
            vec![("f", "nve".to_string()), ("f", "dai".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("jveding", "决定", &readings, &valid),
            vec![("f", "jue".to_string()), ("f", "ding".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("xvyao", "需要", &readings, &valid),
            vec![("f", "xu".to_string()), ("f", "yao".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("xvanze", "选择", &readings, &valid),
            vec![("f", "xuan".to_string()), ("f", "ze".to_string())]
        );
        // Explicit separator fast path normalizes `lue` -> `lve`
        assert_eq!(
            full_pinyin_to_units("ce'lue", "策略", &readings, &valid),
            vec![("f", "ce".to_string()), ("f", "lve".to_string())]
        );
        // Greedy fallback normalizes `lue` -> `lve` when candidate text has unknown chars
        assert_eq!(
            full_pinyin_to_units("celue", "？？", &readings, &valid),
            vec![("f", "ce".to_string()), ("f", "lve".to_string())]
        );
    }

    #[test]
    fn test_semicolon_double_pinyin_sequence() {
        let valid = sample_valid_syllables();
        let units = double_pinyin_to_units(PinyinLayout::Mspy, "m;tm", &valid);
        assert_eq!(
            units,
            vec![("f", "ming".to_string()), ("f", "tian".to_string())]
        );
    }

    fn encode_for_test(layout: PinyinLayout, syl: &str) -> Option<(u8, u8)> {
        if let Some(pair) = match (layout, syl) {
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "a",
            ) => Some((b'a', b'a')),
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "e",
            ) => Some((b'e', b'e')),
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "o",
            ) => Some((b'o', b'o')),
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "ai",
            ) => Some((b'a', b'i')),
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "an",
            ) => Some((b'a', b'n')),
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "ao",
            ) => Some((b'a', b'o')),
            (
                PinyinLayout::Zrm | PinyinLayout::Flypy | PinyinLayout::Mspy | PinyinLayout::Sogou,
                "ang",
            ) => Some((b'a', b'h')),
            (PinyinLayout::Jiajia, "ang") => Some((b'a', b'g')),
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "ei",
            ) => Some((b'e', b'i')),
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "en",
            ) => Some((b'e', b'n')),
            (
                PinyinLayout::Zrm | PinyinLayout::Flypy | PinyinLayout::Mspy | PinyinLayout::Sogou,
                "er",
            ) => Some((b'e', b'r')),
            (PinyinLayout::Jiajia, "er") => Some((b'e', b'q')),
            (
                PinyinLayout::Zrm | PinyinLayout::Flypy | PinyinLayout::Mspy | PinyinLayout::Sogou,
                "eng",
            ) => Some((b'e', b'g')),
            (PinyinLayout::Jiajia, "eng") => Some((b'e', b't')),
            (
                PinyinLayout::Zrm
                | PinyinLayout::Flypy
                | PinyinLayout::Mspy
                | PinyinLayout::Sogou
                | PinyinLayout::Jiajia,
                "ou",
            ) => Some((b'o', b'u')),
            (PinyinLayout::Abc, s) => ZERO_ABC
                .iter()
                .find(|&&(_, _, z)| z == s)
                .map(|&(a, b, _)| (a, b)),
            (PinyinLayout::Ziguang, s) => ZERO_ZIGUANG
                .iter()
                .find(|&&(_, _, z)| z == s)
                .map(|&(a, b, _)| (a, b)),
            _ => None,
        } {
            return Some(pair);
        }

        let spec = layout_spec(layout)?;
        let (ini, fin) = if let Some(rest) = syl.strip_prefix("zh") {
            (spec.zh_key, rest)
        } else if let Some(rest) = syl.strip_prefix("ch") {
            (spec.ch_key, rest)
        } else if let Some(rest) = syl.strip_prefix("sh") {
            (spec.sh_key, rest)
        } else {
            let b0 = syl.as_bytes()[0];
            if matches!(b0, b'a' | b'e' | b'o') {
                return None;
            }
            (b0, &syl[1..])
        };

        let search_fin = match (syl.as_bytes()[0], fin) {
            (b'j' | b'q' | b'x' | b'y', "ue") => "ve",
            _ => fin,
        };
        for k in (b'a'..=b'z').chain(std::iter::once(b';')) {
            if (spec.finals_fn)(k).contains(&search_fin) || (spec.finals_fn)(k).contains(&fin) {
                return Some((ini, k));
            }
        }
        None
    }

    #[test]
    fn test_exhaustive_roundtrip_all_7_layouts() {
        let mut valid = sample_valid_syllables();
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("unified_vocab.json");
        if fixture.is_file() {
            let vocab: serde_json::Value =
                serde_json::from_reader(std::fs::File::open(fixture).unwrap()).unwrap();
            for unit in vocab["units"].as_array().unwrap() {
                if let Some(syl) = unit.as_str().unwrap().strip_prefix("f:") {
                    valid.insert(syl.to_owned());
                }
            }
        }
        let skip = [
            "hm", "hng", "m", "n", "ng", "lo", "nou", "den", "kei", "dia", "tei", "fiao", "chua",
        ];
        for layout in [
            PinyinLayout::Zrm,
            PinyinLayout::Flypy,
            PinyinLayout::Mspy,
            PinyinLayout::Sogou,
            PinyinLayout::Abc,
            PinyinLayout::Ziguang,
            PinyinLayout::Jiajia,
        ] {
            let mut tested = 0usize;
            for syl in &valid {
                if skip.contains(&syl.as_str()) {
                    continue;
                }
                if let Some((k1, k2)) = encode_for_test(layout, syl) {
                    let dec = dp_pair_to_syllable(layout, k1, k2, &valid);
                    assert_eq!(
                        dec.as_deref(),
                        Some(syl.as_str()),
                        "Roundtrip mismatch on {:?}: {} -> ({},{}) -> {:?}",
                        layout,
                        syl,
                        k1 as char,
                        k2 as char,
                        dec
                    );
                    tested += 1;
                }
            }
            assert!(
                tested >= 90,
                "Too few syllables tested for {:?}: {}",
                layout,
                tested
            );
        }
    }

    #[test]
    fn test_full_pinyin_separator_with_abbrev_and_partial_split() {
        let mut valid = sample_valid_syllables();
        for s in ["fang", "an", "xi", "xian", "shi"] {
            valid.insert(s.to_string());
        }
        let mut readings: HashMap<String, Vec<String>> = HashMap::new();
        readings.insert("方".to_string(), vec!["fang".to_string()]);
        readings.insert("案".to_string(), vec!["an".to_string()]);
        readings.insert("西".to_string(), vec!["xi".to_string()]);
        readings.insert("安".to_string(), vec!["an".to_string()]);
        readings.insert("市".to_string(), vec!["shi".to_string()]);

        // `fang'a` -> 方案: second slot must be partial `a` (for `an`), not full syllable `a`
        let u1 = full_pinyin_to_units("fang'a", "方案", &readings, &valid);
        assert_eq!(u1, vec![("f", "fang".to_string()), ("p", "a".to_string())]);

        // `xian'shi` -> 西安市: 3-char lead_text aligns to [xi, an, shi], not 2-slot [xian, shi]
        let u2 = full_pinyin_to_units("xian'shi", "西安市", &readings, &valid);
        assert_eq!(
            u2,
            vec![
                ("f", "xi".to_string()),
                ("f", "an".to_string()),
                ("f", "shi".to_string())
            ]
        );
    }

    #[test]
    fn test_greedy_split_preserves_zh_ch_sh_initials() {
        let mut valid = sample_valid_syllables();
        valid.insert("wen".to_string());
        valid.insert("n".to_string());
        valid.insert("o".to_string());
        let readings: HashMap<String, Vec<String>> = HashMap::new();
        // `zhonwen` (fuzzy `zhong` -> `zhon`) falls back to greedy split; must keep `zh` intact
        // and must NOT emit bogus full syllables `f:o`/`f:n`
        let u = full_pinyin_to_units("zhonwen", "中文", &readings, &valid);
        assert_eq!(u[0], ("p", "zh".to_string()));
        assert_eq!(u.last().unwrap(), &("f", "wen".to_string()));
        assert!(!u.iter().any(|(k, s)| *k == "f" && (s == "o" || s == "n")));
    }

    #[test]
    fn test_full_pinyin_fuzzy_transposition_alignment() {
        let valid = sample_valid_syllables();
        let mut readings: HashMap<String, Vec<String>> = HashMap::new();
        readings.insert("我".to_string(), vec!["wo".to_string()]);
        readings.insert("们".to_string(), vec!["men".to_string()]);
        readings.insert("有".to_string(), vec!["you".to_string()]);
        readings.insert("钱".to_string(), vec!["qian".to_string()]);
        readings.insert("今".to_string(), vec!["jin".to_string()]);
        readings.insert("天".to_string(), vec!["tian".to_string()]);
        readings.insert("想".to_string(), vec!["xiang".to_string()]);
        readings.insert("去".to_string(), vec!["qu".to_string()]);

        let u1 = full_pinyin_to_units("womenyuoqian", "我们有钱", &readings, &valid);
        assert_eq!(
            u1,
            vec![
                ("f", "wo".to_string()),
                ("f", "men".to_string()),
                ("f", "you".to_string()),
                ("f", "qian".to_string()),
            ]
        );
        let u2 = full_pinyin_to_units("jintianxaingqu", "今天想去", &readings, &valid);
        assert_eq!(
            u2,
            vec![
                ("f", "jin".to_string()),
                ("f", "tian".to_string()),
                ("f", "xiang".to_string()),
                ("f", "qu".to_string()),
            ]
        );
    }

    #[test]
    fn test_double_pinyin_respects_apostrophe_chunks_and_layout_tokens() {
        let valid = sample_valid_syllables();
        let u = double_pinyin_to_units(PinyinLayout::Zrm, "w'm'jntm", &valid);
        assert_eq!(
            u,
            vec![
                ("p", "w".to_string()),
                ("p", "m".to_string()),
                ("f", "jin".to_string()),
                ("f", "tian".to_string()),
            ]
        );
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "my_abacus_pinyin"),
            PinyinLayout::Full
        );
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "double_pinyin_abc"),
            PinyinLayout::Abc
        );
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "double_pinyin_zg"),
            PinyinLayout::Ziguang
        );
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "double_pinyin_ms"),
            PinyinLayout::Mspy
        );
        assert_eq!(
            PinyinLayout::from_name_or_schema("auto", "double_pinyin_jj"),
            PinyinLayout::Jiajia
        );
    }

    #[test]
    fn test_polyphone_exact_full_beats_earlier_partial() {
        let valid = sample_valid_syllables();
        let mut readings: HashMap<String, Vec<String>> = HashMap::new();
        readings.insert("对".into(), vec!["dui".into()]);
        readings.insert("称".into(), vec!["cheng".into(), "chen".into()]);
        readings.insert("勾".into(), vec!["gou".into()]);
        readings.insert("勒".into(), vec!["lei".into(), "le".into()]);
        readings.insert("心".into(), vec!["xin".into()]);
        readings.insert("宽".into(), vec!["kuan".into()]);
        readings.insert("体".into(), vec!["ti".into()]);
        readings.insert("胖".into(), vec!["pang".into(), "pan".into()]);
        readings.insert("解".into(), vec!["jie".into()]);
        readings.insert("剖".into(), vec!["pou".into(), "po".into()]);
        readings.insert("水".into(), vec!["shui".into()]);
        readings.insert("杉".into(), vec!["shan".into(), "sha".into()]);
        // Non-ASCII reading entry must never panic on UTF-8 slicing
        readings.insert("绿".into(), vec!["lü".into(), "lv".into()]);

        assert_eq!(
            full_pinyin_to_units("duichen", "对称", &readings, &valid),
            vec![("f", "dui".to_string()), ("f", "chen".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("goule", "勾勒", &readings, &valid),
            vec![("f", "gou".to_string()), ("f", "le".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("xinkuantipan", "心宽体胖", &readings, &valid),
            vec![
                ("f", "xin".to_string()),
                ("f", "kuan".to_string()),
                ("f", "ti".to_string()),
                ("f", "pan".to_string()),
            ]
        );
        assert_eq!(
            full_pinyin_to_units("jiepo", "解剖", &readings, &valid),
            vec![("f", "jie".to_string()), ("f", "po".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("shuisha", "水杉", &readings, &valid),
            vec![("f", "shui".to_string()), ("f", "sha".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("lv", "绿", &readings, &valid),
            vec![("f", "lv".to_string())]
        );
    }

    #[test]
    fn test_invalid_dp_pair_returns_none_and_emits_partial_initial() {
        let valid = sample_valid_syllables();
        // Invalid Mandarin syllables must return None rather than bogus strings like "bou"/"fian"
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'b', b'b', &valid),
            None
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'f', b'm', &valid),
            None
        );
        assert_eq!(
            dp_pair_to_syllable(PinyinLayout::Zrm, b'b', b'p', &valid),
            None
        );
        assert_eq!(
            double_pinyin_to_units(PinyinLayout::Zrm, "bbni", &valid),
            vec![("p", "b".to_string()), ("f", "ni".to_string())]
        );
    }

    #[test]
    fn test_dp_fallback_splitter_recovers_ng_boundaries() {
        let mut valid = sample_valid_syllables();
        for s in [
            "nan", "nang", "san", "sang", "han", "hang", "fan", "fang", "guo", "lai",
        ] {
            valid.insert(s.to_string());
        }
        let readings: HashMap<String, Vec<String>> = HashMap::new();
        assert_eq!(
            full_pinyin_to_units("nanguo", "？？", &readings, &valid),
            vec![("f", "nan".to_string()), ("f", "guo".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("sanguo", "？？", &readings, &valid),
            vec![("f", "san".to_string()), ("f", "guo".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("hanguo", "？？", &readings, &valid),
            vec![("f", "han".to_string()), ("f", "guo".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("fanguolai", "？？？", &readings, &valid),
            vec![
                ("f", "fan".to_string()),
                ("f", "guo".to_string()),
                ("f", "lai".to_string()),
            ]
        );
    }

    #[test]
    fn test_full_pinyin_fuzzy_transposition_with_uv_aliases() {
        let valid = sample_valid_syllables();
        let mut readings: HashMap<String, Vec<String>> = HashMap::new();
        readings.insert("策".into(), vec!["ce".into()]);
        readings.insert("略".into(), vec!["lve".into()]);
        readings.insert("选".into(), vec!["xuan".into()]);
        readings.insert("择".into(), vec!["ze".into()]);

        assert_eq!(
            full_pinyin_to_units("celeu", "策略", &readings, &valid),
            vec![("f", "ce".to_string()), ("f", "lve".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("xavn", "选", &readings, &valid),
            vec![("f", "xuan".to_string())]
        );
        assert_eq!(
            full_pinyin_to_units("xavnze", "选择", &readings, &valid),
            vec![("f", "xuan".to_string()), ("f", "ze".to_string())]
        );
    }
}
