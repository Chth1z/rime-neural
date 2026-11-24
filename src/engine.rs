use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use ort::environment::Environment;
use ort::session::{
    HasSelectedOutputs, OutputSelector, RunOptions, Session, builder::GraphOptimizationLevel,
};
use ort::value::TensorRef;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::metaspace::PrependScheme;

use crate::gram::{WanxiangGram, sanitize_max_length, sanitize_penalties};
use crate::pinyin::{PinyinLayout, input_to_units};

const CONTEXT_CHARS: usize = 128;
const LRU_CAPACITY: usize = 256;
const KV_CACHE_CAPACITY: usize = 16;

#[inline]
fn canon_f64_bits(x: f64) -> u64 {
    let v = if !x.is_finite() || x == 0.0 { 0.0 } else { x };
    v.to_bits()
}

#[inline]
fn canon_f32_bits(x: f32) -> u32 {
    let v = if !x.is_finite() || x == 0.0 { 0.0 } else { x };
    v.to_bits()
}

#[derive(Deserialize)]
struct MlSubConfig {
    #[serde(default = "default_old_chars")]
    old_chars: usize,
    #[serde(default)]
    pinyin: HashMap<String, Vec<String>>,
}

fn default_old_chars() -> usize {
    6163
}

#[derive(Deserialize)]
struct UnifiedVocabFile {
    units: Vec<String>,
    #[serde(default)]
    chars: Vec<String>,
    readings: HashMap<String, Vec<String>>,
    #[serde(default = "default_max_slots")]
    max_slots: usize,
    #[serde(default = "default_ctx_tokens")]
    choice_context_tokens: usize,
    #[serde(default = "default_cand_tokens")]
    choice_candidate_tokens: usize,
    #[serde(default)]
    choice_compact: bool,
    #[serde(default)]
    ml: Option<MlSubConfig>,
}

fn default_max_slots() -> usize {
    24
}
fn default_ctx_tokens() -> usize {
    128
}
fn default_cand_tokens() -> usize {
    16
}

#[derive(Clone, Copy)]
pub struct FusionWeights {
    pub w_logit: f64,
    pub w_rank: f64,
    pub w_gram: f64,
    pub w_char_lp: f64,
    pub w_ctx_lp: f64,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            w_logit: 1.0,
            w_rank: -1.4,
            w_gram: 0.05,
            w_char_lp: 0.15,
            w_ctx_lp: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct P2cConfig {
    /// Minimum Pass-1 `char_logits` margin to trigger KV-cached Pass-2 cross-encoder check (<= 0 disables Pass-2).
    pub verify_margin: f32,
    /// Minimum Pass-2 cross-encoder logit lead over Rime's best candidate to promote P2C to #1.
    pub top1_logit_diff: f32,
    /// Minimum Wanxiang `.gram` per-char score delta floor to allow #1 promotion.
    pub top1_min_gram_diff: f32,
    /// Minimum Pass-2 cross-encoder logit lead to insert P2C safely at #2.
    pub top2_logit_diff: f32,
    /// Legacy Pass-1-only fallback margins (used ONLY on non-KV whole graphs).
    pub legacy_top1_margin: f32,
    pub legacy_top2_margin: f32,
}

impl Default for P2cConfig {
    fn default() -> Self {
        Self {
            verify_margin: 0.20,
            top1_logit_diff: 2.0,
            top1_min_gram_diff: -2.0,
            top2_logit_diff: 0.5,
            legacy_top1_margin: 1.0,
            legacy_top2_margin: 0.45,
        }
    }
}

#[derive(Clone)]
pub struct InitConfig {
    pub model_dir: PathBuf,
    pub ort_dll: PathBuf,
    pub gram_path: Option<PathBuf>,
    pub gram_collocation_penalty: f64,
    pub gram_non_collocation_penalty: f64,
    pub gram_max_length: usize,
    pub model_file: Option<String>,
    pub threads: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GramRequest {
    /// Use the default `.gram` configured at engine initialization.
    Default,
    /// Explicitly disable `.gram` scoring for this schema.
    Disabled,
    /// Evaluate with an explicit `.gram` path and per-schema penalties / collocation window.
    Custom {
        paths: [Option<PathBuf>; 2],
        collocation_penalty: u64,
        non_collocation_penalty: u64,
        max_length: usize,
    },
}

impl GramRequest {
    pub fn custom(
        paths: [Option<PathBuf>; 2],
        collocation_penalty: f64,
        non_collocation_penalty: f64,
        max_length: usize,
    ) -> Self {
        let (col, non_col) = sanitize_penalties(collocation_penalty, non_collocation_penalty);
        Self::Custom {
            paths,
            collocation_penalty: canon_f64_bits(col),
            non_collocation_penalty: canon_f64_bits(non_col),
            max_length: sanitize_max_length(max_length),
        }
    }
}

#[derive(Clone)]
pub struct RerankResult {
    pub order: Vec<usize>,
    pub p2c_text: Option<String>,
    pub p2c_pos: u8, // 0 = none, 1 = top-1, 2 = top-2
    pub logits: Vec<f32>,
    pub gram_deltas: Vec<f32>,
    pub char_lps: Vec<f32>,
    pub has_ctx: bool,
    pub has_gram: bool,
}

#[derive(Clone)]
struct CachedEval {
    logits: Vec<f32>,
    gram_deltas: Vec<f32>,
    char_lps: Vec<f32>,
    p2c_text: Option<String>,
    p2c_margin: f32,
    p2c_verified: Option<(f32, f32)>,
    used_fused_kv: bool,
    has_gram: bool,
    last_used: u64,
}

#[derive(Default, Debug, Hash, PartialEq, Eq, Clone)]
struct ResolvedGramKey {
    path: Option<PathBuf>,
    col_bits: u64,
    non_col_bits: u64,
    max_length: usize,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
struct CacheKey {
    ctx_tail: String,
    raw_input: String,
    layout: PinyinLayout,
    weights_bits: [u64; 5],
    p2c_verify_margin_bits: u32,
    gram_key: ResolvedGramKey,
    cands: Vec<String>,
}

struct CharVocab {
    chars: Vec<String>,
    char_to_id: HashMap<char, usize>,
    /// All character IDs (`cid < chars.len()`, including extended/Traditional chars) per syllable.
    syl_to_all_cids: HashMap<String, Vec<usize>>,
    old_chars: usize,
    max_slots: usize,
}

impl CharVocab {
    /// Decode candidate character log-probabilities and greedy P2C synthesis directly from ORT's
    /// borrowed `char_logits` slice without copying the full `[n_slots, vocab_chars]` tensor.
    #[allow(clippy::too_many_arguments)]
    fn decode_char_logits(
        &self,
        c_slice: &[f32],
        shape: (usize, usize),
        units: &[(&'static str, String)],
        cands: &[String],
        readings: &HashMap<String, Vec<String>>,
        orig_units_len: usize,
        shifted_lcp: bool,
    ) -> (Vec<f32>, Option<String>, f32) {
        let (n_slots, vocab_chars) = shape;
        let neutral = || (vec![0.0; cands.len()], None, -100.0);
        if shifted_lcp
            || n_slots == 0
            || n_slots != orig_units_len
            || n_slots > self.max_slots
            || vocab_chars < self.old_chars
            || c_slice.len() < n_slots.saturating_mul(vocab_chars)
            || units.iter().any(|(kind, _)| *kind != "f")
            || cands.is_empty()
            || cands.iter().any(|c| c.chars().count() != n_slots)
        {
            return neutral();
        }

        let mut cand_chars: Vec<_> = cands.iter().map(|c| c.chars()).collect();
        let mut scores = vec![0.0f64; cands.len()];
        let mut generated = String::new();
        let mut gen_score = 0.0;
        let mut ch_buf = [0u8; 4];
        for (j, (_, syl)) in units.iter().enumerate() {
            let Some(ids) = self.syl_to_all_cids.get(syl) else {
                return neutral();
            };
            let allowed = &ids[..ids.partition_point(|&id| id < vocab_chars)];
            let row = &c_slice[j * vocab_chars..(j + 1) * vocab_chars];
            let mut max_v = f32::NEG_INFINITY;
            let mut best_core = None;
            for &id in allowed {
                let v = row[id];
                if v.is_finite() && v > max_v {
                    max_v = v;
                    // IDs are sorted, so the last core maximum is the best synthesis choice.
                    if id < self.old_chars {
                        best_core = Some(id);
                    }
                }
            }
            if !max_v.is_finite() {
                return neutral();
            }
            if let Some(id) = best_core {
                generated.push_str(&self.chars[id]);
                gen_score += f64::from(row[id]) - f64::from(max_v);
            }

            // Every aligned sentence has the same per-slot log normalizer. Centering on the
            // row maximum preserves all score differences, including those for extended IDs.
            // Only the fixed -15 log-probability for an unscored character needs the partition.
            let mut missing_score = None;
            for (chars, score) in cand_chars.iter_mut().zip(&mut scores) {
                let ch = chars.next().unwrap();
                if let Some(&id) = self.char_to_id.get(&ch)
                    && allowed.binary_search(&id).is_ok()
                    && row[id].is_finite()
                {
                    *score += f64::from(row[id]) - f64::from(max_v);
                } else {
                    let key: &str = ch.encode_utf8(&mut ch_buf);
                    if readings.get(key).is_some_and(|rlist| !rlist.contains(syl)) {
                        // A valid alternate segmentation cannot be scored on these slots.
                        return neutral();
                    }
                    *score += *missing_score.get_or_insert_with(|| {
                        let sum: f64 = allowed
                            .iter()
                            .map(|&id| row[id])
                            .filter(|v| v.is_finite())
                            .map(|v| (f64::from(v) - f64::from(max_v)).exp())
                            .sum();
                        sum.ln() - 15.0
                    });
                }
            }
        }

        let baseline = scores[0];
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let char_lps = scores
            .into_iter()
            .map(|s| ((s - baseline) / n_slots as f64) as f32)
            .collect();
        if generated.chars().count() == n_slots && !cands.contains(&generated) {
            let margin = ((gen_score - best) / n_slots as f64) as f32;
            (char_lps, Some(generated), margin)
        } else {
            (char_lps, None, -100.0)
        }
    }
}

struct ScratchBufs {
    ids: Vec<i64>,
    uids: Vec<i64>,
    slots: Vec<i64>,
    markers: Vec<i64>,
    att: Vec<i64>,
    mm: Vec<bool>,
    cand_str: String,
}

impl ScratchBufs {
    fn new(max_slots: usize) -> Self {
        Self {
            ids: Vec::with_capacity(256),
            uids: Vec::with_capacity(256),
            slots: Vec::with_capacity(max_slots.max(4)),
            markers: Vec::with_capacity(16),
            att: Vec::with_capacity(256),
            mm: Vec::with_capacity(16),
            cand_str: String::with_capacity(64),
        }
    }
}

type ForwardResult = (Vec<f32>, Vec<f32>, Option<String>, f32);

struct ContextKv {
    p: usize,
    enc_k: Vec<f32>,
    enc_v: Vec<f32>,
    choice_k: Vec<f32>,
    choice_v: Vec<f32>,
}

pub struct EngineStats {
    pub calls: u64,
    pub cache_hits: u64,
    pub kv_hits: u64,
    pub p2c_pass2_calls: u64,
    pub total_infer_ms: f64,
    pub fused_kv: bool,
    pub has_gram: bool,
    pub enc_layers: usize,
    pub choice_compact: bool,
    pub model_name: String,
}

pub struct NeuralEngine {
    model_dir: PathBuf,
    session: Session,
    use_fused_kv: bool,
    has_char_logits: bool,
    enc_layers: usize,
    choice_layers: usize,
    num_heads: usize,
    head_dim: usize,
    present_names: [&'static str; 4],
    z_enc: Vec<f32>,
    z_ch: Vec<f32>,
    true_mask: Vec<bool>,
    scratch: ScratchBufs,
    ctx_run_opts: Option<RunOptions<HasSelectedOutputs>>,
    key_run_opts: Option<RunOptions<HasSelectedOutputs>>,
    pass2_run_opts: Option<RunOptions<HasSelectedOutputs>>,
    tokenizer: Tokenizer,
    tokenizer_no_prepend: Option<Tokenizer>,
    special_tokens: Vec<String>,
    special_first_bytes: [bool; 256],
    forbidden_special_ids: HashSet<i64>,
    choice_compact: bool,
    full_unit_ids: HashMap<String, i64>,
    partial_unit_ids: HashMap<String, i64>,
    char_vocab: CharVocab,
    readings: HashMap<String, Vec<String>>,
    valid_syllables: HashSet<String>,
    max_slots: usize,
    context_tokens: usize,
    candidate_tokens: usize,
    cls_id: i64,
    sep_id: i64,
    mask_id: i64,
    unk_id: i64,
    gram_cache: HashMap<PathBuf, WanxiangGram>,
    default_gram_path: Option<PathBuf>,
    default_gram_penalties: (f64, f64),
    default_gram_max_len: usize,
    last_ctx_str: String,
    last_ctx_ids: Vec<i64>,
    kv_cache: VecDeque<(Vec<i64>, ContextKv)>,
    lru_map: HashMap<CacheKey, CachedEval>,
    lru_clock: u64,
    pub stats: EngineStats,
}

static ORT_ENV: OnceLock<Environment> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());
static ENGINE: OnceLock<Mutex<NeuralEngine>> = OnceLock::new();
static ASYNC_STARTED: AtomicBool = AtomicBool::new(false);
static LAST_INIT_ERROR: Mutex<Option<String>> = Mutex::new(None);

fn set_last_init_error(err: Option<String>) {
    if let Ok(mut guard) = LAST_INIT_ERROR.lock() {
        *guard = err;
    }
}

pub fn last_init_error() -> Option<String> {
    LAST_INIT_ERROR.lock().ok().and_then(|g| g.clone())
}

/// Retryable ORT environment initialization: transient errors (e.g. temporarily missing DLL)
/// are not cached permanently in `ORT_ENV`. Caller holds `INIT_LOCK`.
fn ensure_ort_env(ort_dll: &Path) -> Result<&'static Environment, String> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let env = ort::init_from(ort_dll)
        .map_err(|e| format!("ort::init_from({}) failed: {}", ort_dll.display(), e))?
        .with_name("rime_neural")
        .with_telemetry(false)
        .build()
        .map_err(|e| format!("ort env build failed: {}", e))?;
    let _ = ORT_ENV.set(env);
    ORT_ENV
        .get()
        .ok_or_else(|| "ORT_ENV initialization race".to_string())
}

fn resolve_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested.clamp(1, 16);
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .clamp(4, 8)
}

fn resolve_model_path(model_dir: &Path, override_name: Option<&str>) -> Result<PathBuf, String> {
    if let Some(name) = override_name.filter(|s| !s.is_empty()) {
        let p = model_dir.join(name);
        if p.is_file() {
            return Ok(p);
        }
        return Err(format!("configured model_file not found: {}", p.display()));
    }
    for cand in [
        "laya.one.fused.int8g.onnx",
        "laya.one.int8g.onnx",
        "laya.one.w4.onnx",
        "laya.int8g.onnx",
    ] {
        let p = model_dir.join(cand);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(format!(
        "no supported ONNX model found in {}",
        model_dir.display()
    ))
}

/// Recursively set `PrependScheme::Never` on any `Metaspace` pre-tokenizer (including inside
/// `PreTokenizerWrapper::Sequence`). Returns `true` if at least one `Metaspace` was updated.
fn disable_metaspace_prepend(pt: &mut PreTokenizerWrapper) -> bool {
    match pt {
        PreTokenizerWrapper::Metaspace(ms) => {
            ms.prepend_scheme = PrependScheme::Never;
            true
        }
        PreTokenizerWrapper::Sequence(seq) => {
            let mut updated = false;
            for child in seq.as_mut() {
                if disable_metaspace_prepend(child) {
                    updated = true;
                }
            }
            updated
        }
        _ => false,
    }
}

fn disable_metaspace_prepend_json(val: &mut serde_json::Value) -> bool {
    let Some(obj) = val.as_object_mut() else {
        return false;
    };
    if obj.get("type").and_then(|v| v.as_str()) == Some("Metaspace") {
        obj.insert(
            "prepend_scheme".to_string(),
            serde_json::Value::String("never".to_string()),
        );
        return true;
    }
    if let Some(arr) = obj.get_mut("pretokenizers").and_then(|v| v.as_array_mut()) {
        let mut updated = false;
        for child in arr {
            if disable_metaspace_prepend_json(child) {
                updated = true;
            }
        }
        return updated;
    }
    false
}

/// Build the no-prepend candidate tokenizer for `choice_compact` models by cloning the loaded
/// tokenizer and updating `Metaspace.prepend_scheme` in memory (handles both standalone
/// `Metaspace` and composite `Sequence` pre-tokenizers without re-parsing 32 MB JSON).
fn build_no_prepend_tokenizer(base: &Tokenizer, tok_path: &Path) -> Result<Tokenizer, String> {
    if let Some(mut pt) = base.get_pre_tokenizer().cloned()
        && disable_metaspace_prepend(&mut pt)
    {
        let mut cloned = base.clone();
        cloned.with_pre_tokenizer(Some(pt));
        return Ok(cloned);
    }
    // Fallback for non-Metaspace JSON layouts
    let raw = std::fs::read_to_string(tok_path)
        .map_err(|e| format!("read {}: {}", tok_path.display(), e))?;
    let mut val: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse tokenizer.json: {}", e))?;
    if let Some(pt) = val.get_mut("pre_tokenizer") {
        let _ = disable_metaspace_prepend_json(pt);
    }
    let bytes = serde_json::to_vec(&val).map_err(|e| format!("serialize tokenizer.json: {}", e))?;
    Tokenizer::from_bytes(&bytes).map_err(|e| format!("build no-prepend tokenizer: {}", e))
}

fn build_first_byte_table(special_tokens: &[String]) -> [bool; 256] {
    let mut tbl = [false; 256];
    for tok in special_tokens {
        if let Some(&b) = tok.as_bytes().first() {
            tbl[b as usize] = true;
        }
    }
    tbl
}

/// Extract control/special added tokens from the tokenizer so untrusted UIA context and candidate
/// strings cannot inject raw control token IDs (`<pad>`, `<bos>`, `<eos>`, `<mask>`,
/// `<start_of_turn>`, `<end_of_turn>`, `<unused*>`, `[@BOS@]`, `[CLS]`, etc.) while preserving
/// legitimate non-special vocabulary tokens (such as Gemma's HTML `<code>`/`<b>`/`<table>` tokens)
/// and `<unk>` OOV fallback IDs.
fn collect_special_tokens(tokenizer: &Tokenizer) -> (Vec<String>, [bool; 256], HashSet<i64>) {
    let mut strs: Vec<String> = Vec::new();
    let mut seen_str: HashSet<String> = HashSet::new();
    let mut ids: HashSet<i64> = HashSet::new();
    let unk_id = tokenizer.token_to_id("<unk>").map(|id| id as i64);

    for (id, added) in tokenizer.get_added_tokens_decoder() {
        let content = added.content.as_str();
        let is_control_like = added.special
            || content == "[@BOS@]"
            || content.starts_with("<unused")
            || content == "<2mass>";
        if is_control_like && !content.is_empty() {
            if Some(id as i64) != unk_id && content != "<unk>" {
                ids.insert(id as i64);
            }
            if seen_str.insert(added.content.clone()) {
                strs.push(added.content);
            }
        }
    }
    for fallback in [
        "<bos>",
        "<eos>",
        "<mask>",
        "<pad>",
        "<start_of_turn>",
        "<end_of_turn>",
        "[@BOS@]",
    ] {
        if let Some(id) = tokenizer.token_to_id(fallback) {
            ids.insert(id as i64);
        }
        if seen_str.insert(fallback.to_string()) {
            strs.push(fallback.to_string());
        }
    }
    // Longest tokens first so `<start_of_turn>` is stripped cleanly before any shorter substring
    strs.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let triggers = build_first_byte_table(&strs);
    (strs, triggers, ids)
}

fn introspect_kv_dims(session: &Session) -> (bool, usize, usize, usize, usize) {
    let mut has_enc_k = false;
    let mut has_ch_k = false;
    let mut has_pm = false;
    let mut enc_layers = 12usize;
    let mut choice_layers = 2usize;
    let mut num_heads = 6usize;
    let mut head_dim = 64usize;

    for inp in session.inputs() {
        match inp.name() {
            "past_enc_k" => {
                has_enc_k = true;
                if let Some(shape) = inp.dtype().tensor_shape()
                    && shape.len() == 5
                {
                    if shape[0] > 0 {
                        enc_layers = shape[0] as usize;
                    }
                    if shape[2] > 0 {
                        num_heads = shape[2] as usize;
                    }
                    if shape[4] > 0 {
                        head_dim = shape[4] as usize;
                    }
                }
            }
            "past_choice_k" => {
                has_ch_k = true;
                if let Some(shape) = inp.dtype().tensor_shape()
                    && shape.len() == 5
                    && shape[0] > 0
                {
                    choice_layers = shape[0] as usize;
                }
            }
            "past_mask" => {
                has_pm = true;
            }
            _ => {}
        }
    }

    (
        has_enc_k && has_ch_k && has_pm,
        enc_layers,
        choice_layers,
        num_heads,
        head_dim,
    )
}

/// Attach or preload a `.gram` mapping off-lock so the calling UI thread and active `rerank()`
/// keystrokes are never blocked on disk/mmap I/O.
fn maybe_attach_gram(cfg: &InitConfig) {
    let Some(ref gpath) = cfg.gram_path else {
        return;
    };
    let Some(mutex) = ENGINE.get() else {
        return;
    };
    let new_p = sanitize_penalties(
        cfg.gram_collocation_penalty,
        cfg.gram_non_collocation_penalty,
    );
    let new_max = sanitize_max_length(cfg.gram_max_length);

    // Phase 1: check briefly under lock whether this path is already mapped.
    let already_cached = {
        let Ok(mut eng) = mutex.lock() else {
            return;
        };
        if eng.gram_cache.contains_key(gpath) {
            eng.default_gram_path = Some(gpath.clone());
            eng.default_gram_penalties = new_p;
            eng.default_gram_max_len = new_max;
            eng.stats.has_gram = true;
            true
        } else {
            false
        }
    };
    if already_cached {
        return;
    }

    // Phase 2: perform blocking filesystem + Win32 mmap I/O strictly outside `ENGINE` lock.
    if !gpath.is_file() {
        return;
    }
    let Ok(g) = WanxiangGram::open(gpath) else {
        return;
    };

    // Phase 3: insert into cache under a brief lock.
    if let Ok(mut eng) = mutex.lock() {
        eng.gram_cache.entry(gpath.clone()).or_insert(g);
        eng.default_gram_path = Some(gpath.clone());
        eng.default_gram_penalties = new_p;
        eng.default_gram_max_len = new_max;
        eng.stats.has_gram = true;
    }
}

pub fn init_engine(cfg: &InitConfig) -> Result<(), String> {
    let _guard = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if ENGINE.get().is_some() {
        maybe_attach_gram(cfg);
        return Ok(());
    }
    let res = init_engine_inner(cfg);
    match &res {
        Ok(()) => set_last_init_error(None),
        Err(e) => set_last_init_error(Some(e.clone())),
    }
    res
}

fn init_engine_inner(cfg: &InitConfig) -> Result<(), String> {
    let ort_env = ensure_ort_env(&cfg.ort_dll)?;
    let onnx_path = resolve_model_path(&cfg.model_dir, cfg.model_file.as_deref())?;
    let model_name = onnx_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown.onnx")
        .to_string();

    let tok_path = cfg.model_dir.join("tokenizer.json");
    let vocab_path = cfg.model_dir.join("unified_vocab.json");

    let vocab_raw = std::fs::read_to_string(&vocab_path)
        .map_err(|e| format!("read {}: {}", vocab_path.display(), e))?;
    let uv: UnifiedVocabFile = serde_json::from_str(&vocab_raw)
        .map_err(|e| format!("parse {}: {}", vocab_path.display(), e))?;

    let mut full_unit_ids = HashMap::with_capacity(450);
    let mut partial_unit_ids = HashMap::with_capacity(64);
    let mut valid_syllables = HashSet::new();
    for (idx, name) in uv.units.into_iter().enumerate() {
        let id = idx as i64;
        if let Some(syl) = name.strip_prefix("f:") {
            valid_syllables.insert(syl.to_string());
            full_unit_ids.insert(syl.to_string(), id);
        } else if let Some(part) = name.strip_prefix("p:") {
            partial_unit_ids.insert(part.to_string(), id);
        }
    }

    let mut char_to_id: HashMap<char, usize> = HashMap::with_capacity(uv.chars.len());
    for (idx, s) in uv.chars.iter().enumerate() {
        if let Some(ch) = s.chars().next()
            && s.chars().count() == 1
        {
            char_to_id.insert(ch, idx);
        }
    }

    let old_chars = uv
        .ml
        .as_ref()
        .map(|m| m.old_chars)
        .unwrap_or(uv.chars.len())
        .min(uv.chars.len());

    // Union `uv.readings` and `uv.ml.pinyin` (preserving order and deduplicating per char) so
    // delta `ml.pinyin` tables never erase base `uv.readings` entries, while `old_chars` strictly
    // caps Simplified P2C generation within each sorted list of character IDs.
    let mut merged_readings = uv.readings;
    if let Some(ref ml) = uv.ml {
        for (ch, rlist) in &ml.pinyin {
            let entry = merged_readings.entry(ch.clone()).or_default();
            for r in rlist {
                if !entry.iter().any(|existing| existing == r) {
                    entry.push(r.clone());
                }
            }
        }
    }

    let mut syl_to_all_cids: HashMap<String, Vec<usize>> = HashMap::with_capacity(450);
    for (ch_str, rlist) in &merged_readings {
        if let Some(ch) = ch_str.chars().next()
            && ch_str.chars().count() == 1
            && let Some(&cid) = char_to_id.get(&ch)
        {
            for syl in rlist {
                syl_to_all_cids.entry(syl.clone()).or_default().push(cid);
            }
        }
    }
    for cids in syl_to_all_cids.values_mut() {
        cids.sort_unstable();
        cids.dedup();
    }

    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("load {}: {}", tok_path.display(), e))?;
    let (special_tokens, special_first_bytes, forbidden_special_ids) =
        collect_special_tokens(&tokenizer);
    let tokenizer_no_prepend = if uv.choice_compact {
        Some(build_no_prepend_tokenizer(&tokenizer, &tok_path)?)
    } else {
        None
    };

    let cls_id = tokenizer
        .token_to_id("<bos>")
        .ok_or_else(|| "missing <bos> token".to_string())? as i64;
    let sep_id = tokenizer
        .token_to_id("<eos>")
        .ok_or_else(|| "missing <eos> token".to_string())? as i64;
    let mask_id = tokenizer
        .token_to_id("<mask>")
        .ok_or_else(|| "missing <mask> token".to_string())? as i64;
    let unk_id = tokenizer.token_to_id("<unk>").unwrap_or(0) as i64;

    let threads = resolve_threads(cfg.threads);
    let session = Session::builder(ort_env)
        .map_err(|e| format!("session builder: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("opt level: {}", e))?
        .with_parallel_execution(false)
        .map_err(|e| format!("parallel exec: {}", e))?
        .with_intra_threads(threads)
        .map_err(|e| format!("intra threads: {}", e))?
        .with_inter_threads(1)
        .map_err(|e| format!("inter threads: {}", e))?
        .with_inter_op_spinning(false)
        .map_err(|e| format!("inter spinning: {}", e))?
        .commit_from_file(&onnx_path)
        .map_err(|e| format!("load {}: {}", onnx_path.display(), e))?;

    let (use_fused_kv, enc_layers, choice_layers, num_heads, head_dim) =
        introspect_kv_dims(&session);
    let output_names: HashSet<&str> = session.outputs().iter().map(|o| o.name()).collect();
    if !output_names.contains("logits") {
        return Err(format!(
            "ONNX model {} is missing required output 'logits'",
            model_name
        ));
    }
    let has_char_logits = output_names.contains("char_logits");
    let has_named_present = output_names.contains("present_enc_k");
    let present_names = if has_named_present {
        [
            "present_enc_k",
            "present_enc_v",
            "present_choice_k",
            "present_choice_v",
        ]
    } else {
        ["present_0", "present_1", "present_2", "present_3"]
    };
    if use_fused_kv {
        for req in present_names {
            if !output_names.contains(req) {
                return Err(format!(
                    "fused KV ONNX model {} is missing required output '{}'",
                    model_name, req
                ));
            }
        }
    }

    // Pre-build selective RunOptions so ORT skips unused heads & 3 MB/keystroke KV-output copies
    let (ctx_run_opts, key_run_opts, pass2_run_opts) = if use_fused_kv {
        let ctx_sel = OutputSelector::no_default()
            .with(present_names[0])
            .with(present_names[1])
            .with(present_names[2])
            .with(present_names[3]);
        let ctx_opts = RunOptions::new()
            .map_err(|e| format!("ctx RunOptions: {}", e))?
            .with_outputs(ctx_sel);

        let mut key_sel = OutputSelector::no_default().with("logits");
        if has_char_logits {
            key_sel = key_sel.with("char_logits");
        }
        let key_opts = RunOptions::new()
            .map_err(|e| format!("key RunOptions: {}", e))?
            .with_outputs(key_sel);

        let p2_sel = OutputSelector::no_default().with("logits");
        let p2_opts = RunOptions::new()
            .map_err(|e| format!("pass2 RunOptions: {}", e))?
            .with_outputs(p2_sel);

        (Some(ctx_opts), Some(key_opts), Some(p2_opts))
    } else {
        (None, None, None)
    };

    let gram_penalties = sanitize_penalties(
        cfg.gram_collocation_penalty,
        cfg.gram_non_collocation_penalty,
    );
    let gram_max_len = sanitize_max_length(cfg.gram_max_length);
    let mut gram_cache = HashMap::new();
    let mut default_gram_path = None;
    if let Some(gp) = cfg.gram_path.as_deref()
        && gp.is_file()
        && let Ok(g) = WanxiangGram::open(gp)
    {
        gram_cache.insert(gp.to_path_buf(), g);
        default_gram_path = Some(gp.to_path_buf());
    }
    let has_gram = default_gram_path.is_some();

    let max_slots = uv.max_slots.max(1);
    let context_tokens = uv.choice_context_tokens;
    let candidate_tokens = uv.choice_candidate_tokens.max(1);

    let z_enc = vec![0.0f32; enc_layers * num_heads * head_dim];
    let z_ch = vec![0.0f32; choice_layers * num_heads * head_dim];
    let true_mask = vec![true; context_tokens + 2];

    let mut engine = NeuralEngine {
        model_dir: cfg.model_dir.clone(),
        session,
        use_fused_kv,
        has_char_logits,
        enc_layers,
        choice_layers,
        num_heads,
        head_dim,
        present_names,
        z_enc,
        z_ch,
        true_mask,
        scratch: ScratchBufs::new(max_slots),
        ctx_run_opts,
        key_run_opts,
        pass2_run_opts,
        tokenizer,
        tokenizer_no_prepend,
        special_tokens,
        special_first_bytes,
        forbidden_special_ids,
        choice_compact: uv.choice_compact,
        full_unit_ids,
        partial_unit_ids,
        char_vocab: CharVocab {
            chars: uv.chars,
            char_to_id,
            syl_to_all_cids,
            old_chars,
            max_slots,
        },
        readings: merged_readings,
        valid_syllables,
        max_slots,
        context_tokens,
        candidate_tokens,
        cls_id,
        sep_id,
        mask_id,
        unk_id,
        gram_cache,
        default_gram_path,
        default_gram_penalties: gram_penalties,
        default_gram_max_len: gram_max_len,
        last_ctx_str: String::new(),
        last_ctx_ids: Vec::new(),
        kv_cache: VecDeque::with_capacity(KV_CACHE_CAPACITY + 2),
        lru_map: HashMap::with_capacity(LRU_CAPACITY + 16),
        lru_clock: 0,
        stats: EngineStats {
            calls: 0,
            cache_hits: 0,
            kv_hits: 0,
            p2c_pass2_calls: 0,
            total_infer_ms: 0.0,
            fused_kv: use_fused_kv,
            has_gram,
            enc_layers,
            choice_compact: uv.choice_compact,
            model_name,
        },
    };

    // Single untimed warmup pass (also pre-populates empty-context KV cache!)
    let warmup_cands = vec!["你好".to_string(), "拟好".to_string()];
    engine
        .evaluate(
            "",
            "nihk",
            PinyinLayout::Zrm,
            &warmup_cands,
            FusionWeights::default(),
            0.0,
            &GramRequest::Default,
        )
        .map_err(|e| format!("model warmup: {e}"))?;
    engine.lru_map.clear();
    engine.lru_clock = 0;
    engine.stats.calls = 0;
    engine.stats.cache_hits = 0;
    engine.stats.kv_hits = 0;
    engine.stats.p2c_pass2_calls = 0;
    engine.stats.total_infer_ms = 0.0;

    let _ = ENGINE.set(Mutex::new(engine));
    Ok(())
}

struct AsyncInitGuard {
    succeeded: bool,
}

impl Drop for AsyncInitGuard {
    fn drop(&mut self) {
        if !self.succeeded {
            // Finalize fallback error BEFORE releasing `ASYNC_STARTED` so a retrying thread's
            // `set_last_init_error(None)` cannot race with this cleanup.
            if last_init_error().is_none() {
                set_last_init_error(Some(
                    "background init thread exited unexpectedly".to_string(),
                ));
            }
            ASYNC_STARTED.store(false, Ordering::Release);
        }
    }
}

pub fn init_engine_async(cfg: InitConfig) {
    if ENGINE.get().is_some() {
        // Offload lazy `.gram` attachment/penalty update so the calling UI thread never blocks on disk I/O
        if cfg.gram_path.is_some() {
            let _ = std::thread::Builder::new()
                .name("rime-neural-gram".to_string())
                .spawn(move || maybe_attach_gram(&cfg));
        }
        return;
    }
    if ASYNC_STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    set_last_init_error(None);
    let spawned = std::thread::Builder::new()
        .name("rime-neural-init".to_string())
        .spawn(move || {
            let mut guard = AsyncInitGuard { succeeded: false };
            if init_engine(&cfg).is_ok() {
                guard.succeeded = true;
            }
        });
    if let Err(e) = spawned {
        set_last_init_error(Some(format!("spawn init thread failed: {}", e)));
        ASYNC_STARTED.store(false, Ordering::Release);
    }
}

pub fn is_ready() -> bool {
    ENGINE.get().is_some_and(|m| !m.is_poisoned())
}

pub fn model_dir_path() -> Option<PathBuf> {
    let mutex = ENGINE.get()?;
    let eng = mutex.lock().ok()?;
    Some(eng.model_dir.clone())
}

pub fn active_feedback_meta() -> (String, bool) {
    let Some(mutex) = ENGINE.get() else {
        return (String::new(), false);
    };
    let Ok(eng) = mutex.lock() else {
        return (String::new(), false);
    };
    (eng.stats.model_name.clone(), eng.stats.has_gram)
}

fn utf8_tail(s: &str, max_chars: usize) -> &str {
    if max_chars == 0 {
        return "";
    }
    let total = s.chars().count();
    if total <= max_chars {
        return s;
    }
    let skip = total - max_chars;
    match s.char_indices().nth(skip) {
        Some((byte_idx, _)) => &s[byte_idx..],
        None => "",
    }
}

fn common_prefix_chars(cands: &[String]) -> usize {
    if cands.is_empty() {
        return 0;
    }
    let first = cands[0].as_str();
    let mut lcp = first.chars().count();
    let mut min_len = lcp;
    for c in &cands[1..] {
        let mut matched = 0usize;
        let mut c_len = 0usize;
        let mut diverged = false;
        let mut fi = first.chars();
        for bc in c.chars() {
            c_len += 1;
            if !diverged {
                if fi.next() == Some(bc) {
                    matched += 1;
                } else {
                    diverged = true;
                }
            }
        }
        min_len = min_len.min(c_len);
        lcp = lcp.min(matched);
    }
    if lcp >= min_len { 0 } else { lcp }
}

fn rank_order_from_features(
    logits: &[f32],
    gram_deltas: &[f32],
    char_lps: &[f32],
    has_ctx: f64,
    weights: FusionWeights,
) -> Vec<usize> {
    let clean_l = |x: f32| if x.is_finite() { x as f64 } else { -1e6 };
    let clean_f = |x: f32| if x.is_finite() { x as f64 } else { 0.0 };
    let max_l = logits
        .iter()
        .map(|&x| clean_l(x))
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = logits.iter().map(|&x| (clean_l(x) - max_l).exp()).sum();
    let lse = if sum_exp > 0.0 && sum_exp.is_finite() {
        max_l + sum_exp.ln()
    } else {
        max_l
    };

    let mut scored: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let lp = clean_l(x) - lse;
            let gd = gram_deltas.get(i).copied().map(clean_f).unwrap_or(0.0);
            let clp = char_lps.get(i).copied().map(clean_f).unwrap_or(0.0);
            let score = weights.w_logit * lp
                + weights.w_rank * ((i as f64) + 1.0).ln()
                + weights.w_gram * gd
                + weights.w_char_lp * clp
                + weights.w_ctx_lp * lp * has_ctx;
            let safe_score = if score.is_finite() {
                score
            } else {
                f64::NEG_INFINITY
            };
            (i, safe_score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().map(|(idx, _)| idx).collect()
}

fn sanitize_special_tokens<'a>(
    text: &'a str,
    special_tokens: &[String],
    first_bytes: &[bool; 256],
) -> Cow<'a, str> {
    if special_tokens.is_empty() || !text.bytes().any(|b| first_bytes[b as usize]) {
        return Cow::Borrowed(text);
    }
    let mut owned: Option<String> = None;
    for tok in special_tokens {
        let cur = owned.as_deref().unwrap_or(text);
        if cur.contains(tok.as_str()) {
            owned = Some(cur.replace(tok.as_str(), " "));
        }
    }
    match owned {
        Some(s) => Cow::Owned(s),
        None => Cow::Borrowed(text),
    }
}

impl NeuralEngine {
    #[inline]
    fn lookup_unit_id(&self, kind: &str, text: &str) -> i64 {
        if kind == "f" {
            self.full_unit_ids.get(text).copied().unwrap_or(0)
        } else {
            self.partial_unit_ids.get(text).copied().unwrap_or(0)
        }
    }

    fn encode_with_tok(&self, tok: &Tokenizer, text: &str) -> Vec<i64> {
        let cleaned =
            sanitize_special_tokens(text, &self.special_tokens, &self.special_first_bytes);
        match tok.encode_fast(cleaned.as_ref(), false) {
            Ok(enc) => enc
                .get_ids()
                .iter()
                .map(|&x| x as i64)
                .filter(|id| !self.forbidden_special_ids.contains(id))
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    fn encode_ids(&self, text: &str) -> Vec<i64> {
        self.encode_with_tok(&self.tokenizer, text)
    }

    fn encode_candidate_ids(&mut self, text: &str) -> Vec<i64> {
        if self.choice_compact {
            let tok = self
                .tokenizer_no_prepend
                .as_ref()
                .unwrap_or(&self.tokenizer);
            let mut ids = self.encode_with_tok(tok, text);
            if ids.is_empty() {
                ids.push(self.unk_id);
            }
            ids
        } else {
            self.scratch.cand_str.clear();
            self.scratch.cand_str.push(' ');
            self.scratch.cand_str.push_str(text);
            let cleaned = sanitize_special_tokens(
                &self.scratch.cand_str,
                &self.special_tokens,
                &self.special_first_bytes,
            );
            match self.tokenizer.encode_fast(cleaned.as_ref(), false) {
                Ok(enc) => enc
                    .get_ids()
                    .iter()
                    .map(|&x| x as i64)
                    .filter(|id| !self.forbidden_special_ids.contains(id))
                    .collect(),
                Err(_) => Vec::new(),
            }
        }
    }

    fn build_context_token_ids(&self, ctx_final: &str) -> Vec<i64> {
        let mut ids: Vec<i64> = Vec::with_capacity(self.context_tokens + 2);
        ids.push(self.cls_id);
        if !ctx_final.is_empty() {
            let cids = self.encode_ids(ctx_final);
            let start = cids.len().saturating_sub(self.context_tokens);
            ids.extend_from_slice(&cids[start..]);
        }
        ids.push(self.sep_id);
        ids
    }

    fn build_context_token_ids_cached(&mut self, ctx_final: &str) -> Vec<i64> {
        if !self.last_ctx_ids.is_empty() && self.last_ctx_str == ctx_final {
            return self.last_ctx_ids.clone();
        }
        let ids = self.build_context_token_ids(ctx_final);
        self.last_ctx_str.clear();
        self.last_ctx_str.push_str(ctx_final);
        self.last_ctx_ids.clear();
        self.last_ctx_ids.extend_from_slice(&ids);
        ids
    }

    fn compute_context_kv(&mut self, ctx_ids: &[i64]) -> Result<ContextKv, String> {
        let p = ctx_ids.len();
        self.scratch.uids.clear();
        self.scratch.uids.resize(p, 0);
        let dummy_false = [false];
        let dummy_zero = [0i64];

        let t_ids = TensorRef::from_array_view(([1usize, p], ctx_ids))
            .map_err(|e| format!("ctx ids: {}", e))?;
        let t_uid = TensorRef::from_array_view(([1usize, p], self.scratch.uids.as_slice()))
            .map_err(|e| format!("ctx uid: {}", e))?;
        let t_ek = TensorRef::from_array_view((
            [self.enc_layers, 1, self.num_heads, 1, self.head_dim],
            self.z_enc.as_slice(),
        ))
        .map_err(|e| format!("ek: {}", e))?;
        let t_ev = TensorRef::from_array_view((
            [self.enc_layers, 1, self.num_heads, 1, self.head_dim],
            self.z_enc.as_slice(),
        ))
        .map_err(|e| format!("ev: {}", e))?;
        let t_ck = TensorRef::from_array_view((
            [self.choice_layers, 1, self.num_heads, 1, self.head_dim],
            self.z_ch.as_slice(),
        ))
        .map_err(|e| format!("ck: {}", e))?;
        let t_cv = TensorRef::from_array_view((
            [self.choice_layers, 1, self.num_heads, 1, self.head_dim],
            self.z_ch.as_slice(),
        ))
        .map_err(|e| format!("cv: {}", e))?;
        let t_pm = TensorRef::from_array_view(([1usize, 1], dummy_false.as_slice()))
            .map_err(|e| format!("pm: {}", e))?;
        let t_mp = TensorRef::from_array_view(([1usize, 1], dummy_zero.as_slice()))
            .map_err(|e| format!("mp: {}", e))?;
        let t_sp = TensorRef::from_array_view(([1usize, 1], dummy_zero.as_slice()))
            .map_err(|e| format!("sp: {}", e))?;

        let inputs = ort::inputs![
            "input_ids" => t_ids,
            "unit_ids" => t_uid,
            "past_enc_k" => t_ek,
            "past_enc_v" => t_ev,
            "past_choice_k" => t_ck,
            "past_choice_v" => t_cv,
            "past_mask" => t_pm,
            "marker_pos" => t_mp,
            "slot_pos" => t_sp,
        ];

        let t0 = Instant::now();
        let outputs = if let Some(ref opts) = self.ctx_run_opts {
            self.session
                .run_with_options(inputs, opts)
                .map_err(|e| format!("ctx run: {}", e))?
        } else {
            self.session
                .run(inputs)
                .map_err(|e| format!("ctx run: {}", e))?
        };
        self.stats.total_infer_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let (_, ek) = outputs
            .get(self.present_names[0])
            .ok_or_else(|| format!("missing ONNX output: {}", self.present_names[0]))?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("out ek: {}", e))?;
        let (_, ev) = outputs
            .get(self.present_names[1])
            .ok_or_else(|| format!("missing ONNX output: {}", self.present_names[1]))?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("out ev: {}", e))?;
        let (_, ck) = outputs
            .get(self.present_names[2])
            .ok_or_else(|| format!("missing ONNX output: {}", self.present_names[2]))?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("out ck: {}", e))?;
        let (_, cv) = outputs
            .get(self.present_names[3])
            .ok_or_else(|| format!("missing ONNX output: {}", self.present_names[3]))?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("out cv: {}", e))?;

        Ok(ContextKv {
            p,
            enc_k: ek.to_vec(),
            enc_v: ev.to_vec(),
            choice_k: ck.to_vec(),
            choice_v: cv.to_vec(),
        })
    }

    fn get_or_compute_kv_index(&mut self, ctx_final: &str) -> Result<usize, String> {
        let ctx_ids = self.build_context_token_ids_cached(ctx_final);
        if let Some(pos) = self.kv_cache.iter().position(|(k, _)| k == &ctx_ids) {
            self.stats.kv_hits += 1;
            let last = self.kv_cache.len() - 1;
            if pos != last
                && let Some(entry) = self.kv_cache.remove(pos)
            {
                self.kv_cache.push_back(entry);
            }
            return Ok(self.kv_cache.len() - 1);
        }
        let kv = self.compute_context_kv(&ctx_ids)?;
        if self.kv_cache.len() >= KV_CACHE_CAPACITY {
            // Keep empty context (`[cls, sep]`, len == 2) pinned; evict LRU non-empty entry
            if let Some(evict) = self.kv_cache.iter().position(|(k, _)| k.len() > 2) {
                self.kv_cache.remove(evict);
            } else {
                self.kv_cache.pop_front();
            }
        }
        self.kv_cache.push_back((ctx_ids, kv));
        Ok(self.kv_cache.len() - 1)
    }

    fn resolve_gram_request(
        &self,
        req: &GramRequest,
    ) -> (ResolvedGramKey, Option<(f64, f64, usize)>) {
        let active = match req {
            GramRequest::Disabled => None,
            GramRequest::Default => self.default_gram_path.as_ref().map(|path| {
                let (col, non_col) = self.default_gram_penalties;
                (path, col, non_col, self.default_gram_max_len)
            }),
            GramRequest::Custom {
                paths,
                collocation_penalty,
                non_collocation_penalty,
                max_length,
            } => paths
                .iter()
                .flatten()
                .find(|path| self.gram_cache.contains_key(*path))
                .map(|path| {
                    (
                        path,
                        f64::from_bits(*collocation_penalty),
                        f64::from_bits(*non_collocation_penalty),
                        *max_length,
                    )
                }),
        };
        if let Some((path, col, non_col, max_length)) = active {
            (
                ResolvedGramKey {
                    path: Some(path.clone()),
                    col_bits: canon_f64_bits(col),
                    non_col_bits: canon_f64_bits(non_col),
                    max_length,
                },
                Some((col, non_col, max_length)),
            )
        } else {
            (ResolvedGramKey::default(), None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &mut self,
        context: &str,
        raw_input: &str,
        layout: PinyinLayout,
        cands: &[String],
        weights: FusionWeights,
        p2c_verify_margin: f32,
        gram_req: &GramRequest,
    ) -> Result<CachedEval, String> {
        self.stats.calls += 1;
        let ctx_tail = utf8_tail(context, CONTEXT_CHARS);
        let (gram_key, gram_params) = self.resolve_gram_request(gram_req);
        let key = CacheKey {
            ctx_tail: ctx_tail.to_string(),
            raw_input: raw_input.to_string(),
            layout,
            weights_bits: [
                canon_f64_bits(weights.w_logit),
                canon_f64_bits(weights.w_rank),
                canon_f64_bits(weights.w_gram),
                canon_f64_bits(weights.w_char_lp),
                canon_f64_bits(weights.w_ctx_lp),
            ],
            p2c_verify_margin_bits: canon_f32_bits(p2c_verify_margin),
            gram_key,
            cands: cands.to_vec(),
        };
        if let Some(entry) = self.lru_map.get_mut(&key) {
            self.stats.cache_hits += 1;
            self.lru_clock = self.lru_clock.wrapping_add(1);
            entry.last_used = self.lru_clock;
            return Ok(entry.clone());
        }

        let mut units = input_to_units(
            layout,
            raw_input,
            &cands[0],
            &self.readings,
            &self.valid_syllables,
        );
        let orig_units_len = units.len();

        let mut cand_toks: Vec<Vec<i64>> = Vec::with_capacity(cands.len());
        for c in cands {
            cand_toks.push(self.encode_candidate_ids(c));
        }

        let mut eff_ctx = ctx_tail.to_string();
        let mut shifted_lcp = false;
        if cand_toks.iter().any(|t| t.len() > self.candidate_tokens) || units.len() > self.max_slots
        {
            let lcp = common_prefix_chars(cands);
            // Only shift `units` by `lcp` when 1:1 char-to-unit alignment holds
            if lcp > 0 && lcp < units.len() && units.len() == cands[0].chars().count() {
                shifted_lcp = true;
                let prefix_str: String = cands[0].chars().take(lcp).collect();
                eff_ctx.push_str(&prefix_str);
                cand_toks.clear();
                for c in cands {
                    let tail: String = c.chars().skip(lcp).collect();
                    cand_toks.push(self.encode_candidate_ids(&tail));
                }
                units = units.split_off(lcp);
            }
        }

        let ctx_final = utf8_tail(&eff_ctx, CONTEXT_CHARS).to_string();
        let (kv_idx_opt, (logits, char_lps, p2c_text, p2c_margin)) = if self.use_fused_kv {
            let kv_idx = self.get_or_compute_kv_index(&ctx_final)?;
            let out = self.run_fused_kv_at(
                kv_idx,
                &units,
                &cand_toks,
                Some((cands, orig_units_len, shifted_lcp)),
            )?;
            (Some(kv_idx), out)
        } else {
            let out = self.run_whole_graph(
                &ctx_final,
                &units,
                &cand_toks,
                cands,
                orig_units_len,
                shifted_lcp,
            )?;
            (None, out)
        };

        // Compute exact `.gram` deltas (~0.004 ms/cand) using per-schema params and retain `raw_g`
        let active_gram_ref = key
            .gram_key
            .path
            .as_ref()
            .and_then(|p| self.gram_cache.get(p))
            .zip(gram_params);
        let has_gram = active_gram_ref.is_some();
        let (raw_g, gram_deltas): (Vec<f64>, Vec<f32>) =
            if let Some((g, (col, non_col, max_len))) = active_gram_ref {
                let rg: Vec<f64> = cands
                    .iter()
                    .map(|c| g.score_sentence_per_char(c, col, non_col, max_len))
                    .collect();
                let g0 = rg.first().copied().unwrap_or(0.0);
                let gd = rg.iter().map(|&x| (x - g0) as f32).collect();
                (rg, gd)
            } else {
                (Vec::new(), vec![0.0f32; cands.len()])
            };

        // Optional KV-cached Pass-2 Cross-Encoder + .gram verification of promising novel P2C candidates
        let mut p2c_verified: Option<(f32, f32)> = None;
        if let Some(kv_idx) = kv_idx_opt
            && p2c_verify_margin > 0.0
            && p2c_margin.is_finite()
            && p2c_margin > p2c_verify_margin
            && let Some(ref p2c_str) = p2c_text
        {
            let has_ctx_f = if ctx_tail.trim().chars().count() >= 2 {
                1.0
            } else {
                0.0
            };
            let rank0 =
                rank_order_from_features(&logits, &gram_deltas, &char_lps, has_ctx_f, weights);
            let best_idx = rank0[0];
            let mut probe_toks: Vec<Vec<i64>> = Vec::with_capacity(3);
            probe_toks.push(cand_toks[best_idx].clone());
            if rank0.len() >= 2 {
                probe_toks.push(cand_toks[rank0[1]].clone());
            }
            probe_toks.push(self.encode_candidate_ids(p2c_str));

            if let Ok((l2, _, _, _)) = self.run_fused_kv_at(kv_idx, &units, &probe_toks, None)
                && let (Some(&l_best), Some(&l_p2c)) = (l2.first(), l2.last())
                && l_best.is_finite()
                && l_p2c.is_finite()
            {
                let ldiff = l_p2c - l_best;
                self.stats.p2c_pass2_calls += 1;
                let gdiff = if let Some((g, (col, non_col, max_len))) = key
                    .gram_key
                    .path
                    .as_ref()
                    .and_then(|p| self.gram_cache.get(p))
                    .zip(gram_params)
                {
                    (g.score_sentence_per_char(p2c_str, col, non_col, max_len) - raw_g[best_idx])
                        as f32
                } else {
                    0.0
                };
                if ldiff.is_finite() && gdiff.is_finite() {
                    p2c_verified = Some((ldiff, gdiff));
                }
            }
        }

        self.lru_clock = self.lru_clock.wrapping_add(1);
        let res = CachedEval {
            logits,
            gram_deltas,
            char_lps,
            p2c_text,
            p2c_margin,
            p2c_verified,
            used_fused_kv: self.use_fused_kv,
            has_gram,
            last_used: self.lru_clock,
        };

        if self.lru_map.len() >= LRU_CAPACITY
            && let Some(evict_key) = self
                .lru_map
                .iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| k.clone())
        {
            self.lru_map.remove(&evict_key);
        }
        self.lru_map.insert(key, res.clone());
        Ok(res)
    }

    fn run_fused_kv_at(
        &mut self,
        kv_idx: usize,
        units: &[(&'static str, String)],
        cand_toks: &[Vec<i64>],
        decode_p2c: Option<(&[String], usize, bool)>,
    ) -> Result<ForwardResult, String> {
        let want_char_logits = decode_p2c.is_some();

        self.scratch.ids.clear();
        self.scratch.uids.clear();
        self.scratch.slots.clear();
        self.scratch.markers.clear();

        for (kind, text) in units.iter().take(self.max_slots) {
            self.scratch.slots.push(self.scratch.ids.len() as i64);
            self.scratch.ids.push(self.mask_id);
            let uid = if *kind == "f" {
                self.full_unit_ids.get(text.as_str()).copied().unwrap_or(0)
            } else {
                self.partial_unit_ids
                    .get(text.as_str())
                    .copied()
                    .unwrap_or(0)
            };
            self.scratch.uids.push(uid);
        }
        self.scratch.ids.push(self.sep_id);
        self.scratch.uids.push(0);

        for ctok in cand_toks {
            self.scratch.markers.push(self.scratch.ids.len() as i64);
            if !self.choice_compact {
                self.scratch.ids.push(self.mask_id);
                self.scratch.uids.push(0);
            }
            let take = ctok.len().min(self.candidate_tokens);
            self.scratch.ids.extend_from_slice(&ctok[..take]);
            self.scratch.uids.resize(self.scratch.ids.len(), 0);
        }
        self.scratch.ids.push(self.sep_id);
        self.scratch.uids.push(0);

        let seq_len = self.scratch.ids.len();
        let n_markers = self.scratch.markers.len();
        let real_slots = self.scratch.slots.len();
        if self.scratch.slots.is_empty() {
            self.scratch.slots.push(0);
        }
        let n_slots_in = self.scratch.slots.len();

        let kv = &self.kv_cache[kv_idx].1;
        if self.true_mask.len() < kv.p {
            self.true_mask.resize(kv.p, true);
        }
        let t_ids = TensorRef::from_array_view(([1usize, seq_len], self.scratch.ids.as_slice()))
            .map_err(|e| format!("ids: {}", e))?;
        let t_uid = TensorRef::from_array_view(([1usize, seq_len], self.scratch.uids.as_slice()))
            .map_err(|e| format!("uid: {}", e))?;
        let t_ek = TensorRef::from_array_view((
            [self.enc_layers, 1, self.num_heads, kv.p, self.head_dim],
            kv.enc_k.as_slice(),
        ))
        .map_err(|e| format!("ek: {}", e))?;
        let t_ev = TensorRef::from_array_view((
            [self.enc_layers, 1, self.num_heads, kv.p, self.head_dim],
            kv.enc_v.as_slice(),
        ))
        .map_err(|e| format!("ev: {}", e))?;
        let t_ck = TensorRef::from_array_view((
            [self.choice_layers, 1, self.num_heads, kv.p, self.head_dim],
            kv.choice_k.as_slice(),
        ))
        .map_err(|e| format!("ck: {}", e))?;
        let t_cv = TensorRef::from_array_view((
            [self.choice_layers, 1, self.num_heads, kv.p, self.head_dim],
            kv.choice_v.as_slice(),
        ))
        .map_err(|e| format!("cv: {}", e))?;
        let t_pm = TensorRef::from_array_view(([1usize, kv.p], &self.true_mask[..kv.p]))
            .map_err(|e| format!("pm: {}", e))?;
        let t_mp =
            TensorRef::from_array_view(([1usize, n_markers], self.scratch.markers.as_slice()))
                .map_err(|e| format!("mp: {}", e))?;
        let t_sp =
            TensorRef::from_array_view(([1usize, n_slots_in], self.scratch.slots.as_slice()))
                .map_err(|e| format!("sp: {}", e))?;

        let inputs = ort::inputs![
            "input_ids" => t_ids,
            "unit_ids" => t_uid,
            "past_enc_k" => t_ek,
            "past_enc_v" => t_ev,
            "past_choice_k" => t_ck,
            "past_choice_v" => t_cv,
            "past_mask" => t_pm,
            "marker_pos" => t_mp,
            "slot_pos" => t_sp,
        ];

        let run_opts = if want_char_logits && self.has_char_logits {
            self.key_run_opts.as_ref()
        } else {
            self.pass2_run_opts.as_ref()
        };

        let t0 = Instant::now();
        let outputs = if let Some(opts) = run_opts {
            self.session
                .run_with_options(inputs, opts)
                .map_err(|e| format!("key run: {}", e))?
        } else {
            self.session
                .run(inputs)
                .map_err(|e| format!("key run: {}", e))?
        };
        self.stats.total_infer_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let (_, l_slice) = outputs
            .get("logits")
            .ok_or_else(|| "missing ONNX output: logits".to_string())?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("logits: {}", e))?;
        if l_slice.len() < n_markers {
            return Err(format!(
                "logits length {} < n_markers {}",
                l_slice.len(),
                n_markers
            ));
        }
        let logits: Vec<f32> = l_slice[..n_markers].to_vec();

        if let Some((cands, orig_units_len, shifted_lcp)) = decode_p2c {
            if self.has_char_logits
                && let Some(char_val) = outputs.get("char_logits")
            {
                let (c_shape, c_slice) = char_val
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("char_logits: {}", e))?;
                let vocab_chars =
                    usize::try_from(c_shape.last().copied().unwrap_or(0)).unwrap_or(0);
                let (char_lps, p2c_text, p2c_margin) = self.char_vocab.decode_char_logits(
                    c_slice,
                    (real_slots, vocab_chars),
                    units,
                    cands,
                    &self.readings,
                    orig_units_len,
                    shifted_lcp,
                );
                return Ok((logits, char_lps, p2c_text, p2c_margin));
            }
            return Ok((logits, vec![0.0f32; cands.len()], None, -100.0));
        }

        Ok((logits, Vec::new(), None, -100.0))
    }

    fn run_whole_graph(
        &mut self,
        ctx_final: &str,
        units: &[(&'static str, String)],
        cand_toks: &[Vec<i64>],
        cands: &[String],
        orig_units_len: usize,
        shifted_lcp: bool,
    ) -> Result<ForwardResult, String> {
        let ctx_ids = self.build_context_token_ids_cached(ctx_final);
        self.scratch.ids.clear();
        self.scratch.ids.extend_from_slice(&ctx_ids);
        self.scratch.uids.clear();
        self.scratch.uids.resize(self.scratch.ids.len(), 0);
        self.scratch.slots.clear();
        self.scratch.markers.clear();

        for (kind, text) in units.iter().take(self.max_slots) {
            self.scratch.slots.push(self.scratch.ids.len() as i64);
            self.scratch.ids.push(self.mask_id);
            let uid = self.lookup_unit_id(kind, text);
            self.scratch.uids.push(uid);
        }
        self.scratch.ids.push(self.sep_id);
        self.scratch.uids.push(0);

        for ctok in cand_toks {
            self.scratch.markers.push(self.scratch.ids.len() as i64);
            if !self.choice_compact {
                self.scratch.ids.push(self.mask_id);
            }
            let take = ctok.len().min(self.candidate_tokens);
            self.scratch.ids.extend_from_slice(&ctok[..take]);
        }
        self.scratch.ids.push(self.sep_id);
        self.scratch.uids.resize(self.scratch.ids.len(), 0);

        let seq_len = self.scratch.ids.len();
        let n_markers = self.scratch.markers.len();
        let real_slots = self.scratch.slots.len();
        if self.scratch.slots.is_empty() {
            self.scratch.slots.push(0);
        }
        let n_slots_in = self.scratch.slots.len();

        self.scratch.att.clear();
        self.scratch.att.resize(seq_len, 1);
        self.scratch.mm.clear();
        self.scratch.mm.resize(n_markers, true);
        let t_ids = TensorRef::from_array_view(([1usize, seq_len], self.scratch.ids.as_slice()))
            .map_err(|e| format!("ids: {}", e))?;
        let t_att = TensorRef::from_array_view(([1usize, seq_len], self.scratch.att.as_slice()))
            .map_err(|e| format!("att: {}", e))?;
        let t_uid = TensorRef::from_array_view(([1usize, seq_len], self.scratch.uids.as_slice()))
            .map_err(|e| format!("uid: {}", e))?;
        let t_mp =
            TensorRef::from_array_view(([1usize, n_markers], self.scratch.markers.as_slice()))
                .map_err(|e| format!("mp: {}", e))?;
        let t_mm = TensorRef::from_array_view(([1usize, n_markers], self.scratch.mm.as_slice()))
            .map_err(|e| format!("mm: {}", e))?;
        let t_sp =
            TensorRef::from_array_view(([1usize, n_slots_in], self.scratch.slots.as_slice()))
                .map_err(|e| format!("sp: {}", e))?;

        let t0 = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => t_ids,
                "attention_mask" => t_att,
                "unit_ids" => t_uid,
                "marker_pos" => t_mp,
                "marker_mask" => t_mm,
                "slot_pos" => t_sp,
            ])
            .map_err(|e| format!("whole run: {}", e))?;
        self.stats.total_infer_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let (_, l_slice) = outputs
            .get("logits")
            .ok_or_else(|| "missing ONNX output: logits".to_string())?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("logits: {}", e))?;
        if l_slice.len() < n_markers {
            return Err(format!(
                "logits length {} < n_markers {}",
                l_slice.len(),
                n_markers
            ));
        }
        let logits: Vec<f32> = l_slice[..n_markers].to_vec();

        if self.has_char_logits
            && let Some(char_val) = outputs.get("char_logits")
        {
            let (c_shape, c_slice) = char_val
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("char_logits: {}", e))?;
            let vocab_chars = usize::try_from(c_shape.last().copied().unwrap_or(0)).unwrap_or(0);
            let (char_lps, p2c_text, p2c_margin) = self.char_vocab.decode_char_logits(
                c_slice,
                (real_slots, vocab_chars),
                units,
                cands,
                &self.readings,
                orig_units_len,
                shifted_lcp,
            );
            Ok((logits, char_lps, p2c_text, p2c_margin))
        } else {
            Ok((logits, vec![0.0f32; cands.len()], None, -100.0))
        }
    }
}

/// Pure P2C placement decision with strict finiteness guards on Pass-2 logit/gram deltas and
/// legacy Pass-1 margins.
fn select_p2c_placement(
    p2c_text: Option<&str>,
    p2c_margin: f32,
    p2c_verified: Option<(f32, f32)>,
    used_fused_kv: bool,
    has_ctx: bool,
    p2c_cfg: P2cConfig,
) -> (Option<String>, u8) {
    let Some(txt) = p2c_text else {
        return (None, 0);
    };
    if let Some((ldiff, gdiff)) = p2c_verified {
        if !ldiff.is_finite() || !gdiff.is_finite() {
            return (None, 0);
        }
        if p2c_cfg.top1_logit_diff > 0.0
            && ldiff > p2c_cfg.top1_logit_diff
            && gdiff > p2c_cfg.top1_min_gram_diff
        {
            (Some(txt.to_string()), 1)
        } else if p2c_cfg.top2_logit_diff > 0.0
            && ldiff > p2c_cfg.top2_logit_diff
            && gdiff > (p2c_cfg.top1_min_gram_diff - 1.5)
        {
            (Some(txt.to_string()), 2)
        } else {
            (None, 0)
        }
    } else if !used_fused_kv && has_ctx && p2c_margin.is_finite() {
        if p2c_cfg.legacy_top1_margin > 0.0 && p2c_margin > p2c_cfg.legacy_top1_margin {
            (Some(txt.to_string()), 1)
        } else if p2c_cfg.legacy_top2_margin > 0.0 && p2c_margin > p2c_cfg.legacy_top2_margin {
            (Some(txt.to_string()), 2)
        } else {
            (None, 0)
        }
    } else {
        (None, 0)
    }
}

pub fn rerank(
    context: &str,
    raw_input: &str,
    layout: PinyinLayout,
    cands: &[String],
    weights: FusionWeights,
    p2c_cfg: P2cConfig,
    gram_req: &GramRequest,
) -> Result<RerankResult, String> {
    let ctx_tail = utf8_tail(context, CONTEXT_CHARS);
    let has_ctx_bool = ctx_tail.trim().chars().count() >= 2;
    if cands.len() < 2 {
        return Ok(RerankResult {
            order: (1..=cands.len()).collect(),
            p2c_text: None,
            p2c_pos: 0,
            logits: vec![0.0; cands.len()],
            gram_deltas: vec![0.0; cands.len()],
            char_lps: vec![0.0; cands.len()],
            has_ctx: has_ctx_bool,
            has_gram: false,
        });
    }
    let mutex = ENGINE.get().ok_or_else(|| "engine not ready".to_string())?;
    let mut eng = mutex
        .lock()
        .map_err(|_| "engine mutex poisoned".to_string())?;
    let eval = eng.evaluate(
        context,
        raw_input,
        layout,
        cands,
        weights,
        p2c_cfg.verify_margin,
        gram_req,
    )?;

    let has_ctx = if has_ctx_bool { 1.0 } else { 0.0 };
    let order0 = rank_order_from_features(
        &eval.logits,
        &eval.gram_deltas,
        &eval.char_lps,
        has_ctx,
        weights,
    );

    let (p2c_text, p2c_pos) = select_p2c_placement(
        eval.p2c_text.as_deref(),
        eval.p2c_margin,
        eval.p2c_verified,
        eval.used_fused_kv,
        has_ctx_bool,
        p2c_cfg,
    );

    Ok(RerankResult {
        order: order0.into_iter().map(|idx| idx + 1).collect(),
        p2c_text,
        p2c_pos,
        logits: eval.logits,
        gram_deltas: eval.gram_deltas,
        char_lps: eval.char_lps,
        has_ctx: has_ctx_bool,
        has_gram: eval.has_gram,
    })
}

pub struct StatsSnapshot {
    pub ready: bool,
    pub calls: u64,
    pub cache_hits: u64,
    pub kv_hits: u64,
    pub p2c_pass2_calls: u64,
    pub avg_ms: f64,
    pub fused_kv: bool,
    pub has_gram: bool,
    pub enc_layers: usize,
    pub choice_compact: bool,
    pub model_name: String,
    pub last_error: Option<String>,
}

pub fn get_stats() -> StatsSnapshot {
    let Some(mutex) = ENGINE.get() else {
        return StatsSnapshot {
            ready: false,
            calls: 0,
            cache_hits: 0,
            kv_hits: 0,
            p2c_pass2_calls: 0,
            avg_ms: 0.0,
            fused_kv: false,
            has_gram: false,
            enc_layers: 0,
            choice_compact: false,
            model_name: String::new(),
            last_error: last_init_error(),
        };
    };
    let Ok(eng) = mutex.lock() else {
        return StatsSnapshot {
            ready: false,
            calls: 0,
            cache_hits: 0,
            kv_hits: 0,
            p2c_pass2_calls: 0,
            avg_ms: 0.0,
            fused_kv: false,
            has_gram: false,
            enc_layers: 0,
            choice_compact: false,
            model_name: String::new(),
            last_error: Some("engine mutex poisoned".to_string()),
        };
    };
    let infer_calls = eng.stats.calls.saturating_sub(eng.stats.cache_hits);
    let avg_ms = if infer_calls > 0 {
        eng.stats.total_infer_ms / (infer_calls as f64)
    } else {
        0.0
    };
    StatsSnapshot {
        ready: true,
        calls: eng.stats.calls,
        cache_hits: eng.stats.cache_hits,
        kv_hits: eng.stats.kv_hits,
        p2c_pass2_calls: eng.stats.p2c_pass2_calls,
        avg_ms,
        fused_kv: eng.stats.fused_kv,
        has_gram: eng.stats.has_gram,
        enc_layers: eng.stats.enc_layers,
        choice_compact: eng.stats.choice_compact,
        model_name: eng.stats.model_name.clone(),
        last_error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::AddedToken;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::metaspace::Metaspace;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

    fn test_char_vocab() -> CharVocab {
        // Indices 0..4 are core Simplified (`old_chars = 4`), index 4 (`東`, `dong`) is extended
        // (`cid >= old_chars`), indices 5..9 test alternate segmentation (`方/案/反/感`).
        let chars: Vec<String> = vec![
            "你".into(),
            "拟".into(),
            "好".into(),
            "号".into(),
            "東".into(),
            "方".into(),
            "案".into(),
            "反".into(),
            "感".into(),
        ];
        let mut char_to_id = HashMap::new();
        for (i, s) in chars.iter().enumerate() {
            char_to_id.insert(s.chars().next().unwrap(), i);
        }
        let mut syl_to_all_cids = HashMap::new();
        syl_to_all_cids.insert("ni".to_string(), vec![0, 1]);
        syl_to_all_cids.insert("hao".to_string(), vec![2, 3, 4]);
        syl_to_all_cids.insert("fang".to_string(), vec![5]);
        syl_to_all_cids.insert("an".to_string(), vec![6]);
        CharVocab {
            chars,
            char_to_id,
            syl_to_all_cids,
            old_chars: 4,
            max_slots: 24,
        }
    }

    fn test_readings() -> HashMap<String, Vec<String>> {
        let mut r = HashMap::new();
        r.insert("你".into(), vec!["ni".into()]);
        r.insert("拟".into(), vec!["ni".into()]);
        r.insert("好".into(), vec!["hao".into()]);
        r.insert("号".into(), vec!["hao".into()]);
        r.insert("東".into(), vec!["hao".into()]);
        r.insert("方".into(), vec!["fang".into()]);
        r.insert("案".into(), vec!["an".into()]);
        r.insert("反".into(), vec!["fan".into()]);
        r.insert("感".into(), vec!["gan".into()]);
        r
    }

    #[test]
    fn test_common_prefix_and_utf8_tail() {
        let cands = vec!["我们今天去公园".to_string(), "我们今天去广场".to_string()];
        assert_eq!(common_prefix_chars(&cands), 5);
        let identical = vec!["你好".to_string(), "你好".to_string()];
        assert_eq!(common_prefix_chars(&identical), 0);
        assert_eq!(utf8_tail("你好世界测试", 4), "世界测试");
        assert_eq!(utf8_tail("短", 4), "短");
        assert_eq!(utf8_tail("abc", 0), "");
        assert_eq!(utf8_tail("你好", 0), "");
    }

    #[test]
    fn test_rank_order_respects_fusion_weights_and_sanitizes_nan() {
        let logits = [0.0f32, 3.0, -1.0];
        let grams = [0.0f32, 0.0, 0.0];
        let clps = [0.0f32, 0.0, 0.0];
        let order = rank_order_from_features(&logits, &grams, &clps, 0.0, FusionWeights::default());
        assert_eq!(order, vec![1, 0, 2]);

        // NaN / +Inf / -Inf logits must not crash or sort NaN above finite logits
        let nan_logits = [0.0f32, f32::NAN, 2.0];
        let order_nan =
            rank_order_from_features(&nan_logits, &grams, &clps, 0.0, FusionWeights::default());
        assert_eq!(order_nan[0], 2);
        assert_eq!(order_nan[2], 1);

        // `w_ctx_lp` amplifies neural logits only when `has_ctx = 1.0`
        let w_ctx = FusionWeights {
            w_ctx_lp: 2.0,
            ..FusionWeights::default()
        };
        let close_logits = [0.0f32, 0.8];
        let o_no_ctx =
            rank_order_from_features(&close_logits, &[0.0, 0.0], &[0.0, 0.0], 0.0, w_ctx);
        let o_with_ctx =
            rank_order_from_features(&close_logits, &[0.0, 0.0], &[0.0, 0.0], 1.0, w_ctx);
        assert_eq!(o_no_ctx[0], 0);
        assert_eq!(o_with_ctx[0], 1);
    }

    #[test]
    fn test_decode_char_logits_suppresses_p2c_on_length_mismatch_and_alt_seg() {
        let cv = test_char_vocab();
        let readings = test_readings();
        let units = vec![("f", "ni".to_string()), ("f", "hao".to_string())];
        // Vocab width 9; strongly prefers "拟号" (cid 1, cid 3) and scores extended `東` (cid 4) well
        let mut c_slice = vec![0.0f32; 2 * 9];
        c_slice[1] = 5.0; // slot 0: 拟
        c_slice[9 + 3] = 5.0; // slot 1: 号
        c_slice[9 + 4] = 4.5; // slot 1: extended `東` (cid=4 >= old_chars=4)

        // Same-length candidates -> emits P2C "拟号" with positive margin
        let cands_ok = vec!["你好".to_string(), "你号".to_string()];
        let (clps, p2c, margin) =
            cv.decode_char_logits(&c_slice, (2, 9), &units, &cands_ok, &readings, 2, false);
        assert_eq!(p2c.as_deref(), Some("拟号"));
        assert!(margin > 0.0);
        assert!(clps[1] > clps[0]);

        // Extended char (`cid >= old_chars`) uses true logit instead of `-15.0` penalty, while
        // P2C synthesis remains capped to `cid < old_chars` (`拟号`, never `拟東`).
        let cands_ext = vec!["你好".to_string(), "你東".to_string()];
        let (clps_ext, p2c_ext, _) =
            cv.decode_char_logits(&c_slice, (2, 9), &units, &cands_ext, &readings, 2, false);
        assert_eq!(p2c_ext.as_deref(), Some("拟号"));
        assert!(
            clps_ext[1] > 0.0,
            "extended char with high logit should beat baseline, got {}",
            clps_ext[1]
        );

        // Mismatched candidate length (3 chars vs 2 slots) -> must suppress P2C completely
        let cands_mismatch = vec!["你好啊".to_string(), "你好".to_string()];
        let (clps2, p2c2, margin2) = cv.decode_char_logits(
            &c_slice,
            (2, 9),
            &units,
            &cands_mismatch,
            &readings,
            2,
            false,
        );
        assert_eq!(p2c2, None);
        assert_eq!(margin2, -100.0);
        assert_eq!(clps2, vec![0.0, 0.0]);

        // Equal-length alternate segmentation (`方案` vs `反感` on `fangan`): must not penalize `反感` by -15/slot
        let units_fangan = vec![("f", "fang".to_string()), ("f", "an".to_string())];
        let cands_alt = vec!["方案".to_string(), "反感".to_string()];
        let (clps_alt, p2c_alt, _) = cv.decode_char_logits(
            &c_slice,
            (2, 9),
            &units_fangan,
            &cands_alt,
            &readings,
            2,
            false,
        );
        assert_eq!(clps_alt, vec![0.0, 0.0]);
        assert_eq!(p2c_alt, None);
    }

    #[test]
    fn test_extended_character_logits_share_candidate_and_p2c_scale() {
        let cv = test_char_vocab();
        let readings = test_readings();
        let units = [("f", "ni".to_string()), ("f", "hao".to_string())];
        let mut scores = [0.0; 18];
        scores[1] = 5.0;
        scores[9 + 3] = 5.0;
        scores[9 + 4] = 15.0;
        let cands = ["你好".into(), "你東".into()];
        let (clps, generated, margin) =
            cv.decode_char_logits(&scores, (2, 9), &units, &cands, &readings, 2, false);
        assert_eq!(clps, [0.0, 7.5]);
        assert_eq!(generated.as_deref(), Some("拟号"));
        assert_eq!(margin, -2.5);

        // A slot with only extended finite logits can still score existing candidates.
        scores[9 + 2] = f32::NAN;
        scores[9 + 3] = f32::NAN;
        let cands = ["你東".into(), "拟東".into()];
        let (clps, generated, _) =
            cv.decode_char_logits(&scores, (2, 9), &units, &cands, &readings, 2, false);
        assert_eq!(clps, [0.0, 2.5]);
        assert_eq!(generated, None);

        // For an unknown character the fixed -15 log probability uses all three hao IDs.
        let cands = ["你好".into(), "你□".into()];
        let (clps, _, _) =
            cv.decode_char_logits(&[0.0; 18], (2, 9), &units, &cands, &readings, 2, false);
        assert!((clps[1] - (3.0f32.ln() - 15.0) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_select_p2c_placement_guards_non_finite_and_thresholds() {
        let cfg = P2cConfig::default();
        // +Inf / NaN Pass-2 deltas must never promote P2C
        assert_eq!(
            select_p2c_placement(
                Some("拟号"),
                1.0,
                Some((f32::INFINITY, 0.0)),
                true,
                true,
                cfg
            ),
            (None, 0)
        );
        assert_eq!(
            select_p2c_placement(Some("拟号"), 1.0, Some((3.0, f32::NAN)), true, true, cfg),
            (None, 0)
        );
        // Valid Pass-2 #1 promotion
        assert_eq!(
            select_p2c_placement(Some("拟号"), 1.0, Some((2.5, -1.0)), true, true, cfg),
            (Some("拟号".to_string()), 1)
        );
        // Fails #1 gram floor (-2.5 < -2.0), passes #2 gram floor (-2.5 > -3.5)
        assert_eq!(
            select_p2c_placement(Some("拟号"), 1.0, Some((2.5, -2.5)), true, true, cfg),
            (Some("拟号".to_string()), 2)
        );
        // Fails #2 gram floor (-4.0 < -3.5)
        assert_eq!(
            select_p2c_placement(Some("拟号"), 1.0, Some((2.5, -4.0)), true, true, cfg),
            (None, 0)
        );
        // Fused KV without Pass-2 verification never falls back to legacy margin
        assert_eq!(
            select_p2c_placement(Some("拟号"), 5.0, None, true, true, cfg),
            (None, 0)
        );
    }

    #[test]
    fn test_sanitize_and_collect_special_tokens_preserves_html_and_strips_non_angle_specials() {
        let model = WordLevel::builder()
            .vocab(
                [("<unk>".to_string(), 0u32), ("hello".to_string(), 1u32)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("<unk>".to_string())
            .build()
            .expect("wordlevel");
        let mut tok = Tokenizer::new(model);
        tok.add_special_tokens(&[
            AddedToken::from("<bos>", true),
            AddedToken::from("<eos>", true),
            AddedToken::from("<start_of_turn>", true),
            AddedToken::from("[CLS]", true),
        ]);
        tok.add_tokens(&[
            AddedToken::from("<code>", false),
            AddedToken::from("<b>", false),
            AddedToken::from("<unused0>", false),
            AddedToken::from("[@BOS@]", false),
        ]);
        let (specials, triggers, forbidden) = collect_special_tokens(&tok);
        assert!(specials.iter().any(|s| s == "<bos>"));
        assert!(specials.iter().any(|s| s == "[CLS]"));
        assert!(specials.iter().any(|s| s == "<unused0>"));
        assert!(specials.iter().any(|s| s == "[@BOS@]"));
        assert!(
            !specials.iter().any(|s| s == "<code>" || s == "<b>"),
            "non-special HTML tokens must not be stripped: {:?}",
            specials
        );
        assert!(
            !forbidden.contains(&0),
            "<unk> must not be blocked in forbidden_special_ids"
        );
        assert_eq!(
            sanitize_special_tokens("普通文本", &specials, &triggers).as_ref(),
            "普通文本"
        );
        assert_eq!(
            sanitize_special_tokens("<code>Vec<T></code>", &specials, &triggers).as_ref(),
            "<code>Vec<T></code>"
        );
        assert_eq!(
            sanitize_special_tokens(
                "a<bos>b<start_of_turn>c[CLS]d<unused0>e[@BOS@]f",
                &specials,
                &triggers
            )
            .as_ref(),
            "a b c d e f"
        );
    }

    #[test]
    fn test_build_no_prepend_tokenizer_handles_sequence_pretokenizer() {
        let model = WordLevel::builder()
            .vocab([("<unk>".to_string(), 0u32)].into_iter().collect())
            .unk_token("<unk>".to_string())
            .build()
            .expect("wordlevel");
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true)),
        ]))));
        let no_prep =
            build_no_prepend_tokenizer(&tok, Path::new("nonexistent.json")).expect("clone seq");
        let Some(PreTokenizerWrapper::Sequence(seq)) = no_prep.get_pre_tokenizer() else {
            panic!("expected Sequence pre-tokenizer");
        };
        let Some(PreTokenizerWrapper::Metaspace(ms)) = seq.as_ref().get(1) else {
            panic!("expected inner Metaspace");
        };
        assert_eq!(ms.prepend_scheme, PrependScheme::Never);
    }

    #[test]
    fn test_cache_key_distinguishes_fusion_weights_and_gram_params() {
        assert_eq!(canon_f64_bits(-0.0), canon_f64_bits(0.0));
        assert_eq!(canon_f32_bits(-0.0), canon_f32_bits(0.0));
        let w1 = FusionWeights::default();
        let w2 = FusionWeights { w_rank: -0.5, ..w1 };
        let g1 = ResolvedGramKey {
            path: Some(PathBuf::from("a.gram")),
            col_bits: canon_f64_bits(-15.5),
            non_col_bits: canon_f64_bits(-4.0),
            max_length: 5,
        };
        let g2 = ResolvedGramKey {
            col_bits: canon_f64_bits(-14.0),
            ..g1.clone()
        };
        let k1 = CacheKey {
            ctx_tail: "你好".to_string(),
            raw_input: "nihk".to_string(),
            layout: PinyinLayout::Zrm,
            weights_bits: [
                canon_f64_bits(w1.w_logit),
                canon_f64_bits(w1.w_rank),
                canon_f64_bits(w1.w_gram),
                canon_f64_bits(w1.w_char_lp),
                canon_f64_bits(w1.w_ctx_lp),
            ],
            p2c_verify_margin_bits: canon_f32_bits(0.2),
            gram_key: g1,
            cands: vec!["你好".to_string()],
        };
        let k2 = CacheKey {
            weights_bits: [
                canon_f64_bits(w2.w_logit),
                canon_f64_bits(w2.w_rank),
                canon_f64_bits(w2.w_gram),
                canon_f64_bits(w2.w_char_lp),
                canon_f64_bits(w2.w_ctx_lp),
            ],
            ..k1.clone()
        };
        let k3 = CacheKey {
            gram_key: g2,
            ..k1.clone()
        };
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
    }
}
