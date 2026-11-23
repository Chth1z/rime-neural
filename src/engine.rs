use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use ort::environment::Environment;
use ort::session::{
    builder::GraphOptimizationLevel, HasSelectedOutputs, OutputSelector, RunOptions, Session,
};
use ort::value::{Tensor, TensorRef};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::gram::WanxiangGram;
use crate::pinyin::{input_to_units, PinyinLayout};

const CONTEXT_CHARS: usize = 128;
const LRU_CAPACITY: usize = 256;
const KV_CACHE_CAPACITY: usize = 16;

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
    pub model_file: Option<String>,
    pub threads: usize,
}

#[derive(Clone)]
pub struct RerankResult {
    pub order: Vec<usize>,
    pub p2c_text: Option<String>,
    pub p2c_pos: u8, // 0 = none, 1 = top-1, 2 = top-2
    pub logits: Vec<f32>,
    pub gram_deltas: Vec<f32>,
    pub char_lps: Vec<f32>,
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
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct CacheKey {
    ctx_tail: String,
    raw_input: String,
    layout: PinyinLayout,
    p2c_verify_margin_bits: u32,
    cands: Vec<String>,
}

struct CharVocab {
    chars: Vec<String>,
    char_to_id: HashMap<char, usize>,
    syl_to_cids: HashMap<String, Vec<usize>>,
    old_chars: usize,
    max_slots: usize,
}

impl CharVocab {
    /// Decode candidate character log-probabilities and greedy P2C synthesis directly from ORT's
    /// borrowed `char_logits` slice without copying the full `[n_slots, vocab_chars]` tensor.
    fn decode_char_logits(
        &self,
        c_slice: &[f32],
        shape: (usize, usize),
        units: &[(&'static str, String)],
        cands: &[String],
        orig_units_len: usize,
        shifted_lcp: bool,
    ) -> (Vec<f32>, Option<String>, f32) {
        let (n_slots, vocab_chars) = shape;
        let mut char_lps = vec![0.0f32; cands.len()];
        let mut p2c_text: Option<String> = None;
        let mut p2c_margin: f32 = -100.0;

        let can_decode = !shifted_lcp
            && orig_units_len > 0
            && orig_units_len <= self.max_slots
            && n_slots == orig_units_len
            && vocab_chars >= self.old_chars
            && !self.chars.is_empty()
            && c_slice.len() >= n_slots.saturating_mul(vocab_chars);

        if !can_decode {
            return (char_lps, p2c_text, p2c_margin);
        }

        let mut gen_chars: Vec<char> = Vec::with_capacity(n_slots);
        let mut slot_lse: Vec<f32> = Vec::with_capacity(n_slots);
        let mut gen_sum = 0.0f32;
        let mut all_full = true;

        for (j, (kind, syl)) in units.iter().enumerate() {
            if *kind != "f" {
                all_full = false;
            }
            let row = &c_slice[j * vocab_chars..(j + 1) * vocab_chars];
            if let Some(allowed) = self.syl_to_cids.get(syl) {
                let mut max_v = f32::NEG_INFINITY;
                let mut best_cid = allowed[0];
                for &cid in allowed {
                    let v = row[cid];
                    if v > max_v {
                        max_v = v;
                        best_cid = cid;
                    }
                }
                let sum_e: f32 = allowed.iter().map(|&cid| (row[cid] - max_v).exp()).sum();
                let lse = max_v + sum_e.ln();
                gen_sum += max_v - lse;
                if let Some(ch) = self.chars[best_cid].chars().next() {
                    gen_chars.push(ch);
                }
                slot_lse.push(lse);
            } else {
                all_full = false;
                slot_lse.push(0.0);
            }
        }

        if all_full && gen_chars.len() == n_slots {
            let all_same_len = cands.iter().all(|c| c.chars().count() == n_slots);
            let mut raw_clp = Vec::with_capacity(cands.len());
            for c in cands {
                let mut s = 0.0f32;
                let mut cnt = 0usize;
                for (j, ch) in c.chars().take(n_slots).enumerate() {
                    cnt += 1;
                    let row = &c_slice[j * vocab_chars..(j + 1) * vocab_chars];
                    let lp = self
                        .char_to_id
                        .get(&ch)
                        .and_then(|&cid| {
                            let allowed = self.syl_to_cids.get(&units[j].1)?;
                            allowed
                                .binary_search(&cid)
                                .ok()
                                .map(|_| row[cid] - slot_lse[j])
                        })
                        .unwrap_or(-15.0);
                    s += lp;
                }
                if cnt < n_slots {
                    s += -15.0 * ((n_slots - cnt) as f32);
                }
                raw_clp.push(s / (n_slots as f32));
            }
            if all_same_len {
                let c0 = raw_clp[0];
                for (i, &v) in raw_clp.iter().enumerate() {
                    char_lps[i] = v - c0;
                }
            }
            let max_cand_clp = raw_clp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let gen_str: String = gen_chars.into_iter().collect();
            let gen_mean = gen_sum / (n_slots as f32);
            if !cands.iter().any(|c| c == &gen_str) {
                p2c_margin = gen_mean - max_cand_clp;
                p2c_text = Some(gen_str);
            }
        }

        (char_lps, p2c_text, p2c_margin)
    }
}

struct ScratchBufs {
    ids: Vec<i64>,
    uids: Vec<i64>,
    slots: Vec<i64>,
    markers: Vec<i64>,
}

impl ScratchBufs {
    fn new(max_slots: usize) -> Self {
        Self {
            ids: Vec::with_capacity(256),
            uids: Vec::with_capacity(256),
            slots: Vec::with_capacity(max_slots.max(4)),
            markers: Vec::with_capacity(16),
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
    mask: Vec<bool>,
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
    scratch: ScratchBufs,
    ctx_run_opts: Option<RunOptions<HasSelectedOutputs>>,
    key_run_opts: Option<RunOptions<HasSelectedOutputs>>,
    pass2_run_opts: Option<RunOptions<HasSelectedOutputs>>,
    tokenizer: Tokenizer,
    tokenizer_no_prepend: Option<Tokenizer>,
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
    gram: Option<WanxiangGram>,
    kv_cache: VecDeque<(String, ContextKv)>,
    lru_map: HashMap<CacheKey, CachedEval>,
    lru_order: VecDeque<CacheKey>,
    pub stats: EngineStats,
}

static ORT_ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
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

fn ensure_ort_env(ort_dll: &Path) -> Result<&'static Environment, String> {
    ORT_ENV
        .get_or_init(|| {
            ort::init_from(ort_dll)
                .map_err(|e| format!("ort::init_from({}) failed: {}", ort_dll.display(), e))?
                .with_name("rime_neural")
                .with_telemetry(false)
                .build()
                .map_err(|e| format!("ort env build failed: {}", e))
        })
        .as_ref()
        .map_err(|e| e.clone())
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

fn build_no_prepend_tokenizer(tok_path: &Path) -> Result<Tokenizer, String> {
    let raw = std::fs::read_to_string(tok_path)
        .map_err(|e| format!("read {}: {}", tok_path.display(), e))?;
    let mut val: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse tokenizer.json: {}", e))?;
    if let Some(pt) = val.get_mut("pre_tokenizer").and_then(|v| v.as_object_mut()) {
        pt.insert(
            "prepend_scheme".to_string(),
            serde_json::Value::String("never".to_string()),
        );
    }
    let bytes = serde_json::to_vec(&val).map_err(|e| format!("serialize tokenizer.json: {}", e))?;
    Tokenizer::from_bytes(&bytes).map_err(|e| format!("build no-prepend tokenizer: {}", e))
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
                if let Some(shape) = inp.dtype().tensor_shape() {
                    if shape.len() == 5 {
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
            }
            "past_choice_k" => {
                has_ch_k = true;
                if let Some(shape) = inp.dtype().tensor_shape() {
                    if shape.len() == 5 && shape[0] > 0 {
                        choice_layers = shape[0] as usize;
                    }
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

pub fn init_engine(cfg: &InitConfig) -> Result<(), String> {
    if ENGINE.get().is_some() {
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
        if let Some(ch) = s.chars().next() {
            if s.chars().count() == 1 {
                char_to_id.insert(ch, idx);
            }
        }
    }

    let old_chars = uv
        .ml
        .as_ref()
        .map(|m| m.old_chars)
        .unwrap_or(uv.chars.len())
        .min(uv.chars.len());
    let pinyin_src = uv
        .ml
        .as_ref()
        .filter(|m| !m.pinyin.is_empty())
        .map(|m| &m.pinyin)
        .unwrap_or(&uv.readings);

    let mut syl_to_cids: HashMap<String, Vec<usize>> = HashMap::with_capacity(450);
    for (ch_str, rlist) in pinyin_src {
        if let Some(ch) = ch_str.chars().next() {
            if let Some(&cid) = char_to_id.get(&ch) {
                if cid < old_chars {
                    for syl in rlist {
                        syl_to_cids.entry(syl.clone()).or_default().push(cid);
                    }
                }
            }
        }
    }
    // Sort & deduplicate each syllable's char-id list once so hot-path lookup uses O(log N) binary_search with zero HashMap allocs
    for cids in syl_to_cids.values_mut() {
        cids.sort_unstable();
        cids.dedup();
    }

    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("load {}: {}", tok_path.display(), e))?;
    let tokenizer_no_prepend = if uv.choice_compact {
        Some(build_no_prepend_tokenizer(&tok_path)?)
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
        .with_intra_threads(threads)
        .map_err(|e| format!("intra threads: {}", e))?
        .with_inter_threads(1)
        .map_err(|e| format!("inter threads: {}", e))?
        .commit_from_file(&onnx_path)
        .map_err(|e| format!("load {}: {}", onnx_path.display(), e))?;

    let (use_fused_kv, enc_layers, choice_layers, num_heads, head_dim) =
        introspect_kv_dims(&session);
    let has_char_logits = session.outputs().iter().any(|o| o.name() == "char_logits");
    let has_named_present = session
        .outputs()
        .iter()
        .any(|o| o.name() == "present_enc_k");
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

    let gram = cfg.gram_path.as_deref().and_then(|gp| {
        if gp.is_file() {
            WanxiangGram::open(
                gp,
                cfg.gram_collocation_penalty,
                cfg.gram_non_collocation_penalty,
            )
            .ok()
        } else {
            None
        }
    });
    let has_gram = gram.is_some();

    let z_enc = vec![0.0f32; enc_layers * num_heads * head_dim];
    let z_ch = vec![0.0f32; choice_layers * num_heads * head_dim];

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
        scratch: ScratchBufs::new(uv.max_slots),
        ctx_run_opts,
        key_run_opts,
        pass2_run_opts,
        tokenizer,
        tokenizer_no_prepend,
        choice_compact: uv.choice_compact,
        full_unit_ids,
        partial_unit_ids,
        char_vocab: CharVocab {
            chars: uv.chars,
            char_to_id,
            syl_to_cids,
            old_chars,
            max_slots: uv.max_slots,
        },
        readings: uv.readings,
        valid_syllables,
        max_slots: uv.max_slots,
        context_tokens: uv.choice_context_tokens,
        candidate_tokens: uv.choice_candidate_tokens,
        cls_id,
        sep_id,
        mask_id,
        unk_id,
        gram,
        kv_cache: VecDeque::with_capacity(KV_CACHE_CAPACITY + 2),
        lru_map: HashMap::with_capacity(LRU_CAPACITY + 16),
        lru_order: VecDeque::with_capacity(LRU_CAPACITY + 16),
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
    let _ = engine.evaluate(
        "",
        "nihk",
        PinyinLayout::Zrm,
        &warmup_cands,
        FusionWeights::default(),
        0.0,
    );
    engine.lru_map.clear();
    engine.lru_order.clear();
    engine.stats.calls = 0;
    engine.stats.cache_hits = 0;
    engine.stats.kv_hits = 0;
    engine.stats.p2c_pass2_calls = 0;
    engine.stats.total_infer_ms = 0.0;

    let _ = ENGINE.set(Mutex::new(engine));
    Ok(())
}

pub fn init_engine_async(cfg: InitConfig) {
    if ENGINE.get().is_some() {
        return;
    }
    if ASYNC_STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("rime-neural-init".to_string())
        .spawn(move || {
            if init_engine(&cfg).is_err() {
                ASYNC_STARTED.store(false, Ordering::Release);
            }
        });
}

pub fn is_ready() -> bool {
    ENGINE.get().is_some()
}

pub fn model_dir_path() -> Option<PathBuf> {
    let mutex = ENGINE.get()?;
    let eng = mutex.lock().ok()?;
    Some(eng.model_dir.clone())
}

fn utf8_tail(s: &str, max_chars: usize) -> &str {
    let total = s.chars().count();
    if total <= max_chars {
        return s;
    }
    let skip = total - max_chars;
    match s.char_indices().nth(skip) {
        Some((byte_idx, _)) => &s[byte_idx..],
        None => s,
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
    if lcp >= min_len {
        0
    } else {
        lcp
    }
}

fn rank_order_from_features(
    logits: &[f32],
    gram_deltas: &[f32],
    char_lps: &[f32],
    has_ctx: f64,
    weights: FusionWeights,
) -> Vec<usize> {
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp: f64 = logits.iter().map(|&x| ((x as f64) - max_l).exp()).sum();
    let lse = max_l + sum_exp.ln();

    let mut scored: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let lp = (x as f64) - lse;
            let score = weights.w_logit * lp
                + weights.w_rank * ((i as f64) + 1.0).ln()
                + weights.w_gram * (gram_deltas[i] as f64)
                + weights.w_char_lp * (char_lps[i] as f64)
                + weights.w_ctx_lp * lp * has_ctx;
            (i, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().map(|(idx, _)| idx).collect()
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

    fn encode_ids(&self, text: &str) -> Vec<i64> {
        let cleaned: Cow<'_, str> = if text.contains("<mask>") {
            Cow::Owned(text.replace("<mask>", " "))
        } else {
            Cow::Borrowed(text)
        };
        match self.tokenizer.encode(cleaned.as_ref(), false) {
            Ok(enc) => enc.get_ids().iter().map(|&x| x as i64).collect(),
            Err(_) => Vec::new(),
        }
    }

    fn encode_candidate_ids(&self, text: &str) -> Vec<i64> {
        if self.choice_compact {
            let cleaned: Cow<'_, str> = if text.contains("<mask>") {
                Cow::Owned(text.replace("<mask>", " "))
            } else {
                Cow::Borrowed(text)
            };
            let tok = self
                .tokenizer_no_prepend
                .as_ref()
                .unwrap_or(&self.tokenizer);
            let mut ids: Vec<i64> = match tok.encode(cleaned.as_ref(), false) {
                Ok(enc) => enc.get_ids().iter().map(|&x| x as i64).collect(),
                Err(_) => Vec::new(),
            };
            if ids.is_empty() {
                ids.push(self.unk_id);
            }
            ids
        } else {
            let mut s = String::with_capacity(text.len() + 1);
            s.push(' ');
            s.push_str(text);
            self.encode_ids(&s)
        }
    }

    fn compute_context_kv(&mut self, ctx_final: &str) -> Result<ContextKv, String> {
        let mut ids: Vec<i64> = Vec::with_capacity(self.context_tokens + 2);
        ids.push(self.cls_id);
        if !ctx_final.is_empty() {
            let cids = self.encode_ids(ctx_final);
            let start = cids.len().saturating_sub(self.context_tokens);
            ids.extend_from_slice(&cids[start..]);
        }
        ids.push(self.sep_id);
        let p = ids.len();
        let uids = vec![0i64; p];

        let t_ids =
            Tensor::from_array(([1usize, p], ids)).map_err(|e| format!("ctx ids: {}", e))?;
        let t_uid =
            Tensor::from_array(([1usize, p], uids)).map_err(|e| format!("ctx uid: {}", e))?;
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
        let t_pm =
            Tensor::from_array(([1usize, 1], vec![false])).map_err(|e| format!("pm: {}", e))?;
        let t_mp =
            Tensor::from_array(([1usize, 1], vec![0i64])).map_err(|e| format!("mp: {}", e))?;
        let t_sp =
            Tensor::from_array(([1usize, 1], vec![0i64])).map_err(|e| format!("sp: {}", e))?;

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

        let (_, ek) = outputs[self.present_names[0]]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("out ek: {}", e))?;
        let (_, ev) = outputs[self.present_names[1]]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("out ev: {}", e))?;
        let (_, ck) = outputs[self.present_names[2]]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("out ck: {}", e))?;
        let (_, cv) = outputs[self.present_names[3]]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("out cv: {}", e))?;

        Ok(ContextKv {
            p,
            enc_k: ek.to_vec(),
            enc_v: ev.to_vec(),
            choice_k: ck.to_vec(),
            choice_v: cv.to_vec(),
            mask: vec![true; p],
        })
    }

    fn get_or_compute_kv_index(&mut self, ctx_final: &str) -> Result<usize, String> {
        if let Some(pos) = self.kv_cache.iter().position(|(k, _)| k == ctx_final) {
            self.stats.kv_hits += 1;
            // Promote to MRU back (unless already at back)
            let last = self.kv_cache.len() - 1;
            if pos != last {
                if let Some(entry) = self.kv_cache.remove(pos) {
                    self.kv_cache.push_back(entry);
                }
            }
            return Ok(self.kv_cache.len() - 1);
        }
        let kv = self.compute_context_kv(ctx_final)?;
        if self.kv_cache.len() >= KV_CACHE_CAPACITY {
            // Keep empty context pinned; evict LRU non-empty entry
            if let Some(evict) = self.kv_cache.iter().position(|(k, _)| !k.is_empty()) {
                self.kv_cache.remove(evict);
            } else {
                self.kv_cache.pop_front();
            }
        }
        self.kv_cache.push_back((ctx_final.to_string(), kv));
        Ok(self.kv_cache.len() - 1)
    }

    fn evaluate(
        &mut self,
        context: &str,
        raw_input: &str,
        layout: PinyinLayout,
        cands: &[String],
        weights: FusionWeights,
        p2c_verify_margin: f32,
    ) -> Result<CachedEval, String> {
        self.stats.calls += 1;
        let ctx_tail = utf8_tail(context, CONTEXT_CHARS);
        let key = CacheKey {
            ctx_tail: ctx_tail.to_string(),
            raw_input: raw_input.to_string(),
            layout,
            p2c_verify_margin_bits: p2c_verify_margin.to_bits(),
            cands: cands.to_vec(),
        };
        if let Some(cached) = self.lru_map.get(&key).cloned() {
            self.stats.cache_hits += 1;
            // True LRU promotion
            if let Some(pos) = self.lru_order.iter().position(|k| k == &key) {
                if pos + 1 != self.lru_order.len() {
                    if let Some(k) = self.lru_order.remove(pos) {
                        self.lru_order.push_back(k);
                    }
                }
            }
            return Ok(cached);
        }

        let mut units = input_to_units(
            layout,
            raw_input,
            &cands[0],
            &self.readings,
            &self.valid_syllables,
        );
        let orig_units_len = units.len();

        let mut cand_toks: Vec<Vec<i64>> =
            cands.iter().map(|c| self.encode_candidate_ids(c)).collect();

        let mut eff_ctx = ctx_tail.to_string();
        let mut shifted_lcp = false;
        if cand_toks.iter().any(|t| t.len() > self.candidate_tokens) || units.len() > self.max_slots
        {
            let lcp = common_prefix_chars(cands);
            if lcp > 0 && lcp < units.len() {
                shifted_lcp = true;
                let prefix_str: String = cands[0].chars().take(lcp).collect();
                eff_ctx.push_str(&prefix_str);
                cand_toks = cands
                    .iter()
                    .map(|c| {
                        let tail: String = c.chars().skip(lcp).collect();
                        self.encode_candidate_ids(&tail)
                    })
                    .collect();
                units = units.split_off(lcp);
            }
        }

        let ctx_final = utf8_tail(&eff_ctx, CONTEXT_CHARS).to_string();
        let (logits, char_lps, p2c_text, p2c_margin) = if self.use_fused_kv {
            self.run_fused_kv(
                &ctx_final,
                &units,
                &cand_toks,
                Some((cands, orig_units_len, shifted_lcp)),
            )?
        } else {
            self.run_whole_graph(
                &ctx_final,
                &units,
                &cand_toks,
                cands,
                orig_units_len,
                shifted_lcp,
            )?
        };

        // Compute exact Wanxiang .gram deltas (~0.01 ms, 0 MB extra RAM)
        let gram_deltas: Vec<f32> = if let Some(ref g) = self.gram {
            let raw_g: Vec<f64> = cands.iter().map(|c| g.score_sentence_per_char(c)).collect();
            let g0 = raw_g.first().copied().unwrap_or(0.0);
            raw_g.into_iter().map(|x| (x - g0) as f32).collect()
        } else {
            vec![0.0f32; cands.len()]
        };

        // Optional KV-cached Pass-2 Cross-Encoder + .gram verification of promising novel P2C candidates
        // Triggers on only ~7% of sentences; reuses warm prefix KV-cache + Pass-1 candidate tokens and fetches ONLY `logits`.
        let mut p2c_verified: Option<(f32, f32)> = None;
        if self.use_fused_kv && p2c_verify_margin > 0.0 && p2c_margin > p2c_verify_margin {
            if let Some(ref p2c_str) = p2c_text {
                let has_ctx_f = if context.trim().chars().count() >= 2 {
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

                if let Ok((l2, _, _, _)) = self.run_fused_kv(&ctx_final, &units, &probe_toks, None)
                {
                    if let (Some(&l_best), Some(&l_p2c)) = (l2.first(), l2.last()) {
                        self.stats.p2c_pass2_calls += 1;
                        let gdiff = if let Some(ref g) = self.gram {
                            (g.score_sentence_per_char(p2c_str)
                                - g.score_sentence_per_char(&cands[best_idx]))
                                as f32
                        } else {
                            0.0
                        };
                        p2c_verified = Some((l_p2c - l_best, gdiff));
                    }
                }
            }
        }

        let res = CachedEval {
            logits,
            gram_deltas,
            char_lps,
            p2c_text,
            p2c_margin,
            p2c_verified,
            used_fused_kv: self.use_fused_kv,
        };

        if self.lru_map.len() >= LRU_CAPACITY {
            if let Some(old) = self.lru_order.pop_front() {
                self.lru_map.remove(&old);
            }
        }
        self.lru_order.push_back(key.clone());
        self.lru_map.insert(key, res.clone());
        Ok(res)
    }

    fn run_fused_kv(
        &mut self,
        ctx_final: &str,
        units: &[(&'static str, String)],
        cand_toks: &[Vec<i64>],
        decode_p2c: Option<(&[String], usize, bool)>,
    ) -> Result<ForwardResult, String> {
        let kv_idx = self.get_or_compute_kv_index(ctx_final)?;
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
        let t_pm = TensorRef::from_array_view(([1usize, kv.p], kv.mask.as_slice()))
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

        let (_, l_slice) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("logits: {}", e))?;
        let logits: Vec<f32> = l_slice.iter().take(n_markers).copied().collect();

        if let Some((cands, orig_units_len, shifted_lcp)) = decode_p2c {
            if self.has_char_logits {
                let (c_shape, c_slice) = outputs["char_logits"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("char_logits: {}", e))?;
                let vocab_chars = c_shape.last().copied().unwrap_or(0) as usize;
                let (char_lps, p2c_text, p2c_margin) = self.char_vocab.decode_char_logits(
                    c_slice,
                    (real_slots, vocab_chars),
                    units,
                    cands,
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
        let mut ids: Vec<i64> = Vec::with_capacity(256);
        ids.push(self.cls_id);
        if !ctx_final.is_empty() {
            let cids = self.encode_ids(ctx_final);
            let start = cids.len().saturating_sub(self.context_tokens);
            ids.extend_from_slice(&cids[start..]);
        }
        ids.push(self.sep_id);

        let mut uids: Vec<i64> = vec![0; ids.len()];
        let mut slots: Vec<i64> = Vec::with_capacity(self.max_slots);
        for (kind, text) in units.iter().take(self.max_slots) {
            slots.push(ids.len() as i64);
            ids.push(self.mask_id);
            uids.push(self.lookup_unit_id(kind, text));
        }
        ids.push(self.sep_id);
        uids.push(0);

        let mut markers: Vec<i64> = Vec::with_capacity(cand_toks.len());
        for ctok in cand_toks {
            markers.push(ids.len() as i64);
            if !self.choice_compact {
                ids.push(self.mask_id);
            }
            let take = ctok.len().min(self.candidate_tokens);
            ids.extend_from_slice(&ctok[..take]);
        }
        ids.push(self.sep_id);
        uids.resize(ids.len(), 0);

        let seq_len = ids.len();
        let n_markers = markers.len();
        let real_slots = slots.len();
        if slots.is_empty() {
            slots.push(0);
        }
        let n_slots_in = slots.len();

        let t_ids =
            Tensor::from_array(([1usize, seq_len], ids)).map_err(|e| format!("ids: {}", e))?;
        let t_att = Tensor::from_array(([1usize, seq_len], vec![1i64; seq_len]))
            .map_err(|e| format!("att: {}", e))?;
        let t_uid =
            Tensor::from_array(([1usize, seq_len], uids)).map_err(|e| format!("uid: {}", e))?;
        let t_mp =
            Tensor::from_array(([1usize, n_markers], markers)).map_err(|e| format!("mp: {}", e))?;
        let t_mm = Tensor::from_array(([1usize, n_markers], vec![true; n_markers]))
            .map_err(|e| format!("mm: {}", e))?;
        let t_sp =
            Tensor::from_array(([1usize, n_slots_in], slots)).map_err(|e| format!("sp: {}", e))?;

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

        let (_, l_slice) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("logits: {}", e))?;
        let logits: Vec<f32> = l_slice.iter().take(n_markers).copied().collect();

        if self.has_char_logits {
            let (c_shape, c_slice) = outputs["char_logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("char_logits: {}", e))?;
            let vocab_chars = c_shape.last().copied().unwrap_or(0) as usize;
            let (char_lps, p2c_text, p2c_margin) = self.char_vocab.decode_char_logits(
                c_slice,
                (real_slots, vocab_chars),
                units,
                cands,
                orig_units_len,
                shifted_lcp,
            );
            Ok((logits, char_lps, p2c_text, p2c_margin))
        } else {
            Ok((logits, vec![0.0f32; cands.len()], None, -100.0))
        }
    }
}

pub fn rerank(
    context: &str,
    raw_input: &str,
    layout: PinyinLayout,
    cands: &[String],
    weights: FusionWeights,
    p2c_cfg: P2cConfig,
) -> Result<RerankResult, String> {
    if cands.len() < 2 {
        return Ok(RerankResult {
            order: (1..=cands.len()).collect(),
            p2c_text: None,
            p2c_pos: 0,
            logits: vec![0.0; cands.len()],
            gram_deltas: vec![0.0; cands.len()],
            char_lps: vec![0.0; cands.len()],
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
    )?;

    let has_ctx = if context.trim().chars().count() >= 2 {
        1.0
    } else {
        0.0
    };
    let order0 = rank_order_from_features(
        &eval.logits,
        &eval.gram_deltas,
        &eval.char_lps,
        has_ctx,
        weights,
    );

    // P2C synthesis gating:
    //   1. Fused KV-cache graphs: strictly require KV-cached Pass-2 Cross-Encoder + .gram verification
    //      (setting `verify_margin <= 0` cleanly disables P2C on KV graphs without silent legacy fallback)
    //   2. Legacy non-KV whole graphs: context-gated Pass-1 `char_logits` margin fallback
    let (p2c_text, p2c_pos) = if let Some(ref txt) = eval.p2c_text {
        if let Some((ldiff, gdiff)) = eval.p2c_verified {
            if p2c_cfg.top1_logit_diff > 0.0
                && ldiff > p2c_cfg.top1_logit_diff
                && gdiff > p2c_cfg.top1_min_gram_diff
            {
                (Some(txt.clone()), 1u8)
            } else if p2c_cfg.top2_logit_diff > 0.0
                && ldiff > p2c_cfg.top2_logit_diff
                && gdiff > (p2c_cfg.top1_min_gram_diff - 1.5)
            {
                (Some(txt.clone()), 2u8)
            } else {
                (None, 0u8)
            }
        } else if !eval.used_fused_kv && has_ctx > 0.5 {
            if p2c_cfg.legacy_top1_margin > 0.0 && eval.p2c_margin > p2c_cfg.legacy_top1_margin {
                (Some(txt.clone()), 1u8)
            } else if p2c_cfg.legacy_top2_margin > 0.0
                && eval.p2c_margin > p2c_cfg.legacy_top2_margin
            {
                (Some(txt.clone()), 2u8)
            } else {
                (None, 0u8)
            }
        } else {
            (None, 0u8)
        }
    } else {
        (None, 0u8)
    };

    Ok(RerankResult {
        order: order0.into_iter().map(|idx| idx + 1).collect(),
        p2c_text,
        p2c_pos,
        logits: eval.logits,
        gram_deltas: eval.gram_deltas,
        char_lps: eval.char_lps,
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

    #[test]
    fn test_common_prefix_and_utf8_tail() {
        let cands = vec!["我们今天去公园".to_string(), "我们今天去广场".to_string()];
        assert_eq!(common_prefix_chars(&cands), 5);
        let identical = vec!["你好".to_string(), "你好".to_string()];
        assert_eq!(common_prefix_chars(&identical), 0);
        assert_eq!(utf8_tail("你好世界测试", 4), "世界测试");
        assert_eq!(utf8_tail("短", 4), "短");
    }

    #[test]
    fn test_rank_order_respects_fusion_weights() {
        let logits = [0.0f32, 3.0, -1.0];
        let grams = [0.0f32, 0.0, 0.0];
        let clps = [0.0f32, 0.0, 0.0];
        let order = rank_order_from_features(&logits, &grams, &clps, 0.0, FusionWeights::default());
        assert_eq!(order, vec![1, 0, 2]);
    }
}
