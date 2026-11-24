use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024; // 5 MiB auto-rotation cap
const FEEDBACK_SCHEMA_VERSION: u8 = 2;

static LOG_LOCK: Mutex<()> = Mutex::new(());

/// Zero-plaintext numeric feedback record for offline fusion weight tuning (`scripts/tune-fusion.py`).
/// Stores only numerical model features and candidate index outcomes — never user text, keystrokes,
/// candidate strings, or UIA screen context.
#[derive(Serialize)]
pub struct FeedbackRecord<'a> {
    pub v: u8,
    pub ts: u64,
    pub sample_weight: f32,
    pub schema: &'a str,
    pub has_ctx: bool,
    pub num_cands: usize,
    pub logits: Vec<f32>,
    pub gram_deltas: Vec<f32>,
    pub char_lps: Vec<f32>,
    pub shown_top1_idx: i32,
    pub chosen_idx: i32,
    pub p2c_shown: bool,
    pub p2c_chosen: bool,
    pub chosen_rank: i32,
    pub reverted: bool,
}

impl<'a> FeedbackRecord<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_numeric(
        schema: &'a str,
        has_ctx: bool,
        logits: &[f32],
        gram_deltas: &[f32],
        char_lps: &[f32],
        shown_top1_idx: i32,
        chosen_idx: i32,
        p2c_shown: bool,
        p2c_chosen: bool,
        chosen_rank: i32,
        reverted: bool,
    ) -> Self {
        let n = logits.len();
        let clean = |xs: &[f32]| -> Vec<f32> {
            let mut v: Vec<f32> = xs
                .iter()
                .take(n)
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect();
            v.resize(n, 0.0);
            v
        };
        Self {
            v: FEEDBACK_SCHEMA_VERSION,
            ts: now_unix_secs(),
            sample_weight: 1.0,
            schema,
            has_ctx,
            num_cands: n,
            logits: clean(logits),
            gram_deltas: clean(gram_deltas),
            char_lps: clean(char_lps),
            shown_top1_idx,
            chosen_idx,
            p2c_shown,
            p2c_chosen,
            chosen_rank,
            reverted,
        }
    }
}

fn rotated_path(log_path: &Path) -> PathBuf {
    let mut os = OsString::from(log_path.as_os_str());
    os.push(".1");
    PathBuf::from(os)
}

pub fn append_feedback(log_path: &Path, rec: &FeedbackRecord<'_>) -> Result<(), String> {
    if rec.num_cands < 2 {
        return Ok(());
    }
    let _guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(parent) = log_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(meta) = fs::metadata(log_path) {
        if meta.len() >= MAX_LOG_BYTES {
            let rotated = rotated_path(log_path);
            let _ = fs::remove_file(&rotated);
            let _ = fs::rename(log_path, &rotated);
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| format!("open {}: {}", log_path.display(), e))?;
    let mut line = serde_json::to_string(rec).map_err(|e| format!("serialize feedback: {}", e))?;
    line.push('\n');
    file.write_all(line.as_bytes())
        .map_err(|e| format!("write feedback: {}", e))
}

pub fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_record_contains_zero_plaintext_and_sanitizes_non_finite() {
        let rec = FeedbackRecord::new_numeric(
            "rime_frost_double_pinyin",
            true,
            &[1.0, f32::NAN, f32::INFINITY],
            &[0.0, -0.5],
            &[-0.1, -0.2, -0.3],
            1,
            0,
            false,
            false,
            2,
            true,
        );
        let json = serde_json::to_string(&rec).expect("serialize");
        assert!(json.contains("\"v\":2"));
        assert!(
            !json.contains("null"),
            "non-finite floats must not serialize as null: {}",
            json
        );
        assert!(!json.contains("raw_input"));
        assert!(!json.contains("chosen_text"));
        assert!(!json.contains("context"));
        assert_eq!(rec.logits, vec![1.0, 0.0, 0.0]);
        assert_eq!(rec.gram_deltas, vec![0.0, -0.5, 0.0]);
    }

    #[test]
    fn test_append_feedback_atomic_jsonl_and_rotation_path() {
        let tmp = std::env::temp_dir().join("rime_neural_test_fb.jsonl");
        let _ = fs::remove_file(&tmp);
        let rec = FeedbackRecord::new_numeric(
            "rime_frost",
            false,
            &[0.5, -0.5],
            &[0.0, 0.1],
            &[-0.1, -0.2],
            0,
            0,
            false,
            false,
            1,
            false,
        );
        append_feedback(&tmp, &rec).expect("append");
        let content = fs::read_to_string(&tmp).expect("read");
        let _ = fs::remove_file(&tmp);
        assert!(content.ends_with('\n'));
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).expect("parse jsonl");
        assert_eq!(parsed["v"], 2);
        assert_eq!(parsed["num_cands"], 2);
        assert_eq!(
            rotated_path(Path::new("C:\\Users\\测试\\feedback.jsonl")),
            PathBuf::from("C:\\Users\\测试\\feedback.jsonl.1")
        );
    }
}
