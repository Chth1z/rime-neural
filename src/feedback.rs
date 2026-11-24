use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024; // 5 MiB auto-rotation cap
const FEEDBACK_SCHEMA_VERSION: u8 = 2;
const MAX_FEEDBACK_CANDS: usize = 64;
const MAX_SCHEMA_BYTES: usize = 128;
const MAX_MODEL_BYTES: usize = 64;

static LOG_LOCK: Mutex<()> = Mutex::new(());

/// Truncate a UTF-8 string to at most `max_bytes` on a valid char boundary.
fn truncate_utf8_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Zero-plaintext numeric feedback record for offline fusion weight tuning (`scripts/tune-fusion.py`).
/// Stores only numerical model features and candidate index outcomes — never user text, keystrokes,
/// candidate strings, or UIA screen context.
#[derive(Serialize)]
pub struct FeedbackRecord<'a> {
    pub v: u8,
    pub ts: u64,
    pub sample_weight: f32,
    pub schema: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    pub model: &'a str,
    pub has_gram: bool,
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
        model: &'a str,
        has_gram: bool,
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
        let n = logits.len().min(MAX_FEEDBACK_CANDS);
        let n_i32 = n as i32;
        let clean = |xs: &[f32]| -> Vec<f32> {
            let mut v: Vec<f32> = xs
                .iter()
                .take(n)
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect();
            v.resize(n, 0.0);
            v
        };
        let shown_top1_idx = if (-2..n_i32).contains(&shown_top1_idx) {
            shown_top1_idx
        } else {
            -1
        };
        let chosen_idx = if (-1..n_i32).contains(&chosen_idx) {
            chosen_idx
        } else {
            -1
        };
        let reverted = reverted && (shown_top1_idx > 0 || shown_top1_idx == -2) && chosen_idx == 0;
        Self {
            v: FEEDBACK_SCHEMA_VERSION,
            ts: now_unix_secs(),
            sample_weight: 1.0,
            schema: truncate_utf8_bytes(schema, MAX_SCHEMA_BYTES),
            model: truncate_utf8_bytes(model, MAX_MODEL_BYTES),
            has_gram,
            has_ctx,
            num_cands: n,
            logits: clean(logits),
            gram_deltas: clean(gram_deltas),
            char_lps: clean(char_lps),
            shown_top1_idx,
            chosen_idx,
            p2c_shown,
            p2c_chosen,
            chosen_rank: chosen_rank.max(0),
            reverted,
        }
    }
}

fn rotated_path(log_path: &Path) -> PathBuf {
    let mut os = OsString::from(log_path.as_os_str());
    os.push(".1");
    PathBuf::from(os)
}

#[cfg(windows)]
fn is_reparse_point(meta: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    use windows::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
    (meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT.0) != 0
}

#[cfg(not(windows))]
fn is_reparse_point(_meta: &fs::Metadata) -> bool {
    false
}

/// Fail-closed check refusing symlinks, Windows reparse points, directories, and unexpected
/// metadata errors (only `ErrorKind::NotFound` is accepted for non-existent paths).
fn reject_symlink_or_dir(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            if meta.file_type().is_symlink() || is_reparse_point(&meta) {
                return Err(format!(
                    "refusing to write symlink/reparse point: {}",
                    path.display()
                ));
            }
            if !meta.file_type().is_file() {
                return Err(format!(
                    "refusing to write non-regular file path: {}",
                    path.display()
                ));
            }
            Ok(())
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!(
            "metadata check failed for {}: {}",
            path.display(),
            e
        )),
    }
}

fn reject_symlink_dir(dir: &Path) -> Result<(), String> {
    match fs::symlink_metadata(dir) {
        Ok(meta) => {
            if meta.file_type().is_symlink() || is_reparse_point(&meta) {
                return Err(format!(
                    "refusing to write under symlink/reparse directory: {}",
                    dir.display()
                ));
            }
            if !meta.is_dir() {
                return Err(format!("parent path is not a directory: {}", dir.display()));
            }
            Ok(())
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!(
            "parent metadata failed for {}: {}",
            dir.display(),
            e
        )),
    }
}

enum RotateError {
    Hard(String),
    Transient(String),
}

fn atomic_rotate(src: &Path, dst: &Path) -> Result<(), RotateError> {
    reject_symlink_or_dir(dst).map_err(RotateError::Hard)?;
    fs::rename(src, dst).map_err(|e| {
        let msg = format!("rename {}: {}", src.display(), e);
        // A reader or antivirus may briefly lock either name during a Windows replacement.
        if cfg!(windows) && matches!(e.raw_os_error(), Some(32 | 33)) {
            RotateError::Transient(msg)
        } else {
            RotateError::Hard(msg)
        }
    })
}

fn open_append_no_reparse(log_path: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT.0);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(log_path)
        .map_err(|e| format!("open {}: {}", log_path.display(), e))?;
    {
        // File::metadata queries the opened handle/fd, so a replacement after the path check
        // cannot turn this append into a write through a symlink or reparse point.
        let meta = file
            .metadata()
            .map_err(|e| format!("opened file metadata {}: {}", log_path.display(), e))?;
        if !meta.is_file() || is_reparse_point(&meta) {
            return Err(format!(
                "refusing to write non-regular file or reparse point handle: {}",
                log_path.display()
            ));
        }
    }
    Ok(file)
}

fn append_feedback_with_cap(
    log_path: &Path,
    rec: &FeedbackRecord<'_>,
    max_bytes: u64,
) -> Result<(), String> {
    if rec.num_cands < 2 {
        return Ok(());
    }
    let _guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(parent) = log_path.parent()
        && !parent.as_os_str().is_empty()
    {
        reject_symlink_dir(parent)?;
        fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {}", parent.display(), e))?;
        reject_symlink_dir(parent)?;
    }
    reject_symlink_or_dir(log_path)?;
    if let Ok(meta) = fs::symlink_metadata(log_path)
        && meta.len() >= max_bytes
    {
        let rotated = rotated_path(log_path);
        match atomic_rotate(log_path, &rotated) {
            Ok(()) => {}
            Err(RotateError::Hard(msg)) => return Err(msg),
            Err(RotateError::Transient(msg)) => {
                // Allow brief overflow during transient reader locks, but enforce a strict 2x
                // hard cap so disk usage can never grow unboundedly.
                if meta.len() >= max_bytes.saturating_mul(2) {
                    return Err(format!(
                        "feedback log exceeded hard 2x cap after rotation lock: {}",
                        msg
                    ));
                }
            }
        }
    }
    reject_symlink_or_dir(log_path)?;
    let mut file = open_append_no_reparse(log_path)?;
    let mut line = serde_json::to_string(rec).map_err(|e| format!("serialize feedback: {}", e))?;
    line.push('\n');
    file.write_all(line.as_bytes())
        .map_err(|e| format!("write feedback: {}", e))
}

pub fn append_feedback(log_path: &Path, rec: &FeedbackRecord<'_>) -> Result<(), String> {
    append_feedback_with_cap(log_path, rec, MAX_LOG_BYTES)
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
            "laya.one.fused.int8g.onnx",
            true,
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
        assert!(json.contains("\"model\":\"laya.one.fused.int8g.onnx\""));
        assert!(json.contains("\"has_gram\":true"));
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
    fn test_record_caps_oversized_inputs_and_clamps_out_of_range_indices() {
        let big_schema = "s".repeat(500);
        let big_model = "m".repeat(200);
        let big_logits = vec![1.0f32; 200];
        let rec = FeedbackRecord::new_numeric(
            &big_schema,
            &big_model,
            false,
            false,
            &big_logits,
            &[],
            &[],
            99, // >= MAX_FEEDBACK_CANDS (64) -> must clamp to -1
            70, // >= MAX_FEEDBACK_CANDS (64) -> must clamp to -1
            false,
            false,
            -5,
            true,
        );
        assert_eq!(rec.schema.len(), MAX_SCHEMA_BYTES);
        assert_eq!(rec.model.len(), MAX_MODEL_BYTES);
        assert_eq!(rec.num_cands, MAX_FEEDBACK_CANDS);
        assert_eq!(rec.logits.len(), MAX_FEEDBACK_CANDS);
        assert_eq!(rec.shown_top1_idx, -1);
        assert_eq!(rec.chosen_idx, -1);
        assert_eq!(rec.chosen_rank, 0);
        assert!(!rec.reverted);
    }

    #[test]
    fn test_append_feedback_atomic_jsonl_and_rotation() {
        let tmp = std::env::temp_dir().join("rime_neural_test_fb_rot.jsonl");
        let rot = rotated_path(&tmp);
        let _ = fs::remove_file(&tmp);
        let _ = fs::remove_file(&rot);

        let rec = FeedbackRecord::new_numeric(
            "rime_frost",
            "laya.one.fused.int8g.onnx",
            true,
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
        append_feedback_with_cap(&tmp, &rec, 128).expect("append 1");
        // Second write exceeds 128-byte cap and triggers atomic rotation to `.1`
        append_feedback_with_cap(&tmp, &rec, 128).expect("append 2");

        let cur = fs::read_to_string(&tmp).expect("read current");
        let old = fs::read_to_string(&rot).expect("read rotated");
        let _ = fs::remove_file(&tmp);
        let _ = fs::remove_file(&rot);

        assert!(cur.ends_with('\n'));
        assert!(old.ends_with('\n'));
        let parsed: serde_json::Value = serde_json::from_str(cur.trim()).expect("parse jsonl");
        assert_eq!(parsed["v"], 2);
        assert_eq!(parsed["num_cands"], 2);
        assert_eq!(
            rotated_path(Path::new("C:\\Users\\测试\\feedback.jsonl")),
            PathBuf::from("C:\\Users\\测试\\feedback.jsonl.1")
        );
    }

    #[test]
    fn test_append_feedback_rejects_directory_at_log_or_rotated_destination() {
        let dir = std::env::temp_dir().join("rime_neural_test_fb_dir_guard");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");
        let log = dir.join("feedback.jsonl");
        let rot_dir = rotated_path(&log);
        fs::create_dir_all(&rot_dir).expect("mkdir .1");

        let rec = FeedbackRecord::new_numeric(
            "rime_frost",
            "m.onnx",
            false,
            false,
            &[0.5, -0.5],
            &[0.0, 0.0],
            &[0.0, 0.0],
            0,
            0,
            false,
            false,
            1,
            false,
        );
        // First append succeeds (below cap); second hits rotation against directory `.1` -> hard Err
        append_feedback_with_cap(&log, &rec, 64).expect("initial write");
        let err = append_feedback_with_cap(&log, &rec, 64);
        assert!(err.is_err(), "expected hard failure when .1 is a directory");

        // Directory at log path itself -> hard Err
        assert!(append_feedback_with_cap(&rot_dir, &rec, 64).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
