use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024; // 5 MiB auto-rotation cap

#[derive(Serialize)]
pub struct FeedbackRecord<'a> {
    pub ts: u64,
    pub sample_weight: f32,
    pub schema: &'a str,
    pub raw_input: &'a str,
    pub context: String,
    pub cands: &'a [String],
    pub logits: &'a [f32],
    pub gram_deltas: &'a [f32],
    pub char_lps: &'a [f32],
    pub p2c_cand: Option<&'a str>,
    pub shown_top1: &'a str,
    pub chosen_text: &'a str,
    pub chosen_rank: i32,
    pub reverted: bool,
}

/// Redact long digit/ASCII runs (phone numbers, tokens, URLs, emails) from stored context snippets.
fn sanitize_context(ctx: &str) -> String {
    let mut out = String::with_capacity(ctx.len());
    let mut ascii_run = String::new();
    let flush = |run: &mut String, dst: &mut String| {
        if run.is_empty() {
            return;
        }
        let has_digit = run.bytes().any(|b| b.is_ascii_digit());
        if (has_digit && run.len() >= 4)
            || run.len() >= 12
            || run.contains("://")
            || run.contains('@')
        {
            dst.push_str("<REDACTED>");
        } else {
            dst.push_str(run);
        }
        run.clear();
    };
    for ch in ctx.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '@' | ':' | '/' | '.' | '-' | '_') {
            ascii_run.push(ch);
        } else {
            flush(&mut ascii_run, &mut out);
            out.push(ch);
        }
    }
    flush(&mut ascii_run, &mut out);
    // Keep at most last 64 chars
    let total = out.chars().count();
    if total > 64 {
        let skip = total - 64;
        if let Some((bidx, _)) = out.char_indices().nth(skip) {
            return out[bidx..].to_string();
        }
    }
    out
}

pub fn append_feedback(log_path: &Path, rec: &FeedbackRecord<'_>) -> Result<(), String> {
    if let Some(parent) = log_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(meta) = fs::metadata(log_path) {
        if meta.len() >= MAX_LOG_BYTES {
            let rotated = PathBuf::from(format!("{}.1", log_path.display()));
            let _ = fs::remove_file(&rotated);
            let _ = fs::rename(log_path, &rotated);
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| format!("open {}: {}", log_path.display(), e))?;
    let line = serde_json::to_string(rec).map_err(|e| format!("serialize feedback: {}", e))?;
    writeln!(file, "{}", line).map_err(|e| format!("write feedback: {}", e))
}

pub fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn sanitize_for_log(ctx: &str) -> String {
    sanitize_context(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_redacts_secrets() {
        assert_eq!(
            sanitize_context("我的手机是13812345678请记下"),
            "我的手机是<REDACTED>请记下"
        );
        assert_eq!(
            sanitize_context("邮箱abc@example.com发过去"),
            "邮箱<REDACTED>发过去"
        );
        assert_eq!(
            sanitize_context("访问https://secret.example.com/tok"),
            "访问<REDACTED>"
        );
        assert_eq!(sanitize_context("用AI跑大模型"), "用AI跑大模型");
    }
}
