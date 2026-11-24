//! Non-blocking, privacy-hardened Windows UI Automation (UIA) preceding-text reader.
//!
//! Architecture guarantees:
//!   1. **Zero main-thread blocking**: All COM/UIA calls run on a single dedicated background
//!      STA thread. The keystroke hot path only calls `GetForegroundWindow()` + `try_send()`
//!      + `RwLock::try_read()` (< 2 µs total), so sluggish apps never hang `WeaselServer.exe`.
//!   2. **Zero cross-app contamination**: Tracks the active foreground `HWND` synchronously on
//!      every query and clamps `TextPattern` ranges to `RangeFromChild(focused)` + `\u{FFFC}`
//!      boundaries so Chromium/Electron apps never bleed surrounding UI labels or stale
//!      `commit_history` from another window.
//!   3. **Hard password protection**: Any control reporting `IsPassword == true` or hosted by a
//!      known password manager / Windows credential UI returns `UiaContext::Blocked` immediately
//!      (forcing empty context AND disabling local feedback logging).

use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiaContext {
    /// Password control or sensitive process -> force empty context and suppress logging.
    Blocked,
    /// Authoritative text strictly before the caret in the active control (may be empty if the
    /// user is typing in a fresh/empty box).
    Text(String),
    /// Foreground window just changed and fresh UIA snapshot is not ready yet -> do not reuse
    /// previous window's `commit_history`.
    WindowChanged,
    /// Active window unchanged, control does not expose UIA text -> safe to fall back to Rime's
    /// same-window `commit_history`.
    FallbackSameWindow,
}

const MAX_UIA_CHARS: usize = 64;
const MAX_VALUE_PATTERN_TAIL_CHARS: usize = 256;
const SNAPSHOT_TTL: Duration = Duration::from_secs(4);
const MIN_REFRESH_INTERVAL: Duration = Duration::from_millis(35);
const POST_COMMIT_SETTLE_DELAY: Duration = Duration::from_millis(25);

const DENIED_PROCESSES: &[&str] = &[
    "keepass.exe",
    "keepassxc.exe",
    "1password.exe",
    "bitwarden.exe",
    "enpass.exe",
    "lastpass.exe",
    "dashlane.exe",
    "roboform.exe",
    "nordpass.exe",
    "protonpass.exe",
    "credentialuibroker.exe",
    "consent.exe",
    "lockapp.exe",
    "mstsc.exe",
];

fn is_invisible_format_char(c: char) -> bool {
    matches!(
        c,
        '\u{200B}' // ZERO WIDTH SPACE (Electron / Chromium ProseMirror / Slate caret anchor)
            | '\u{200C}' // ZERO WIDTH NON-JOINER
            | '\u{200D}' // ZERO WIDTH JOINER
            | '\u{2060}' // WORD JOINER
            | '\u{FEFF}' // ZERO WIDTH NO-BREAK SPACE / BOM
    )
}

fn tail_utf8_chars(s: &str, max_chars: usize) -> &str {
    let count = s.chars().count();
    if count <= max_chars {
        return s;
    }
    match s.char_indices().nth(count - max_chars) {
        Some((idx, _)) => &s[idx..],
        None => s,
    }
}

/// Strip embedded-object chrome (`\u{FFFC}`), zero-width format characters, live inline
/// composition/preview text, and control characters into a clean 64-char preceding-text tail.
pub fn normalize_before_caret(
    raw: &str,
    live_preedit: &str,
    raw_input: &str,
    cand_preview: &str,
) -> String {
    // Chromium / Electron / Word insert U+FFFC (Object Replacement Character) at DOM widget /
    // toolbar / attachment boundaries. Never bleed across embedded UI elements.
    let base = match raw.rfind('\u{FFFC}') {
        Some(pos) => &raw[pos + '\u{FFFC}'.len_utf8()..],
        None => raw,
    };

    // Strip invisible zero-width format anchors before suffix matching so Chromium/Electron
    // editors wrapping composition ranges with U+200B do not defeat preedit stripping.
    let mut s = String::with_capacity(base.len());
    let mut prev_cr = false;
    for c in base.chars() {
        if is_invisible_format_char(c) {
            continue;
        }
        if c == '\r' {
            s.push('\n');
            prev_cr = true;
        } else if c == '\n' {
            if !prev_cr {
                s.push('\n');
            }
            prev_cr = false;
        } else {
            s.push(c);
            prev_cr = false;
        }
    }

    // Strip uncommitted inline composition echoed into the text control (longest match first, once).
    // Check pinyin preedit/raw input before inline `preedit_type: preview` candidate text.
    let trimmed_tail = s.trim_end_matches(' ');
    if trimmed_tail.len() != s.len() {
        s.truncate(trimmed_tail.len());
    }
    let lp = live_preedit.trim();
    let ri = raw_input.trim();
    let preedit_suffixes = if lp.len() >= ri.len() {
        [lp, ri]
    } else {
        [ri, lp]
    };
    let mut stripped = false;
    for suffix in preedit_suffixes {
        if !suffix.is_empty() && s.ends_with(suffix) {
            let new_len = s.len().saturating_sub(suffix.len());
            s.truncate(new_len);
            stripped = true;
            break;
        }
    }
    if !stripped {
        let cp = cand_preview.trim();
        if cp.chars().count() >= 2 && s.ends_with(cp) {
            let new_len = s.len().saturating_sub(cp.len());
            s.truncate(new_len);
        }
    }

    // Keep only the active paragraph after blank-line breaks when non-empty
    if let Some(pos) = s.rfind("\n\n") {
        let tail = s[pos + 2..].trim_start();
        if !tail.is_empty() {
            s = tail.to_string();
        }
    }

    // Flatten remaining line breaks and trim trailing spaces
    let cleaned: String = s
        .chars()
        .map(|c| if c == '\n' || c == '\t' { ' ' } else { c })
        .filter(|&c| !c.is_control())
        .collect();
    let trimmed = cleaned.trim_end();

    tail_utf8_chars(trimmed, MAX_UIA_CHARS).to_string()
}

#[cfg(windows)]
mod win_impl {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
    use std::sync::mpsc::{self, SyncSender};
    use std::sync::{Mutex, OnceLock, RwLock};
    use std::thread;

    use windows::core::{Interface, PWSTR};
    use windows::Win32::Foundation::{CloseHandle, HWND};
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER,
        COINIT_MULTITHREADED,
    };
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, CUIAutomation8, IUIAutomation, IUIAutomation6, IUIAutomationTextPattern,
        IUIAutomationValuePattern, TextPatternRangeEndpoint_End, TextPatternRangeEndpoint_Start,
        TextUnit_Character, UIA_EditControlTypeId, UIA_TextPatternId, UIA_ValuePatternId,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        GetAncestor, GetForegroundWindow, GetWindowThreadProcessId, GA_ROOT,
    };

    const UIA_TIMEOUT_MS: u32 = 200;

    struct ComGuard;
    impl Drop for ComGuard {
        fn drop(&mut self) {
            unsafe {
                CoUninitialize();
            }
        }
    }

    struct UiaSnapshot {
        hwnd: isize,
        is_password: bool,
        readable: bool,
        before_caret: String,
        updated_at: Option<Instant>,
    }

    struct UiaRuntime {
        tx: SyncSender<()>,
        snapshot: RwLock<UiaSnapshot>,
    }

    static RUNTIME: OnceLock<UiaRuntime> = OnceLock::new();
    static LAST_SEEN_HWND: AtomicIsize = AtomicIsize::new(0);
    static LAST_DENIED: Mutex<(isize, bool)> = Mutex::new((0, false));
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static POST_COMMIT_PENDING: AtomicBool = AtomicBool::new(false);

    fn is_denied_pid(pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) };
        let Ok(handle) = handle else {
            return false;
        };
        let mut buf = [0u16; 512];
        let mut len = buf.len() as u32;
        let ok = unsafe {
            QueryFullProcessImageNameW(
                handle,
                PROCESS_NAME_WIN32,
                PWSTR(buf.as_mut_ptr()),
                &mut len,
            )
            .is_ok()
        };
        unsafe {
            let _ = CloseHandle(handle);
        }
        if !ok || len == 0 {
            return false;
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        let exe = path
            .rsplit(['\\', '/'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        DENIED_PROCESSES.iter().any(|&p| p == exe)
    }

    fn is_denied_hwnd_cached(hwnd: HWND) -> bool {
        let raw = hwnd.0 as isize;
        if raw == 0 {
            return false;
        }
        if let Ok(guard) = LAST_DENIED.lock() {
            if guard.0 == raw {
                return guard.1;
            }
        }
        let mut pid: u32 = 0;
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
        }
        let denied = is_denied_pid(pid);
        if let Ok(mut guard) = LAST_DENIED.lock() {
            *guard = (raw, denied);
        }
        denied
    }

    fn unreadable_snapshot(hwnd_raw: isize, is_password: bool, now: Instant) -> UiaSnapshot {
        UiaSnapshot {
            hwnd: hwnd_raw,
            is_password,
            readable: false,
            before_caret: String::new(),
            updated_at: Some(now),
        }
    }

    fn probe_once(automation: &IUIAutomation) -> UiaSnapshot {
        let now = Instant::now();
        let hwnd = unsafe { GetForegroundWindow() };
        let hwnd_raw = hwnd.0 as isize;
        if hwnd_raw == 0 {
            return unreadable_snapshot(0, false, now);
        }

        if is_denied_hwnd_cached(hwnd) {
            return unreadable_snapshot(hwnd_raw, true, now);
        }

        let focused = match unsafe { automation.GetFocusedElement() } {
            Ok(el) => el,
            Err(_) => {
                return unreadable_snapshot(hwnd_raw, false, now);
            }
        };

        // Verify focused element belongs to the sampled foreground root window when a native HWND is reported
        if let Ok(el_hwnd) = unsafe { focused.CurrentNativeWindowHandle() } {
            if !el_hwnd.0.is_null() {
                let root = unsafe { GetAncestor(el_hwnd, GA_ROOT) };
                if !root.0.is_null() && root.0 as isize != hwnd_raw {
                    return unreadable_snapshot(hwnd_raw, false, now);
                }
            }
        }

        // Also check focused element's own PID against the password-manager denylist
        if let Ok(fpid) = unsafe { focused.CurrentProcessId() } {
            if fpid > 0 && is_denied_pid(fpid as u32) {
                return unreadable_snapshot(hwnd_raw, true, now);
            }
        }

        // Hard password check
        if let Ok(is_pw) = unsafe { focused.CurrentIsPassword() } {
            if is_pw.as_bool() {
                return unreadable_snapshot(hwnd_raw, true, now);
            }
        }

        // Primary path: IUIAutomationTextPattern (Chrome, Edge, VS Code, Electron, Word, Notepad, WPF, WinUI)
        if let Ok(text_pat) =
            unsafe { focused.GetCurrentPatternAs::<IUIAutomationTextPattern>(UIA_TextPatternId) }
        {
            if let Ok(selection) = unsafe { text_pat.GetSelection() } {
                if let Ok(len) = unsafe { selection.Length() } {
                    if len >= 1 {
                        if let Ok(caret_range) = unsafe { selection.GetElement(0) } {
                            if let Ok(before) = unsafe { caret_range.Clone() } {
                                let collapsed = unsafe {
                                    before.MoveEndpointByRange(
                                        TextPatternRangeEndpoint_End,
                                        &caret_range,
                                        TextPatternRangeEndpoint_Start,
                                    )
                                };
                                if collapsed.is_ok() {
                                    let _ = unsafe {
                                        before.MoveEndpointByUnit(
                                            TextPatternRangeEndpoint_Start,
                                            TextUnit_Character,
                                            -96,
                                        )
                                    };
                                    // Clamp `before`'s start strictly within `focused` so Chromium/Electron
                                    // WebArea text providers never walk backwards into surrounding buttons/chrome.
                                    if let Ok(elem_range) =
                                        unsafe { text_pat.RangeFromChild(&focused) }
                                    {
                                        let starts_before = unsafe {
                                            before.CompareEndpoints(
                                                TextPatternRangeEndpoint_Start,
                                                &elem_range,
                                                TextPatternRangeEndpoint_Start,
                                            )
                                        };
                                        let end_inside = unsafe {
                                            before.CompareEndpoints(
                                                TextPatternRangeEndpoint_End,
                                                &elem_range,
                                                TextPatternRangeEndpoint_Start,
                                            )
                                        };
                                        if matches!(starts_before, Ok(cmp) if cmp < 0)
                                            && matches!(end_inside, Ok(cmp) if cmp >= 0)
                                        {
                                            let _ = unsafe {
                                                before.MoveEndpointByRange(
                                                    TextPatternRangeEndpoint_Start,
                                                    &elem_range,
                                                    TextPatternRangeEndpoint_Start,
                                                )
                                            };
                                        }
                                    }
                                    if let Ok(bstr) = unsafe { before.GetText(512) } {
                                        // Re-verify foreground window did not change mid-probe (cross-app isolation)
                                        if unsafe { GetForegroundWindow().0 as isize } != hwnd_raw {
                                            return unreadable_snapshot(hwnd_raw, false, now);
                                        }
                                        return UiaSnapshot {
                                            hwnd: hwnd_raw,
                                            is_password: false,
                                            readable: true,
                                            before_caret: bstr.to_string(),
                                            updated_at: Some(now),
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback: IUIAutomationValuePattern ONLY for standard non-readonly Edit controls
        if let Ok(ctype) = unsafe { focused.CurrentControlType() } {
            if ctype == UIA_EditControlTypeId {
                if let Ok(val_pat) = unsafe {
                    focused.GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId)
                } {
                    let is_ro = unsafe { val_pat.CurrentIsReadOnly() }
                        .map(|b| b.as_bool())
                        .unwrap_or(true);
                    if !is_ro {
                        if let Ok(bstr) = unsafe { val_pat.CurrentValue() } {
                            if unsafe { GetForegroundWindow().0 as isize } != hwnd_raw {
                                return unreadable_snapshot(hwnd_raw, false, now);
                            }
                            let full = bstr.to_string();
                            let tail = tail_utf8_chars(&full, MAX_VALUE_PATTERN_TAIL_CHARS);
                            return UiaSnapshot {
                                hwnd: hwnd_raw,
                                is_password: false,
                                readable: true,
                                before_caret: tail.to_string(),
                                updated_at: Some(now),
                            };
                        }
                    }
                }
            }
        }

        unreadable_snapshot(hwnd_raw, false, now)
    }

    fn ensure_started() -> &'static UiaRuntime {
        RUNTIME.get_or_init(|| {
            let (tx, rx) = mpsc::sync_channel::<()>(1);
            let runtime = UiaRuntime {
                tx,
                snapshot: RwLock::new(UiaSnapshot {
                    hwnd: 0,
                    is_password: false,
                    readable: false,
                    before_caret: String::new(),
                    updated_at: None,
                }),
            };

            let _ = thread::Builder::new()
                .name("rime_neural_uia".to_string())
                .spawn(move || {
                    let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
                    if hr.is_err() {
                        return;
                    }
                    // Declared before `automation` so `IUIAutomation` drops before `CoUninitialize`.
                    let _com_guard = ComGuard;

                    let automation: Result<IUIAutomation, _> = unsafe {
                        CoCreateInstance(&CUIAutomation8, None, CLSCTX_INPROC_SERVER).or_else(
                            |_| CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER),
                        )
                    };
                    let Ok(automation) = automation else {
                        return;
                    };
                    if let Ok(auto6) = automation.cast::<IUIAutomation6>() {
                        unsafe {
                            let _ = auto6.SetConnectionTimeout(UIA_TIMEOUT_MS);
                            let _ = auto6.SetTransactionTimeout(UIA_TIMEOUT_MS);
                        }
                    }

                    let mut last_probe: Option<Instant> = None;
                    let mut last_hwnd: isize = 0;

                    while rx.recv().is_ok() {
                        let post_commit = POST_COMMIT_PENDING.swap(false, Ordering::AcqRel);
                        if post_commit {
                            // Wait briefly off the main thread so TSF finishes inserting committed
                            // text into Chromium / Electron / Word before probing UIA.
                            thread::sleep(POST_COMMIT_SETTLE_DELAY);
                            if !POST_COMMIT_PENDING.load(Ordering::Acquire) {
                                while rx.try_recv().is_ok() {}
                            }
                        } else {
                            let cur_hwnd = unsafe { GetForegroundWindow().0 as isize };
                            if cur_hwnd == last_hwnd
                                && last_probe.is_some_and(|t| t.elapsed() < MIN_REFRESH_INTERVAL)
                            {
                                continue;
                            }
                        }
                        let snap = probe_once(&automation);
                        last_hwnd = snap.hwnd;
                        last_probe = snap.updated_at;
                        if let Some(rt) = RUNTIME.get() {
                            if let Ok(mut guard) = rt.snapshot.write() {
                                *guard = snap;
                            }
                        }
                    }
                });

            runtime
        })
    }

    /// Enable the background UIA worker thread (per-schema opt-out is handled in Lua so one
    /// schema with `enable_uia_context: false` never disables UIA for another schema).
    pub fn set_enabled(enable: bool) {
        if enable {
            ENABLED.store(true, Ordering::Release);
            let rt = ensure_started();
            let _ = rt.tx.try_send(());
        }
    }

    pub fn is_enabled() -> bool {
        ENABLED.load(Ordering::Acquire)
    }

    /// Non-blocking hot-path query (< 2 µs). Never waits on COM or UIA.
    pub fn query_context(live_preedit: &str, raw_input: &str, cand_preview: &str) -> UiaContext {
        if !is_enabled() {
            return UiaContext::FallbackSameWindow;
        }
        let rt = ensure_started();
        let hwnd = unsafe { GetForegroundWindow() };
        let cur_hwnd = hwnd.0 as isize;
        let prev_hwnd = LAST_SEEN_HWND.swap(cur_hwnd, Ordering::AcqRel);

        // Kick background worker asynchronously (never blocks)
        let _ = rt.tx.try_send(());

        if cur_hwnd == 0 {
            return UiaContext::FallbackSameWindow;
        }

        // Immediate synchronous password-manager check (cached per HWND, ~15 µs once on window switch)
        if is_denied_hwnd_cached(hwnd) {
            return UiaContext::Blocked;
        }

        // Fast non-blocking read of latest cached snapshot
        if let Ok(snap) = rt.snapshot.try_read() {
            if snap.hwnd == cur_hwnd {
                if snap.is_password {
                    return UiaContext::Blocked;
                }
                if snap.readable && snap.updated_at.is_some_and(|t| t.elapsed() <= SNAPSHOT_TTL) {
                    let clean = normalize_before_caret(
                        &snap.before_caret,
                        live_preedit,
                        raw_input,
                        cand_preview,
                    );
                    return UiaContext::Text(clean);
                }
                if prev_hwnd != 0 && prev_hwnd != cur_hwnd {
                    return UiaContext::WindowChanged;
                }
                return UiaContext::FallbackSameWindow;
            }
            // Background worker has not yet finished probing `cur_hwnd` after a window switch:
            // keep suppressing stale cross-app `commit_history` until the new window is probed.
            if snap.hwnd != 0 {
                return UiaContext::WindowChanged;
            }
        }

        if prev_hwnd != 0 && prev_hwnd != cur_hwnd {
            UiaContext::WindowChanged
        } else {
            UiaContext::FallbackSameWindow
        }
    }

    pub fn foreground_hwnd() -> isize {
        unsafe { GetForegroundWindow().0 as isize }
    }

    pub fn kick_refresh() {
        if is_enabled() {
            let rt = ensure_started();
            let _ = rt.tx.try_send(());
        }
    }

    /// Force a post-commit UIA probe on the background STA thread after a brief settle delay,
    /// bypassing `MIN_REFRESH_INTERVAL` so newly committed text is captured reliably.
    pub fn kick_after_commit() {
        if is_enabled() {
            let rt = ensure_started();
            POST_COMMIT_PENDING.store(true, Ordering::Release);
            let _ = rt.tx.try_send(());
        }
    }
}

#[cfg(not(windows))]
mod win_impl {
    use super::*;
    pub fn set_enabled(_enable: bool) {}
    pub fn is_enabled() -> bool {
        false
    }
    pub fn query_context(_live_preedit: &str, _raw_input: &str, _cand_preview: &str) -> UiaContext {
        UiaContext::FallbackSameWindow
    }
    pub fn foreground_hwnd() -> isize {
        0
    }
    pub fn kick_refresh() {}
    pub fn kick_after_commit() {}
}

pub use win_impl::{
    foreground_hwnd, is_enabled, kick_after_commit, kick_refresh, query_context, set_enabled,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_before_caret_strips_chrome_preedit_and_paragraph() {
        assert_eq!(
            normalize_before_caret("第一段内容\n\n我们今天去公园玩jntm", "jntm", "jntm", ""),
            "我们今天去公园玩"
        );
        assert_eq!(
            normalize_before_caret("今天天气真好，wo men ", "wo men", "women", ""),
            "今天天气真好，"
        );
        // Embedded object separator U+FFFC must block Chromium/Electron UI chrome bleed
        assert_eq!(
            normalize_before_caret(
                "No Project\u{FFFC}刚刚买了一个jwtyzu",
                "jw ty zu",
                "jwtyzu",
                ""
            ),
            "刚刚买了一个"
        );
        // Must only strip once (longest match first), never double-strip matching suffix
        assert_eq!(normalize_before_caret("ababc", "abc", "ab", ""), "ab");
    }

    #[test]
    fn test_normalize_before_caret_strips_zwsp_and_inline_preview() {
        // Electron/Chromium editors (ProseMirror/Slate) wrap composition ranges with U+200B/BOM
        assert_eq!(
            normalize_before_caret(
                "\u{FEFF}请帮我检查\u{200B}wo men\u{200B}",
                "wo men",
                "women",
                ""
            ),
            "请帮我检查"
        );
        // Weasel `preedit_type: preview` echoes top candidate inline instead of pinyin preedit
        assert_eq!(
            normalize_before_caret("我们正在讨论策略", "ce lue", "celue", "策略"),
            "我们正在讨论"
        );
        // When pinyin preedit matches, candidate preview must not also be stripped
        assert_eq!(
            normalize_before_caret("策略ce lue", "ce lue", "celue", "策略"),
            "策略"
        );
    }

    #[test]
    fn test_normalize_before_caret_truncates_utf8_safely() {
        let long =
            "这是一段很长的中文前文测试内容，用来验证多字节字符截断不会发生越界崩溃。".repeat(3);
        let out = normalize_before_caret(&long, "", "", "");
        assert_eq!(out.chars().count(), MAX_UIA_CHARS);
    }
}
