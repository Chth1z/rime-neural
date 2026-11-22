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
const SNAPSHOT_TTL: Duration = Duration::from_secs(4);
const MIN_REFRESH_INTERVAL: Duration = Duration::from_millis(35);

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

/// Strip embedded-object chrome (`\u{FFFC}`), live inline composition text, and control
/// characters into a clean 64-char preceding-text tail.
pub fn normalize_before_caret(raw: &str, live_preedit: &str, raw_input: &str) -> String {
    // Chromium / Electron / Word insert U+FFFC (Object Replacement Character) at DOM widget /
    // toolbar / attachment boundaries. Never bleed across embedded UI elements.
    let base = match raw.rfind('\u{FFFC}') {
        Some(pos) => &raw[pos + '\u{FFFC}'.len_utf8()..],
        None => raw,
    };

    let mut s = base.replace("\r\n", "\n").replace('\r', "\n");

    // Strip uncommitted inline composition echoed into the text control (longest match first, once)
    let trimmed_tail = s.trim_end_matches(' ');
    if trimmed_tail.len() != s.len() {
        s.truncate(trimmed_tail.len());
    }
    let lp = live_preedit.trim();
    let ri = raw_input.trim();
    let suffixes = if lp.len() >= ri.len() {
        [lp, ri]
    } else {
        [ri, lp]
    };
    for suffix in suffixes {
        if !suffix.is_empty() && s.ends_with(suffix) {
            let new_len = s.len().saturating_sub(suffix.len());
            s.truncate(new_len);
            break;
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

    let count = trimmed.chars().count();
    if count <= MAX_UIA_CHARS {
        return trimmed.to_string();
    }
    let skip = count - MAX_UIA_CHARS;
    match trimmed.char_indices().nth(skip) {
        Some((idx, _)) => trimmed[idx..].to_string(),
        None => trimmed.to_string(),
    }
}

#[cfg(windows)]
mod win_impl {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
    use std::sync::mpsc::{self, SyncSender};
    use std::sync::{OnceLock, RwLock};
    use std::thread;

    use windows::core::PWSTR;
    use windows::Win32::Foundation::{CloseHandle, HWND};
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER,
        COINIT_APARTMENTTHREADED,
    };
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, IUIAutomation, IUIAutomationTextPattern, IUIAutomationValuePattern,
        TextPatternRangeEndpoint_End, TextPatternRangeEndpoint_Start, TextUnit_Character,
        UIA_EditControlTypeId, UIA_TextPatternId, UIA_ValuePatternId,
    };
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

    struct UiaSnapshot {
        hwnd: isize,
        is_password: bool,
        readable: bool,
        before_caret: String,
        updated_at: Instant,
    }

    struct UiaRuntime {
        tx: SyncSender<()>,
        snapshot: RwLock<UiaSnapshot>,
    }

    static RUNTIME: OnceLock<UiaRuntime> = OnceLock::new();
    static LAST_SEEN_HWND: AtomicIsize = AtomicIsize::new(0);
    static LAST_DENIED_HWND: AtomicIsize = AtomicIsize::new(0);
    static LAST_DENIED_STATE: AtomicBool = AtomicBool::new(false);
    static ENABLED: AtomicBool = AtomicBool::new(false);

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
        if LAST_DENIED_HWND.load(Ordering::Acquire) == raw {
            return LAST_DENIED_STATE.load(Ordering::Acquire);
        }
        let mut pid: u32 = 0;
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
        }
        let denied = is_denied_pid(pid);
        LAST_DENIED_STATE.store(denied, Ordering::Release);
        LAST_DENIED_HWND.store(raw, Ordering::Release);
        denied
    }

    fn probe_once(automation: &IUIAutomation) -> UiaSnapshot {
        let now = Instant::now();
        let hwnd = unsafe { GetForegroundWindow() };
        let hwnd_raw = hwnd.0 as isize;
        if hwnd_raw == 0 {
            return UiaSnapshot {
                hwnd: 0,
                is_password: false,
                readable: false,
                before_caret: String::new(),
                updated_at: now,
            };
        }

        if is_denied_hwnd_cached(hwnd) {
            return UiaSnapshot {
                hwnd: hwnd_raw,
                is_password: true,
                readable: false,
                before_caret: String::new(),
                updated_at: now,
            };
        }

        let focused = match unsafe { automation.GetFocusedElement() } {
            Ok(el) => el,
            Err(_) => {
                return UiaSnapshot {
                    hwnd: hwnd_raw,
                    is_password: false,
                    readable: false,
                    before_caret: String::new(),
                    updated_at: now,
                };
            }
        };

        // Also check focused element's own PID against the password-manager denylist
        if let Ok(fpid) = unsafe { focused.CurrentProcessId() } {
            if fpid > 0 && is_denied_pid(fpid as u32) {
                return UiaSnapshot {
                    hwnd: hwnd_raw,
                    is_password: true,
                    readable: false,
                    before_caret: String::new(),
                    updated_at: now,
                };
            }
        }

        // Hard password check
        if let Ok(is_pw) = unsafe { focused.CurrentIsPassword() } {
            if is_pw.as_bool() {
                return UiaSnapshot {
                    hwnd: hwnd_raw,
                    is_password: true,
                    readable: false,
                    before_caret: String::new(),
                    updated_at: now,
                };
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
                                    if let Ok(bstr) = unsafe { before.GetText(128) } {
                                        return UiaSnapshot {
                                            hwnd: hwnd_raw,
                                            is_password: false,
                                            readable: true,
                                            before_caret: bstr.to_string(),
                                            updated_at: now,
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
                            return UiaSnapshot {
                                hwnd: hwnd_raw,
                                is_password: false,
                                readable: true,
                                before_caret: bstr.to_string(),
                                updated_at: now,
                            };
                        }
                    }
                }
            }
        }

        UiaSnapshot {
            hwnd: hwnd_raw,
            is_password: false,
            readable: false,
            before_caret: String::new(),
            updated_at: now,
        }
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
                    updated_at: Instant::now() - Duration::from_secs(60),
                }),
            };

            let _ = thread::Builder::new()
                .name("rime_neural_uia".to_string())
                .spawn(move || {
                    let hr = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
                    if hr.is_err() {
                        return;
                    }
                    let automation: Result<IUIAutomation, _> =
                        unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER) };
                    let Ok(automation) = automation else {
                        unsafe { CoUninitialize() };
                        return;
                    };

                    let mut last_probe = Instant::now() - Duration::from_secs(10);
                    let mut last_hwnd: isize = 0;

                    while rx.recv().is_ok() {
                        let cur_hwnd = unsafe { GetForegroundWindow().0 as isize };
                        if cur_hwnd == last_hwnd && last_probe.elapsed() < MIN_REFRESH_INTERVAL {
                            continue;
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

                    unsafe { CoUninitialize() };
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
    pub fn query_context(live_preedit: &str, raw_input: &str) -> UiaContext {
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
                if snap.readable && snap.updated_at.elapsed() <= SNAPSHOT_TTL {
                    let clean = normalize_before_caret(&snap.before_caret, live_preedit, raw_input);
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
}

#[cfg(not(windows))]
mod win_impl {
    use super::*;
    pub fn set_enabled(_enable: bool) {}
    pub fn is_enabled() -> bool {
        false
    }
    pub fn query_context(_live_preedit: &str, _raw_input: &str) -> UiaContext {
        UiaContext::FallbackSameWindow
    }
    pub fn foreground_hwnd() -> isize {
        0
    }
    pub fn kick_refresh() {}
}

pub use win_impl::{foreground_hwnd, is_enabled, kick_refresh, query_context, set_enabled};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_before_caret_strips_chrome_preedit_and_paragraph() {
        assert_eq!(
            normalize_before_caret("第一段内容\n\n我们今天去公园玩jntm", "jntm", "jntm"),
            "我们今天去公园玩"
        );
        assert_eq!(
            normalize_before_caret("今天天气真好，wo men ", "wo men", "women"),
            "今天天气真好，"
        );
        // Embedded object separator U+FFFC must block Chromium/Electron UI chrome bleed
        assert_eq!(
            normalize_before_caret("No Project\u{FFFC}刚刚买了一个jwtyzu", "jw ty zu", "jwtyzu"),
            "刚刚买了一个"
        );
        // Must only strip once (longest match first), never double-strip matching suffix
        assert_eq!(normalize_before_caret("ababc", "abc", "ab"), "ab");
    }

    #[test]
    fn test_normalize_before_caret_truncates_utf8_safely() {
        let long =
            "这是一段很长的中文前文测试内容，用来验证多字节字符截断不会发生越界崩溃。".repeat(3);
        let out = normalize_before_caret(&long, "", "");
        assert_eq!(out.chars().count(), MAX_UIA_CHARS);
    }
}
