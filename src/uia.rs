//! Non-blocking, privacy-hardened Windows UI Automation (UIA) preceding-text reader.
//!
//! Architecture guarantees:
//! 1. **Zero main-thread blocking**: All COM/UIA calls run on a single dedicated background
//!    MTA thread (`COINIT_MULTITHREADED`, 200 ms `IUIAutomation6` connection/transaction
//!    timeouts). The keystroke hot path only calls `GetForegroundWindow()`, `try_send()`,
//!    and a brief `RwLock::read()` (writer never holds the lock during COM I/O), so sluggish apps
//!    never hang `WeaselServer.exe`.
//! 2. **Zero cross-app contamination**: Tracks the active foreground `HWND` synchronously on
//!    every query, verifies focused-element root ancestry and post-read foreground `HWND`, and
//!    clamps `TextPattern` ranges to `RangeFromChild(focused)` + `\u{FFFC}` boundaries so
//!    Chromium/Electron apps never bleed surrounding UI labels or stale `commit_history`.
//! 3. **Fail-closed password protection**: Any control reporting `IsPassword == true`, hosted by
//!    a password manager / credential UI / remote-session client, or running under an
//!    inaccessible Protected Process (`ERROR_ACCESS_DENIED`) returns `UiaContext::Blocked`
//!    immediately (forcing empty context AND disabling local feedback logging). Errors from
//!    `CurrentIsPassword()` refuse UIA text extraction rather than failing open.

use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiaContext {
    /// Password control or sensitive process -> force empty context and suppress logging.
    Blocked,
    /// Authoritative text strictly before the caret in the active control (may be empty if the
    /// user is typing in a fresh/empty box). `preedit_stripped` is `true` when inline pinyin/preview
    /// composition text was already stripped during normalization.
    Text {
        text: String,
        preedit_stripped: bool,
    },
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
    "keeper.exe",
    "keeperpasswordmanager.exe",
    "pwsafe.exe",
    "authy desktop.exe",
    "winauth.exe",
    "credentialuibroker.exe",
    "consent.exe",
    "lockapp.exe",
    "mstsc.exe",
    "msrdc.exe",
    "putty.exe",
    "xshell.exe",
    "mobaxterm.exe",
    "finalshell.exe",
];

fn is_invisible_format_char(c: char) -> bool {
    matches!(
        c,
        '\u{200B}' // ZERO WIDTH SPACE (Electron / Chromium ProseMirror / Slate caret anchor)
            | '\u{200C}' // ZERO WIDTH NON-JOINER
            | '\u{200D}' // ZERO WIDTH JOINER
            | '\u{2060}' // WORD JOINER
            | '\u{FEFF}' // ZERO WIDTH NO-BREAK SPACE / BOM
            | '\u{FFFD}' // REPLACEMENT CHARACTER (split UTF-16 surrogate artifact)
    )
}

fn is_mask_bullet(c: char) -> bool {
    matches!(c, '•' | '●' | '◦' | '▪' | '∗' | '*')
}

/// Heuristic detector for custom `ValuePattern` password fields that omit `IsPassword = true`
/// and expose masked bullets (`••••••••` or `********`).
pub fn looks_like_masked_password(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return false;
    }
    let mut non_ws = 0usize;
    let mut bullets = 0usize;
    let mut max_run = 0usize;
    let mut cur_run = 0usize;
    for c in trimmed.chars() {
        if c.is_whitespace() {
            cur_run = 0;
            continue;
        }
        non_ws += 1;
        if is_mask_bullet(c) {
            bullets += 1;
            cur_run += 1;
            if cur_run > max_run {
                max_run = cur_run;
            }
        } else {
            cur_run = 0;
        }
    }
    max_run >= 4 && bullets * 2 >= non_ws
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

/// Find the byte end of the last blank line (`\n[ \t]*\n`) in `s`, if any.
fn last_blank_line_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut last_end: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'\n' {
                last_end = Some(j + 1);
                i = j;
                continue;
            }
        }
        i += 1;
    }
    last_end
}

/// Strip embedded-object chrome (`\u{FFFC}`), zero-width format characters, live inline
/// composition/preview text, and control characters into a clean 64-char preceding-text tail.
/// Returns `(cleaned_text, preedit_stripped)`.
pub fn normalize_before_caret(
    raw: &str,
    live_preedit: &str,
    raw_input: &str,
    cand_preview: &str,
) -> (String, bool) {
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
            stripped = true;
        }
    }

    // Keep only the active paragraph after blank-line breaks (`\n[ \t]*\n`) when non-empty
    if let Some(pos) = last_blank_line_end(&s) {
        let tail = s[pos..].trim_start();
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

    (
        tail_utf8_chars(trimmed, MAX_UIA_CHARS).to_string(),
        stripped,
    )
}

#[cfg(windows)]
mod win_impl {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
    use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
    use std::sync::{Mutex, OnceLock, RwLock};
    use std::thread;

    use windows::Win32::Foundation::{CloseHandle, ERROR_ACCESS_DENIED, HWND};
    use windows::Win32::System::Com::{
        CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx,
        CoUninitialize,
    };
    use windows::Win32::System::Threading::{
        OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
        QueryFullProcessImageNameW,
    };
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, CUIAutomation8, IUIAutomation, IUIAutomation6, IUIAutomationTextPattern,
        IUIAutomationValuePattern, TextPatternRangeEndpoint_End, TextPatternRangeEndpoint_Start,
        TextUnit_Character, UIA_EditControlTypeId, UIA_TextPatternId, UIA_ValuePatternId,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        GA_ROOT, GetAncestor, GetForegroundWindow, GetWindowThreadProcessId,
    };
    use windows::core::{Interface, PWSTR};

    const UIA_TIMEOUT_MS: u32 = 200;

    struct ComGuard;
    impl Drop for ComGuard {
        fn drop(&mut self) {
            // SAFETY: Paired with a successful `CoInitializeEx` on this worker thread; declared
            // before COM interface handles so those release before `CoUninitialize`.
            unsafe {
                CoUninitialize();
            }
        }
    }

    struct WorkerAliveGuard;
    impl Drop for WorkerAliveGuard {
        fn drop(&mut self) {
            WORKER_ALIVE.store(false, Ordering::Release);
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
    // Cache keyed by `(hwnd, pid)` so OS HWND recycling to a denied process cannot hit stale false.
    static LAST_DENIED: Mutex<(isize, u32, bool)> = Mutex::new((0, 0, false));
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static WORKER_ALIVE: AtomicBool = AtomicBool::new(false);
    static POST_COMMIT_PENDING: AtomicBool = AtomicBool::new(false);

    fn is_denied_pid(pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        // SAFETY: `OpenProcess` with `PROCESS_QUERY_LIMITED_INFORMATION` is read-only.
        let handle = match unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) } {
            Ok(h) => h,
            Err(e) => {
                // Fail closed on Protected Processes / hardened credential vaults (`consent.exe`,
                // `lockapp.exe`, PPL security UIs) where even limited query access is denied.
                return e.code() == ERROR_ACCESS_DENIED.to_hresult();
            }
        };
        let mut buf = [0u16; 512];
        let mut len = buf.len() as u32;
        // SAFETY: `handle` is a valid process handle and `buf` holds `len` writable `u16` slots.
        let ok = unsafe {
            QueryFullProcessImageNameW(
                handle,
                PROCESS_NAME_WIN32,
                PWSTR(buf.as_mut_ptr()),
                &mut len,
            )
            .is_ok()
        };
        // SAFETY: `handle` is closed exactly once.
        unsafe {
            let _ = CloseHandle(handle);
        }
        if !ok || len == 0 {
            // Fail closed if process image path could not be queried on an open handle.
            return true;
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
        let mut pid: u32 = 0;
        // SAFETY: `hwnd` is a window handle from `GetForegroundWindow`; `pid` is a valid out-ptr.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
        }
        if let Ok(guard) = LAST_DENIED.lock()
            && guard.0 == raw
            && guard.1 == pid
        {
            return guard.2;
        }
        let denied = is_denied_pid(pid);
        if let Ok(mut guard) = LAST_DENIED.lock() {
            *guard = (raw, pid, denied);
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
        // SAFETY: `GetForegroundWindow` takes no arguments and returns the active window handle.
        let hwnd = unsafe { GetForegroundWindow() };
        let hwnd_raw = hwnd.0 as isize;
        if hwnd_raw == 0 {
            return unreadable_snapshot(0, false, now);
        }

        if is_denied_hwnd_cached(hwnd) {
            return unreadable_snapshot(hwnd_raw, true, now);
        }

        // SAFETY: `automation` is a valid COM `IUIAutomation` pointer initialized on this MTA thread.
        let focused = match unsafe { automation.GetFocusedElement() } {
            Ok(el) => el,
            Err(_) => {
                return unreadable_snapshot(hwnd_raw, false, now);
            }
        };

        // Verify focused element belongs to the sampled foreground root window when a native HWND is reported
        // SAFETY: `focused` is a valid COM `IUIAutomationElement`.
        if let Ok(el_hwnd) = unsafe { focused.CurrentNativeWindowHandle() }
            && !el_hwnd.0.is_null()
        {
            // SAFETY: `el_hwnd` is a non-null HWND returned by UIA.
            let root = unsafe { GetAncestor(el_hwnd, GA_ROOT) };
            if !root.0.is_null() && root.0 as isize != hwnd_raw {
                return unreadable_snapshot(hwnd_raw, false, now);
            }
        }

        // Also check focused element's own PID against the password-manager denylist
        // SAFETY: `focused` is a valid COM `IUIAutomationElement`.
        if let Ok(fpid) = unsafe { focused.CurrentProcessId() }
            && fpid > 0
            && is_denied_pid(fpid as u32)
        {
            return unreadable_snapshot(hwnd_raw, true, now);
        }

        // Hard password check (fail closed on COM error: never extract text when password state is unknown)
        // SAFETY: `focused` is a valid COM `IUIAutomationElement`.
        match unsafe { focused.CurrentIsPassword() } {
            Ok(is_pw) if is_pw.as_bool() => {
                return unreadable_snapshot(hwnd_raw, true, now);
            }
            Ok(_) => {}
            Err(_) => {
                return unreadable_snapshot(hwnd_raw, false, now);
            }
        }

        // Primary path: IUIAutomationTextPattern (Chrome, Edge, VS Code, Electron, Word, Notepad, WPF, WinUI)
        // SAFETY: All UIA COM interface calls operate on validated COM smart pointers on this MTA thread,
        // and `GetForegroundWindow` is thread-safe.
        let text_pat_res: Option<UiaSnapshot> = unsafe {
            if let Ok(text_pat) =
                focused.GetCurrentPatternAs::<IUIAutomationTextPattern>(UIA_TextPatternId)
                && let Ok(selection) = text_pat.GetSelection()
                && let Ok(len) = selection.Length()
                && len >= 1
                && let Ok(caret_range) = selection.GetElement(0)
                && let Ok(before) = caret_range.Clone()
                && before
                    .MoveEndpointByRange(
                        TextPatternRangeEndpoint_End,
                        &caret_range,
                        TextPatternRangeEndpoint_Start,
                    )
                    .is_ok()
                && before
                    .MoveEndpointByUnit(TextPatternRangeEndpoint_Start, TextUnit_Character, -96)
                    .is_ok()
            {
                let mut out_of_element = false;
                // Clamp `before`'s start strictly within `focused` so Chromium/Electron
                // WebArea text providers never walk backwards into surrounding buttons/chrome.
                if let Ok(elem_range) = text_pat.RangeFromChild(&focused) {
                    let starts_before = before.CompareEndpoints(
                        TextPatternRangeEndpoint_Start,
                        &elem_range,
                        TextPatternRangeEndpoint_Start,
                    );
                    let end_inside = before.CompareEndpoints(
                        TextPatternRangeEndpoint_End,
                        &elem_range,
                        TextPatternRangeEndpoint_Start,
                    );
                    if matches!(end_inside, Ok(cmp) if cmp < 0) {
                        out_of_element = true;
                    } else if matches!(starts_before, Ok(cmp) if cmp < 0)
                        && matches!(end_inside, Ok(cmp) if cmp >= 0)
                    {
                        let _ = before.MoveEndpointByRange(
                            TextPatternRangeEndpoint_Start,
                            &elem_range,
                            TextPatternRangeEndpoint_Start,
                        );
                    }
                }
                if !out_of_element && let Ok(bstr) = before.GetText(512) {
                    if GetForegroundWindow().0 as isize != hwnd_raw {
                        Some(unreadable_snapshot(hwnd_raw, false, now))
                    } else {
                        let txt = bstr.to_string();
                        if looks_like_masked_password(&txt) {
                            Some(unreadable_snapshot(hwnd_raw, true, now))
                        } else {
                            Some(UiaSnapshot {
                                hwnd: hwnd_raw,
                                is_password: false,
                                readable: true,
                                before_caret: txt,
                                updated_at: Some(now),
                            })
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };
        if let Some(snap) = text_pat_res {
            return snap;
        }

        // Fallback: IUIAutomationValuePattern ONLY for standard non-readonly Edit controls
        // SAFETY: `focused` and `val_pat` are validated COM pointers on this MTA thread.
        let val_pat_res: Option<UiaSnapshot> = unsafe {
            if let Ok(ctype) = focused.CurrentControlType()
                && ctype == UIA_EditControlTypeId
                && let Ok(val_pat) =
                    focused.GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId)
            {
                let is_ro = val_pat
                    .CurrentIsReadOnly()
                    .map(|b| b.as_bool())
                    .unwrap_or(true);
                if !is_ro && let Ok(bstr) = val_pat.CurrentValue() {
                    if GetForegroundWindow().0 as isize != hwnd_raw {
                        Some(unreadable_snapshot(hwnd_raw, false, now))
                    } else {
                        let full = bstr.to_string();
                        if looks_like_masked_password(&full) {
                            Some(unreadable_snapshot(hwnd_raw, true, now))
                        } else {
                            let tail = tail_utf8_chars(&full, MAX_VALUE_PATTERN_TAIL_CHARS);
                            Some(UiaSnapshot {
                                hwnd: hwnd_raw,
                                is_password: false,
                                readable: true,
                                before_caret: tail.to_string(),
                                updated_at: Some(now),
                            })
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };
        if let Some(snap) = val_pat_res {
            return snap;
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
                    // SAFETY: Initialize COM MTA on this dedicated UIA worker thread.
                    let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
                    if hr.is_err() {
                        return;
                    }
                    // Declared before `automation` so `IUIAutomation` drops before `CoUninitialize`.
                    let _com_guard = ComGuard;

                    // SAFETY: COM is initialized on this thread; instantiates standard Windows UIA.
                    let automation: Result<IUIAutomation, _> = unsafe {
                        CoCreateInstance(&CUIAutomation8, None, CLSCTX_INPROC_SERVER).or_else(
                            |_| CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER),
                        )
                    };
                    let Ok(automation) = automation else {
                        return;
                    };
                    if let Ok(auto6) = automation.cast::<IUIAutomation6>() {
                        // SAFETY: Valid `IUIAutomation6` pointer on initialized COM MTA thread.
                        unsafe {
                            let _ = auto6.SetConnectionTimeout(UIA_TIMEOUT_MS);
                            let _ = auto6.SetTransactionTimeout(UIA_TIMEOUT_MS);
                        }
                    }

                    WORKER_ALIVE.store(true, Ordering::Release);
                    let _alive_guard = WorkerAliveGuard;

                    let mut last_probe: Option<Instant> = None;
                    let mut last_hwnd: isize = 0;
                    let mut deferred_until: Option<Instant> = None;

                    loop {
                        let woke = match deferred_until.take() {
                            Some(deadline) => {
                                let wait = deadline.saturating_duration_since(Instant::now());
                                if wait.is_zero() {
                                    true
                                } else {
                                    match rx.recv_timeout(wait) {
                                        Ok(()) | Err(RecvTimeoutError::Timeout) => true,
                                        Err(RecvTimeoutError::Disconnected) => break,
                                    }
                                }
                            }
                            None => rx.recv().is_ok(),
                        };
                        if !woke {
                            break;
                        }

                        let post_commit = POST_COMMIT_PENDING.swap(false, Ordering::AcqRel);
                        if post_commit {
                            // Wait briefly off the main thread so TSF finishes inserting committed
                            // text into Chromium / Electron / Word before probing UIA.
                            thread::sleep(POST_COMMIT_SETTLE_DELAY);
                            if !POST_COMMIT_PENDING.load(Ordering::Acquire) {
                                while rx.try_recv().is_ok() {}
                            }
                        } else {
                            // SAFETY: `GetForegroundWindow` is safe to invoke on any thread.
                            let cur_hwnd = unsafe { GetForegroundWindow().0 as isize };
                            if cur_hwnd == last_hwnd
                                && let Some(t) = last_probe
                            {
                                let elapsed = t.elapsed();
                                if elapsed < MIN_REFRESH_INTERVAL {
                                    // Schedule a trailing probe instead of dropping the trigger,
                                    // so same-window focus moves (e.g. into a password input)
                                    // are always picked up within <= 35 ms.
                                    deferred_until = Some(t + MIN_REFRESH_INTERVAL);
                                    continue;
                                }
                            }
                        }

                        let snap = probe_once(&automation);
                        last_hwnd = snap.hwnd;
                        last_probe = snap.updated_at;
                        if let Some(rt) = RUNTIME.get()
                            && let Ok(mut guard) = rt.snapshot.write()
                        {
                            *guard = snap;
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

    pub fn is_worker_alive() -> bool {
        WORKER_ALIVE.load(Ordering::Acquire)
    }

    /// Non-blocking hot-path query (< 2 µs). Returns `(context, foreground_hwnd)` sampled
    /// atomically from the same `GetForegroundWindow()` call. Never waits on COM or UIA.
    pub fn query_context(
        live_preedit: &str,
        raw_input: &str,
        cand_preview: &str,
    ) -> (UiaContext, isize) {
        // SAFETY: `GetForegroundWindow` takes no arguments and never blocks.
        let hwnd = unsafe { GetForegroundWindow() };
        let cur_hwnd = hwnd.0 as isize;
        if !is_enabled() {
            return (UiaContext::FallbackSameWindow, cur_hwnd);
        }
        let rt = ensure_started();
        let prev_hwnd = LAST_SEEN_HWND.swap(cur_hwnd, Ordering::AcqRel);

        // Kick background worker asynchronously (never blocks)
        let _ = rt.tx.try_send(());

        if cur_hwnd == 0 {
            return (UiaContext::FallbackSameWindow, 0);
        }

        // Immediate synchronous password-manager / PPL check (cached per (HWND, PID))
        if is_denied_hwnd_cached(hwnd) {
            return (UiaContext::Blocked, cur_hwnd);
        }

        // Read latest cached snapshot (writer only holds lock for a ~100 ns struct move, never I/O)
        if let Ok(snap) = rt.snapshot.read() {
            if snap.hwnd == cur_hwnd {
                if snap.is_password {
                    return (UiaContext::Blocked, cur_hwnd);
                }
                if snap.readable && snap.updated_at.is_some_and(|t| t.elapsed() <= SNAPSHOT_TTL) {
                    let (text, preedit_stripped) = normalize_before_caret(
                        &snap.before_caret,
                        live_preedit,
                        raw_input,
                        cand_preview,
                    );
                    return (
                        UiaContext::Text {
                            text,
                            preedit_stripped,
                        },
                        cur_hwnd,
                    );
                }
                if prev_hwnd != 0 && prev_hwnd != cur_hwnd {
                    return (UiaContext::WindowChanged, cur_hwnd);
                }
                return (UiaContext::FallbackSameWindow, cur_hwnd);
            }
            // Background worker has not yet finished probing `cur_hwnd` after a window switch:
            // keep suppressing stale cross-app `commit_history` until the new window is probed.
            if snap.hwnd != 0 {
                return (UiaContext::WindowChanged, cur_hwnd);
            }
        }

        if prev_hwnd != 0 && prev_hwnd != cur_hwnd {
            (UiaContext::WindowChanged, cur_hwnd)
        } else {
            (UiaContext::FallbackSameWindow, cur_hwnd)
        }
    }

    pub fn foreground_hwnd() -> isize {
        // SAFETY: `GetForegroundWindow` is safe to invoke on any thread.
        unsafe { GetForegroundWindow().0 as isize }
    }

    pub fn kick_refresh() {
        if is_enabled() {
            let rt = ensure_started();
            let _ = rt.tx.try_send(());
        }
    }

    /// Force a post-commit UIA probe on the background MTA thread after a brief settle delay,
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
    pub fn is_worker_alive() -> bool {
        false
    }
    pub fn query_context(
        _live_preedit: &str,
        _raw_input: &str,
        _cand_preview: &str,
    ) -> (UiaContext, isize) {
        (UiaContext::FallbackSameWindow, 0)
    }
    pub fn foreground_hwnd() -> isize {
        0
    }
    pub fn kick_refresh() {}
    pub fn kick_after_commit() {}
}

pub use win_impl::{
    foreground_hwnd, is_enabled, is_worker_alive, kick_after_commit, kick_refresh, query_context,
    set_enabled,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_before_caret_strips_chrome_preedit_and_paragraph() {
        assert_eq!(
            normalize_before_caret("第一段内容\n\n我们今天去公园玩jntm", "jntm", "jntm", ""),
            ("我们今天去公园玩".to_string(), true)
        );
        assert_eq!(
            normalize_before_caret("今天天气真好，wo men ", "wo men", "women", ""),
            ("今天天气真好，".to_string(), true)
        );
        // Whitespace-only blank lines (`\n   \n`) must also break paragraphs
        assert_eq!(
            normalize_before_caret("旧段落\r\n   \r\n新段落内容", "", "", ""),
            ("新段落内容".to_string(), false)
        );
        // Embedded object separator U+FFFC must block Chromium/Electron UI chrome bleed
        assert_eq!(
            normalize_before_caret(
                "No Project\u{FFFC}刚刚买了一个jwtyzu",
                "jw ty zu",
                "jwtyzu",
                ""
            ),
            ("刚刚买了一个".to_string(), true)
        );
        // Must only strip once (longest match first), never double-strip matching suffix
        assert_eq!(
            normalize_before_caret("ababc", "abc", "ab", ""),
            ("ab".to_string(), true)
        );
    }

    #[test]
    fn test_normalize_before_caret_strips_zwsp_replacement_and_inline_preview() {
        // Electron/Chromium editors wrap composition ranges with U+200B/BOM; split surrogates emit U+FFFD
        assert_eq!(
            normalize_before_caret(
                "\u{FEFF}\u{FFFD}请帮我检查\u{200B}wo men\u{200B}",
                "wo men",
                "women",
                ""
            ),
            ("请帮我检查".to_string(), true)
        );
        // Weasel `preedit_type: preview` echoes top candidate inline instead of pinyin preedit
        assert_eq!(
            normalize_before_caret("我们正在讨论策略", "ce lue", "celue", "策略"),
            ("我们正在讨论".to_string(), true)
        );
        // When pinyin preedit matches, candidate preview must not also be stripped
        assert_eq!(
            normalize_before_caret("策略ce lue", "ce lue", "celue", "策略"),
            ("策略".to_string(), true)
        );
    }

    #[test]
    fn test_looks_like_masked_password() {
        assert!(looks_like_masked_password("••••••••"));
        assert!(looks_like_masked_password("********"));
        assert!(!looks_like_masked_password("普通中文句子***"));
        assert!(!looks_like_masked_password(""));
        assert!(!looks_like_masked_password("a*b*c"));
    }

    #[test]
    fn test_normalize_before_caret_truncates_utf8_safely() {
        let long =
            "这是一段很长的中文前文测试内容，用来验证多字节字符截断不会发生越界崩溃。".repeat(3);
        let (out, stripped) = normalize_before_caret(&long, "", "", "");
        assert!(!stripped);
        assert_eq!(out.chars().count(), MAX_UIA_CHARS);
    }
}
