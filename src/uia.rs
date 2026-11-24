//! Non-blocking, privacy-hardened Windows UI Automation (UIA) preceding-text reader.
//!
//! Architecture guarantees:
//! 1. **Zero main-thread blocking**: All COM/UIA calls run on a single dedicated background
//!    MTA thread (`COINIT_MULTITHREADED`, 200 ms `IUIAutomation6` connection/transaction
//!    timeouts). The keystroke hot path only samples the foreground HWND/PID, calls `try_send()`,
//!    and a brief `RwLock::read()` (writer never holds the lock during COM I/O), so sluggish apps
//!    never hang `WeaselServer.exe`.
//! 2. **Zero cross-app contamination**: Tracks the active foreground `HWND` synchronously on
//!    every query, verifies focused-element root ancestry and post-read foreground `HWND`, and
//!    clamps `TextPattern` ranges to `RangeFromChild(&focused)` (falling back to `DocumentRange`) + `\u{FFFC}` boundaries so
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
const SNAPSHOT_TTL: Duration = Duration::from_secs(4);
const MIN_REFRESH_INTERVAL: Duration = Duration::from_millis(35);
const POST_COMMIT_SETTLE_DELAY: Duration = Duration::from_millis(25);

const DENIED_PROCESSES: &[&str] = &[
    // Password managers & 2FA vaults
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
    // Windows credential / elevation / lock surfaces
    "credentialuibroker.exe",
    "consent.exe",
    "lockapp.exe",
    // Remote desktop & SSH / terminal clients
    "mstsc.exe",
    "msrdc.exe",
    "putty.exe",
    "xshell.exe",
    "mobaxterm.exe",
    "finalshell.exe",
    "tabby.exe",
    "termius.exe",
    "securecrt.exe",
    "windterm.exe",
    "anydesk.exe",
    "teamviewer.exe",
    "todesk.exe",
    "sunloginclient.exe",
    "rustdesk.exe",
    // Local shell / terminal emulators (CLI secrets & unmasked sudo/ssh prompts)
    "windowsterminal.exe",
    "conhost.exe",
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "mintty.exe",
];

const DENIED_PROCESS_PREFIXES: &[&str] = &[
    "mobaxterm_",
    "mobaxterm-",
    "keepass-",
    "keepass_",
    "keepassxc-",
    "keepassxc_",
    "1password-",
];

fn is_denied_exe_name(exe_lower: &str) -> bool {
    DENIED_PROCESSES.contains(&exe_lower)
        || (exe_lower.ends_with(".exe")
            && DENIED_PROCESS_PREFIXES
                .iter()
                .any(|&prefix| exe_lower.starts_with(prefix)))
}

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
    matches!(c, '•' | '●' | '◦' | '▪' | '∗' | '*' | '∙' | '⬤' | '⦁')
}

/// Heuristic detector for custom `ValuePattern` / `TextPattern` password fields that omit
/// `IsPassword = true` and expose masked bullets (`••••` or `****`).
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
    (non_ws >= 2 && bullets == non_ws) || (max_run >= 4 && bullets * 2 >= non_ws)
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

/// Strip a stale partial inline-preedit prefix (from an earlier keystroke captured asynchronously
/// before the current keystroke arrived) when `s` ends with a non-trivial ASCII prefix of `lp` or
/// `ri` separated from preceding context by start-of-string, whitespace, or a non-ASCII boundary.
fn strip_stale_preedit_prefix(s: &mut String, lp: &str, ri: &str) -> bool {
    let mut best_cut: Option<usize> = None;
    for target in [lp, ri] {
        if target.len() < 3 || !target.is_ascii() {
            continue;
        }
        let max_k = target.len().saturating_sub(1).min(s.len());
        for k in (2..=max_k).rev() {
            if !s.is_char_boundary(s.len() - k) {
                continue;
            }
            let prefix = &target[..k];
            if s.ends_with(prefix) {
                let cut = s.len() - k;
                let boundary_ok = cut == 0
                    || s[..cut]
                        .chars()
                        .next_back()
                        .is_some_and(|c| !c.is_ascii_alphanumeric());
                if boundary_ok && best_cut.is_none_or(|prev| cut < prev) {
                    best_cut = Some(cut);
                }
                break;
            }
        }
    }
    if let Some(cut) = best_cut {
        s.truncate(cut);
        true
    } else {
        false
    }
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
    // Trim ASCII spaces, tabs, and U+00A0 NBSP emitted by contenteditable editors before matching.
    let trimmed_tail = s.trim_end_matches([' ', '\t', '\u{00A0}']);
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
    if !stripped && strip_stale_preedit_prefix(&mut s, lp, ri) {
        stripped = true;
    }
    if !stripped {
        let cp = cand_preview.trim();
        if cp.chars().count() >= 2 && s.ends_with(cp) {
            let new_len = s.len().saturating_sub(cp.len());
            s.truncate(new_len);
            stripped = true;
        }
    }

    // Keep strictly the active paragraph after blank-line breaks (`\n[ \t]*\n`): if the user
    // started a fresh paragraph after a blank line, return empty instead of bleeding prior text.
    if let Some(pos) = last_blank_line_end(&s) {
        s = s[pos..].trim_start().to_string();
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
    use std::sync::{OnceLock, RwLock};
    use std::thread;

    use windows::Win32::Foundation::{CloseHandle, ERROR_INSUFFICIENT_BUFFER};
    use windows::Win32::System::Com::{
        CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx,
        CoUninitialize,
    };
    use windows::Win32::System::Threading::{
        OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
        QueryFullProcessImageNameW,
    };
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, CUIAutomation8, IUIAutomation, IUIAutomation6, IUIAutomationElement,
        IUIAutomationFocusChangedEventHandler, IUIAutomationFocusChangedEventHandler_Impl,
        IUIAutomationTextPattern, IUIAutomationValuePattern, TextPatternRangeEndpoint_End,
        TextPatternRangeEndpoint_Start, TextUnit_Character, UIA_EditControlTypeId,
        UIA_TextPatternId, UIA_ValuePatternId,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        GA_ROOT, GetAncestor, GetForegroundWindow, GetWindowThreadProcessId,
    };
    use windows::core::{Interface, PWSTR, implement};

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
            if let Some(rt) = RUNTIME.get() {
                rt.invalidate((0, 0), false);
            }
        }
    }

    struct FocusHandlerGuard<'a> {
        automation: &'a IUIAutomation,
        handler: IUIAutomationFocusChangedEventHandler,
    }

    impl Drop for FocusHandlerGuard<'_> {
        fn drop(&mut self) {
            // SAFETY: Unregisters the focus handler registered on this MTA `automation` before
            // `automation` and COM shut down.
            unsafe {
                let _ = self
                    .automation
                    .RemoveFocusChangedEventHandler(&self.handler);
            }
        }
    }

    #[implement(IUIAutomationFocusChangedEventHandler)]
    struct FocusChangedHandler;

    impl IUIAutomationFocusChangedEventHandler_Impl for FocusChangedHandler_Impl {
        fn HandleFocusChangedEvent(
            &self,
            _sender: Option<&IUIAutomationElement>,
        ) -> windows::core::Result<()> {
            if !is_enabled() {
                return Ok(());
            }
            if let Some(rt) = RUNTIME.get() {
                // Suppress context and feedback until the worker checks the newly focused control.
                rt.invalidate(foreground_window(), true);
                let _ = rt.tx.try_send(());
            }
            Ok(())
        }
    }

    struct UiaSnapshot {
        window: (isize, u32),
        blocked: bool,
        before_caret: Option<String>,
        updated_at: Instant,
    }

    struct UiaRuntime {
        tx: SyncSender<()>,
        // Publish the focus epoch and its snapshot together.
        snapshot: RwLock<(u64, UiaSnapshot)>,
    }

    impl UiaRuntime {
        fn invalidate(&self, window: (isize, u32), blocked: bool) {
            let (epoch, snap) = &mut *self.snapshot.write().unwrap_or_else(|e| e.into_inner());
            *epoch = epoch.wrapping_add(1);
            *snap = unreadable_snapshot(window, blocked);
        }

        fn publish(&self, epoch: u64, result: UiaSnapshot) {
            let (current, snap) = &mut *self.snapshot.write().unwrap_or_else(|e| e.into_inner());
            if *current == epoch {
                *snap = result;
            }
        }
    }

    static RUNTIME: OnceLock<UiaRuntime> = OnceLock::new();
    static LAST_SEEN_HWND: AtomicIsize = AtomicIsize::new(0);
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static WORKER_ALIVE: AtomicBool = AtomicBool::new(false);
    static POST_COMMIT_PENDING: AtomicBool = AtomicBool::new(false);

    fn is_denied_pid(pid: u32) -> bool {
        if pid == 0 {
            return true;
        }
        // SAFETY: `OpenProcess` with `PROCESS_QUERY_LIMITED_INFORMATION` is read-only.
        let handle = match unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) } {
            Ok(h) => h,
            Err(_) => return true, // Unable to classify the process.
        };
        let mut cap = 512usize;
        let mut path_opt: Option<String> = None;
        while cap <= 32_768 {
            let mut buf = vec![0u16; cap];
            let mut len = buf.len() as u32;
            // SAFETY: `handle` is a valid process handle and `buf` holds `len` writable `u16` slots.
            match unsafe {
                QueryFullProcessImageNameW(
                    handle,
                    PROCESS_NAME_WIN32,
                    PWSTR(buf.as_mut_ptr()),
                    &mut len,
                )
            } {
                Ok(()) if len > 0 => {
                    path_opt = Some(String::from_utf16_lossy(&buf[..len as usize]));
                    break;
                }
                Err(e) if e.code() == ERROR_INSUFFICIENT_BUFFER.to_hresult() => {
                    cap *= 4;
                }
                _ => break,
            }
        }
        // SAFETY: `handle` is closed exactly once.
        unsafe {
            let _ = CloseHandle(handle);
        }
        let Some(path) = path_opt else {
            // Fail closed if process image path could not be queried on an open handle.
            return true;
        };
        let exe = path
            .rsplit(['\\', '/'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        is_denied_exe_name(&exe)
    }

    fn foreground_window() -> (isize, u32) {
        // SAFETY: The HWND comes from Windows and `pid` is writable for the call.
        unsafe {
            let hwnd = GetForegroundWindow();
            let mut pid = 0;
            if !hwnd.0.is_null() {
                GetWindowThreadProcessId(hwnd, Some(&mut pid));
            }
            (hwnd.0 as isize, pid)
        }
    }

    fn unreadable_snapshot(window: (isize, u32), blocked: bool) -> UiaSnapshot {
        UiaSnapshot {
            window,
            blocked,
            before_caret: None,
            updated_at: Instant::now(),
        }
    }

    fn probe_once(automation: &IUIAutomation) -> UiaSnapshot {
        let window = foreground_window();
        let (hwnd_raw, foreground_pid) = window;
        if hwnd_raw == 0 || is_denied_pid(foreground_pid) {
            return unreadable_snapshot(window, true);
        }

        // SAFETY: `automation` is a valid COM `IUIAutomation` pointer initialized on this MTA thread.
        let focused = match unsafe { automation.GetFocusedElement() } {
            Ok(el) => el,
            // Foreground PID passed the denylist above; apps without UIA exposure are unreadable,
            // not password controls.
            Err(_) => return unreadable_snapshot(window, false),
        };

        // Windowless controls legitimately report HWND 0. When an HWND is present its
        // root must match; a failed ownership or password query cannot classify the control.
        // SAFETY: `focused` is a valid UIA element; GetAncestor takes its native HWND.
        unsafe {
            let Ok(el_hwnd) = focused.CurrentNativeWindowHandle() else {
                return unreadable_snapshot(window, true);
            };
            let Ok(pid) = focused.CurrentProcessId() else {
                return unreadable_snapshot(window, true);
            };
            if pid <= 0 || (pid as u32 != foreground_pid && is_denied_pid(pid as u32)) {
                return unreadable_snapshot(window, true);
            }
            if el_hwnd.0.is_null() {
                if foreground_pid != pid as u32 {
                    return unreadable_snapshot(window, true);
                }
            } else if GetAncestor(el_hwnd, GA_ROOT).0 as isize != hwnd_raw {
                return unreadable_snapshot(window, true);
            }
            if !matches!(focused.CurrentIsPassword(), Ok(pw) if !pw.as_bool()) {
                return unreadable_snapshot(window, true);
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
                && len == 1
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
                // Prefer `RangeFromChild(&focused)` so Chromium/Electron `WebArea` providers stay
                // bounded to the focused box; fall back to `DocumentRange()` when `focused` owns
                // the TextPattern directly.
                let Ok(bounds) = text_pat
                    .RangeFromChild(&focused)
                    .or_else(|_| text_pat.DocumentRange())
                else {
                    return unreadable_snapshot(window, false);
                };
                let start = before.CompareEndpoints(
                    TextPatternRangeEndpoint_Start,
                    &bounds,
                    TextPatternRangeEndpoint_Start,
                );
                let end_vs_start = before.CompareEndpoints(
                    TextPatternRangeEndpoint_End,
                    &bounds,
                    TextPatternRangeEndpoint_Start,
                );
                let end_vs_end = before.CompareEndpoints(
                    TextPatternRangeEndpoint_End,
                    &bounds,
                    TextPatternRangeEndpoint_End,
                );
                if !matches!(end_vs_start, Ok(n) if n >= 0)
                    || !matches!(end_vs_end, Ok(n) if n <= 0)
                    || !matches!(start, Ok(n) if n >= 0 || before.MoveEndpointByRange(
                        TextPatternRangeEndpoint_Start, &bounds, TextPatternRangeEndpoint_Start
                    ).is_ok())
                {
                    Some(unreadable_snapshot(window, false))
                } else if let Ok(bstr) = before.GetText(512) {
                    if GetForegroundWindow().0 as isize != hwnd_raw {
                        Some(unreadable_snapshot(window, true))
                    } else {
                        let txt = bstr.to_string();
                        if looks_like_masked_password(&txt) {
                            Some(unreadable_snapshot(window, true))
                        } else {
                            Some(UiaSnapshot {
                                window,
                                blocked: false,
                                before_caret: Some(txt),
                                updated_at: Instant::now(),
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

        // ValuePattern exposes the entire field, without a selection or caret. It can establish
        // a masked or empty field, but a nonempty value cannot supply preceding text.
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
                        Some(unreadable_snapshot(window, true))
                    } else {
                        if bstr.is_empty() {
                            Some(UiaSnapshot {
                                window,
                                blocked: false,
                                before_caret: Some(String::new()),
                                updated_at: Instant::now(),
                            })
                        } else {
                            Some(unreadable_snapshot(
                                window,
                                looks_like_masked_password(&bstr.to_string()),
                            ))
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

        unreadable_snapshot(window, false)
    }

    fn ensure_started() -> &'static UiaRuntime {
        RUNTIME.get_or_init(|| {
            let (tx, rx) = mpsc::sync_channel::<()>(1);
            let runtime = UiaRuntime {
                tx,
                snapshot: RwLock::new((0, unreadable_snapshot((0, 0), false))),
            };

            let _ = thread::Builder::new()
                .name("rime_neural_uia".to_string())
                .spawn(move || {
                    let rt = RUNTIME.wait();
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

                    // Register focus-change listener so same-window focus moves (e.g. Tab into
                    // `<input type="password">`) invalidate stale snapshots immediately.
                    let focus_handler: IUIAutomationFocusChangedEventHandler =
                        FocusChangedHandler.into();
                    // SAFETY: Valid MTA `automation` and COM handler implementation.
                    let _focus_guard = unsafe {
                        automation
                            .AddFocusChangedEventHandler(None, &focus_handler)
                            .ok()
                            .map(|()| FocusHandlerGuard {
                                automation: &automation,
                                handler: focus_handler,
                            })
                    };

                    WORKER_ALIVE.store(true, Ordering::Release);
                    let _alive_guard = WorkerAliveGuard;

                    let mut last_probe: Option<Instant> = None;
                    let mut last_hwnd: isize = 0;
                    let mut last_epoch = 0;
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
                        if !is_enabled() {
                            continue;
                        }

                        let post_commit = POST_COMMIT_PENDING.swap(false, Ordering::AcqRel);
                        let epoch = if post_commit {
                            // Wait briefly off the main thread so TSF finishes inserting committed
                            // text into Chromium / Electron / Word before probing UIA.
                            thread::sleep(POST_COMMIT_SETTLE_DELAY);
                            rt.snapshot.read().unwrap_or_else(|e| e.into_inner()).0
                        } else {
                            let ep = rt.snapshot.read().unwrap_or_else(|e| e.into_inner()).0;
                            if ep == last_epoch {
                                // SAFETY: `GetForegroundWindow` is safe to invoke on any thread.
                                let cur_hwnd = unsafe { GetForegroundWindow().0 as isize };
                                if cur_hwnd == last_hwnd
                                    && let Some(t) = last_probe
                                {
                                    let elapsed = t.elapsed();
                                    if elapsed < MIN_REFRESH_INTERVAL {
                                        deferred_until = Some(t + MIN_REFRESH_INTERVAL);
                                        continue;
                                    }
                                }
                            }
                            ep
                        };

                        let snap = probe_once(&automation);
                        last_hwnd = snap.window.0;
                        last_probe = Some(snap.updated_at);
                        last_epoch = epoch;
                        // Avoid overwriting a fresh post-commit refresh with an older in-flight probe.
                        if !post_commit && POST_COMMIT_PENDING.load(Ordering::Acquire) {
                            continue;
                        }
                        rt.publish(epoch, snap);
                    }
                });

            runtime
        })
    }

    /// Enable or disable the background UIA context worker. Disabling clears cached snapshots
    /// immediately so no stale screen text remains in memory.
    pub fn set_enabled(enable: bool) {
        if enable {
            ENABLED.store(true, Ordering::Release);
            let rt = ensure_started();
            let _ = rt.tx.try_send(());
        } else {
            ENABLED.store(false, Ordering::Release);
            if let Some(rt) = RUNTIME.get() {
                rt.invalidate((0, 0), false);
            }
        }
    }

    pub fn is_enabled() -> bool {
        ENABLED.load(Ordering::Acquire)
    }

    pub fn is_worker_alive() -> bool {
        WORKER_ALIVE.load(Ordering::Acquire)
    }

    /// Return the sampled foreground HWND and the focus epoch of the snapshot used for context.
    pub fn query_context(
        live_preedit: &str,
        raw_input: &str,
        cand_preview: &str,
    ) -> (UiaContext, isize, u64) {
        let window = foreground_window();
        let (cur_hwnd, pid) = window;
        if !is_enabled() {
            return (UiaContext::FallbackSameWindow, cur_hwnd, 0);
        }
        let rt = ensure_started();
        let prev_hwnd = LAST_SEEN_HWND.swap(cur_hwnd, Ordering::AcqRel);

        // Kick background worker asynchronously (never blocks)
        let _ = rt.tx.try_send(());

        if cur_hwnd == 0 {
            return (UiaContext::WindowChanged, 0, 0);
        }

        if pid == 0 {
            return (UiaContext::Blocked, cur_hwnd, 0);
        }
        if !is_worker_alive() {
            return (UiaContext::WindowChanged, cur_hwnd, 0);
        }

        // Read latest cached snapshot (writer only holds lock for a ~100 ns struct move, never I/O)
        let cache = rt.snapshot.read().unwrap_or_else(|e| e.into_inner());
        let (epoch, snap) = &*cache;
        let context = if snap.window == window {
            if snap.blocked {
                UiaContext::Blocked
            } else if snap.updated_at.elapsed() > SNAPSHOT_TTL {
                UiaContext::WindowChanged
            } else if let Some(raw) = &snap.before_caret {
                let (text, preedit_stripped) =
                    normalize_before_caret(raw, live_preedit, raw_input, cand_preview);
                UiaContext::Text {
                    text,
                    preedit_stripped,
                }
            } else if prev_hwnd != 0 && prev_hwnd != cur_hwnd {
                UiaContext::WindowChanged
            } else {
                UiaContext::FallbackSameWindow
            }
        } else {
            UiaContext::WindowChanged
        };
        (context, cur_hwnd, *epoch)
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn in_flight_probe_cannot_replace_new_focus_at_same_hwnd() {
            let window = (123, 456);
            let (tx, _rx) = mpsc::sync_channel(1);
            let rt = UiaRuntime {
                tx,
                snapshot: RwLock::new((0, unreadable_snapshot(window, false))),
            };
            let old_probe = unreadable_snapshot(window, false);
            rt.invalidate(window, true);
            rt.publish(0, old_probe);
            assert!(rt.snapshot.read().unwrap().1.blocked);
            rt.publish(1, unreadable_snapshot(window, false));
            assert!(!rt.snapshot.read().unwrap().1.blocked);
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
    ) -> (UiaContext, isize, u64) {
        (UiaContext::FallbackSameWindow, 0, 0)
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
        assert!(looks_like_masked_password("••"));
        assert!(looks_like_masked_password("∙∙∙"));
        assert!(!looks_like_masked_password("普通中文句子***"));
        assert!(!looks_like_masked_password(""));
        assert!(!looks_like_masked_password("a*b*c"));
    }

    #[test]
    fn test_normalize_new_paragraph_nbsp_and_stale_preedit_prefix() {
        // Fresh paragraph after blank line must return empty instead of bleeding prior paragraph
        assert_eq!(
            normalize_before_caret("旧段落内容\n\njntm", "jntm", "jntm", ""),
            ("".to_string(), true)
        );
        // Contenteditable editors emit trailing NBSP (U+00A0) after inline composition
        assert_eq!(
            normalize_before_caret("今天wo men\u{00A0}", "wo men", "women", ""),
            ("今天".to_string(), true)
        );
        // 1-keystroke lagged UIA snapshot ending with strict prefix of current preedit
        assert_eq!(
            normalize_before_caret("你好wo mn", "wo mn j", "womnj", ""),
            ("你好".to_string(), true)
        );
        // Must not strip inside an existing English word when boundary is alphanumeric
        assert_eq!(
            normalize_before_caret("hello", "lo world", "loworld", ""),
            ("hello".to_string(), false)
        );
    }

    #[test]
    fn test_denied_exe_name_matches_terminals_and_portable_versions() {
        assert!(is_denied_exe_name("windowsterminal.exe"));
        assert!(is_denied_exe_name("pwsh.exe"));
        assert!(is_denied_exe_name("mobaxterm_personal_24.2.exe"));
        assert!(is_denied_exe_name("keepass-2.57.exe"));
        assert!(!is_denied_exe_name("notepad.exe"));
        assert!(!is_denied_exe_name("code.exe"));
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
