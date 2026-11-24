# Security Policy

## Supported Versions

Security fixes are applied to the latest minor release on `main`:

| Version | Supported |
| :--- | :---: |
| `1.2.x` | Yes |
| `< 1.2` | No |

## Reporting a Vulnerability

If you discover a security or privacy vulnerability (such as UIA context leakage across sensitive controls/windows, plaintext persistence in feedback logs, unsafe FFI memory access, or installer integrity/supply-chain issues), please report it privately rather than opening a public issue:

- Use [GitHub Private Vulnerability Reporting](https://github.com/Chth1z/rime-neural/security/advisories/new) on `Chth1z/rime-neural`.
- Include reproduction steps, affected Windows / Weasel / `rime-neural` versions, and any relevant redacted logs.

We aim to acknowledge reports within 7 days and coordinate a fix before public disclosure.

## Threat Model & Security Boundaries

### 1. In-Process IME Execution
- `rime_neural.dll` runs inside `WeaselServer.exe` as a Lua 5.4 C module loaded via `package.loadlib` and pinned with `GetModuleHandleExA(PIN)`.
- The module entry point verifies Lua 5.4 ABI compatibility (`lua_newuserdatauv` + `504.0 <= lua_version < 505.0`) before publishing its functions. The native bindings use `extern "C-unwind"`, and inference runs offline.

### 2. UIA Context Privacy (Fail-Closed)
- UIA reading runs on a dedicated background MTA thread with 200 ms connection/transaction timeouts and an `IUIAutomationFocusChangedEventHandler` for immediate snapshot invalidation on focus changes.
- Text reads are clamped to `RangeFromChild(&focused)` (falling back to `DocumentRange`), capped at 64 characters before the caret, and stripped of control/special model tokens before tokenization.
- The reader fails closed (returns empty context and suspends feedback logging) on:
  - Controls reporting `IsPassword == true`, access-denied `IsPassword` queries, or masked bullet/asterisk text;
  - Protected Processes (`OpenProcess` returning `ERROR_ACCESS_DENIED`) or unresolvable PIDs;
  - Processes matching the built-in denylist (password managers/authenticators, credential/UAC dialogs, remote desktop/SSH clients, and local terminals).

### 3. Local Feedback Log Integrity
- `feedback.jsonl` writes only numeric model features and candidate indices (`v: 2`); user text, keystrokes, candidate strings, and screen context are not persisted.
- The logger checks the immediate parent directory and `.1` rotation destination by path. Before appending, it opens the log with `FILE_FLAG_OPEN_REPARSE_POINT` and checks the file type and reparse attribute on that handle.
- The logger rotates at 5 MiB by atomic replacement and retains one `.1` backup. During a transient sharing violation it permits an append only while the active file is below 10 MiB.

### 4. Installer Supply-Chain Verification
- `scripts/install.ps1` enforces HTTPS (`--proto =https --proto-redir =https`), verifies SHA-256 checksums against external `SHA256SUMS.txt` for release bundles and pinned SHA-256 digests for ONNX models/tokenizers/vocabularies, verifies pinned SHA-512/SHA-256 digests and PE32+ x64 headers for `onnxruntime.dll`, and extracts archives exclusively through an allowlisted ZipSlip-checked extractor.
