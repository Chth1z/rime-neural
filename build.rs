//! Zero-dependency Windows `VS_VERSION_INFO` (`.res`) resource emitter for `rime_neural.dll`.
//! Embeds `FileVersion`, `ProductVersion`, `ProductName`, `FileDescription`, `OriginalFilename`,
//! and `LegalCopyright` synced directly from `Cargo.toml` with zero external crates or `rc.exe`.

use std::env;
use std::fs;
use std::path::PathBuf;

fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_utf16z(out: &mut Vec<u8>, s: &str) {
    for u in s.encode_utf16() {
        push_u16(out, u);
    }
    push_u16(out, 0);
}

fn pad4(out: &mut Vec<u8>) {
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

fn build_string_entry(key: &str, value: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    let val_utf16_words = value.encode_utf16().count() + 1;
    push_u16(&mut buf, 0); // wLength placeholder
    push_u16(&mut buf, val_utf16_words as u16); // wValueLength (UTF-16 words incl. NUL)
    push_u16(&mut buf, 1); // wType = 1 (text)
    push_utf16z(&mut buf, key);
    pad4(&mut buf);
    push_utf16z(&mut buf, value);
    let len = buf.len() as u16;
    buf[0..2].copy_from_slice(&len.to_le_bytes());
    buf
}

fn build_string_table(lang_cp: &str, entries: &[(&str, &str)]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_u16(&mut buf, 0); // wLength placeholder
    push_u16(&mut buf, 0); // wValueLength = 0
    push_u16(&mut buf, 1); // wType = 1 (text)
    push_utf16z(&mut buf, lang_cp);
    for &(k, v) in entries {
        pad4(&mut buf);
        buf.extend_from_slice(&build_string_entry(k, v));
    }
    let len = buf.len() as u16;
    buf[0..2].copy_from_slice(&len.to_le_bytes());
    buf
}

fn build_string_file_info(entries: &[(&str, &str)]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_u16(&mut buf, 0); // wLength placeholder
    push_u16(&mut buf, 0); // wValueLength = 0
    push_u16(&mut buf, 1); // wType = 1
    push_utf16z(&mut buf, "StringFileInfo");
    pad4(&mut buf);
    buf.extend_from_slice(&build_string_table("040904B0", entries));
    let len = buf.len() as u16;
    buf[0..2].copy_from_slice(&len.to_le_bytes());
    buf
}

fn build_var_file_info() -> Vec<u8> {
    let mut var = Vec::new();
    push_u16(&mut var, 0); // wLength placeholder
    push_u16(&mut var, 4); // wValueLength = 4 bytes (LANGID + codepage)
    push_u16(&mut var, 0); // wType = 0 (binary)
    push_utf16z(&mut var, "Translation");
    pad4(&mut var);
    push_u16(&mut var, 0x0409); // US English
    push_u16(&mut var, 0x04B0); // Unicode (UTF-16LE)
    let var_len = var.len() as u16;
    var[0..2].copy_from_slice(&var_len.to_le_bytes());

    let mut vfi = Vec::new();
    push_u16(&mut vfi, 0);
    push_u16(&mut vfi, 0);
    push_u16(&mut vfi, 1);
    push_utf16z(&mut vfi, "VarFileInfo");
    pad4(&mut vfi);
    vfi.extend_from_slice(&var);
    let vfi_len = vfi.len() as u16;
    vfi[0..2].copy_from_slice(&vfi_len.to_le_bytes());
    vfi
}

fn build_vs_version_info(major: u16, minor: u16, patch: u16, version_str: &str) -> Vec<u8> {
    let file_version_dot = format!("{major}.{minor}.{patch}.0");
    let desc = "In-process ONNX neural sentence reranker for Windows Rime / Weasel";
    let entries = [
        ("CompanyName", "Chth1z"),
        ("FileDescription", desc),
        ("FileVersion", file_version_dot.as_str()),
        ("InternalName", "rime_neural.dll"),
        ("LegalCopyright", "Apache-2.0"),
        ("OriginalFilename", "rime_neural.dll"),
        ("ProductName", "rime-neural"),
        ("ProductVersion", version_str),
    ];

    let mut vi = Vec::new();
    push_u16(&mut vi, 0); // wLength placeholder
    push_u16(&mut vi, 52); // wValueLength = sizeof(VS_FIXEDFILEINFO)
    push_u16(&mut vi, 0); // wType = 0 (binary)
    push_utf16z(&mut vi, "VS_VERSION_INFO");
    pad4(&mut vi);

    // VS_FIXEDFILEINFO (52 bytes)
    let ver_ms = ((major as u32) << 16) | (minor as u32);
    let ver_ls = (patch as u32) << 16;
    push_u32(&mut vi, 0xFEEF_04BD); // dwSignature
    push_u32(&mut vi, 0x0001_0000); // dwStrucVersion
    push_u32(&mut vi, ver_ms); // dwFileVersionMS
    push_u32(&mut vi, ver_ls); // dwFileVersionLS
    push_u32(&mut vi, ver_ms); // dwProductVersionMS
    push_u32(&mut vi, ver_ls); // dwProductVersionLS
    push_u32(&mut vi, 0x0000_003F); // dwFileFlagsMask
    push_u32(&mut vi, 0x0000_0000); // dwFileFlags
    push_u32(&mut vi, 0x0004_0004); // dwFileOS = VOS_NT_WINDOWS32
    push_u32(&mut vi, 0x0000_0002); // dwFileType = VFT_DLL
    push_u32(&mut vi, 0x0000_0000); // dwFileSubtype
    push_u32(&mut vi, 0x0000_0000); // dwFileDateMS
    push_u32(&mut vi, 0x0000_0000); // dwFileDateLS

    pad4(&mut vi);
    vi.extend_from_slice(&build_string_file_info(&entries));
    pad4(&mut vi);
    vi.extend_from_slice(&build_var_file_info());

    let total_len = vi.len() as u16;
    vi[0..2].copy_from_slice(&total_len.to_le_bytes());
    vi
}

fn build_res_file(version_data: &[u8]) -> Vec<u8> {
    let mut res = Vec::new();
    // Standard 32-byte Win32 .res empty header (identifies 32-bit resource format)
    push_u32(&mut res, 0); // DataSize = 0
    push_u32(&mut res, 32); // HeaderSize = 32
    push_u16(&mut res, 0xFFFF);
    push_u16(&mut res, 0x0000); // Type = 0
    push_u16(&mut res, 0xFFFF);
    push_u16(&mut res, 0x0000); // Name = 0
    push_u32(&mut res, 0); // DataVersion
    push_u16(&mut res, 0); // MemoryFlags
    push_u16(&mut res, 0); // LanguageId
    push_u32(&mut res, 0); // Version
    push_u32(&mut res, 0); // Characteristics

    // RT_VERSION (16), ID 1 (VS_VERSION_INFO), Language 0x0409 (en-US)
    push_u32(&mut res, version_data.len() as u32); // DataSize
    push_u32(&mut res, 32); // HeaderSize = 32
    push_u16(&mut res, 0xFFFF);
    push_u16(&mut res, 16); // RT_VERSION
    push_u16(&mut res, 0xFFFF);
    push_u16(&mut res, 1); // VS_VERSION_INFO ID = 1
    push_u32(&mut res, 0); // DataVersion
    push_u16(&mut res, 0x0030); // MemoryFlags = MOVEABLE | PURE
    push_u16(&mut res, 0x0409); // LanguageId = US English
    push_u32(&mut res, 0); // Version
    push_u32(&mut res, 0); // Characteristics
    res.extend_from_slice(version_data);
    pad4(&mut res);
    res
}

fn main() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed=CARGO_PKG_VERSION");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os != "windows" || target_env != "msvc" {
        return;
    }

    let major: u16 = env::var("CARGO_PKG_VERSION_MAJOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let minor: u16 = env::var("CARGO_PKG_VERSION_MINOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let patch: u16 = env::var("CARGO_PKG_VERSION_PATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let ver_str =
        env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| format!("{major}.{minor}.{patch}"));

    let vi = build_vs_version_info(major, minor, patch, &ver_str);
    let res = build_res_file(&vi);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let res_path = out_dir.join("rime_neural_version.res");
    fs::write(&res_path, res).expect("failed to write version .res");
    println!("cargo:rustc-link-arg-cdylib={}", res_path.display());
}
