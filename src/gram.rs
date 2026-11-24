use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;

const GENERIC_READ: u32 = 0x8000_0000;
const FILE_SHARE_READ: u32 = 0x0000_0001;
const FILE_SHARE_DELETE: u32 = 0x0000_0004;
const OPEN_EXISTING: u32 = 3;
const FILE_FLAG_RANDOM_ACCESS: u32 = 0x1000_0000;
const PAGE_READONLY: u32 = 0x02;
const FILE_MAP_READ: u32 = 0x0004;
const INVALID_HANDLE_VALUE: *mut c_void = -1isize as *mut c_void;
const GRAM_MAGIC_V1: &[u8] = b"Rime::Grammar/1.0\0";

unsafe extern "system" {
    fn CreateFileW(
        lpFileName: *const u16,
        dwDesiredAccess: u32,
        dwShareMode: u32,
        lpSecurityAttributes: *const c_void,
        dwCreationDisposition: u32,
        dwFlagsAndAttributes: u32,
        hTemplateFile: *mut c_void,
    ) -> *mut c_void;
    fn GetFileSizeEx(hFile: *mut c_void, lpFileSize: *mut i64) -> i32;
    fn CreateFileMappingW(
        hFile: *mut c_void,
        lpFileMappingAttributes: *const c_void,
        flProtect: u32,
        dwMaximumSizeHigh: u32,
        dwMaximumSizeLow: u32,
        lpName: *const u16,
    ) -> *mut c_void;
    fn MapViewOfFile(
        hFileMappingObject: *mut c_void,
        dwDesiredAccess: u32,
        dwFileOffsetHigh: u32,
        dwFileOffsetLow: u32,
        dwNumberOfBytesToMap: usize,
    ) -> *mut c_void;
    fn UnmapViewOfFile(lpBaseAddress: *const c_void) -> i32;
    fn CloseHandle(hObject: *mut c_void) -> i32;
}

/// Zero-dependency read-only memory-mapped reader for `Rime::Grammar/1.0` (`.gram`) files
/// (such as `wanxiang-lts-zh-hans.gram`). Shares the OS page cache with `rime.dll`
/// (0 MB extra physical RAM) and scores a candidate sentence in ~0.004 ms via a single-pass
/// forward trie walk with zero per-char allocations.
pub struct WanxiangGram {
    _file: *mut c_void,
    _mapping: *mut c_void,
    view: *const u8,
    da_ptr: *const u32,
    da_len: usize,
    collocation_penalty: f64,
    non_collocation_penalty: f64,
}

// SAFETY: `WanxiangGram` holds read-only Win32 file/mapping handles and an immutable `PAGE_READONLY`
// memory view that is never mutated through this struct; handles are closed once in `Drop`.
unsafe impl Send for WanxiangGram {}
// SAFETY: Concurrent shared reads of the immutable `PAGE_READONLY` mapping are thread-safe.
unsafe impl Sync for WanxiangGram {}

impl Drop for WanxiangGram {
    fn drop(&mut self) {
        // SAFETY: Each handle/view was obtained from Win32 and is unmapped/closed in reverse order
        // exactly once when `WanxiangGram` is dropped.
        unsafe {
            if !self.view.is_null() {
                UnmapViewOfFile(self.view as *const c_void);
            }
            if !self._mapping.is_null() {
                CloseHandle(self._mapping);
            }
            if !self._file.is_null() && self._file != INVALID_HANDLE_VALUE {
                CloseHandle(self._file);
            }
        }
    }
}

/// Exact encoding used by `lotem/librime-octagram/src/gram_encoding.cc`.
/// Produces at most 6 bytes per Unicode scalar value (no heap allocation).
#[inline]
fn encode_char_stack(ch: char) -> ([u8; 6], u8) {
    let mut buf = [0u8; 6];
    let mut len = 0usize;
    let u = ch as u32;
    if u < 0x80 {
        buf[0] = if u == 0 { 0xE0 } else { u as u8 };
        len = 1;
    } else if (0x4000..0xA000).contains(&u) {
        if (u & 0xFF) == 0 {
            buf[0] = 0xE1;
            buf[1] = ((u >> 8) + 0x40) as u8;
        } else {
            buf[0] = ((u >> 8) + 0x40) as u8;
            buf[1] = (u & 0xFF) as u8;
        }
        len = 2;
    } else {
        let mut bits = 32i32;
        let mut tmp = u;
        while bits > 0 && (tmp & 0xFE00_0000) == 0 {
            bits -= 7;
            tmp <<= 7;
        }
        let mut nb = (bits + 6) / 7;
        buf[len] = (0xE0 | nb) as u8;
        len += 1;
        while nb > 0 {
            nb -= 1;
            buf[len] = (((tmp >> 25) & 0x7F) | 0x80) as u8;
            len += 1;
            tmp <<= 7;
        }
    }
    (buf, len as u8)
}

/// Parse and validate `Rime::Grammar/1.0` header fields, returning `(da_off, da_size)`.
fn validate_gram_header(bytes: &[u8]) -> Result<(usize, usize), String> {
    if bytes.len() < 64 {
        return Err(format!("invalid .gram size: {}", bytes.len()));
    }
    if !bytes.starts_with(GRAM_MAGIC_V1) {
        return Err("not a Rime::Grammar/1.0 file".to_string());
    }
    let da_size = u32::from_le_bytes(bytes[36..40].try_into().unwrap()) as usize;
    if da_size == 0 {
        return Err("corrupt double-array size (0)".to_string());
    }
    let rel_off = i32::from_le_bytes(bytes[40..44].try_into().unwrap());
    if rel_off < 4 {
        return Err("corrupt double-array offset".to_string());
    }
    let rel_off_usize = usize::try_from(rel_off).map_err(|_| "corrupt double-array offset")?;
    let da_off = 40usize
        .checked_add(rel_off_usize)
        .ok_or_else(|| "corrupt double-array offset overflow".to_string())?;
    let da_bytes = da_size
        .checked_mul(4)
        .ok_or_else(|| "corrupt double-array size overflow".to_string())?;
    let da_end = da_off
        .checked_add(da_bytes)
        .ok_or_else(|| "corrupt double-array end overflow".to_string())?;
    if da_off < 44 || !da_off.is_multiple_of(4) || da_end > bytes.len() {
        return Err("corrupt double-array bounds".to_string());
    }
    Ok((da_off, da_size))
}

#[inline(always)]
fn da_has_leaf(u: u32) -> bool {
    (u & (1 << 8)) != 0
}

#[inline(always)]
fn da_value(u: u32) -> u32 {
    u & 0x7FFF_FFFF
}

#[inline(always)]
fn da_label(u: u32) -> u32 {
    u & 0x8000_00FF
}

#[inline(always)]
fn da_offset(u: u32) -> usize {
    ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize
}

/// Score a sentence over a borrowed Darts-Clone double-array slice using a single-pass forward
/// walk (bit-identical to `octagram.cc`'s per-position `(cl, wl)` scan, with 3–4× fewer root walks).
fn score_slice_per_char(
    da: &[u32],
    text: &str,
    collocation_penalty: f64,
    non_collocation_penalty: f64,
) -> f64 {
    let mut encoded: [([u8; 6], u8); 48] = [([0; 6], 0); 48];
    let mut n = 0usize;
    let mut overflow: Vec<([u8; 6], u8)> = Vec::new();
    for ch in text.chars() {
        let e = encode_char_stack(ch);
        if n < encoded.len() {
            encoded[n] = e;
        } else {
            if overflow.is_empty() {
                overflow.extend_from_slice(&encoded);
            }
            overflow.push(e);
        }
        n += 1;
    }
    if n <= 1 || da.is_empty() {
        return 0.0;
    }
    let slice: &[([u8; 6], u8)] = if overflow.is_empty() {
        &encoded[..n]
    } else {
        &overflow
    };

    let mut best_stack = [non_collocation_penalty; 48];
    let mut best_heap: Vec<f64> = Vec::new();
    let best: &mut [f64] = if n <= best_stack.len() {
        &mut best_stack[..n]
    } else {
        best_heap.resize(n, non_collocation_penalty);
        &mut best_heap
    };

    let root_u = da[0];
    for s in 0..(n - 1) {
        let mut node_pos: usize = 0;
        let mut u = root_u;
        let mut ok = true;
        // Walk first character `slice[s]` (length-1 prefix; no leaf scored for single char).
        let (ref buf0, blen0) = slice[s];
        for &b in &buf0[..blen0 as usize] {
            node_pos ^= da_offset(u) ^ (b as usize);
            let Some(&next_u) = da.get(node_pos) else {
                ok = false;
                break;
            };
            u = next_u;
            if da_label(u) != (b as u32) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        // Walk characters `s + 1 .. min(s + 5, n)` once; each matched n-gram `[s..e)` (2..=5)
        // updates `best[i]` for every split point `i in (s + 1)..e`.
        let max_e = (s + 5).min(n);
        for e in (s + 2)..=max_e {
            let (ref buf, blen) = slice[e - 1];
            for &b in &buf[..blen as usize] {
                node_pos ^= da_offset(u) ^ (b as usize);
                let Some(&next_u) = da.get(node_pos) else {
                    ok = false;
                    break;
                };
                u = next_u;
                if da_label(u) != (b as u32) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
            if da_has_leaf(u)
                && let Some(&leaf) = da.get(node_pos ^ da_offset(u))
            {
                let val = (da_value(leaf) as f64) / 10000.0 + collocation_penalty;
                for bslot in &mut best[(s + 1)..e] {
                    if val > *bslot {
                        *bslot = val;
                    }
                }
            }
        }
    }

    let mut total = 0.0f64;
    for &v in &best[1..n] {
        total += v;
    }
    total / ((n - 1) as f64)
}

impl WanxiangGram {
    pub fn open(
        path: &Path,
        collocation_penalty: f64,
        non_collocation_penalty: f64,
    ) -> Result<Self, String> {
        let collocation_penalty = if collocation_penalty.is_finite() {
            collocation_penalty
        } else {
            -15.5
        };
        let non_collocation_penalty = if non_collocation_penalty.is_finite() {
            non_collocation_penalty
        } else {
            -4.0
        };
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `wide` is NUL-terminated UTF-16. All Win32 handles are checked for validity and
        // released on every error path before returning. `CreateFileMappingW` pins the file size
        // before `GetFileSizeEx` + `MapViewOfFile`, and `validate_gram_header` verifies alignment
        // and bounds before constructing `da_ptr` (`view.add(da_off) as *const u32`).
        unsafe {
            let file = CreateFileW(
                wide.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_DELETE,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_RANDOM_ACCESS,
                std::ptr::null_mut(),
            );
            if file.is_null() || file == INVALID_HANDLE_VALUE {
                return Err(format!("CreateFileW failed: {}", path.display()));
            }
            let mapping = CreateFileMappingW(
                file,
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                CloseHandle(file);
                return Err("CreateFileMappingW failed".to_string());
            }
            let mut file_size: i64 = 0;
            if GetFileSizeEx(file, &mut file_size) == 0 || file_size < 64 {
                CloseHandle(mapping);
                CloseHandle(file);
                return Err(format!("invalid .gram size: {}", file_size));
            }
            let Ok(mapped_len) = usize::try_from(file_size) else {
                CloseHandle(mapping);
                CloseHandle(file);
                return Err(format!("invalid .gram size: {}", file_size));
            };
            if mapped_len > isize::MAX as usize {
                CloseHandle(mapping);
                CloseHandle(file);
                return Err(format!(
                    "invalid .gram size (exceeds isize::MAX): {}",
                    file_size
                ));
            }
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) as *const u8;
            if view.is_null() {
                CloseHandle(mapping);
                CloseHandle(file);
                return Err("MapViewOfFile failed".to_string());
            }

            let bytes = std::slice::from_raw_parts(view, mapped_len);
            let (da_off, da_size) = match validate_gram_header(bytes) {
                Ok(v) => v,
                Err(e) => {
                    UnmapViewOfFile(view as *const c_void);
                    CloseHandle(mapping);
                    CloseHandle(file);
                    return Err(e);
                }
            };

            Ok(Self {
                _file: file,
                _mapping: mapping,
                view,
                da_ptr: view.add(da_off) as *const u32,
                da_len: da_size,
                collocation_penalty,
                non_collocation_penalty,
            })
        }
    }

    pub fn update_penalties(&mut self, collocation_penalty: f64, non_collocation_penalty: f64) {
        if collocation_penalty.is_finite() {
            self.collocation_penalty = collocation_penalty;
        }
        if non_collocation_penalty.is_finite() {
            self.non_collocation_penalty = non_collocation_penalty;
        }
    }

    #[inline]
    fn da_slice(&self) -> &[u32] {
        // SAFETY: `self.da_ptr` is 4-byte aligned (validated via `da_off % 4 == 0` and page-aligned
        // `MapViewOfFile` base) and valid for `self.da_len` `u32` elements for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.da_ptr, self.da_len) }
    }

    /// Mean per-character transition log-weight in Wanxiang's grammar trie (matching `octagram.cc`).
    pub fn score_sentence_per_char(&self, text: &str) -> f64 {
        score_slice_per_char(
            self.da_slice(),
            text,
            self.collocation_penalty,
            self.non_collocation_penalty,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference double-loop matching `octagram.cc` directly for equivalence testing.
    fn reference_score_slice(
        da: &[u32],
        text: &str,
        collocation_penalty: f64,
        non_collocation_penalty: f64,
    ) -> f64 {
        let slice: Vec<([u8; 6], u8)> = text.chars().map(encode_char_stack).collect();
        if slice.len() <= 1 || da.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f64;
        for i in 1..slice.len() {
            let mut best = non_collocation_penalty;
            let max_cl = 4.min(i);
            for cl in 1..=max_cl {
                let mut node_pos = 0usize;
                let mut u = da[0];
                let mut ok = true;
                for &(ref buf, blen) in &slice[(i - cl)..i] {
                    for &b in &buf[..blen as usize] {
                        node_pos ^= da_offset(u) ^ (b as usize);
                        let Some(&next_u) = da.get(node_pos) else {
                            ok = false;
                            break;
                        };
                        u = next_u;
                        if da_label(u) != (b as u32) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                let max_wl = (5 - cl).min(slice.len() - i);
                for wl in 1..=max_wl {
                    let (ref buf, blen) = slice[i + wl - 1];
                    for &b in &buf[..blen as usize] {
                        node_pos ^= da_offset(u) ^ (b as usize);
                        let Some(&next_u) = da.get(node_pos) else {
                            ok = false;
                            break;
                        };
                        u = next_u;
                        if da_label(u) != (b as u32) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                    if da_has_leaf(u)
                        && let Some(&leaf) = da.get(node_pos ^ da_offset(u))
                    {
                        let val = (da_value(leaf) as f64) / 10000.0 + collocation_penalty;
                        if val > best {
                            best = val;
                        }
                    }
                }
            }
            total += best;
        }
        total / ((slice.len() - 1) as f64)
    }

    #[test]
    fn test_wanxiang_gram_if_present_and_single_pass_equivalence() {
        let Ok(appdata) = std::env::var("APPDATA") else {
            return;
        };
        let p = Path::new(&appdata)
            .join("Rime")
            .join("wanxiang-lts-zh-hans.gram");
        if !p.is_file() {
            return;
        }
        let gram = WanxiangGram::open(&p, -15.5, -4.0).expect("open gram");
        let sentences = [
            "所有按键都在传送带上循环滚动",
            "所有案件都在传送带上循环滚动",
            "今天天气真不错我们一起去公园散步吧",
            "端侧神经整句重排模型可以在本地实时运行并且无需任何云端请求也不上传隐私",
            "你好",
            "a",
            "",
        ];
        for s in sentences {
            let fast = gram.score_sentence_per_char(s);
            let ref_s = reference_score_slice(gram.da_slice(), s, -15.5, -4.0);
            assert_eq!(
                fast.to_bits(),
                ref_s.to_bits(),
                "single-pass mismatch on '{}': fast={} ref={}",
                s,
                fast,
                ref_s
            );
        }
        let s1 = gram.score_sentence_per_char("所有按键都在传送带上循环滚动");
        let s2 = gram.score_sentence_per_char("所有案件都在传送带上循环滚动");
        assert!(
            s2 > s1,
            "expected 案件 ({}) > 按键 ({}) in raw N-gram",
            s2,
            s1
        );
    }

    #[test]
    fn test_encode_char_stack_ranges() {
        // NUL -> 0xE0
        let (b, l) = encode_char_stack('\0');
        assert_eq!((&b[..l as usize], l), (&[0xE0][..], 1));
        // ASCII
        let (b, l) = encode_char_stack('A');
        assert_eq!((&b[..l as usize], l), (&b"A"[..], 1));
        // CJK low byte == 0 (U+4E00 '一')
        let (b, l) = encode_char_stack('\u{4E00}');
        assert_eq!((&b[..l as usize], l), (&[0xE1, 0x8E][..], 2));
        // CJK low byte != 0 (U+4E2D '中')
        let (b, l) = encode_char_stack('中');
        assert_eq!((&b[..l as usize], l), (&[0x8E, 0x2D][..], 2));
        // Supplementary plane emoji (U+1F600)
        let (_, l) = encode_char_stack('😀');
        assert!((2..=6).contains(&l));
    }

    #[test]
    fn test_rejects_malformed_gram_headers() {
        // Too short
        assert!(validate_gram_header(&[0u8; 32]).is_err());

        // Wrong version (2.0)
        let mut buf = vec![0u8; 128];
        buf[..18].copy_from_slice(b"Rime::Grammar/2.0\0");
        buf[36..40].copy_from_slice(&4u32.to_le_bytes());
        buf[40..44].copy_from_slice(&4i32.to_le_bytes());
        assert!(validate_gram_header(&buf).is_err());

        // da_size == 0
        buf[..18].copy_from_slice(GRAM_MAGIC_V1);
        buf[36..40].copy_from_slice(&0u32.to_le_bytes());
        assert!(validate_gram_header(&buf).is_err());

        // Negative rel_off overlapping header
        buf[36..40].copy_from_slice(&4u32.to_le_bytes());
        buf[40..44].copy_from_slice(&(-40i32).to_le_bytes());
        assert!(validate_gram_header(&buf).is_err());

        // rel_off = i32::MAX (overflow guard)
        buf[40..44].copy_from_slice(&i32::MAX.to_le_bytes());
        assert!(validate_gram_header(&buf).is_err());

        // Unaligned da_off (rel_off = 5 -> da_off = 45)
        buf[40..44].copy_from_slice(&5i32.to_le_bytes());
        assert!(validate_gram_header(&buf).is_err());

        // da_size = u32::MAX (byte-size overflow / out-of-bounds)
        buf[40..44].copy_from_slice(&4i32.to_le_bytes());
        buf[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(validate_gram_header(&buf).is_err());

        // File-level open rejection test
        buf[36..40].copy_from_slice(&4u32.to_le_bytes());
        buf[40..44].copy_from_slice(&(-40i32).to_le_bytes());
        let tmp = std::env::temp_dir().join("rime_neural_test_corrupt_off.gram");
        std::fs::write(&tmp, &buf).expect("write tmp gram");
        let res = WanxiangGram::open(&tmp, -15.5, -4.0);
        let _ = std::fs::remove_file(&tmp);
        assert!(res.is_err(), "expected corrupt negative rel_off to fail");
    }
}
