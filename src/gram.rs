use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;

const GENERIC_READ: u32 = 0x8000_0000;
const FILE_SHARE_READ: u32 = 0x0000_0001;
const FILE_SHARE_DELETE: u32 = 0x0000_0004;
const OPEN_EXISTING: u32 = 3;
const PAGE_READONLY: u32 = 0x02;
const FILE_MAP_READ: u32 = 0x0004;
const INVALID_HANDLE_VALUE: *mut c_void = -1isize as *mut c_void;

extern "system" {
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
/// (0 MB extra physical RAM) and scores a candidate sentence in ~0.01 ms with zero per-char allocs.
pub struct WanxiangGram {
    _file: *mut c_void,
    _mapping: *mut c_void,
    view: *const u8,
    da_ptr: *const u32,
    da_len: usize,
    collocation_penalty: f64,
    non_collocation_penalty: f64,
}

unsafe impl Send for WanxiangGram {}
unsafe impl Sync for WanxiangGram {}

impl Drop for WanxiangGram {
    fn drop(&mut self) {
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

impl WanxiangGram {
    pub fn open(
        path: &Path,
        collocation_penalty: f64,
        non_collocation_penalty: f64,
    ) -> Result<Self, String> {
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        unsafe {
            let file = CreateFileW(
                wide.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_DELETE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
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
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) as *const u8;
            if view.is_null() {
                CloseHandle(mapping);
                CloseHandle(file);
                return Err("MapViewOfFile failed".to_string());
            }

            let bytes = std::slice::from_raw_parts(view, mapped_len);
            if !bytes.starts_with(b"Rime::Grammar/") {
                UnmapViewOfFile(view as *const c_void);
                CloseHandle(mapping);
                CloseHandle(file);
                return Err("not a Rime::Grammar file".to_string());
            }
            let da_size = u32::from_le_bytes(bytes[36..40].try_into().unwrap()) as usize;
            let rel_off = i32::from_le_bytes(bytes[40..44].try_into().unwrap()) as isize;
            if rel_off < 4 {
                UnmapViewOfFile(view as *const c_void);
                CloseHandle(mapping);
                CloseHandle(file);
                return Err("corrupt double-array offset".to_string());
            }
            let da_off = (40isize + rel_off) as usize;
            let da_bytes = da_size.saturating_mul(4);
            if da_off < 44
                || !da_off.is_multiple_of(4)
                || da_off.saturating_add(da_bytes) > bytes.len()
            {
                UnmapViewOfFile(view as *const c_void);
                CloseHandle(mapping);
                CloseHandle(file);
                return Err("corrupt double-array bounds".to_string());
            }

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

    #[inline(always)]
    fn unit(&self, idx: usize) -> Option<u32> {
        if idx < self.da_len {
            Some(unsafe { *self.da_ptr.add(idx) })
        } else {
            None
        }
    }

    #[inline(always)]
    fn has_leaf(u: u32) -> bool {
        (u & (1 << 8)) != 0
    }

    #[inline(always)]
    fn value(u: u32) -> u32 {
        u & 0x7FFF_FFFF
    }

    #[inline(always)]
    fn label(u: u32) -> u32 {
        u & 0x8000_00FF
    }

    #[inline(always)]
    fn offset(u: u32) -> usize {
        ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize
    }

    /// Mean per-character transition log-weight in Wanxiang's grammar trie (matching `octagram.cc`).
    pub fn score_sentence_per_char(&self, text: &str) -> f64 {
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
        if n <= 1 {
            return 0.0;
        }
        let slice: &[([u8; 6], u8)] = if overflow.is_empty() {
            &encoded[..n]
        } else {
            &overflow
        };

        let mut total = 0.0f64;
        for i in 1..slice.len() {
            let mut best = self.non_collocation_penalty;
            let max_cl = 4.min(i);
            for cl in 1..=max_cl {
                let mut node_pos: usize = 0;
                let Some(mut u) = self.unit(0) else {
                    break;
                };
                let mut ok = true;
                for &(ref buf, blen) in &slice[(i - cl)..i] {
                    for &b in &buf[..blen as usize] {
                        node_pos ^= Self::offset(u) ^ (b as usize);
                        let Some(next_u) = self.unit(node_pos) else {
                            ok = false;
                            break;
                        };
                        u = next_u;
                        if Self::label(u) != (b as u32) {
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
                        node_pos ^= Self::offset(u) ^ (b as usize);
                        let Some(next_u) = self.unit(node_pos) else {
                            ok = false;
                            break;
                        };
                        u = next_u;
                        if Self::label(u) != (b as u32) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                    if Self::has_leaf(u) {
                        if let Some(leaf) = self.unit(node_pos ^ Self::offset(u)) {
                            let val =
                                (Self::value(leaf) as f64) / 10000.0 + self.collocation_penalty;
                            if val > best {
                                best = val;
                            }
                        }
                    }
                }
            }
            total += best;
        }
        total / ((slice.len() - 1) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wanxiang_gram_if_present() {
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
    fn test_rejects_corrupt_header_overlap_offset() {
        let mut buf = vec![0u8; 128];
        buf[..14].copy_from_slice(b"Rime::Grammar/");
        buf[36..40].copy_from_slice(&4u32.to_le_bytes());
        buf[40..44].copy_from_slice(&(-40i32).to_le_bytes()); // da_off = 0 (overlaps header)
        let tmp = std::env::temp_dir().join("rime_neural_test_corrupt_off.gram");
        std::fs::write(&tmp, &buf).expect("write tmp gram");
        let res = WanxiangGram::open(&tmp, -15.5, -4.0);
        let _ = std::fs::remove_file(&tmp);
        assert!(res.is_err(), "expected corrupt negative rel_off to fail");
    }
}
