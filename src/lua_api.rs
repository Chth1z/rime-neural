use std::ffi::{c_char, c_double, c_int, c_longlong, c_void};
use std::sync::OnceLock;

pub type LuaState = *mut c_void;
/// Lua 5.4 C function ABI using `extern "C-unwind"` so Lua 5.4 C++ exceptions / SEH unwinding
/// (e.g. `LUA_ERRMEM` raised by allocating APIs) unwind cleanly instead of aborting across FFI.
pub type LuaCFunction = unsafe extern "C-unwind" fn(LuaState) -> c_int;
pub type LuaInteger = c_longlong;
pub type LuaNumber = c_double;

pub const LUA_TBOOLEAN: c_int = 1;
pub const LUA_TSTRING: c_int = 4;
pub const LUA_TTABLE: c_int = 5;

/// Pin `rime_neural.dll` for the lifetime of the host process (`WeaselServer.exe`) so that
/// `lua_close()` during Rime redeploy (`FreeLibrary` on `_CLIBS`) never unmaps the DLL while
/// ONNX Runtime thread pools, the async loader thread, or the UIA worker thread are active, and
/// subsequent redeploys reuse the already-warmed process singleton in 0 ms.
///
/// Returns `true` if the module handle was successfully pinned.
#[cfg(windows)]
pub unsafe fn pin_self_module(addr_inside_dll: *const c_void) -> bool {
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleHandleExA,
    };
    use windows::core::PCSTR;

    let mut hmod = HMODULE::default();
    // SAFETY: `addr_inside_dll` points to code inside this DLL; `GetModuleHandleExA` with
    // `GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN` pins the
    // module in the host process address space and writes the handle into `hmod`.
    let ok = unsafe {
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            PCSTR(addr_inside_dll as *const u8),
            &mut hmod,
        )
    };
    ok.is_ok() && !hmod.is_invalid()
}

#[cfg(unix)]
pub unsafe fn pin_self_module(addr_inside_dll: *const c_void) -> bool {
    // SAFETY: `libc::Dl_info` is a plain C struct of pointers/integers where all-zeros is valid.
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: `addr_inside_dll` points to code inside this shared library; `dladdr` resolves
    // its loaded path and `dlopen(..., RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE)` pins it in-process
    // so Rime redeploys (`lua_close` -> `dlclose`) never unmap the live ONNX Runtime engine.
    let ok = unsafe { libc::dladdr(addr_inside_dll, &mut info) };
    if ok == 0 || info.dli_fname.is_null() {
        return false;
    }
    // SAFETY: `info.dli_fname` is a NUL-terminated path returned by `dladdr`.
    let handle = unsafe {
        libc::dlopen(
            info.dli_fname,
            libc::RTLD_NOW | libc::RTLD_NOLOAD | libc::RTLD_NODELETE,
        )
    };
    !handle.is_null()
}

#[cfg(not(any(windows, unix)))]
pub unsafe fn pin_self_module(_addr_inside_dll: *const c_void) -> bool {
    false
}

/// Minimal Lua 5.4 C API table dynamically resolved from the host `rime.dll` / `librime.dll` /
/// `lua54.dll`. Requires zero `.lib` import linkage and works with official unmodified Weasel.
pub struct LuaApi {
    pub version: unsafe extern "C-unwind" fn(LuaState) -> LuaNumber,
    pub checkstack: unsafe extern "C-unwind" fn(LuaState, c_int) -> c_int,
    pub gettop: unsafe extern "C-unwind" fn(LuaState) -> c_int,
    pub createtable: unsafe extern "C-unwind" fn(LuaState, c_int, c_int),
    pub pushcclosure: unsafe extern "C-unwind" fn(LuaState, LuaCFunction, c_int),
    pub rawget: unsafe extern "C-unwind" fn(LuaState, c_int) -> c_int,
    pub rawset: unsafe extern "C-unwind" fn(LuaState, c_int),
    pub lua_type: unsafe extern "C-unwind" fn(LuaState, c_int) -> c_int,
    pub tolstring: unsafe extern "C-unwind" fn(LuaState, c_int, *mut usize) -> *const c_char,
    pub pushlstring: unsafe extern "C-unwind" fn(LuaState, *const c_char, usize) -> *const c_char,
    pub pushboolean: unsafe extern "C-unwind" fn(LuaState, c_int),
    pub pushinteger: unsafe extern "C-unwind" fn(LuaState, LuaInteger),
    pub pushnumber: unsafe extern "C-unwind" fn(LuaState, LuaNumber),
    pub pushnil: unsafe extern "C-unwind" fn(LuaState),
    pub toboolean: unsafe extern "C-unwind" fn(LuaState, c_int) -> c_int,
    pub tointegerx: unsafe extern "C-unwind" fn(LuaState, c_int, *mut c_int) -> LuaInteger,
    pub tonumberx: unsafe extern "C-unwind" fn(LuaState, c_int, *mut c_int) -> LuaNumber,
    pub rawlen: unsafe extern "C-unwind" fn(LuaState, c_int) -> u64,
    pub rawgeti: unsafe extern "C-unwind" fn(LuaState, c_int, LuaInteger) -> c_int,
    pub rawseti: unsafe extern "C-unwind" fn(LuaState, c_int, LuaInteger),
    pub settop: unsafe extern "C-unwind" fn(LuaState, c_int),
}

static API: OnceLock<LuaApi> = OnceLock::new();

#[cfg(windows)]
pub unsafe fn get_api() -> Result<&'static LuaApi, &'static str> {
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    use windows::core::PCSTR;

    if let Some(api) = API.get() {
        return Ok(api);
    }

    // Probe candidate host modules in order and require the Lua 5.4-exclusive symbol
    // `lua_newuserdatauv` BEFORE ever resolving or invoking `lua_version` (in Lua 5.2/5.3,
    // `lua_version` returned `const lua_Number*` in RAX instead of `double` in XMM0).
    let candidates: [Option<&'static std::ffi::CStr>; 4] = [
        Some(c"rime.dll"),
        Some(c"librime.dll"),
        Some(c"lua54.dll"),
        None, // current host executable
    ];
    let mut host = HMODULE::default();
    let mut found_any_module = false;
    for name in candidates {
        let pcstr = match name {
            Some(cstr) => PCSTR(cstr.as_ptr() as *const u8),
            None => PCSTR::null(),
        };
        // SAFETY: `pcstr` is either null (current module) or a static NUL-terminated ASCII string.
        if let Ok(h) = unsafe { GetModuleHandleA(pcstr) }
            && !h.is_invalid()
        {
            if name.is_some() {
                found_any_module = true;
            }
            // SAFETY: `h` is a valid loaded module handle and `c"lua_newuserdatauv"` is NUL-terminated.
            let has_lua54_marker = unsafe {
                GetProcAddress(h, PCSTR(c"lua_newuserdatauv".as_ptr() as *const u8)).is_some()
            };
            // SAFETY: `h` is valid and `c"lua_version"` is NUL-terminated.
            let has_version =
                unsafe { GetProcAddress(h, PCSTR(c"lua_version".as_ptr() as *const u8)).is_some() };
            if has_lua54_marker && has_version {
                host = h;
                break;
            }
        }
    }
    if host.is_invalid() {
        return Err(if found_any_module {
            "host module does not export Lua 5.4 C API (missing lua_newuserdatauv / lua_version)"
        } else {
            "Lua 5.4 host module (rime.dll / librime.dll / lua54.dll) not found"
        });
    }

    macro_rules! sym {
        ($name:literal) => {{
            // SAFETY: `host` is a valid loaded module handle verified to export Lua 5.4's
            // `lua_newuserdatauv`, and `concat!($name, "\0")` is NUL-terminated.
            let Some(p) = (unsafe { GetProcAddress(host, PCSTR(concat!($name, "\0").as_ptr())) })
            else {
                return Err(concat!("missing symbol in Lua 5.4 host: ", $name));
            };
            #[allow(clippy::missing_transmute_annotations)]
            // SAFETY: Transmuting non-null `FARPROC` to the matching Lua 5.4 `extern "C-unwind"`
            // signature; Lua 5.4 ABI is additionally verified via `check_lua54_abi` before use.
            unsafe {
                std::mem::transmute(p)
            }
        }};
    }
    let api = LuaApi {
        version: sym!("lua_version"),
        checkstack: sym!("lua_checkstack"),
        gettop: sym!("lua_gettop"),
        createtable: sym!("lua_createtable"),
        pushcclosure: sym!("lua_pushcclosure"),
        rawget: sym!("lua_rawget"),
        rawset: sym!("lua_rawset"),
        lua_type: sym!("lua_type"),
        tolstring: sym!("lua_tolstring"),
        pushlstring: sym!("lua_pushlstring"),
        pushboolean: sym!("lua_pushboolean"),
        pushinteger: sym!("lua_pushinteger"),
        pushnumber: sym!("lua_pushnumber"),
        pushnil: sym!("lua_pushnil"),
        toboolean: sym!("lua_toboolean"),
        tointegerx: sym!("lua_tointegerx"),
        tonumberx: sym!("lua_tonumberx"),
        rawlen: sym!("lua_rawlen"),
        rawgeti: sym!("lua_rawgeti"),
        rawseti: sym!("lua_rawseti"),
        settop: sym!("lua_settop"),
    };
    Ok(API.get_or_init(|| api))
}

#[cfg(unix)]
pub unsafe fn get_api() -> Result<&'static LuaApi, &'static str> {
    if let Some(api) = API.get() {
        return Ok(api);
    }

    // Locate the exact shared object providing Lua 5.4's exclusive `lua_newuserdatauv` symbol,
    // pin that provider handle with `RTLD_NODELETE`, and resolve every symbol strictly from that
    // same provider (verifying `dli_fbase` identity) so a host that also loaded Lua 5.3 earlier
    // in global scope can never mix incompatible ABIs.
    // SAFETY: `RTLD_DEFAULT` searches loaded global symbols; `c"lua_newuserdatauv"` is NUL-terminated.
    let marker = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"lua_newuserdatauv".as_ptr()) };
    if marker.is_null() {
        return Err("host process does not export Lua 5.4 C API (missing lua_newuserdatauv)");
    }
    // SAFETY: `libc::Dl_info` is a plain C struct of pointers/integers where all-zeros is valid.
    let mut marker_info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: `marker` is a valid resolved function address inside the host's Lua 5.4 provider.
    if unsafe { libc::dladdr(marker, &mut marker_info) } == 0
        || marker_info.dli_fbase.is_null()
        || marker_info.dli_fname.is_null()
    {
        return Err("failed to identify Lua 5.4 symbol provider via dladdr");
    }

    // Open and pin the exact provider DSO (`RTLD_NOLOAD | RTLD_NODELETE`); fall back to the
    // main executable handle (`dlopen(NULL, ...)`) when Lua 5.4 is statically linked into it.
    // SAFETY: `marker_info.dli_fname` is a NUL-terminated path returned by `dladdr`.
    let mut provider = unsafe {
        libc::dlopen(
            marker_info.dli_fname,
            libc::RTLD_NOW | libc::RTLD_NOLOAD | libc::RTLD_NODELETE,
        )
    };
    if provider.is_null() {
        // SAFETY: Passing NULL to `dlopen` returns a handle to the main host executable.
        provider = unsafe { libc::dlopen(std::ptr::null(), libc::RTLD_NOW) };
    }
    if provider.is_null() {
        return Err("failed to open Lua 5.4 symbol provider handle");
    }
    let expected_base = marker_info.dli_fbase;

    macro_rules! sym {
        ($name:literal) => {{
            // SAFETY: `provider` is a valid handle returned by `dlopen` for the Lua 5.4 provider,
            // and `concat!($name, "\0")` is NUL-terminated.
            let p =
                unsafe { libc::dlsym(provider, concat!($name, "\0").as_ptr() as *const c_char) };
            if p.is_null() {
                return Err(concat!("missing symbol in Lua 5.4 host: ", $name));
            }
            // SAFETY: `libc::Dl_info` is a plain C struct where all-zeros is valid.
            let mut sym_info: libc::Dl_info = unsafe { std::mem::zeroed() };
            // SAFETY: `p` is a non-null symbol address returned by `dlsym`.
            if unsafe { libc::dladdr(p, &mut sym_info) } == 0 || sym_info.dli_fbase != expected_base
            {
                return Err(concat!("ambiguous Lua symbol provider for: ", $name));
            }
            #[allow(clippy::missing_transmute_annotations)]
            // SAFETY: Transmuting verified single-provider Lua 5.4 symbol pointer to matching
            // `extern "C-unwind"` signature.
            unsafe {
                std::mem::transmute(p)
            }
        }};
    }

    let api = LuaApi {
        version: sym!("lua_version"),
        checkstack: sym!("lua_checkstack"),
        gettop: sym!("lua_gettop"),
        createtable: sym!("lua_createtable"),
        pushcclosure: sym!("lua_pushcclosure"),
        rawget: sym!("lua_rawget"),
        rawset: sym!("lua_rawset"),
        lua_type: sym!("lua_type"),
        tolstring: sym!("lua_tolstring"),
        pushlstring: sym!("lua_pushlstring"),
        pushboolean: sym!("lua_pushboolean"),
        pushinteger: sym!("lua_pushinteger"),
        pushnumber: sym!("lua_pushnumber"),
        pushnil: sym!("lua_pushnil"),
        toboolean: sym!("lua_toboolean"),
        tointegerx: sym!("lua_tointegerx"),
        tonumberx: sym!("lua_tonumberx"),
        rawlen: sym!("lua_rawlen"),
        rawgeti: sym!("lua_rawgeti"),
        rawseti: sym!("lua_rawseti"),
        settop: sym!("lua_settop"),
    };
    Ok(API.get_or_init(|| api))
}

#[cfg(not(any(windows, unix)))]
pub unsafe fn get_api() -> Result<&'static LuaApi, &'static str> {
    Err("dynamic Lua 5.4 host resolution is only supported on Windows and Unix")
}

/// Verify that the host Lua VM is strictly Lua 5.4 (`504.0 <= ver < 505.0`), matching our 64-bit
/// `lua_Unsigned` / `lua_Integer` and `lua_version` C ABI.
pub unsafe fn check_lua54_abi(api: &LuaApi, l: LuaState) -> bool {
    // SAFETY: Caller verified `lua_newuserdatauv` exists before resolving `api`, and `l` is a
    // valid `lua_State*`. In Lua 5.4, `lua_version(L)` returns `lua_Number` (`504.0`).
    let ver = unsafe { (api.version)(l) };
    ver.is_finite() && (504.0..505.0).contains(&ver)
}

/// Strict string read: checks `LUA_TSTRING` before calling `lua_tolstring` so numeric Lua stack
/// slots are never mutated in-place and no Lua GC allocation / unwind can occur.
pub unsafe fn read_str(api: &LuaApi, l: LuaState, idx: c_int) -> Option<String> {
    // SAFETY: `l` is a valid `lua_State*`. Checking `LUA_TSTRING` guarantees `lua_tolstring`
    // returns a direct pointer into the existing Lua string without mutating the stack or allocating.
    if unsafe { (api.lua_type)(l, idx) } != LUA_TSTRING {
        return None;
    }
    let mut len: usize = 0;
    // SAFETY: Slot `idx` is verified to be `LUA_TSTRING`.
    let ptr = unsafe { (api.tolstring)(l, idx, &mut len) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `lua_tolstring` returns a valid pointer to `len` readable bytes owned by the Lua state
    // for at least as long as the string remains on the Lua stack.
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    Some(String::from_utf8_lossy(bytes).into_owned())
}

pub unsafe fn push_str(api: &LuaApi, l: LuaState, s: &str) {
    // SAFETY: `l` is a valid `lua_State*` with stack headroom; `s` is valid for `s.len()` bytes.
    unsafe {
        (api.pushlstring)(l, s.as_ptr() as *const c_char, s.len());
    }
}

pub unsafe fn pop(api: &LuaApi, l: LuaState, n: c_int) {
    // SAFETY: `l` is a valid `lua_State*` with at least `n` stack elements.
    unsafe {
        (api.settop)(l, -n - 1);
    }
}

/// Set a string-keyed field on the table currently at the top of the Lua stack (`-1`) using
/// `lua_rawset` (bypassing any `__newindex` metamethods).
pub unsafe fn rawset_field<F>(api: &LuaApi, l: LuaState, key: &str, push_val: F)
where
    F: FnOnce(&LuaApi, LuaState),
{
    // SAFETY: Table sits at `-1`; pushing key and value places table at `-3`, key at `-2`,
    // value at `-1`, and `lua_rawset(l, -3)` pops key and value without metamethod dispatch.
    unsafe {
        push_str(api, l, key);
        push_val(api, l);
        (api.rawset)(l, -3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C-unwind" fn dummy_checkstack(_: LuaState, _: c_int) -> c_int {
        1
    }
    unsafe extern "C-unwind" fn dummy_gettop(_: LuaState) -> c_int {
        0
    }
    unsafe extern "C-unwind" fn dummy_createtable(_: LuaState, _: c_int, _: c_int) {}
    unsafe extern "C-unwind" fn dummy_pushcclosure(_: LuaState, _: LuaCFunction, _: c_int) {}
    unsafe extern "C-unwind" fn dummy_rawget(_: LuaState, _: c_int) -> c_int {
        0
    }
    unsafe extern "C-unwind" fn dummy_rawset(_: LuaState, _: c_int) {}
    unsafe extern "C-unwind" fn dummy_type_num(_: LuaState, _: c_int) -> c_int {
        3 // LUA_TNUMBER
    }
    unsafe extern "C-unwind" fn dummy_tolstring(
        _: LuaState,
        _: c_int,
        _: *mut usize,
    ) -> *const c_char {
        panic!("read_str must never call tolstring on non-LUA_TSTRING slots")
    }
    unsafe extern "C-unwind" fn dummy_pushlstring(
        _: LuaState,
        _: *const c_char,
        _: usize,
    ) -> *const c_char {
        std::ptr::null()
    }
    unsafe extern "C-unwind" fn dummy_pushboolean(_: LuaState, _: c_int) {}
    unsafe extern "C-unwind" fn dummy_pushinteger(_: LuaState, _: LuaInteger) {}
    unsafe extern "C-unwind" fn dummy_pushnumber(_: LuaState, _: LuaNumber) {}
    unsafe extern "C-unwind" fn dummy_pushnil(_: LuaState) {}
    unsafe extern "C-unwind" fn dummy_toboolean(_: LuaState, _: c_int) -> c_int {
        0
    }
    unsafe extern "C-unwind" fn dummy_tointegerx(
        _: LuaState,
        _: c_int,
        _: *mut c_int,
    ) -> LuaInteger {
        0
    }
    unsafe extern "C-unwind" fn dummy_tonumberx(_: LuaState, _: c_int, _: *mut c_int) -> LuaNumber {
        0.0
    }
    unsafe extern "C-unwind" fn dummy_rawlen(_: LuaState, _: c_int) -> u64 {
        0
    }
    unsafe extern "C-unwind" fn dummy_rawgeti(_: LuaState, _: c_int, _: LuaInteger) -> c_int {
        0
    }
    unsafe extern "C-unwind" fn dummy_rawseti(_: LuaState, _: c_int, _: LuaInteger) {}
    unsafe extern "C-unwind" fn dummy_settop(_: LuaState, _: c_int) {}

    fn mock_api(version_fn: unsafe extern "C-unwind" fn(LuaState) -> LuaNumber) -> LuaApi {
        LuaApi {
            version: version_fn,
            checkstack: dummy_checkstack,
            gettop: dummy_gettop,
            createtable: dummy_createtable,
            pushcclosure: dummy_pushcclosure,
            rawget: dummy_rawget,
            rawset: dummy_rawset,
            lua_type: dummy_type_num,
            tolstring: dummy_tolstring,
            pushlstring: dummy_pushlstring,
            pushboolean: dummy_pushboolean,
            pushinteger: dummy_pushinteger,
            pushnumber: dummy_pushnumber,
            pushnil: dummy_pushnil,
            toboolean: dummy_toboolean,
            tointegerx: dummy_tointegerx,
            tonumberx: dummy_tonumberx,
            rawlen: dummy_rawlen,
            rawgeti: dummy_rawgeti,
            rawseti: dummy_rawseti,
            settop: dummy_settop,
        }
    }

    #[test]
    fn test_check_lua54_abi_bounds_and_strict_read_str() {
        unsafe extern "C-unwind" fn v504(_: LuaState) -> LuaNumber {
            504.0
        }
        unsafe extern "C-unwind" fn v503(_: LuaState) -> LuaNumber {
            503.0
        }
        unsafe extern "C-unwind" fn v505(_: LuaState) -> LuaNumber {
            505.0
        }
        unsafe extern "C-unwind" fn vnan(_: LuaState) -> LuaNumber {
            f64::NAN
        }

        let api_ok = mock_api(v504);
        // SAFETY: Mock function pointers ignore the null `LuaState`.
        unsafe {
            assert!(check_lua54_abi(&api_ok, std::ptr::null_mut()));
            assert!(!check_lua54_abi(&mock_api(v503), std::ptr::null_mut()));
            assert!(!check_lua54_abi(&mock_api(v505), std::ptr::null_mut()));
            assert!(!check_lua54_abi(&mock_api(vnan), std::ptr::null_mut()));
            // Must return None without touching `dummy_tolstring` on LUA_TNUMBER
            assert_eq!(read_str(&api_ok, std::ptr::null_mut(), 1), None);
        }
    }
}
