use std::ffi::{c_char, c_double, c_int, c_longlong, c_void};
use std::sync::{Mutex, OnceLock};

pub type LuaState = *mut c_void;
pub type LuaCFunction = unsafe extern "C" fn(LuaState) -> c_int;
pub type LuaInteger = c_longlong;
pub type LuaNumber = c_double;

pub const LUA_TBOOLEAN: c_int = 1;
pub const LUA_TSTRING: c_int = 4;
pub const LUA_TTABLE: c_int = 5;

const GET_MODULE_HANDLE_EX_FLAG_PIN: u32 = 0x0000_0001;
const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;

unsafe extern "system" {
    fn GetModuleHandleA(lpModuleName: *const c_char) -> *mut c_void;
    fn GetModuleHandleExA(
        dwFlags: u32,
        lpModuleName: *const c_void,
        phModule: *mut *mut c_void,
    ) -> c_int;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *const c_void;
}

/// Pin `rime_neural.dll` for the lifetime of the host process (`WeaselServer.exe`) so that
/// `lua_close()` during Rime redeploy (`FreeLibrary` on `_CLIBS`) never unmaps the DLL while
/// ONNX Runtime thread pools, the async loader thread, or the UIA worker thread are active, and
/// subsequent redeploys reuse the already-warmed process singleton in 0 ms.
///
/// Returns `true` if the module handle was successfully pinned.
pub unsafe fn pin_self_module(addr_inside_dll: *const c_void) -> bool {
    let mut hmod: *mut c_void = std::ptr::null_mut();
    // SAFETY: `addr_inside_dll` points to code inside this DLL; `GetModuleHandleExA` with
    // `GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN` pins the
    // module in the host process address space and writes the handle into `hmod`.
    let ok = unsafe {
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            addr_inside_dll,
            &mut hmod,
        )
    };
    ok != 0 && !hmod.is_null()
}

/// Minimal Lua 5.4 C API table dynamically resolved from Weasel's loaded `rime.dll`.
/// Requires zero `.lib` import linkage and works with official unmodified Weasel builds.
pub struct LuaApi {
    pub version: unsafe extern "C" fn(LuaState) -> LuaNumber,
    pub checkstack: unsafe extern "C" fn(LuaState, c_int) -> c_int,
    pub gettop: unsafe extern "C" fn(LuaState) -> c_int,
    pub createtable: unsafe extern "C" fn(LuaState, c_int, c_int),
    pub pushcclosure: unsafe extern "C" fn(LuaState, LuaCFunction, c_int),
    pub rawget: unsafe extern "C" fn(LuaState, c_int) -> c_int,
    pub rawset: unsafe extern "C" fn(LuaState, c_int),
    pub lua_type: unsafe extern "C" fn(LuaState, c_int) -> c_int,
    pub tolstring: unsafe extern "C" fn(LuaState, c_int, *mut usize) -> *const c_char,
    pub pushlstring: unsafe extern "C" fn(LuaState, *const c_char, usize) -> *const c_char,
    pub pushboolean: unsafe extern "C" fn(LuaState, c_int),
    pub pushinteger: unsafe extern "C" fn(LuaState, LuaInteger),
    pub pushnumber: unsafe extern "C" fn(LuaState, LuaNumber),
    pub pushnil: unsafe extern "C" fn(LuaState),
    pub toboolean: unsafe extern "C" fn(LuaState, c_int) -> c_int,
    pub tointegerx: unsafe extern "C" fn(LuaState, c_int, *mut c_int) -> LuaInteger,
    pub tonumberx: unsafe extern "C" fn(LuaState, c_int, *mut c_int) -> LuaNumber,
    pub rawlen: unsafe extern "C" fn(LuaState, c_int) -> u64,
    pub rawgeti: unsafe extern "C" fn(LuaState, c_int, LuaInteger) -> c_int,
    pub rawseti: unsafe extern "C" fn(LuaState, c_int, LuaInteger),
    pub settop: unsafe extern "C" fn(LuaState, c_int),
}

// SAFETY: `LuaApi` holds only immutable C function pointers exported by the host `rime.dll`
// which stays loaded for the lifetime of the process.
unsafe impl Sync for LuaApi {}
// SAFETY: Function pointers have no thread affinity.
unsafe impl Send for LuaApi {}

static API: OnceLock<LuaApi> = OnceLock::new();
static API_LOCK: Mutex<()> = Mutex::new(());

pub unsafe fn get_api() -> Result<&'static LuaApi, &'static str> {
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let _guard = API_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(api) = API.get() {
        return Ok(api);
    }
    // SAFETY: `c"rime.dll"` is a valid NUL-terminated string; `GetModuleHandleA` returns a
    // borrowed handle to the host process's loaded `rime.dll` (or null if absent).
    let mut h = unsafe { GetModuleHandleA(c"rime.dll".as_ptr()) };
    if h.is_null() {
        // SAFETY: Fallback to `librime.dll` for non-Weasel Windows librime hosts.
        h = unsafe { GetModuleHandleA(c"librime.dll".as_ptr()) };
    }
    if h.is_null() {
        return Err("rime.dll module handle not found");
    }
    macro_rules! sym {
        ($name:literal) => {{
            // SAFETY: `h` is a valid loaded module handle and `concat!($name, "\0")` is
            // NUL-terminated. Transmuting `FARPROC` to the matching Lua 5.4 C ABI fn pointer
            // is validated against Lua 5.4's `lua_version(L) >= 504.0` before invocation.
            let p = unsafe { GetProcAddress(h, concat!($name, "\0").as_ptr() as *const c_char) };
            if p.is_null() {
                return Err(concat!("missing symbol in rime.dll: ", $name));
            }
            #[allow(clippy::missing_transmute_annotations)]
            // SAFETY: Transmuting non-null `FARPROC` to the matching Lua 5.4 C ABI signature;
            // Lua 5.4 ABI is verified via `check_lua54_abi` before any exported closure runs.
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
    let _ = API.set(api);
    API.get().ok_or("LuaApi initialization race")
}

/// Verify that the host Lua VM is Lua 5.4+ (matching our 64-bit `lua_Unsigned` / `lua_Integer` ABI).
pub unsafe fn check_lua54_abi(api: &LuaApi, l: LuaState) -> bool {
    // SAFETY: Caller passes a valid `lua_State*` and `lua_version` takes `lua_State*` in Lua 5.4
    // (returning `lua_Number` = 504.0).
    let ver = unsafe { (api.version)(l) };
    ver.is_finite() && (504.0..600.0).contains(&ver)
}

/// Strict string read: checks `LUA_TSTRING` before calling `lua_tolstring` so numeric Lua stack
/// slots are never mutated in-place and no Lua GC allocation / `longjmp` can occur.
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
