use std::ffi::{c_char, c_double, c_int, c_longlong, c_void};
use std::sync::OnceLock;

pub type LuaState = *mut c_void;
pub type LuaCFunction = unsafe extern "C" fn(LuaState) -> c_int;
pub type LuaInteger = c_longlong;
pub type LuaNumber = c_double;

const GET_MODULE_HANDLE_EX_FLAG_PIN: u32 = 0x00000001;
const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x00000004;

extern "system" {
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
pub unsafe fn pin_self_module(addr_inside_dll: *const c_void) {
    let mut hmod: *mut c_void = std::ptr::null_mut();
    let _ = GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        addr_inside_dll,
        &mut hmod,
    );
}

/// Minimal Lua 5.4 C API table dynamically resolved from Weasel's loaded `rime.dll`.
/// Requires zero `.lib` import linkage and works with official unmodified Weasel builds.
pub struct LuaApi {
    pub createtable: unsafe extern "C" fn(LuaState, c_int, c_int),
    pub pushcclosure: unsafe extern "C" fn(LuaState, LuaCFunction, c_int),
    pub setfield: unsafe extern "C" fn(LuaState, c_int, *const c_char),
    pub rawget: unsafe extern "C" fn(LuaState, c_int) -> c_int,
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

unsafe impl Sync for LuaApi {}
unsafe impl Send for LuaApi {}

static API: OnceLock<Result<LuaApi, &'static str>> = OnceLock::new();

pub unsafe fn get_api() -> Result<&'static LuaApi, &'static str> {
    API.get_or_init(|| {
        let h = GetModuleHandleA(c"rime.dll".as_ptr());
        if h.is_null() {
            return Err("rime.dll module handle not found");
        }
        macro_rules! sym {
            ($name:literal) => {{
                let p = GetProcAddress(h, concat!($name, "\0").as_ptr() as *const c_char);
                if p.is_null() {
                    return Err(concat!("missing symbol in rime.dll: ", $name));
                }
                #[allow(clippy::missing_transmute_annotations)]
                std::mem::transmute(p)
            }};
        }
        Ok(LuaApi {
            createtable: sym!("lua_createtable"),
            pushcclosure: sym!("lua_pushcclosure"),
            setfield: sym!("lua_setfield"),
            rawget: sym!("lua_rawget"),
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
        })
    })
    .as_ref()
    .map_err(|e| *e)
}

pub unsafe fn read_str(api: &LuaApi, l: LuaState, idx: c_int) -> Option<String> {
    let mut len: usize = 0;
    let ptr = (api.tolstring)(l, idx, &mut len);
    if ptr.is_null() {
        return None;
    }
    let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
    Some(String::from_utf8_lossy(bytes).into_owned())
}

pub unsafe fn push_str(api: &LuaApi, l: LuaState, s: &str) {
    (api.pushlstring)(l, s.as_ptr() as *const c_char, s.len());
}

pub unsafe fn pop(api: &LuaApi, l: LuaState, n: c_int) {
    (api.settop)(l, -n - 1);
}
