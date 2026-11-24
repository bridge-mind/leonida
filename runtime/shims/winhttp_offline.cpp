// winhttp_offline: WinHTTP for a mapped title, with no network.
//
// The package's XCurl.dll and libHttpClient.GDK.dll do their HTTP through WinHTTP: between them they import 16
// functions from winhttp.dll and nothing from Winsock (docs/runtime_imports.md, "WinHTTP offline"). Bound to the host's
// winhttp.dll a mapped title could reach the network, and nothing here goes online. So winhttp.dll is a module the
// shims own (a LoadLibrary of it gets a pseudo-handle, library_loader.cpp) and every function a title can reach is
// answered in-process:
//   - WinHttpOpen, WinHttpConnect and WinHttpOpenRequest return local handles. WinHttpConnect resolves no name and
//     opens no socket.
//   - WinHttpSendRequest fails with ERROR_WINHTTP_CANNOT_CONNECT. On a session opened with WINHTTP_FLAG_ASYNC it
//     returns TRUE and reports the failure the way WinHTTP does, through the status callback:
//     WINHTTP_CALLBACK_STATUS_REQUEST_ERROR with WINHTTP_ASYNC_RESULT { API_SEND_REQUEST, ERROR_WINHTTP_CANNOT_CONNECT },
//     from a thread-pool thread, with the context the request was sent with (a context of 0 keeps the one set with
//     WINHTTP_OPTION_CONTEXT_VALUE).
//   - Reading a response (WinHttpReceiveResponse, WinHttpQueryHeaders, WinHttpQueryDataAvailable, WinHttpReadData) and
//     WinHttpWriteData fail with ERROR_WINHTTP_INCORRECT_HANDLE_STATE: no request ever got that far.
//   - Options, timeouts and request headers are accepted. WINHTTP_OPTION_CONTEXT_VALUE is kept per handle and
//     WINHTTP_OPTION_HANDLE_TYPE answered; WinHttpQueryOption reports ERROR_WINHTTP_INVALID_OPTION for the rest.
//   - A status callback set on a handle is inherited by the handles created under it, as in WinHTTP, and so is
//     the handle's WINHTTP_OPTION_CONTEXT_VALUE as it is when the child is made.
//     WINHTTP_CALLBACK_STATUS_HANDLE_CREATED is reported when a child handle is made, and
//     WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING once per handle, last, after WinHttpCloseHandle and any callback still
//     running for it.
//   - WinHttpCrackUrl only parses a string, so it is the host's.
// Each refused server is logged once ("[rt] winhttp: offline, refused <host>:<port>"), which says what a title tried
// to reach. Only images mapped by the loader see these: they are bound through the registry, the runtime's own code
// calls the host directly.
#include <windows.h>
#include <winhttp.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include "shims.h"

namespace rt::winhttp {
namespace {

enum class Kind { Session, Connect, Request };

struct Handle {
    Kind kind = Kind::Session;
    Handle* parent = nullptr;  // holds one reference on the parent
    bool async = false;
    std::atomic<long> refs{1};  // the handle's own reference, dropped by WinHttpCloseHandle
    std::atomic<bool> closed{false};

    std::mutex mutex;  // guards the three below
    WINHTTP_STATUS_CALLBACK callback = nullptr;
    DWORD notifications = 0;
    DWORD_PTR context = 0;

    std::wstring server;  // Connect: for the log
    INTERNET_PORT port = 0;
};

// Intentionally immortal: a mapped DLL can close handles from DLL_PROCESS_DETACH, after normal statics are gone.
std::mutex& live_mutex() {
    static auto* m = new std::mutex;
    return *m;
}
std::set<Handle*>& live() {
    static auto* s = new std::set<Handle*>;
    return *s;
}

HINTERNET to_hinternet(Handle* h) { return reinterpret_cast<HINTERNET>(h); }

// Takes a reference on a live handle of an accepted kind. On failure `error` says why: ERROR_INVALID_HANDLE for
// something that is not an open handle, ERROR_WINHTTP_INCORRECT_HANDLE_TYPE for the wrong kind.
Handle* acquire(HINTERNET handle, bool session, bool connect, bool request, DWORD& error) {
    Handle* h = reinterpret_cast<Handle*>(handle);
    std::lock_guard lock(live_mutex());
    if (!handle || live().count(h) == 0) {
        error = ERROR_INVALID_HANDLE;
        return nullptr;
    }
    const bool ok = (h->kind == Kind::Session && session) || (h->kind == Kind::Connect && connect) ||
                    (h->kind == Kind::Request && request);
    if (!ok) {
        error = ERROR_WINHTTP_INCORRECT_HANDLE_TYPE;
        return nullptr;
    }
    h->refs.fetch_add(1);
    return h;
}

void notify(Handle* h, DWORD status, void* info, DWORD length) {
    WINHTTP_STATUS_CALLBACK callback;
    DWORD notifications;
    DWORD_PTR context;
    {
        std::lock_guard lock(h->mutex);
        callback = h->callback;
        notifications = h->notifications;
        context = h->context;
    }
    // The WINHTTP_CALLBACK_FLAG_* bits are the status values they select.
    if (callback && callback != WINHTTP_INVALID_STATUS_CALLBACK && (notifications & status) != 0)
        callback(to_hinternet(h), context, status, info, length);
}

void release(Handle* h) {
    while (h && h->refs.fetch_sub(1) == 1) {
        HINTERNET self = to_hinternet(h);
        notify(h, WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING, &self, sizeof(self));
        Handle* parent = h->parent;
        delete h;
        h = parent;  // the child's reference on its parent
    }
}

HINTERNET make_handle(Kind kind, Handle* parent, bool async, LPCWSTR server = nullptr, INTERNET_PORT port = 0) {
    std::wstring host = server ? server : L"";
    auto* h = new (std::nothrow) Handle;
    if (!h) {
        if (parent) release(parent);
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    // Publish connection metadata before live() / HANDLE_CREATED: the callback can use or close the handle inline.
    h->server = std::move(host);
    h->port = port;
    h->kind = kind;
    h->parent = parent;
    h->async = async;
    if (parent) {
        // A callback set on a handle is inherited by the handles created under it, and so is the context value
        // (WINHTTP_OPTION_CONTEXT_VALUE), as the host's winhttp.dll does: what the parent has at this moment.
        std::lock_guard lock(parent->mutex);
        h->callback = parent->callback;
        h->notifications = parent->notifications;
        h->context = parent->context;
    }
    {
        std::lock_guard lock(live_mutex());
        live().insert(h);
    }
    if (parent) {
        h->refs.fetch_add(1);  // held while HANDLE_CREATED runs, so HANDLE_CLOSING comes after it
        HINTERNET self = to_hinternet(h);
        notify(h, WINHTTP_CALLBACK_STATUS_HANDLE_CREATED, &self, sizeof(self));
        const bool closed = h->closed.load();
        release(h);
        if (closed) {
            SetLastError(ERROR_WINHTTP_OPERATION_CANCELLED);
            return nullptr;
        }
    }
    return to_hinternet(h);
}

std::wstring server_of(const Handle* request) {
    const Handle* connect = request ? request->parent : nullptr;
    if (!connect) return L"?";
    return connect->server + L":" + std::to_wstring(connect->port);
}

void log_refused(const Handle* request) {
    static auto* logged = new std::set<std::wstring>;
    static auto* mutex = new std::mutex;
    const std::wstring server = server_of(request);
    std::lock_guard lock(*mutex);
    if (logged->insert(server).second) std::fwprintf(stderr, L"[rt] winhttp: offline, refused %ls\n", server.c_str());
}

void CALLBACK send_failed(PTP_CALLBACK_INSTANCE, void* context) {
    Handle* h = static_cast<Handle*>(context);
    if (!h->closed.load()) {
        WINHTTP_ASYNC_RESULT result{};
        result.dwResult = API_SEND_REQUEST;
        result.dwError = ERROR_WINHTTP_CANNOT_CONNECT;
        notify(h, WINHTTP_CALLBACK_STATUS_REQUEST_ERROR, &result, sizeof(result));
    }
    release(h);  // the reference WinHttpSendRequest took for this callback
}

BOOL fail(DWORD error) {
    SetLastError(error);
    return FALSE;
}

}  // namespace

HINTERNET WINAPI open_session(LPCWSTR, DWORD, LPCWSTR, LPCWSTR, DWORD flags) {
    static std::once_flag once;
    std::call_once(once, [] {
        std::fprintf(stderr, "[rt] winhttp: offline; requests fail with ERROR_WINHTTP_CANNOT_CONNECT\n");
    });
    return make_handle(Kind::Session, nullptr, (flags & WINHTTP_FLAG_ASYNC) != 0);
}

HINTERNET WINAPI open_connection(HINTERNET session, LPCWSTR server, INTERNET_PORT port, DWORD) {
    DWORD error = 0;
    Handle* s = acquire(session, true, false, false, error);
    if (!s) {
        SetLastError(error);
        return nullptr;
    }
    if (!server || !*server) {
        release(s);
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    return make_handle(Kind::Connect, s, s->async, server, port);  // keeps the reference on the session
}

HINTERNET WINAPI open_request(HINTERNET connection, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR*, DWORD) {
    DWORD error = 0;
    Handle* c = acquire(connection, false, true, false, error);
    if (!c) {
        SetLastError(error);
        return nullptr;
    }
    return make_handle(Kind::Request, c, c->async);  // keeps the reference on the connection
}

BOOL WINAPI send_request(HINTERNET request, LPCWSTR, DWORD, LPVOID, DWORD, DWORD, DWORD_PTR context) {
    DWORD error = 0;
    Handle* r = acquire(request, false, false, true, error);
    if (!r) return fail(error);
    if (context) {
        // The context this request's callbacks carry. 0 keeps the one set with WINHTTP_OPTION_CONTEXT_VALUE, as the
        // host's winhttp.dll does (measured on Windows 11: WinHttpQueryOption and every callback still give it).
        std::lock_guard lock(r->mutex);
        r->context = context;
    }
    log_refused(r);
    if (!r->async) {
        release(r);
        return fail(ERROR_WINHTTP_CANNOT_CONNECT);
    }
    // Asynchronous: the failure arrives through the callback, never on the calling thread, as with WinHTTP. The
    // reference taken above is the callback's and is dropped there.
    if (!TrySubmitThreadpoolCallback(&send_failed, r, nullptr)) {
        const DWORD submit_error = GetLastError();
        release(r);
        return fail(submit_error ? submit_error : ERROR_NOT_ENOUGH_MEMORY);
    }
    SetLastError(ERROR_SUCCESS);
    return TRUE;
}

// No request was ever sent, so there is no response to read and nothing to write to.
BOOL WINAPI receive_response(HINTERNET request, LPVOID) {
    DWORD error = 0;
    Handle* r = acquire(request, false, false, true, error);
    if (!r) return fail(error);
    release(r);
    return fail(ERROR_WINHTTP_INCORRECT_HANDLE_STATE);
}

BOOL WINAPI query_data_available(HINTERNET request, LPDWORD) { return receive_response(request, nullptr); }
BOOL WINAPI read_data(HINTERNET request, LPVOID, DWORD, LPDWORD) { return receive_response(request, nullptr); }
BOOL WINAPI write_data(HINTERNET request, LPCVOID, DWORD, LPDWORD) { return receive_response(request, nullptr); }
BOOL WINAPI query_headers(HINTERNET request, DWORD, LPCWSTR, LPVOID, LPDWORD, LPDWORD) {
    return receive_response(request, nullptr);
}

// RFC 7230 tchar grammar for HTTP header field-names (matches native WinHttpAddRequestHeaders):
// printable ASCII 0x21..0x7E excluding HTTP token separators ()<>@,;:\"/[]?={}.
bool is_header_name_char(wchar_t ch) {
    if (ch <= 0x20 || ch >= 0x7F) return false;
    switch (ch) {
        case L'(': case L')': case L'<': case L'>': case L'@':
        case L',': case L';': case L':': case L'\\': case L'"':
        case L'/': case L'[': case L']': case L'?': case L'=':
        case L'{': case L'}':
            return false;
        default:
            return true;
    }
}

// Headers WinHTTP would reject. Measured against the host's winhttp.dll (Windows 10 19045 / Windows 11 26100):
// a null or empty block (count == 0), embedded NULs, a non-empty line with no ':', an empty header name (": v"),
// or any non-token character (whitespace, control, separator, or non-ASCII) in a header name fail with
// ERROR_INVALID_PARAMETER; a non-empty block containing only blank lines ("\r\n") is accepted by the host.
// A length of (DWORD)-1 means the text is NUL-terminated. Nothing is stored: no request is ever sent.
bool header_block_valid(LPCWSTR headers, DWORD length) {
    if (!headers) return false;
    const size_t count = length == static_cast<DWORD>(-1) ? wcslen(headers) : static_cast<size_t>(length);
    if (count == 0) return false;
    size_t i = 0;
    while (i < count) {
        if (headers[i] == L'\0') return false;
        const size_t line_start = i;
        while (i < count && headers[i] != L'\r' && headers[i] != L'\n') {
            if (headers[i] == L'\0') return false;
            ++i;
        }
        const size_t line_end = i;
        if (i < count && headers[i] == L'\r') ++i;
        if (i < count && headers[i] == L'\n') ++i;
        if (line_end == line_start) continue;
        size_t colon = line_start;
        while (colon < line_end && headers[colon] != L':') ++colon;
        if (colon == line_start || colon == line_end) return false;
        for (size_t k = line_start; k < colon; ++k) {
            if (!is_header_name_char(headers[k])) return false;
        }
    }
    return true;
}

BOOL WINAPI add_request_headers(HINTERNET request, LPCWSTR headers, DWORD length, DWORD) {
    DWORD error = 0;
    Handle* r = acquire(request, false, false, true, error);
    if (!r) return fail(error);
    release(r);
    if (!header_block_valid(headers, length)) return fail(ERROR_INVALID_PARAMETER);
    return TRUE;
}

// Measured the same way: valid on session and request handles (connect handles fail with
// ERROR_WINHTTP_INCORRECT_HANDLE_TYPE); 0 and -1 (infinite) are accepted; any other negative value, in any of the
// four, fails with ERROR_INVALID_PARAMETER.
BOOL WINAPI set_timeouts(HINTERNET handle, int resolve, int connect, int send, int receive) {
    DWORD error = 0;
    Handle* h = acquire(handle, true, false, true, error);
    if (!h) return fail(error);
    release(h);
    if (resolve < -1 || connect < -1 || send < -1 || receive < -1) return fail(ERROR_INVALID_PARAMETER);
    return TRUE;
}

BOOL WINAPI set_option(HINTERNET handle, DWORD option, LPVOID buffer, DWORD length) {
    if (!handle) {
        if (option == WINHTTP_OPTION_CONTEXT_VALUE) return fail(ERROR_WINHTTP_INCORRECT_HANDLE_TYPE);
        return TRUE;  // process-wide options: accepted, nothing to apply
    }
    DWORD error = 0;
    Handle* h = acquire(handle, true, true, true, error);
    if (!h) return fail(error);
    if (option == WINHTTP_OPTION_CONTEXT_VALUE) {
        if (!buffer) {
            release(h);
            return fail(ERROR_INVALID_PARAMETER);
        }
        if (length < sizeof(DWORD_PTR)) {
            release(h);
            return fail(ERROR_INSUFFICIENT_BUFFER);
        }
        std::lock_guard lock(h->mutex);
        h->context = *static_cast<const DWORD_PTR*>(buffer);
    }
    release(h);
    return TRUE;
}

BOOL WINAPI query_option(HINTERNET handle, DWORD option, LPVOID buffer, LPDWORD length) {
    if (!length) return fail(ERROR_INVALID_PARAMETER);
    DWORD error = 0;
    Handle* h = acquire(handle, true, true, true, error);
    if (!h) return fail(error);
    DWORD_PTR value_ptr = 0;
    DWORD value_dword = 0;
    const void* value = nullptr;
    DWORD size = 0;
    if (option == WINHTTP_OPTION_CONTEXT_VALUE) {
        std::lock_guard lock(h->mutex);
        value_ptr = h->context;
        value = &value_ptr;
        size = sizeof(value_ptr);
    } else if (option == WINHTTP_OPTION_HANDLE_TYPE) {
        value_dword = h->kind == Kind::Session   ? WINHTTP_HANDLE_TYPE_SESSION
                      : h->kind == Kind::Connect ? WINHTTP_HANDLE_TYPE_CONNECT
                                                 : WINHTTP_HANDLE_TYPE_REQUEST;
        value = &value_dword;
        size = sizeof(value_dword);
    }
    release(h);
    if (!value) return fail(ERROR_WINHTTP_INVALID_OPTION);
    if (!buffer || *length < size) {
        *length = size;
        return fail(ERROR_INSUFFICIENT_BUFFER);
    }
    std::memcpy(buffer, value, size);
    *length = size;
    return TRUE;
}

WINHTTP_STATUS_CALLBACK WINAPI set_status_callback(HINTERNET handle, WINHTTP_STATUS_CALLBACK callback,
                                                   DWORD notifications, DWORD_PTR) {
    if (!handle) {
        SetLastError(ERROR_WINHTTP_INCORRECT_HANDLE_TYPE);
        return WINHTTP_INVALID_STATUS_CALLBACK;
    }
    DWORD error = 0;
    Handle* h = acquire(handle, true, true, true, error);
    if (!h) {
        SetLastError(error);
        return WINHTTP_INVALID_STATUS_CALLBACK;
    }
    WINHTTP_STATUS_CALLBACK previous;
    {
        std::lock_guard lock(h->mutex);
        previous = h->callback;
        h->callback = callback;
        h->notifications = callback ? notifications : 0;
    }
    release(h);
    return previous;
}

BOOL WINAPI close_handle(HINTERNET handle) {
    Handle* h = reinterpret_cast<Handle*>(handle);
    {
        std::lock_guard lock(live_mutex());
        if (!handle || live().erase(h) == 0) return fail(ERROR_INVALID_HANDLE);
    }
    h->closed.store(true);
    release(h);  // the handle's own reference; HANDLE_CLOSING follows when nothing else holds it
    return TRUE;
}

BOOL WINAPI crack_url(LPCWSTR url, DWORD length, DWORD flags, LPURL_COMPONENTS components) {
    // Parsing only: the host's, loaded on first use so the runtime does not depend on winhttp.dll otherwise.
    using CrackUrlFn = BOOL(WINAPI*)(LPCWSTR, DWORD, DWORD, LPURL_COMPONENTS);
    static CrackUrlFn host = [] {
        const HMODULE m = LoadLibraryExW(L"winhttp.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        return m ? reinterpret_cast<CrackUrlFn>(GetProcAddress(m, "WinHttpCrackUrl")) : nullptr;
    }();
    if (!host) return fail(ERROR_PROC_NOT_FOUND);
    return host(url, length, flags, components);
}

}  // namespace rt::winhttp

namespace {
using namespace rt::winhttp;
}  // namespace

SHIM_TABLE(winhttp_offline,
    {"winhttp.dll", "WinHttpOpen", reinterpret_cast<void*>(&open_session)},
    {"winhttp.dll", "WinHttpConnect", reinterpret_cast<void*>(&open_connection)},
    {"winhttp.dll", "WinHttpOpenRequest", reinterpret_cast<void*>(&open_request)},
    {"winhttp.dll", "WinHttpSendRequest", reinterpret_cast<void*>(&send_request)},
    {"winhttp.dll", "WinHttpReceiveResponse", reinterpret_cast<void*>(&receive_response)},
    {"winhttp.dll", "WinHttpQueryDataAvailable", reinterpret_cast<void*>(&query_data_available)},
    {"winhttp.dll", "WinHttpReadData", reinterpret_cast<void*>(&read_data)},
    {"winhttp.dll", "WinHttpWriteData", reinterpret_cast<void*>(&write_data)},
    {"winhttp.dll", "WinHttpQueryHeaders", reinterpret_cast<void*>(&query_headers)},
    {"winhttp.dll", "WinHttpAddRequestHeaders", reinterpret_cast<void*>(&add_request_headers)},
    {"winhttp.dll", "WinHttpSetTimeouts", reinterpret_cast<void*>(&set_timeouts)},
    {"winhttp.dll", "WinHttpSetOption", reinterpret_cast<void*>(&set_option)},
    {"winhttp.dll", "WinHttpQueryOption", reinterpret_cast<void*>(&query_option)},
    {"winhttp.dll", "WinHttpSetStatusCallback", reinterpret_cast<void*>(&set_status_callback)},
    {"winhttp.dll", "WinHttpCloseHandle", reinterpret_cast<void*>(&close_handle)},
    {"winhttp.dll", "WinHttpCrackUrl", reinterpret_cast<void*>(&crack_url)},
);
