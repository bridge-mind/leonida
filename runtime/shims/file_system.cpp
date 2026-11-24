// file_system: the file API of a mapped title, seen through the console's drives.
//
// A GDK title names its files on the console's drives, "GAME:\\data\\x.sps" being one in the installed game folder, so
// the imports it makes of CreateFile*, FindFirstFile*/FindNextFile*/FindClose, GetFileAttributes*/GetFileAttributesEx*,
// DeleteFile*, CreateDirectory*/RemoveDirectory*, MoveFile*, CopyFile*/CopyFile2 and SetFileAttributes* are answered
// here (docs/runtime_imports.md, "Console paths and the write overlay"; the prefixes come from ivall/xcl,
// GPL-2.0-only, as facts). The rules:
//   - "GAME:" (any case, "\\" or "/") maps under --game-dir, "G:\\" is the same folder and "T:\\" a scratch folder in
//     the overlay (RT-218); any other path is a host path and goes to the host;
//   - an open that can write, and DeleteFile, go to the overlay (--overlay, default <game-dir>\\overlay), so the
//     package stays exactly as installed. A write-open of a package file copies it into the overlay first, so
//     OPEN_EXISTING keeps its bytes; a delete of a package file leaves an empty marker under
//     <overlay>\\.gta6rt-deleted, and reads and listings treat the name as gone. The marker is never removed and a
//     deleted name is never copied up: a re-created file is its overlay copy, which reads and listings already
//     prefer, while an open of a deleted name that fails leaves the name deleted;
//   - reads prefer the overlay copy and then the package;
//   - FindFirstFile merges the two folders, the overlay winning on a name, and hides deleted names and the marker
//     folder;
//   - folders are made in the overlay; a removed package folder is hidden by a marker, a moved package file is
//     copied into the overlay and its old name deleted, and attributes are set on the overlay copy;
//   - a path that climbs above the root with ".." or names another drive is refused with ERROR_INVALID_NAME, so a
//     title cannot reach outside the package and the overlay.
// Only images mapped by the loader see these: they are bound through the registry, the runtime's own code calls the
// host directly.
#include <windows.h>
#include <intrin.h>

#include <atomic>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "file_system.h"
#include "shims.h"
#include "startup_trace.h"

namespace rt::fs {
namespace {

constexpr wchar_t kGamePrefix[] = L"game:";
constexpr size_t kGamePrefixLength = 5;
constexpr wchar_t kDeletedDir[] = L".gta6rt-deleted";
constexpr wchar_t kScratchDir[] = L".gta6rt-temp";  // what "T:\" names, inside the overlay
// A removed package folder is marked by this file inside its marker folder: the marker folder itself can already
// exist because files below it were deleted, and that alone does not hide the folder.
constexpr wchar_t kRemovedDirMarker[] = L".gta6rt-dir";
// Copy-up writes here first and then renames into place, so a half-copied file never sits under the title's name.
constexpr wchar_t kCopyDir[] = L".gta6rt-copy";

// The overlay root's own folders, which a title never sees in a listing.
bool internal_dir(const wchar_t* name) {
    return _wcsicmp(name, kDeletedDir) == 0 || _wcsicmp(name, kCopyDir) == 0 || _wcsicmp(name, kScratchDir) == 0 ||
           _wcsicmp(name, kPersistentStorageDir) == 0;
}

std::mutex g_mutex;
std::filesystem::path g_game_dir;
std::filesystem::path g_overlay_dir;

struct Roots {
    std::filesystem::path game, overlay;
};

Roots roots() {
    std::lock_guard lock(g_mutex);
    return {g_game_dir, g_overlay_dir};
}

std::wstring lower(std::wstring s) {
    for (auto& c : s) c = static_cast<wchar_t>(std::towlower(c));
    return s;
}

bool same_name(const wchar_t* a, const wchar_t* b) { return _wcsicmp(a, b) == 0; }

// The narrow API (CreateFileA and friends) takes ANSI code page text, as the CRT does; the loader's own paths are
// handled as wide strings everywhere else.
std::wstring widen_ansi(const char* s) {
    if (!s) return {};
    const int n = MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
    if (n <= 1) return {};
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_ACP, 0, s, -1, w.data(), n);
    return w;
}

void narrow_ansi(const std::wstring& wide, char* out, size_t capacity) {
    if (capacity == 0) return;
    const int n = WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1 || static_cast<size_t>(n) > capacity) {
        out[0] = '\0';
        return;
    }
    WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, out, static_cast<int>(capacity), nullptr, nullptr);
}

enum class Parsed { Host, Console, Refused };

// Splits a console path into the components below the root. A component that names a drive or a stream ("C:", "x:y")
// and a ".." that would leave the root are refused: the package and the overlay are the only folders a title may name.
bool is_separator(wchar_t c) { return c == L'\\' || c == L'/'; }

// True when `path` is `root` or lies below it, compared by name (case-insensitive, either separator).
bool lies_under(const std::wstring& path, const std::filesystem::path& root) {
    std::wstring base = lower(root.native());
    for (auto& c : base) if (c == L'/') c = L'\\';
    while (!base.empty() && base.back() == L'\\') base.pop_back();
    if (base.empty()) return false;
    std::wstring p = lower(path);
    for (auto& c : p) if (c == L'/') c = L'\\';
    return p.compare(0, base.size(), base) == 0 && (p.size() == base.size() || p[base.size()] == L'\\');
}

// "G:\x" is the installed package, like "GAME:\x", and "T:\x" a scratch folder the title may write to, the way the
// other console layers map those two drives (RT-218). Only the executable can say which spelling the title uses, so
// a host drive of the same letter must keep working: a path under --game-dir or the overlay (a package on drive G:,
// or a path GetModuleFileName handed back) stays a host path, and so does every G:/T: path while no roots are
// configured. The scratch folder lives in the overlay, as `.gta6rt-temp`.
enum class Drive { None, Game, Scratch };

Drive console_drive(const std::wstring& path, size_t& prefix_length) {
    if (path.size() >= kGamePrefixLength && lower(path.substr(0, kGamePrefixLength)) == kGamePrefix) {
        prefix_length = kGamePrefixLength;
        return Drive::Game;
    }
    if (path.size() < 2 || path[1] != L':' || (path.size() > 2 && !is_separator(path[2]))) return Drive::None;
    const wchar_t letter = static_cast<wchar_t>(std::towlower(path[0]));
    if (letter != L'g' && letter != L't') return Drive::None;
    const Roots r = roots();
    if (r.game.empty() && r.overlay.empty()) return Drive::None;
    if (lies_under(path, r.game) || lies_under(path, r.overlay)) return Drive::None;
    prefix_length = 2;
    return letter == L'g' ? Drive::Game : Drive::Scratch;
}

Parsed parse_console(const std::wstring& path, std::vector<std::wstring>& parts) {
    size_t prefix_length = 0;
    const Drive drive = console_drive(path, prefix_length);
    if (drive == Drive::None) return Parsed::Host;
    parts.clear();
    if (drive == Drive::Scratch) {
        parts.push_back(kScratchDir);
        const std::filesystem::path overlay = roots().overlay;
        const DWORD last_error = ::GetLastError();
        std::error_code ec;
        if (!overlay.empty()) std::filesystem::create_directories(overlay / kScratchDir, ec);
        ::SetLastError(last_error);
    }
    const size_t floor = parts.size();  // ".." may not climb out of the drive's own root
    std::wstring part;
    auto flush = [&]() -> bool {
        if (part.empty() || part == L".") {
            part.clear();
            return true;
        }
        if (part == L"..") {
            if (parts.size() <= floor) return false;
            parts.pop_back();
        } else if (part.find(L':') != std::wstring::npos) {
            return false;
        } else {
            parts.push_back(part);
        }
        part.clear();
        return true;
    };
    for (size_t i = prefix_length; i < path.size(); ++i) {
        const wchar_t c = path[i];
        if (is_separator(c)) {
            if (!flush()) return Parsed::Refused;
        } else {
            part.push_back(c);
        }
    }
    if (!flush()) return Parsed::Refused;
    // The overlay's own folders are not part of the title's GAME: tree: hidden from its listings (internal_dir) and
    // refused as an explicit path too, in every spelling the loop above normalises (case, slashes, "." and "..").
    // T: still names the scratch folder: its first part is pushed above, not parsed from the title's path.
    if (drive == Drive::Game && !parts.empty() && internal_dir(parts.front().c_str())) return Parsed::Refused;
    return Parsed::Console;
}

std::filesystem::path join(const std::filesystem::path& root, const std::vector<std::wstring>& parts) {
    std::filesystem::path p = root;
    for (const auto& part : parts) p /= part;
    return p;
}

bool drive_root(const std::wstring& path, const std::vector<std::wstring>& parts) {
    size_t prefix_length = 0;
    return parts.empty() || (parts.size() == 1 && console_drive(path, prefix_length) == Drive::Scratch);
}

struct Target {
    std::filesystem::path path;
    DWORD error = 0;  // non-zero: the caller fails with this error and does not touch the host
};

Target fail(DWORD error) {
    Target t;
    t.error = error;
    return t;
}

// The overlay keeps one empty marker per deleted package path, outside the mirrored tree so it cannot collide with a
// file the title creates.
std::filesystem::path marker(const std::filesystem::path& overlay, const std::vector<std::wstring>& parts) {
    return join(overlay / kDeletedDir, parts);
}

// A deleted file leaves an empty file at its marker path and a removed folder leaves kRemovedDirMarker inside its
// marker folder.
bool deleted(const std::filesystem::path& overlay, const std::vector<std::wstring>& parts) {
    if (overlay.empty() || parts.empty()) return false;
    std::error_code ec;
    // Only regular files are deletion markers; parent directories inside .gta6rt-deleted/ exist only to hold markers
    // for nested deleted files and must not hide the corresponding ancestor directories in the package.
    const std::filesystem::path m = marker(overlay, parts);
    return std::filesystem::is_regular_file(m, ec) || std::filesystem::is_regular_file(m / kRemovedDirMarker, ec);
}

// Writes an empty hidden marker file. On failure `error` holds the reason.
bool write_marker(const std::filesystem::path& mark, DWORD& error) {
    std::error_code ec;
    std::filesystem::create_directories(mark.parent_path(), ec);
    if (ec) {
        error = static_cast<DWORD>(ec.value());
        return false;
    }
    const HANDLE h = ::CreateFileW(mark.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                   nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_HIDDEN, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        error = ::GetLastError();
        return false;
    }
    ::CloseHandle(h);
    return true;
}

// Flushes a file the shim wrote on the title's behalf, for MOVEFILE_WRITE_THROUGH. On failure `error` holds the reason.
bool flush_file(const std::filesystem::path& path, DWORD& error) {
    const HANDLE h = ::CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        error = ::GetLastError();
        return false;
    }
    const BOOL ok = ::FlushFileBuffers(h);
    if (!ok) error = ::GetLastError();
    ::CloseHandle(h);
    return ok != FALSE;
}

// What a console path is in the title's view: the overlay copy if there is one, else the package's unless deleted.
enum class Kind { None, File, Directory };
struct View {
    Kind kind = Kind::None;
    bool in_overlay = false;  // the overlay holds the name
    bool in_package = false;  // the package holds the name and it is not deleted
};

View view(const Roots& r, const std::vector<std::wstring>& parts) {
    View v;
    std::error_code ec;
    if (parts.empty()) {  // "GAME:" itself
        v.kind = Kind::Directory;
        v.in_package = !r.game.empty();
        return v;
    }
    if (!r.overlay.empty()) {
        const std::filesystem::path p = join(r.overlay, parts);
        if (std::filesystem::exists(p, ec)) {
            v.in_overlay = true;
            v.kind = std::filesystem::is_directory(p, ec) ? Kind::Directory : Kind::File;
        }
    }
    if (!r.game.empty() && !deleted(r.overlay, parts)) {
        const std::filesystem::path p = join(r.game, parts);
        if (std::filesystem::exists(p, ec)) {
            v.in_package = true;
            if (v.kind == Kind::None) v.kind = std::filesystem::is_directory(p, ec) ? Kind::Directory : Kind::File;
        }
    }
    return v;
}

bool is_folder(const Roots& r, const std::vector<std::wstring>& parts) { return view(r, parts).kind == Kind::Directory; }

// Whether two console paths name the same entry: names compare without case, as the lookups do.
bool same_entry(const std::vector<std::wstring>& a, const std::vector<std::wstring>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!same_name(a[i].c_str(), b[i].c_str())) return false;
    return true;
}

std::vector<std::wstring> parent_of(const std::vector<std::wstring>& parts) {
    return parts.empty() ? parts : std::vector<std::wstring>(parts.begin(), parts.end() - 1);
}

// Whether a folder has nothing in the title's view: no overlay entry and no package entry that is not deleted.
bool empty_in_view(const Roots& r, const std::vector<std::wstring>& parts) {
    std::error_code ec;
    if (!r.overlay.empty()) {
        const std::filesystem::path p = join(r.overlay, parts);
        for (auto it = std::filesystem::directory_iterator(p, ec); !ec && it != std::filesystem::directory_iterator();
             it.increment(ec)) {
            if (parts.empty() && internal_dir(it->path().filename().c_str())) continue;
            return false;
        }
    }
    ec.clear();
    if (!r.game.empty()) {
        const std::filesystem::path p = join(r.game, parts);
        for (auto it = std::filesystem::directory_iterator(p, ec); !ec && it != std::filesystem::directory_iterator();
             it.increment(ec)) {
            std::vector<std::wstring> child = parts;
            child.push_back(it->path().filename().wstring());
            if (!deleted(r.overlay, child)) return false;
        }
    }
    return true;
}

BOOL fail_bool(DWORD error) {
    SetLastError(error);
    return FALSE;
}

Target read_target(const std::wstring& path) {
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(path, parts);
    if (parsed == Parsed::Host) {
        Target t;
        t.path = std::filesystem::path(path);
        return t;
    }
    if (parsed == Parsed::Refused) return fail(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty()) return fail(ERROR_PATH_NOT_FOUND);

    std::error_code ec;
    if (!r.overlay.empty()) {
        const std::filesystem::path overlay_path = join(r.overlay, parts);
        if (std::filesystem::exists(overlay_path, ec)) {
            Target t;
            t.path = overlay_path;
            return t;
        }
    }
    // A package file the title deleted is gone even though the package still holds it.
    if (deleted(r.overlay, parts)) return fail(ERROR_FILE_NOT_FOUND);

    Target t;
    t.path = join(r.game, parts);
    return t;
}

Target write_target(const std::wstring& path, bool copy_up) {
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(path, parts);
    if (parsed == Parsed::Host) {
        Target t;
        t.path = std::filesystem::path(path);
        return t;
    }
    if (parsed == Parsed::Refused) return fail(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.overlay.empty()) return fail(ERROR_PATH_NOT_FOUND);
    if (parts.empty()) return fail(ERROR_INVALID_NAME);  // "GAME:" is a folder, not a file

    // The folder has to exist in the title's view, as CreateDirectory, MoveFile and CopyFile require: without the
    // check a failed open left the missing folders behind, and a create succeeded in a folder that was never made,
    // below a package file (which then read as a folder) and inside a folder the title had removed. The overlay may
    // still lack a folder only the package has, so it is mirrored here, which the title cannot see.
    // T:'s scratch folder is the exception: it creates its parent folders, as merged with RT-218.
    const bool scratch = !parts.empty() && same_name(parts.front().c_str(), kScratchDir);
    if (!scratch && !is_folder(r, parent_of(parts))) return fail(ERROR_PATH_NOT_FOUND);

    Target t;
    t.path = join(r.overlay, parts);

    std::error_code ec;
    std::filesystem::create_directories(t.path.parent_path(), ec);
    if (ec) {
        t.error = static_cast<DWORD>(ec.value());
        return t;
    }

    // A deleted package file is not there to copy: copying it up would bring the name back, and an open that keeps
    // the contents (OPEN_EXISTING, TRUNCATE_EXISTING) has to fail on it instead. CREATE_ALWAYS and CREATE_NEW still
    // make a new overlay file, which reads prefer over the package and so over the marker.
    const std::filesystem::path package_path = join(r.game, parts);
    // One copy-up at a time: a second thread writing the same file waits here and then finds the first one's copy, so
    // it neither fails on the taken name nor collides with that file's open through a rename of its own.
    static std::mutex s_copy_mutex;
    std::unique_lock copy_lock(s_copy_mutex, std::defer_lock);
    if (copy_up) copy_lock.lock();
    if (copy_up && !deleted(r.overlay, parts) && !std::filesystem::exists(t.path, ec) &&
        std::filesystem::is_regular_file(package_path, ec)) {
        // Copied to a private name, then renamed into place: a process killed mid-copy leaves only a hidden leftover,
        // not a truncated file that reads would prefer over the package.
        static unsigned s_copies = 0;  // under s_copy_mutex
        const std::filesystem::path staging = r.overlay / kCopyDir;
        std::filesystem::create_directories(staging, ec);
        const std::filesystem::path tmp =
            staging / (std::to_wstring(::GetCurrentProcessId()) + L"-" + std::to_wstring(++s_copies));
        // Use CopyFileW for staging because MSVC's std::filesystem::copy_file calls CopyFile2, which Wine 9 lacks.
        if (ec) {
            t.error = static_cast<DWORD>(ec.value());
        } else if (!::CopyFileW(package_path.c_str(), tmp.c_str(), FALSE)) {
            t.error = ::GetLastError();
        }
        if (t.error) {
            std::filesystem::remove(tmp, ec);
            return t;
        }
        if (!::MoveFileExW(tmp.c_str(), t.path.c_str(), 0)) {
            // The name taken means the title created the file itself meanwhile (CREATE_NEW and CREATE_ALWAYS don't
            // copy up, so they don't take the lock); that file is what it wrote, so keep it.
            const DWORD error = ::GetLastError();
            std::filesystem::remove(tmp, ec);
            if (!std::filesystem::exists(t.path, ec)) t.error = error;
        }
    }
    return t;
}

// MAXIMUM_ALLOWED names no right and grants every one the file allows, write included, so it is a write-open: opened
// as a read it handed out a handle to the package file itself, and a WriteFile through it changed the package.
bool writes(DWORD access, DWORD disposition) {
    constexpr DWORD kWriteAccess = GENERIC_WRITE | GENERIC_ALL | MAXIMUM_ALLOWED | FILE_WRITE_DATA | FILE_APPEND_DATA |
                                   FILE_WRITE_EA | FILE_WRITE_ATTRIBUTES | DELETE | WRITE_DAC | WRITE_OWNER;
    if ((access & kWriteAccess) != 0) return true;
    return disposition == CREATE_NEW || disposition == CREATE_ALWAYS || disposition == OPEN_ALWAYS ||
           disposition == TRUNCATE_EXISTING;
}

Target open_target(const std::wstring& path, DWORD access, DWORD disposition) {
    if (!writes(access, disposition)) return read_target(path);
    // OPEN_ALWAYS and TRUNCATE_EXISTING read what is already there, and a write-open must not lose a package file's
    // bytes; CREATE_ALWAYS and CREATE_NEW replace it.
    const bool keeps_contents =
        disposition == OPEN_EXISTING || disposition == OPEN_ALWAYS || disposition == TRUNCATE_EXISTING;
    return write_target(path, keeps_contents);
}

// ---- FindFirstFile: the overlay and the package merged into one listing
//
// The host enumerates each folder; the results are merged here, overlay first, so a name that exists in both appears
// once with the overlay's times and size, a name the title deleted does not appear at all, and the overlay's marker
// folder never does.
struct Enumeration {
    std::vector<WIN32_FIND_DATAW> entries;
    size_t next = 0;
};

std::mutex g_enum_mutex;
std::map<HANDLE, std::unique_ptr<Enumeration>> g_enumerations;  // keyed by the pseudo-handle handed to the title

HANDLE register_enumeration(std::unique_ptr<Enumeration> e) {
    std::lock_guard lock(g_enum_mutex);
    const HANDLE h = reinterpret_cast<HANDLE>(e.get());
    g_enumerations.emplace(h, std::move(e));
    return h;
}

bool unregister_enumeration(HANDLE h) {
    std::lock_guard lock(g_enum_mutex);
    return g_enumerations.erase(h) != 0;
}

bool collect(const std::wstring& pattern, const std::vector<std::wstring>& parts, std::vector<WIN32_FIND_DATAW>& out,
             DWORD& error) {
    // "GAME:" and "GAME:\\dir\\" name the folder itself: list everything in it.
    std::vector<std::wstring> dirs;
    std::wstring leaf;
    const bool trailing_separator = !pattern.empty() && (pattern.back() == L'\\' || pattern.back() == L'/');
    if (parts.empty() || trailing_separator) {
        dirs = parts;
        leaf = L"*";
    } else {
        dirs.assign(parts.begin(), parts.end() - 1);
        leaf = parts.back();
    }

    const Roots r = roots();
    if (r.game.empty()) {
        error = ERROR_PATH_NOT_FOUND;
        return false;
    }
    // The folder being listed has to exist in the title's view. The host would still enumerate a package folder the
    // title removed (its "." and ".."), and a folder that is missing is a missing path, not a missing file. A listing
    // below a file is a third answer on the host, ERROR_DIRECTORY.
    const Kind listed = view(r, dirs).kind;
    if (listed != Kind::Directory) {
        error = listed == Kind::File ? ERROR_DIRECTORY : ERROR_PATH_NOT_FOUND;
        return false;
    }

    // Every deletion marker for an entry of this folder lives inside its marker folder, so that folder is read once
    // instead of probing it twice per package entry (deleted()). An entry is deleted when its marker is a file, or a
    // folder holding kRemovedDirMarker: the rule deleted() applies. The lookup folds ASCII letters only, which matches
    // the file system's case-insensitive compare exactly when both names are ASCII. Any other name, a non-ASCII entry
    // or a marker that is not plain ASCII or is a reparse point, goes through deleted() and the file system's rule.
    auto ascii = [](const wchar_t* s) {
        for (; *s; ++s)
            if (*s > 0x7f) return false;
        return true;
    };
    auto ascii_key = [](const wchar_t* s) {
        std::wstring k = s;
        for (auto& c : k)
            if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c - L'A' + L'a');
        return k;
    };
    bool has_markers = false;
    bool markers_exact = true;
    std::unordered_set<std::wstring> deleted_here;
    if (!r.overlay.empty()) {
        const std::filesystem::path marks = marker(r.overlay, dirs);
        WIN32_FIND_DATAW m{};
        const HANDLE mh = ::FindFirstFileW((marks / L"*").c_str(), &m);
        if (mh != INVALID_HANDLE_VALUE) {
            has_markers = true;
            do {
                if (same_name(m.cFileName, L".") || same_name(m.cFileName, L"..")) continue;
                if (!ascii(m.cFileName) || (m.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
                    markers_exact = false;
                    continue;
                }
                if (!(m.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    deleted_here.insert(ascii_key(m.cFileName));
                    continue;
                }
                // A marker folder is a removed folder only with kRemovedDirMarker in it; without one it just holds the
                // markers of a nested delete. A missing sentinel is that ordinary answer (status() reports not_found
                // with an error code set); any other error leaves the decision to deleted() per entry.
                std::error_code ec;
                const auto st = std::filesystem::status(marks / m.cFileName / kRemovedDirMarker, ec);
                if (st.type() == std::filesystem::file_type::regular) deleted_here.insert(ascii_key(m.cFileName));
                else if (st.type() != std::filesystem::file_type::not_found && ec) markers_exact = false;
            } while (::FindNextFileW(mh, &m));
            if (::GetLastError() != ERROR_NO_MORE_FILES) markers_exact = false;  // an incomplete scan
            ::FindClose(mh);
        } else {
            // Only a missing marker folder means "nothing deleted here". Any other failure (a folder that cannot be
            // listed although its markers can be queried by path, for one) leaves the decision to deleted().
            const DWORD e = ::GetLastError();
            if (e != ERROR_FILE_NOT_FOUND && e != ERROR_PATH_NOT_FOUND) {
                has_markers = true;
                markers_exact = false;
            }
        }
    }
    auto entry_deleted = [&](const wchar_t* name) {
        if (!has_markers) return false;
        if (markers_exact && ascii(name)) return deleted_here.count(ascii_key(name)) != 0;
        std::vector<std::wstring> entry_parts = dirs;
        entry_parts.push_back(name);
        return deleted(r.overlay, entry_parts);
    };
    // Names already listed, ordered by _wcsicmp itself, so two names are one entry exactly when same_name says so, in
    // whatever locale the title set (a lower-cased key is not: under .UTF-8 _wcslwr_s folds É and é, _wcsicmp not).
    struct NoCase {
        bool operator()(const std::wstring& a, const std::wstring& b) const { return _wcsicmp(a.c_str(), b.c_str()) < 0; }
    };
    std::set<std::wstring, NoCase> seen;
    for (const auto& e : out) seen.insert(e.cFileName);

    auto add_root = [&](const std::filesystem::path& root, bool overlay_root) {
        if (root.empty()) return;
        std::filesystem::path search = join(root, dirs);
        search /= leaf;
        WIN32_FIND_DATAW found{};
        const HANDLE h = ::FindFirstFileW(search.c_str(), &found);
        if (h == INVALID_HANDLE_VALUE) return;
        do {
            if (overlay_root && internal_dir(found.cFileName)) continue;
            const bool dot = same_name(found.cFileName, L".") || same_name(found.cFileName, L"..");
            if (!overlay_root && !dot && entry_deleted(found.cFileName)) continue;
            if (seen.insert(found.cFileName).second) out.push_back(found);
        } while (::FindNextFileW(h, &found));
        ::FindClose(h);
    };
    add_root(r.overlay, true);
    add_root(r.game, false);

    if (out.empty()) {
        error = ERROR_FILE_NOT_FOUND;
        return false;
    }
    return true;
}

void to_ansi(const WIN32_FIND_DATAW& wide, WIN32_FIND_DATAA* ansi) {
    *ansi = WIN32_FIND_DATAA{};
    ansi->dwFileAttributes = wide.dwFileAttributes;
    ansi->ftCreationTime = wide.ftCreationTime;
    ansi->ftLastAccessTime = wide.ftLastAccessTime;
    ansi->ftLastWriteTime = wide.ftLastWriteTime;
    ansi->nFileSizeHigh = wide.nFileSizeHigh;
    ansi->nFileSizeLow = wide.nFileSizeLow;
    ansi->dwReserved0 = wide.dwReserved0;
    ansi->dwReserved1 = wide.dwReserved1;
    narrow_ansi(wide.cFileName, ansi->cFileName, MAX_PATH);
    narrow_ansi(wide.cAlternateFileName, ansi->cAlternateFileName, 14);
}

}  // namespace

void configure(const std::filesystem::path& game_dir, const std::filesystem::path& overlay_dir) {
    std::lock_guard lock(g_mutex);
    g_game_dir = game_dir;
    g_overlay_dir = overlay_dir;
}

std::filesystem::path game_dir() {
    std::lock_guard lock(g_mutex);
    return g_game_dir;
}

std::filesystem::path overlay_dir() {
    std::lock_guard lock(g_mutex);
    return g_overlay_dir;
}

// gta6rt --trace FILE: a failed open of a guest path goes to the startup trace as a "file_open_failed" event (the
// path the title passed, the Win32 error, the caller as module+RVA), so a boot that stops on a missing or refused
// file names it. Failures only, and at most 256 of them, so a title probing many optional files can't fill the file.
void note_open_failure(const char* api, LPCWSTR name, DWORD error, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::atomic<int> left{256};
    if (left.fetch_sub(1, std::memory_order_relaxed) <= 0) return;
    std::string utf8;
    if (const int n = WideCharToMultiByte(CP_UTF8, 0, name, -1, nullptr, 0, nullptr, nullptr); n > 1) {
        utf8.resize(static_cast<size_t>(n) - 1);
        WideCharToMultiByte(CP_UTF8, 0, name, -1, utf8.data(), n, nullptr, nullptr);
    }
    rt::startup_trace::event(
        "file_open_failed",
        std::string("\"api\":\"") + api + "\",\"path\":" + rt::startup_trace::quote(utf8) +
            ",\"error\":" + std::to_string(error) + ",\"caller\":" +
            rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller))));
}

HANDLE WINAPI create_file_w(LPCWSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES attributes,
                            DWORD disposition, DWORD flags, HANDLE template_file) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const Target t = open_target(name, access, disposition);
    if (t.error) {
        note_open_failure("CreateFileW", name, t.error, _ReturnAddress());
        SetLastError(t.error);
        return INVALID_HANDLE_VALUE;
    }
    const HANDLE h = ::CreateFileW(t.path.c_str(), access, share, attributes, disposition, flags, template_file);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        note_open_failure("CreateFileW", name, error, _ReturnAddress());
        SetLastError(error);
    }
    return h;
}

HANDLE WINAPI create_file_a(LPCSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES attributes,
                            DWORD disposition, DWORD flags, HANDLE template_file) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::wstring wide = widen_ansi(name);
    return create_file_w(wide.c_str(), access, share, attributes, disposition, flags, template_file);
}

HANDLE WINAPI create_file2(LPCWSTR name, DWORD access, DWORD share, DWORD disposition,
                           LPCREATEFILE2_EXTENDED_PARAMETERS params) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const Target t = open_target(name, access, disposition);
    if (t.error) {
        note_open_failure("CreateFile2", name, t.error, _ReturnAddress());
        SetLastError(t.error);
        return INVALID_HANDLE_VALUE;
    }
    const HANDLE h = ::CreateFile2(t.path.c_str(), access, share, disposition, params);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        note_open_failure("CreateFile2", name, error, _ReturnAddress());
        SetLastError(error);
    }
    return h;
}

DWORD WINAPI get_file_attributes_w(LPCWSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_FILE_ATTRIBUTES;
    }
    const Target t = read_target(name);
    if (t.error) {
        SetLastError(t.error);
        return INVALID_FILE_ATTRIBUTES;
    }
    return ::GetFileAttributesW(t.path.c_str());
}

DWORD WINAPI get_file_attributes_a(LPCSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_FILE_ATTRIBUTES;
    }
    const std::wstring wide = widen_ansi(name);
    return get_file_attributes_w(wide.c_str());
}

BOOL WINAPI get_file_attributes_ex_w(LPCWSTR name, GET_FILEEX_INFO_LEVELS level, LPVOID info) {
    if (!name || !info) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const Target t = read_target(name);
    if (t.error) {
        SetLastError(t.error);
        return FALSE;
    }
    return ::GetFileAttributesExW(t.path.c_str(), level, info);
}

BOOL WINAPI get_file_attributes_ex_a(LPCSTR name, GET_FILEEX_INFO_LEVELS level, LPVOID info) {
    if (!name || !info) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::wstring wide = widen_ansi(name);
    return get_file_attributes_ex_w(wide.c_str(), level, info);
}

HANDLE WINAPI find_first_file_w(LPCWSTR pattern, LPWIN32_FIND_DATAW data) {
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(pattern, parts);
    if (parsed == Parsed::Host) return ::FindFirstFileW(pattern, data);
    if (parsed == Parsed::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return INVALID_HANDLE_VALUE;
    }

    std::vector<WIN32_FIND_DATAW> entries;
    DWORD error = 0;
    if (!collect(pattern, parts, entries, error)) {
        SetLastError(error);
        return INVALID_HANDLE_VALUE;
    }
    auto e = std::make_unique<Enumeration>();
    e->entries = std::move(entries);
    e->next = 1;
    *data = e->entries[0];
    return register_enumeration(std::move(e));
}

HANDLE WINAPI find_first_file_ex_w(LPCWSTR pattern, FINDEX_INFO_LEVELS info_level, LPVOID data,
                                   FINDEX_SEARCH_OPS search_op, LPVOID search_filter, DWORD additional_flags) {
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(pattern, parts);
    if (parsed == Parsed::Host)
        return ::FindFirstFileExW(pattern, info_level, data, search_op, search_filter, additional_flags);
    if (parsed == Parsed::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return INVALID_HANDLE_VALUE;
    }
    if (info_level != FindExInfoStandard && info_level != FindExInfoBasic) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    if (search_op != FindExSearchNameMatch && search_op != FindExSearchLimitToDirectories) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }

    std::vector<WIN32_FIND_DATAW> entries;
    DWORD error = 0;
    if (!collect(pattern, parts, entries, error)) {
        SetLastError(error);
        return INVALID_HANDLE_VALUE;
    }
    if (search_op == FindExSearchLimitToDirectories) {
        std::vector<WIN32_FIND_DATAW> dirs_only;
        for (auto& entry : entries)
            if ((entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) dirs_only.push_back(entry);
        entries = std::move(dirs_only);
        if (entries.empty()) {
            SetLastError(ERROR_FILE_NOT_FOUND);
            return INVALID_HANDLE_VALUE;
        }
    }
    if (info_level == FindExInfoBasic) {
        for (auto& entry : entries) entry.cAlternateFileName[0] = L'\0';
    }
    auto e = std::make_unique<Enumeration>();
    e->entries = std::move(entries);
    e->next = 1;
    *static_cast<LPWIN32_FIND_DATAW>(data) = e->entries[0];
    return register_enumeration(std::move(e));
}

HANDLE WINAPI find_first_file_ex_a(LPCSTR pattern, FINDEX_INFO_LEVELS info_level, LPVOID data,
                                   FINDEX_SEARCH_OPS search_op, LPVOID search_filter, DWORD additional_flags) {
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::wstring wide = widen_ansi(pattern);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(wide, parts);
    if (parsed == Parsed::Host)
        return ::FindFirstFileExA(pattern, info_level, data, search_op, search_filter, additional_flags);
    WIN32_FIND_DATAW wide_data{};
    const HANDLE h = find_first_file_ex_w(wide.c_str(), info_level, &wide_data, search_op, search_filter, additional_flags);
    if (h != INVALID_HANDLE_VALUE) to_ansi(wide_data, static_cast<LPWIN32_FIND_DATAA>(data));
    return h;
}

HANDLE WINAPI find_first_file_a(LPCSTR pattern, LPWIN32_FIND_DATAA data) {
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::wstring wide = widen_ansi(pattern);
    WIN32_FIND_DATAW found{};
    const HANDLE h = find_first_file_w(wide.c_str(), &found);
    if (h != INVALID_HANDLE_VALUE) to_ansi(found, data);
    return h;
}

BOOL WINAPI find_next_file_w(HANDLE find, LPWIN32_FIND_DATAW data) {
    if (!find || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    {
        std::lock_guard lock(g_enum_mutex);
        const auto it = g_enumerations.find(find);
        if (it != g_enumerations.end()) {
            Enumeration& e = *it->second;
            if (e.next >= e.entries.size()) {
                SetLastError(ERROR_NO_MORE_FILES);
                return FALSE;
            }
            *data = e.entries[e.next++];
            SetLastError(ERROR_SUCCESS);
            return TRUE;
        }
    }
    return ::FindNextFileW(find, data);  // a host listing
}

BOOL WINAPI find_next_file_a(HANDLE find, LPWIN32_FIND_DATAA data) {
    if (!find || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    WIN32_FIND_DATAW found{};
    const BOOL ok = find_next_file_w(find, &found);
    if (ok) to_ansi(found, data);
    return ok;
}

BOOL WINAPI find_close(HANDLE find) {
    if (!find) {
        SetLastError(ERROR_INVALID_HANDLE);
        return FALSE;
    }
    if (unregister_enumeration(find)) return TRUE;
    return ::FindClose(find);
}

BOOL WINAPI delete_file_w(LPCWSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host) return ::DeleteFileW(name);
    if (parsed == Parsed::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return FALSE;
    }

    const Roots r = roots();
    if (r.overlay.empty() || parts.empty()) {
        SetLastError(ERROR_INVALID_NAME);
        return FALSE;
    }

    bool removed = false;
    std::error_code ec;
    const std::filesystem::path overlay_path = join(r.overlay, parts);
    if (std::filesystem::exists(overlay_path, ec)) {
        if (!::DeleteFileW(overlay_path.c_str())) return FALSE;
        removed = true;
    }
    if (!r.game.empty() && !deleted(r.overlay, parts)) {
        const std::filesystem::path package_path = join(r.game, parts);
        if (std::filesystem::exists(package_path, ec)) {
            if (std::filesystem::is_directory(package_path, ec)) {
                SetLastError(ERROR_ACCESS_DENIED);
                return FALSE;
            }
            // The overlay delete already checked its own attributes. Without an overlay, the package file's
            // read-only bit still applies even though deletion only writes a marker instead of deleting that file.
            if (!removed) {
                const DWORD attributes = ::GetFileAttributesW(package_path.c_str());
                if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_READONLY) != 0)
                    return fail_bool(ERROR_ACCESS_DENIED);
            }
            // The package file stays where it is; the marker is what hides it.
            DWORD error = 0;
            if (!write_marker(marker(r.overlay, parts), error)) return fail_bool(error);
            removed = true;
        }
    }
    if (!removed) {
        SetLastError(ERROR_FILE_NOT_FOUND);
        return FALSE;
    }
    return TRUE;
}

BOOL WINAPI delete_file_a(LPCSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::wstring wide = widen_ansi(name);
    return delete_file_w(wide.c_str());
}

// ---- Folders: created in the overlay, removed by a marker when the package has them

namespace {

BOOL create_directory(LPCWSTR template_name, LPCWSTR name, LPSECURITY_ATTRIBUTES security) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    std::wstring template_path;
    if (template_name) {
        const Target t = read_target(template_name);
        if (t.error) return fail_bool(t.error);
        template_path = t.path.wstring();
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host)
        return template_name ? ::CreateDirectoryExW(template_path.c_str(), name, security)
                             : ::CreateDirectoryW(name, security);
    if (parsed == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty()) return fail_bool(ERROR_PATH_NOT_FOUND);
    if (view(r, parts).kind != Kind::None) return fail_bool(ERROR_ALREADY_EXISTS);  // "GAME:" included
    if (!is_folder(r, parent_of(parts))) return fail_bool(ERROR_PATH_NOT_FOUND);

    const std::filesystem::path target = join(r.overlay, parts);
    std::error_code ec;
    std::filesystem::create_directories(target.parent_path(), ec);
    if (ec) return fail_bool(static_cast<DWORD>(ec.value()));
    return template_name ? ::CreateDirectoryExW(template_path.c_str(), target.c_str(), security)
                         : ::CreateDirectoryW(target.c_str(), security);
}

}  // namespace

BOOL WINAPI create_directory_w(LPCWSTR name, LPSECURITY_ATTRIBUTES security) {
    return create_directory(nullptr, name, security);
}

BOOL WINAPI create_directory_a(LPCSTR name, LPSECURITY_ATTRIBUTES security) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide = widen_ansi(name);
    return create_directory(nullptr, wide.c_str(), security);
}

BOOL WINAPI create_directory_ex_w(LPCWSTR template_name, LPCWSTR name, LPSECURITY_ATTRIBUTES security) {
    if (!template_name) return fail_bool(ERROR_INVALID_PARAMETER);
    return create_directory(template_name, name, security);
}

BOOL WINAPI create_directory_ex_a(LPCSTR template_name, LPCSTR name, LPSECURITY_ATTRIBUTES security) {
    if (!template_name || !name) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide_template = widen_ansi(template_name), wide = widen_ansi(name);
    return create_directory(wide_template.c_str(), wide.c_str(), security);
}

BOOL WINAPI remove_directory_w(LPCWSTR name) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host) return ::RemoveDirectoryW(name);
    if (parsed == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty()) return fail_bool(ERROR_PATH_NOT_FOUND);
    if (drive_root(name, parts)) return fail_bool(ERROR_ACCESS_DENIED);  // "GAME:" and "T:" themselves stay
    const View v = view(r, parts);
    if (v.kind == Kind::None) return fail_bool(ERROR_FILE_NOT_FOUND);
    if (v.kind == Kind::File) return fail_bool(ERROR_DIRECTORY);
    if (!empty_in_view(r, parts)) return fail_bool(ERROR_DIR_NOT_EMPTY);

    if (v.in_overlay && !::RemoveDirectoryW(join(r.overlay, parts).c_str())) return FALSE;  // its error
    if (v.in_package) {
        // The package folder stays where it is; the marker hides it, and its files are already hidden one by one.
        DWORD error = 0;
        if (!write_marker(marker(r.overlay, parts) / kRemovedDirMarker, error)) return fail_bool(error);
    }
    return TRUE;
}

BOOL WINAPI remove_directory_a(LPCSTR name) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide = widen_ansi(name);
    return remove_directory_w(wide.c_str());
}

// ---- Move, copy and attributes: the result lands in the overlay, a moved package name is deleted by a marker

BOOL WINAPI move_file_ex_w(LPCWSTR from, LPCWSTR to, DWORD flags) {
    if (!from) return fail_bool(ERROR_INVALID_PARAMETER);
    std::vector<std::wstring> src, dst;
    const Parsed from_kind = parse_console(from, src);
    const Parsed to_kind = to ? parse_console(to, dst) : Parsed::Host;
    if (from_kind == Parsed::Host && to_kind == Parsed::Host) return ::MoveFileExW(from, to, flags);
    if (from_kind == Parsed::Refused || to_kind == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);
    // A boot-time move or delete is a host feature a title cannot use on the console's drives.
    if (!to || (flags & MOVEFILE_DELAY_UNTIL_REBOOT) != 0) return fail_bool(ERROR_INVALID_PARAMETER);
    // GAME: is its own volume: a move between it and a host path is refused the way a cross-volume move is.
    if (from_kind != to_kind) return fail_bool(ERROR_NOT_SAME_DEVICE);

    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty()) return fail_bool(ERROR_PATH_NOT_FOUND);
    if (drive_root(from, src) || drive_root(to, dst)) return fail_bool(ERROR_ACCESS_DENIED);
    const View from_view = view(r, src);
    if (from_view.kind == Kind::None) return fail_bool(ERROR_FILE_NOT_FOUND);
    const std::filesystem::path overlay_from = join(r.overlay, src), overlay_to = join(r.overlay, dst);
    if (same_entry(src, dst)) {
        // The destination is the source, so it is not "already there". The host succeeds at a move onto the same
        // name and changes nothing, and renames when only the case differs. Here the case of a name is the overlay's
        // when the overlay holds it, so a package file gets an overlay copy under the new spelling (no marker: the
        // listing takes the overlay's entry and skips the package's as the same name); a package folder is left
        // where it is, as for any other move of one.
        if (src == dst) return TRUE;
        if (from_view.kind == Kind::Directory && from_view.in_package) return fail_bool(ERROR_ACCESS_DENIED);
        if (from_view.in_overlay)
            return ::MoveFileExW(overlay_from.c_str(), overlay_to.c_str(), flags & MOVEFILE_WRITE_THROUGH);
        std::error_code same_ec;
        std::filesystem::create_directories(overlay_to.parent_path(), same_ec);
        if (same_ec) return fail_bool(static_cast<DWORD>(same_ec.value()));
        if (!::CopyFileW(join(r.game, src).c_str(), overlay_to.c_str(), TRUE)) return FALSE;
        DWORD flush_error = 0;
        if ((flags & MOVEFILE_WRITE_THROUGH) != 0 && !flush_file(overlay_to, flush_error)) return fail_bool(flush_error);
        return TRUE;
    }

    const View to_view = view(r, dst);
    if (to_view.kind == Kind::Directory) return fail_bool(ERROR_ACCESS_DENIED);
    if (to_view.kind == Kind::File && (flags & MOVEFILE_REPLACE_EXISTING) == 0) return fail_bool(ERROR_ALREADY_EXISTS);
    if (!is_folder(r, parent_of(dst))) return fail_bool(ERROR_PATH_NOT_FOUND);

    std::error_code ec;
    std::filesystem::create_directories(overlay_to.parent_path(), ec);
    if (ec) return fail_bool(static_cast<DWORD>(ec.value()));

    if (from_view.kind == Kind::Directory) {
        // A package folder is left where it is (as DeleteFile does); an overlay-only folder is renamed in place.
        if (from_view.in_package) return fail_bool(ERROR_ACCESS_DENIED);
        if (to_view.kind != Kind::None) return fail_bool(ERROR_ACCESS_DENIED);
        return ::MoveFileExW(overlay_from.c_str(), overlay_to.c_str(), flags & MOVEFILE_WRITE_THROUGH);
    }

    const bool write_through = (flags & MOVEFILE_WRITE_THROUGH) != 0;
    const DWORD keep = flags & (MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    DWORD error = 0;
    if (from_view.in_overlay) {
        if (!::MoveFileExW(overlay_from.c_str(), overlay_to.c_str(), keep | MOVEFILE_COPY_ALLOWED)) return FALSE;
    } else {
        // The host never sees a move here, only a copy and a marker, so MOVEFILE_WRITE_THROUGH is honoured by
        // flushing both before returning, as MoveFileEx does for its own copy and delete.
        const BOOL fail_if_exists = (flags & MOVEFILE_REPLACE_EXISTING) == 0;
        if (!::CopyFileW(join(r.game, src).c_str(), overlay_to.c_str(), fail_if_exists)) return FALSE;
        if (write_through && !flush_file(overlay_to, error)) return fail_bool(error);
    }
    if (from_view.in_package) {
        const std::filesystem::path mark = marker(r.overlay, src);
        if (!write_marker(mark, error)) return fail_bool(error);
        if (write_through && !flush_file(mark, error)) return fail_bool(error);
    }
    return TRUE;
}

BOOL WINAPI move_file_w(LPCWSTR from, LPCWSTR to) {
    if (!from || !to) return fail_bool(ERROR_INVALID_PARAMETER);
    return move_file_ex_w(from, to, MOVEFILE_COPY_ALLOWED);  // MoveFile moves files across volumes
}

BOOL WINAPI move_file_ex_a(LPCSTR from, LPCSTR to, DWORD flags) {
    if (!from) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide_from = widen_ansi(from), wide_to = to ? widen_ansi(to) : std::wstring();
    return move_file_ex_w(wide_from.c_str(), to ? wide_to.c_str() : nullptr, flags);
}

BOOL WINAPI move_file_a(LPCSTR from, LPCSTR to) {
    if (!from || !to) return fail_bool(ERROR_INVALID_PARAMETER);
    return move_file_ex_a(from, to, MOVEFILE_COPY_ALLOWED);
}

namespace {

// Where a copy writes: a host path as given; a GAME: path is checked against the title's view (an existing name
// counts even when only the package has it) and then means its overlay copy.
bool copy_destination(LPCWSTR to, bool fail_if_exists, std::filesystem::path& out, DWORD& error) {
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(to, parts);
    if (parsed == Parsed::Host) {
        out = std::filesystem::path(to);
        return true;
    }
    error = ERROR_INVALID_NAME;
    if (parsed == Parsed::Refused || parts.empty()) return false;
    const Roots r = roots();
    error = ERROR_PATH_NOT_FOUND;
    if (r.game.empty() || r.overlay.empty()) return false;
    const View v = view(r, parts);
    error = ERROR_ACCESS_DENIED;
    if (v.kind == Kind::Directory) return false;
    error = ERROR_FILE_EXISTS;
    if (v.kind == Kind::File && fail_if_exists) return false;
    error = ERROR_PATH_NOT_FOUND;
    if (!is_folder(r, parent_of(parts))) return false;
    out = join(r.overlay, parts);
    std::error_code ec;
    std::filesystem::create_directories(out.parent_path(), ec);
    if (ec) {
        error = static_cast<DWORD>(ec.value());
        return false;
    }
    error = 0;
    return true;
}

// The source and destination a copy opens, or the error it fails with.
bool copy_paths(LPCWSTR from, LPCWSTR to, bool fail_if_exists, std::filesystem::path& source,
                std::filesystem::path& destination, DWORD& error) {
    if (!from || !to) {
        error = ERROR_INVALID_PARAMETER;
        return false;
    }
    const Target t = read_target(from);
    if (t.error) {
        error = t.error;
        return false;
    }
    source = t.path;
    return copy_destination(to, fail_if_exists, destination, error);
}

}  // namespace

BOOL WINAPI copy_file_w(LPCWSTR from, LPCWSTR to, BOOL fail_if_exists) {
    std::filesystem::path source, destination;
    DWORD error = 0;
    if (!copy_paths(from, to, fail_if_exists != FALSE, source, destination, error)) return fail_bool(error);
    return ::CopyFileW(source.c_str(), destination.c_str(), fail_if_exists);
}

BOOL WINAPI copy_file_a(LPCSTR from, LPCSTR to, BOOL fail_if_exists) {
    if (!from || !to) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide_from = widen_ansi(from), wide_to = widen_ansi(to);
    return copy_file_w(wide_from.c_str(), wide_to.c_str(), fail_if_exists);
}

BOOL WINAPI copy_file_ex_w(LPCWSTR from, LPCWSTR to, LPPROGRESS_ROUTINE progress, LPVOID data, LPBOOL cancel,
                           DWORD flags) {
    std::filesystem::path source, destination;
    DWORD error = 0;
    if (!copy_paths(from, to, (flags & COPY_FILE_FAIL_IF_EXISTS) != 0, source, destination, error))
        return fail_bool(error);
    return ::CopyFileExW(source.c_str(), destination.c_str(), progress, data, cancel, flags);
}

BOOL WINAPI copy_file_ex_a(LPCSTR from, LPCSTR to, LPPROGRESS_ROUTINE progress, LPVOID data, LPBOOL cancel,
                           DWORD flags) {
    if (!from || !to) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide_from = widen_ansi(from), wide_to = widen_ansi(to);
    return copy_file_ex_w(wide_from.c_str(), wide_to.c_str(), progress, data, cancel, flags);
}

namespace {

// The host's CopyFile2, or null where the host has none. Wine 9's kernel32 and kernelbase don't export it: Wine's
// loader binds a static import of it to a stub that aborts the process when called ("unimplemented function
// KERNEL32.dll.CopyFile2, aborting"), so a direct ::CopyFile2 call (or MSVC's std::filesystem::copy_file, which uses
// it) kills a title under Wine 9 or a Proton built on it; pe_loader.cpp and loader_tls_test.cpp avoid it for the same
// reason. GetProcAddress returns null there instead, and the fallback below runs.
using CopyFile2Fn = HRESULT(WINAPI*)(PCWSTR, PCWSTR, COPYFILE2_EXTENDED_PARAMETERS*);
std::atomic<bool> g_force_copy_file2_fallback{false};

CopyFile2Fn host_copy_file2() {
    static const CopyFile2Fn fn = [] {
        for (const wchar_t* dll : {L"kernelbase.dll", L"kernel32.dll"}) {
            if (const HMODULE module = GetModuleHandleW(dll))
                if (const FARPROC p = GetProcAddress(module, "CopyFile2")) return reinterpret_cast<CopyFile2Fn>(p);
        }
        return CopyFile2Fn{};
    }();
    return g_force_copy_file2_fallback.load(std::memory_order_relaxed) ? CopyFile2Fn{} : fn;
}

// The fallback's flags, by what CopyFileExW can do with them (values from the SDK's WinBase.h, spelled out because
// some of the names are only declared for newer NTDDI_VERSION targets):
// - passed on: the flags both calls define, with the same meaning;
// - dropped: CopyFile2-only requests about how the copy runs (privileges, resuming a paused copy, no offload, EDP,
//   destination DACL, compressed SMB traffic, pre-allocation, low-free-space mode, sparse copy) that leave the
//   destination's contents as CopyFileExW writes them;
// - everything else changes what the copy produces (0x80 DIRECTORY, 0x8000 SKIP_ALTERNATE_STREAMS, 0x200000
//   OPEN_AND_COPY_REPARSE_POINT, 0x800000 IGNORE_SOURCE_ENCRYPTION) or isn't defined, so the fallback refuses the
//   call with ERROR_NOT_SUPPORTED instead of copying something else.
constexpr DWORD kFallbackPassedFlags = COPY_FILE_FAIL_IF_EXISTS | COPY_FILE_RESTARTABLE |
                                       COPY_FILE_OPEN_SOURCE_FOR_WRITE | COPY_FILE_ALLOW_DECRYPTED_DESTINATION |
                                       COPY_FILE_COPY_SYMLINK | COPY_FILE_NO_BUFFERING;
constexpr DWORD kFallbackDroppedFlags = 0x00002000 /* REQUEST_SECURITY_PRIVILEGES */ |
                                        0x00004000 /* RESUME_FROM_PAUSE */ | 0x00040000 /* NO_OFFLOAD */ |
                                        0x00400000 /* IGNORE_EDP_BLOCK */ |
                                        0x02000000 /* DONT_REQUEST_DEST_WRITE_DAC */ |
                                        0x04000000 /* DISABLE_PRE_ALLOCATION */ |
                                        0x08000000 /* ENABLE_LOW_FREE_SPACE_MODE */ |
                                        0x10000000 /* REQUEST_COMPRESSED_TRAFFIC */ |
                                        0x20000000 /* ENABLE_SPARSE_COPY */ | 0x80000000 /* DISABLE_SPARSE_COPY */;

// Whether the fallback can run this call as CopyFile2 would: a COPYFILE2_EXTENDED_PARAMETERS block (the V2 block's
// extra fields have no CopyFileExW equivalent) and no flag outside the two sets above.
bool fallback_supports(const COPYFILE2_EXTENDED_PARAMETERS* parameters) {
    if (!parameters) return true;
    if (parameters->dwSize != sizeof(COPYFILE2_EXTENDED_PARAMETERS)) return false;
    return (parameters->dwCopyFlags & ~(kFallbackPassedFlags | kFallbackDroppedFlags)) == 0;
}

// CopyFile2 through CopyFileExW, for hosts without it (only for calls fallback_supports). The cancel flag has the
// same type in both (one already set is answered by copy_file2 before this runs). A COPYFILE2 progress routine has
// another signature and message set, so it isn't called here.
HRESULT copy_file2_fallback(const wchar_t* source, const wchar_t* destination,
                            const COPYFILE2_EXTENDED_PARAMETERS* parameters) {
    const DWORD flags = parameters ? parameters->dwCopyFlags & kFallbackPassedFlags : 0;
    BOOL* const cancel = parameters ? parameters->pfCancel : nullptr;
    if (::CopyFileExW(source, destination, nullptr, nullptr, cancel, flags)) return S_OK;
    const DWORD error = GetLastError();
    return HRESULT_FROM_WIN32(error ? error : ERROR_GEN_FAILURE);
}

}  // namespace

void force_copy_file2_fallback(bool force) { g_force_copy_file2_fallback.store(force, std::memory_order_relaxed); }

// The parameter block is checked before the destination is prepared, so a refused call leaves nothing behind: a
// block shorter than COPYFILE2_EXTENDED_PARAMETERS is E_INVALIDARG, as the host's CopyFile2 answers (measured on
// Windows 11 for dwSize 0, 4 and 31, which also left no destination), and a call the fallback can't run as asked is
// ERROR_NOT_SUPPORTED. A longer block goes to the host's CopyFile2, which checks it itself.
HRESULT WINAPI copy_file2(PCWSTR from, PCWSTR to, COPYFILE2_EXTENDED_PARAMETERS* parameters) {
    if (parameters && parameters->dwSize < sizeof(COPYFILE2_EXTENDED_PARAMETERS)) return E_INVALIDARG;
    const CopyFile2Fn host = host_copy_file2();
    if (!host && !fallback_supports(parameters)) return HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
    const bool fail_if_exists = parameters && (parameters->dwCopyFlags & COPY_FILE_FAIL_IF_EXISTS) != 0;
    std::filesystem::path source, destination;
    DWORD error = 0;
    if (!copy_paths(from, to, fail_if_exists, source, destination, error)) return HRESULT_FROM_WIN32(error);
    // A cancel flag already set is answered here, for the host's CopyFile2 as well as the fallback: Windows returns
    // ERROR_REQUEST_ABORTED and writes nothing, while Wine reads the flag in neither call (Wine 11.20's CopyFile2 logs
    // "pfCancel is not supported" and copies, dlls/kernelbase/file.c copy_file; Wine 9 has only CopyFileExW).
    if (parameters && parameters->pfCancel && *parameters->pfCancel) return HRESULT_FROM_WIN32(ERROR_REQUEST_ABORTED);
    if (host) return host(source.c_str(), destination.c_str(), parameters);
    return copy_file2_fallback(source.c_str(), destination.c_str(), parameters);
}

BOOL WINAPI set_file_attributes_w(LPCWSTR name, DWORD attributes) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host) return ::SetFileAttributesW(name, attributes);
    if (parsed == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty()) return fail_bool(ERROR_PATH_NOT_FOUND);
    if (parts.empty()) return fail_bool(ERROR_ACCESS_DENIED);
    const View v = view(r, parts);
    if (v.kind == Kind::None) return fail_bool(ERROR_FILE_NOT_FOUND);

    // The attributes belong to the overlay copy: a package file is copied up and a package folder mirrored first,
    // so the package keeps its own attributes and reads, which prefer the overlay, see the new ones.
    const std::filesystem::path target = join(r.overlay, parts);
    if (!v.in_overlay) {
        std::error_code ec;
        if (v.kind == Kind::Directory) {
            std::filesystem::create_directories(target, ec);
            if (ec) return fail_bool(static_cast<DWORD>(ec.value()));
        } else {
            const Target t = write_target(name, true);
            if (t.error) return fail_bool(t.error);
        }
    }
    return ::SetFileAttributesW(target.c_str(), attributes);
}

BOOL WINAPI set_file_attributes_a(LPCSTR name, DWORD attributes) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide = widen_ansi(name);
    return set_file_attributes_w(wide.c_str(), attributes);
}

}  // namespace rt::fs

namespace {
using namespace rt::fs;

// CreateFile, the FindFirstFile family, the attribute queries and DeleteFile arrive through kernel32/kernelbase
// imports and through the api sets a GDK title is linked against (file-l1-1-0 for the find and attribute calls,
// file-l1-2-0 and file-l2-1-0 for CreateFile).
#define FILE_API(dll)                                                                                    \
    {dll, "CreateFileA", reinterpret_cast<void*>(&create_file_a)},                                       \
        {dll, "CreateFileW", reinterpret_cast<void*>(&create_file_w)},                                   \
        {dll, "CreateFile2", reinterpret_cast<void*>(&create_file2)},                                    \
        {dll, "FindFirstFileA", reinterpret_cast<void*>(&find_first_file_a)},                            \
        {dll, "FindFirstFileW", reinterpret_cast<void*>(&find_first_file_w)},                            \
        {dll, "FindFirstFileExA", reinterpret_cast<void*>(&find_first_file_ex_a)},                       \
        {dll, "FindFirstFileExW", reinterpret_cast<void*>(&find_first_file_ex_w)},                       \
        {dll, "FindNextFileA", reinterpret_cast<void*>(&find_next_file_a)},                              \
        {dll, "FindNextFileW", reinterpret_cast<void*>(&find_next_file_w)},                              \
        {dll, "FindClose", reinterpret_cast<void*>(&find_close)},                                        \
        {dll, "GetFileAttributesA", reinterpret_cast<void*>(&get_file_attributes_a)},                    \
        {dll, "GetFileAttributesW", reinterpret_cast<void*>(&get_file_attributes_w)},                    \
        {dll, "GetFileAttributesExA", reinterpret_cast<void*>(&get_file_attributes_ex_a)},               \
        {dll, "GetFileAttributesExW", reinterpret_cast<void*>(&get_file_attributes_ex_w)},               \
        {dll, "DeleteFileA", reinterpret_cast<void*>(&delete_file_a)},                                   \
        {dll, "DeleteFileW", reinterpret_cast<void*>(&delete_file_w)},                                   \
        {dll, "CreateDirectoryA", reinterpret_cast<void*>(&create_directory_a)},                         \
        {dll, "CreateDirectoryW", reinterpret_cast<void*>(&create_directory_w)},                         \
        {dll, "CreateDirectoryExA", reinterpret_cast<void*>(&create_directory_ex_a)},                    \
        {dll, "CreateDirectoryExW", reinterpret_cast<void*>(&create_directory_ex_w)},                    \
        {dll, "RemoveDirectoryA", reinterpret_cast<void*>(&remove_directory_a)},                         \
        {dll, "RemoveDirectoryW", reinterpret_cast<void*>(&remove_directory_w)},                         \
        {dll, "MoveFileA", reinterpret_cast<void*>(&move_file_a)},                                       \
        {dll, "MoveFileW", reinterpret_cast<void*>(&move_file_w)},                                       \
        {dll, "MoveFileExA", reinterpret_cast<void*>(&move_file_ex_a)},                                  \
        {dll, "MoveFileExW", reinterpret_cast<void*>(&move_file_ex_w)},                                  \
        {dll, "CopyFileA", reinterpret_cast<void*>(&copy_file_a)},                                       \
        {dll, "CopyFileW", reinterpret_cast<void*>(&copy_file_w)},                                       \
        {dll, "CopyFileExA", reinterpret_cast<void*>(&copy_file_ex_a)},                                  \
        {dll, "CopyFileExW", reinterpret_cast<void*>(&copy_file_ex_w)},                                  \
        {dll, "CopyFile2", reinterpret_cast<void*>(&copy_file2)},                                        \
        {dll, "SetFileAttributesA", reinterpret_cast<void*>(&set_file_attributes_a)},                    \
        {dll, "SetFileAttributesW", reinterpret_cast<void*>(&set_file_attributes_w)}
}  // namespace

SHIM_TABLE(file_system,
    FILE_API("kernel32.dll"),
    FILE_API("kernelbase.dll"),
    FILE_API("api-ms-win-core-file-l1-1-0.dll"),
    FILE_API("api-ms-win-core-file-l1-2-0.dll"),
    FILE_API("api-ms-win-core-file-l2-1-0.dll"),
);
