// Checks the console path mapping the file shims install for a mapped title (RT-97).
//
// Every function is resolved through rt::shims::find under the DLL and name an import uses, so the test runs against
// the table the runtime installs rather than a private copy. It works on folders it creates in %TEMP%: a package
// folder that stands in for the game install, an overlay folder, and a "GAME:" path in each variant.
#include <windows.h>

#include <aclapi.h>

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

#include "file_system.h"
#include "shims.h"
#include "startup_trace.h"

namespace {

using CreateFileWFn = HANDLE(WINAPI*)(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
using CreateFileAFn = HANDLE(WINAPI*)(LPCSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
using CreateFile2Fn = HANDLE(WINAPI*)(LPCWSTR, DWORD, DWORD, DWORD, LPCREATEFILE2_EXTENDED_PARAMETERS);
using FindFirstFileWFn = HANDLE(WINAPI*)(LPCWSTR, LPWIN32_FIND_DATAW);
using FindFirstFileAFn = HANDLE(WINAPI*)(LPCSTR, LPWIN32_FIND_DATAA);
using FindFirstFileExWFn =
    HANDLE(WINAPI*)(LPCWSTR, FINDEX_INFO_LEVELS, LPVOID, FINDEX_SEARCH_OPS, LPVOID, DWORD);
using FindFirstFileExAFn =
    HANDLE(WINAPI*)(LPCSTR, FINDEX_INFO_LEVELS, LPVOID, FINDEX_SEARCH_OPS, LPVOID, DWORD);
using FindNextFileWFn = BOOL(WINAPI*)(HANDLE, LPWIN32_FIND_DATAW);
using FindCloseFn = BOOL(WINAPI*)(HANDLE);
using GetFileAttributesWFn = DWORD(WINAPI*)(LPCWSTR);
using GetFileAttributesAFn = DWORD(WINAPI*)(LPCSTR);
using GetFileAttributesExWFn = BOOL(WINAPI*)(LPCWSTR, GET_FILEEX_INFO_LEVELS, LPVOID);
using DeleteFileWFn = BOOL(WINAPI*)(LPCWSTR);
using DeleteFileAFn = BOOL(WINAPI*)(LPCSTR);
using CreateDirectoryWFn = BOOL(WINAPI*)(LPCWSTR, LPSECURITY_ATTRIBUTES);
using CreateDirectoryAFn = BOOL(WINAPI*)(LPCSTR, LPSECURITY_ATTRIBUTES);
using CreateDirectoryExWFn = BOOL(WINAPI*)(LPCWSTR, LPCWSTR, LPSECURITY_ATTRIBUTES);
using RemoveDirectoryWFn = BOOL(WINAPI*)(LPCWSTR);
using RemoveDirectoryAFn = BOOL(WINAPI*)(LPCSTR);
using MoveFileWFn = BOOL(WINAPI*)(LPCWSTR, LPCWSTR);
using MoveFileExWFn = BOOL(WINAPI*)(LPCWSTR, LPCWSTR, DWORD);
using MoveFileAFn = BOOL(WINAPI*)(LPCSTR, LPCSTR);
using CopyFileWFn = BOOL(WINAPI*)(LPCWSTR, LPCWSTR, BOOL);
using CopyFileAFn = BOOL(WINAPI*)(LPCSTR, LPCSTR, BOOL);
using CopyFileExWFn = BOOL(WINAPI*)(LPCWSTR, LPCWSTR, LPPROGRESS_ROUTINE, LPVOID, LPBOOL, DWORD);
using CopyFile2Fn = HRESULT(WINAPI*)(PCWSTR, PCWSTR, COPYFILE2_EXTENDED_PARAMETERS*);
using SetFileAttributesWFn = BOOL(WINAPI*)(LPCWSTR, DWORD);
using SetFileAttributesAFn = BOOL(WINAPI*)(LPCSTR, DWORD);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* dll, const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find(dll, name));
}

void write_path(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    const HANDLE h =
        ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    ::WriteFile(h, text.data(), static_cast<DWORD>(text.size()), &written, nullptr);
    ::CloseHandle(h);
}

std::string read_handle(HANDLE h) {
    std::string out;
    char buffer[256];
    DWORD got = 0;
    while (::ReadFile(h, buffer, sizeof(buffer), &got, nullptr) && got != 0) out.append(buffer, got);
    return out;
}

std::string read_path(const std::filesystem::path& path) {
    const HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return {};
    std::string text = read_handle(h);
    ::CloseHandle(h);
    return text;
}

std::vector<std::wstring> list(FindFirstFileWFn first, FindNextFileWFn next, FindCloseFn close, const wchar_t* pattern) {
    std::vector<std::wstring> names;
    WIN32_FIND_DATAW found{};
    const HANDLE h = first(pattern, &found);
    if (h == INVALID_HANDLE_VALUE) return names;
    names.emplace_back(found.cFileName);
    while (next(h, &found)) names.emplace_back(found.cFileName);
    close(h);
    return names;
}

int count(const std::vector<std::wstring>& names, const wchar_t* name) {
    int n = 0;
    for (const auto& candidate : names)
        if (_wcsicmp(candidate.c_str(), name) == 0) ++n;
    return n;
}

bool contains(const std::vector<std::wstring>& names, const wchar_t* name) { return count(names, name) != 0; }

// Denies the current user `rights` on a folder while it lives and puts the folder's old DACL back when it goes, on
// every way out of the check. Both changes go through one handle opened before the deny. Under Wine the DACL is the
// folder's unix mode: a deny of FILE_LIST_DIRECTORY clears its owner and group read bits (server/file.c, sd_to_mode),
// and Wine opens a folder with open(O_RDONLY) (server/fd.c, open_fd), which then fails with EACCES. A restore through
// the folder's name (SetNamedSecurityInfoW) could not open it again, so the folder stayed unlistable and the closing
// remove_all threw. Windows checks access when a handle is opened, so the handle keeps WRITE_DAC there as well.
class FolderDeny {
public:
    FolderDeny(const std::wstring& folder, DWORD rights) {
        handle_ = ::CreateFileW(folder.c_str(), READ_CONTROL | WRITE_DAC,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                FILE_FLAG_BACKUP_SEMANTICS, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) return;
        if (GetSecurityInfo(handle_, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &old_dacl_,
                            nullptr, &sd_) != ERROR_SUCCESS)
            return;
        EXPLICIT_ACCESS_W deny{};
        deny.grfAccessPermissions = rights;
        deny.grfAccessMode = DENY_ACCESS;
        deny.grfInheritance = NO_INHERITANCE;
        BuildTrusteeWithNameW(&deny.Trustee, const_cast<LPWSTR>(L"CURRENT_USER"));
        PACL new_dacl = nullptr;
        if (SetEntriesInAclW(1, &deny, old_dacl_, &new_dacl) != ERROR_SUCCESS) return;
        denied_ = SetSecurityInfo(handle_, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, new_dacl,
                                  nullptr) == ERROR_SUCCESS;
        LocalFree(new_dacl);
    }
    FolderDeny(const FolderDeny&) = delete;
    FolderDeny& operator=(const FolderDeny&) = delete;
    ~FolderDeny() {
        restore();
        if (sd_) LocalFree(sd_);
        if (handle_ != INVALID_HANDLE_VALUE) ::CloseHandle(handle_);
    }

    bool denied() const { return denied_; }

    // Puts the old DACL back; true when it is back (or was never changed).
    bool restore() {
        if (denied_ && SetSecurityInfo(handle_, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr,
                                       old_dacl_, nullptr) == ERROR_SUCCESS)
            denied_ = false;
        return !denied_;
    }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    PACL old_dacl_ = nullptr;  // points into sd_
    PSECURITY_DESCRIPTOR sd_ = nullptr;
    bool denied_ = false;
};

// Whether the host's own MoveFileW changes a file's name to the same name in another case. Windows does. Wine 11.20
// looks the new name up without case, finds the file itself and returns success without renaming it (server/fd.c,
// set_fd_name: "if (!stat( name, &st ))" with the same inode returns early), so a listing keeps the old spelling.
bool host_renames_case(const std::filesystem::path& folder) {
    write_path(folder / "case_probe.tmp", "probe");
    const BOOL moved = ::MoveFileW((folder / "case_probe.tmp").c_str(), (folder / "CASE_PROBE.TMP").c_str());
    const std::vector<std::wstring> names =
        list(::FindFirstFileW, ::FindNextFileW, ::FindClose, (folder / "*").c_str());
    ::DeleteFileW((folder / "CASE_PROBE.TMP").c_str());
    return moved && std::find(names.begin(), names.end(), std::wstring(L"CASE_PROBE.TMP")) != names.end();
}

void run_checks(const std::filesystem::path& root);

}  // namespace

int main() {
    // A folder of this run's own: Wine hands each fresh prefix the same small process ids, so a folder named after
    // the process id alone can be one an earlier run left behind.
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("leonida_file_system_" + std::to_string(GetCurrentProcessId()) +
                                                  "_" + std::to_string(now.QuadPart));
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    // An exception from one check (std::filesystem throws on a host error) ends the checks, not the run: it is a
    // failure, the folder is still removed and the summary is still printed.
    try {
        run_checks(root);
    } catch (const std::exception& e) {
        check(false, (std::string("the checks run to the end (stopped by: ") + e.what() + ")").c_str());
    }
    std::filesystem::remove_all(root, ec);
    if (ec) std::printf("NOTE could not remove %s: %s\n", root.string().c_str(), ec.message().c_str());
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

namespace {

void run_checks(const std::filesystem::path& root) {
    const std::filesystem::path package = root / "package";
    const std::filesystem::path overlay = root / "overlay";
    write_path(package / "data" / "hello.txt", "package");
    write_path(package / "data" / "dup.txt", "package");
    write_path(package / "data" / "bye.txt", "package");
    write_path(package / "data" / "sub" / "nested.txt", "nested");

    rt::fs::configure(package, overlay);
    check(rt::fs::game_dir() == package, "configure() sets the game folder");
    check(rt::fs::overlay_dir() == overlay, "configure() sets the overlay folder");

    // The table the runtime installs: kernel32 and the api sets a GDK title links against.
    const auto create_w = resolve<CreateFileWFn>("kernel32.dll", "CreateFileW");
    const auto create_a = resolve<CreateFileAFn>("kernelbase.dll", "CreateFileA");
    const auto first_w = resolve<FindFirstFileWFn>("api-ms-win-core-file-l1-1-0.dll", "FindFirstFileW");
    const auto first_a = resolve<FindFirstFileAFn>("kernel32.dll", "FindFirstFileA");
    const auto next_w = resolve<FindNextFileWFn>("kernel32.dll", "FindNextFileW");
    const auto find_close = resolve<FindCloseFn>("kernel32.dll", "FindClose");
    const auto attributes_w = resolve<GetFileAttributesWFn>("kernel32.dll", "GetFileAttributesW");
    const auto attributes_a = resolve<GetFileAttributesAFn>("kernel32.dll", "GetFileAttributesA");
    const auto attributes_ex_w = resolve<GetFileAttributesExWFn>("kernel32.dll", "GetFileAttributesExW");
    const auto delete_w = resolve<DeleteFileWFn>("kernel32.dll", "DeleteFileW");
    const auto delete_a = resolve<DeleteFileAFn>("api-ms-win-core-file-l1-1-0.dll", "DeleteFileA");
    check(create_w && create_a && first_w && first_a && next_w && find_close && attributes_w && attributes_a &&
              attributes_ex_w && delete_w && delete_a,
          "the file API resolves through the registry under kernel32/kernelbase and the api sets");

    // A read of a package file, through the mixed-case prefix and through forward slashes.
    {
        HANDLE h = create_w(L"GAME:\\data\\hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "CreateFileW opens GAME:\\data\\hello.txt from the package");
        check(read_handle(h) == "package", "the package file's bytes come back unchanged");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        h = create_w(L"game:/data/hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "the prefix is case-insensitive and accepts forward slashes");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        h = create_a("GAME:\\data\\hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "CreateFileA takes the same path");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
    }

    // --trace: a failed open of a guest path is a "file_open_failed" event, a successful one is not, and with the
    // trace off nothing is written.
    {
        const std::wstring trace = (root / "open_trace.jsonl").wstring();
        const auto events = [&] {
            std::ifstream in(std::filesystem::path(trace), std::ios::binary);
            std::stringstream text;
            text << in.rdbuf();
            return text.str();
        };
        const auto count = [](const std::string& hay, const std::string& needle) {
            int n = 0;
            for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + 1)) ++n;
            return n;
        };
        HANDLE h = create_w(L"GAME:\\data\\absent_off.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND, "a missing open fails with the trace off");
        check(rt::startup_trace::open(trace.c_str()), "open the startup trace file");
        h = create_w(L"GAME:\\data\\hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "an open that succeeds");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(count(events(), "file_open_failed") == 0, "a successful open writes no file_open_failed event");
        h = create_w(L"GAME:\\data\\absent_on.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "a missing open still fails the same way with the trace on");
        rt::startup_trace::close();
        const std::string log = events();
        check(count(log, "\"kind\":\"file_open_failed\"") == 1, "one file_open_failed event");
        check(log.find("\"path\":\"GAME:\\\\data\\\\absent_on.txt\"") != std::string::npos, "the event names the path");
        check(log.find("\"error\":2,") != std::string::npos, "the event carries error 2");
        check(log.find("\"caller\":\"host:") != std::string::npos, "the event names a caller");
        check(count(log, "absent_off") == 0, "nothing was traced before the trace was opened");
    }

    // Attributes, wide and narrow.
    check(attributes_w(L"GAME:\\data\\hello.txt") != INVALID_FILE_ATTRIBUTES, "GetFileAttributesW stats a package file");
    check(attributes_a("GAME:\\data\\hello.txt") != INVALID_FILE_ATTRIBUTES, "GetFileAttributesA stats a package file");
    const std::filesystem::path host_file = package / "data" / "hello.txt";
    check(attributes_w(host_file.c_str()) != INVALID_FILE_ATTRIBUTES, "a host path is passed through unchanged");

    // A write-open copies the package file into the overlay first, and the package is left alone.
    {
        const HANDLE h =
            create_w(L"GAME:\\data\\hello.txt", GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "a write-open of a package file succeeds");
        DWORD written = 0;
        if (h != INVALID_HANDLE_VALUE) {
            ::WriteFile(h, "overlay", 7, &written, nullptr);
            ::CloseHandle(h);
        }
        check(read_path(package / "data" / "hello.txt") == "package", "the package file is not modified");
        check(read_path(overlay / "data" / "hello.txt") == "overlay", "the write landed in the overlay copy");
        HANDLE r = create_w(L"GAME:\\data\\hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(r != INVALID_HANDLE_VALUE && read_handle(r) == "overlay", "reads prefer the overlay copy");
        if (r != INVALID_HANDLE_VALUE) ::CloseHandle(r);
    }

    // A new file, in a folder the package does not have: the title makes the folder first, as it must on the host.
    {
        const auto make_folder = resolve<CreateDirectoryWFn>("kernel32.dll", "CreateDirectoryW");
        check(make_folder && make_folder(L"GAME:\\save", nullptr) != FALSE, "CreateDirectoryW makes GAME:\\save");
        const HANDLE h = create_w(L"GAME:\\save\\slot1.bin", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "CREATE_ALWAYS makes a new console file");
        DWORD written = 0;
        if (h != INVALID_HANDLE_VALUE) {
            ::WriteFile(h, "12345", 5, &written, nullptr);
            ::CloseHandle(h);
        }
        check(read_path(overlay / "save" / "slot1.bin") == "12345", "the new file is in the overlay");
        check(!std::filesystem::exists(package / "save"), "nothing was created in the package");

        WIN32_FILE_ATTRIBUTE_DATA data{};
        check(attributes_ex_w(L"GAME:\\save\\slot1.bin", GetFileExInfoStandard, &data) != FALSE &&
                  data.nFileSizeLow == 5,
              "GetFileAttributesExW reports the overlay copy's size");
    }

    // One listing, merged, overlay first and without duplicates.
    {
        const std::vector<std::wstring> names = list(first_w, next_w, find_close, L"GAME:\\data\\*");
        check(contains(names, L"hello.txt") && count(names, L"hello.txt") == 1, "the merged listing has hello.txt once");
        check(contains(names, L"dup.txt"), "the merged listing has the package's own files");
        check(contains(names, L"sub"), "the merged listing has the package's folders");
        check(!contains(names, L".gta6rt-deleted"), "the overlay's marker folder is not listed");

        const std::vector<std::wstring> root_names = list(first_w, next_w, find_close, L"GAME:\\");
        check(contains(root_names, L"data") && contains(root_names, L"save"),
              "GAME:\\ lists the package's folder and the overlay's");

        WIN32_FIND_DATAA ansi{};
        const HANDLE h = first_a("GAME:\\data\\*", &ansi);
        check(h != INVALID_HANDLE_VALUE && ansi.cFileName[0] != '\0', "FindFirstFileA fills an ANSI listing too");
        if (h != INVALID_HANDLE_VALUE) find_close(h);

        SetLastError(0);
        WIN32_FIND_DATAW none{};
        const HANDLE missing = first_w(L"GAME:\\missing\\*", &none);
        check(missing == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "a listing inside a folder that does not exist reports ERROR_PATH_NOT_FOUND, as the host does");
    }

    // Names that differ only in case are one entry, as on the host, and the overlay's spelling is the one listed.
    {
        write_path(overlay / "data" / "Mixed.TXT", "overlay");
        write_path(package / "data" / "mixed.txt", "package");
        const std::vector<std::wstring> names = list(first_w, next_w, find_close, L"GAME:\\data\\*");
        check(count(names, L"mixed.txt") == 1, "names that differ only in case are listed once");
        check(std::find(names.begin(), names.end(), std::wstring(L"Mixed.TXT")) != names.end(),
              "and the overlay's spelling wins");
    }

    // A folder with no deletion markers at all (its marker folder does not exist) lists every package entry once.
    {
        for (int i = 0; i < 300; ++i) write_path(package / "many" / ("f" + std::to_string(i) + ".bin"), "x");
        write_path(overlay / "many" / "F7.BIN", "overlay");
        check(!std::filesystem::exists(overlay / ".gta6rt-deleted" / "many"), "GAME:\\many has no marker folder");
        const std::vector<std::wstring> names = list(first_w, next_w, find_close, L"GAME:\\many\\*");
        int files = 0;
        for (const auto& n : names)
            if (n != L"." && n != L"..") ++files;
        check(files == 300 && count(names, L"f7.bin") == 1 && count(names, L"f299.bin") == 1,
              "a folder without markers lists all 300 files, with the overlay's copy once");

        // Once the folder has a marker folder, the listing reads it once: a delete under another ASCII case hides
        // exactly that file and every other file stays.
        check(delete_w(L"GAME:\\many\\F3.BIN") != FALSE, "DeleteFileW hides f3.bin under another case");
        const std::vector<std::wstring> after = list(first_w, next_w, find_close, L"GAME:\\many\\*");
        int left = 0;
        for (const auto& n : after)
            if (n != L"." && n != L"..") ++left;
        check(left == 299 && !contains(after, L"f3.bin") && count(after, L"f30.bin") == 1,
              "with a marker folder the listing hides only the deleted file");
    }

    // Names with non-ASCII letters keep the file system's case rule: deleting U+00E9 hides the package's U+00C9.
    {
        write_path(package / "names" / L"Éclair.txt", "package");
        write_path(package / "names" / "plain.txt", "package");
        check(delete_w(L"GAME:\\names\\éclair.txt") != FALSE, "DeleteFileW takes a name that differs in a non-ASCII case");
        const std::vector<std::wstring> names = list(first_w, next_w, find_close, L"GAME:\\names\\*");
        check(std::find(names.begin(), names.end(), std::wstring(L"Éclair.txt")) == names.end() &&
                  contains(names, L"plain.txt"),
              "and the listing hides it, as the file system's compare does");
    }

    // Duplicates are decided by _wcsicmp (same_name) in the title's locale, not by a lower-cased key: under .UTF-8,
    // _wcslwr_s folds U+00C9 and U+00E9 while _wcsicmp keeps them apart.
    {
        write_path(overlay / "loc" / L"Éclair.txt", "overlay");
        write_path(package / "loc" / L"éclair.txt", "package");
        const char* old = std::setlocale(LC_CTYPE, nullptr);
        const std::string saved = old ? old : "C";
        if (std::setlocale(LC_CTYPE, ".UTF-8")) {
            const int expected = _wcsicmp(L"Éclair.txt", L"éclair.txt") == 0 ? 1 : 2;
            const std::vector<std::wstring> names = list(first_w, next_w, find_close, L"GAME:\\loc\\*");
            int files = 0;
            for (const auto& n : names)
                if (n != L"." && n != L"..") ++files;
            check(files == expected, "under .UTF-8 the merged listing keeps exactly the names _wcsicmp tells apart");
        }
        std::setlocale(LC_CTYPE, saved.c_str());
    }

    // A marker folder that cannot be listed (listing denied, lookups by path still allowed) is not "no markers":
    // the deleted file stays hidden, as GetFileAttributesW already reports it.
    {
        write_path(package / "acl" / "a.txt", "package");
        write_path(package / "acl" / "b.txt", "package");
        check(delete_w(L"GAME:\\acl\\a.txt") != FALSE, "DeleteFileW hides a.txt");
        const std::wstring marks = (overlay / ".gta6rt-deleted" / "acl").wstring();
        FolderDeny deny(marks, FILE_LIST_DIRECTORY);
        const bool denied = deny.denied();
        WIN32_FIND_DATAW probe{};
        const HANDLE ph = ::FindFirstFileW((marks + L"\\*").c_str(), &probe);
        const bool listing_denied = ph == INVALID_HANDLE_VALUE && GetLastError() == ERROR_ACCESS_DENIED;
        if (ph != INVALID_HANDLE_VALUE) ::FindClose(ph);
        if (denied && listing_denied) {
            check(attributes_w(L"GAME:\\acl\\a.txt") == INVALID_FILE_ATTRIBUTES, "a.txt is still missing by path");
            const std::vector<std::wstring> names = list(first_w, next_w, find_close, L"GAME:\\acl\\*");
            check(!contains(names, L"a.txt") && contains(names, L"b.txt"),
                  "a marker folder that cannot be listed still hides its deleted files");
        } else {
            std::printf("SKIP could not deny listing the marker folder\n");
        }
        if (denied) {
            const bool restored = deny.restore();
            const HANDLE again = ::FindFirstFileW((marks + L"\\*").c_str(), &probe);
            check(restored && again != INVALID_HANDLE_VALUE,
                  "the marker folder lists again once its old DACL is back");
            if (again != INVALID_HANDLE_VALUE) ::FindClose(again);
        }
    }

    // A delete hides the package file without removing it.
    {
        check(delete_w(L"GAME:\\data\\dup.txt") != FALSE, "DeleteFileW accepts a package file");
        check(std::filesystem::exists(package / "data" / "dup.txt"), "the package file is still there");
        SetLastError(0);
        check(attributes_w(L"GAME:\\data\\dup.txt") == INVALID_FILE_ATTRIBUTES &&
                  GetLastError() == ERROR_FILE_NOT_FOUND,
              "the deleted name no longer resolves");
        SetLastError(0);
        const HANDLE h =
            create_w(L"GAME:\\data\\dup.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND, "and cannot be opened");
        check(!contains(list(first_w, next_w, find_close, L"GAME:\\data\\*"), L"dup.txt"),
              "and is absent from the listing");

        // A write-open of the hidden name fails, and must leave the name deleted: no copy-up from the package, no
        // overlay file, and still hidden afterwards.
        SetLastError(0);
        const HANDLE write_open =
            create_w(L"GAME:\\data\\dup.txt", GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        check(write_open == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "a write-open of the deleted name fails with ERROR_FILE_NOT_FOUND");
        if (write_open != INVALID_HANDLE_VALUE) ::CloseHandle(write_open);
        check(!std::filesystem::exists(overlay / "data" / "dup.txt"), "the failed write-open left no overlay copy");
        SetLastError(0);
        check(attributes_w(L"GAME:\\data\\dup.txt") == INVALID_FILE_ATTRIBUTES &&
                  GetLastError() == ERROR_FILE_NOT_FOUND,
              "and the name is still deleted");

        // Re-creating it makes it readable again, and it is listed.
        const HANDLE again = create_w(L"GAME:\\data\\dup.txt", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        DWORD written = 0;
        check(again != INVALID_HANDLE_VALUE, "the name can be created again");
        if (again != INVALID_HANDLE_VALUE) {
            ::WriteFile(again, "again", 5, &written, nullptr);
            ::CloseHandle(again);
        }
        check(read_path(overlay / "data" / "dup.txt") == "again", "the new copy is in the overlay");
        check(contains(list(first_w, next_w, find_close, L"GAME:\\data\\*"), L"dup.txt"),
              "and it is listed again");

        check(delete_a("GAME:\\data\\bye.txt") != FALSE, "DeleteFileA takes a console path");
        check(std::filesystem::exists(package / "data" / "bye.txt"), "the narrow delete leaves the package file too");
        SetLastError(0);
        check(delete_w(L"GAME:\\data\\bye.txt") == FALSE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "deleting an already hidden file reports ERROR_FILE_NOT_FOUND");
        SetLastError(0);
        check(delete_w(L"GAME:\\nope\\missing.txt") == FALSE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "deleting a file that never existed reports ERROR_FILE_NOT_FOUND");

        // Deleting a nested package file creates parent directories inside .gta6rt-deleted/; those marker parent
        // directories must not hide the corresponding ancestor directories in the package.
        check(delete_w(L"GAME:\\data\\sub\\nested.txt") != FALSE, "DeleteFileW accepts a nested package file");
        const DWORD sub_attrs = attributes_w(L"GAME:\\data\\sub");
        check(sub_attrs != INVALID_FILE_ATTRIBUTES && (sub_attrs & FILE_ATTRIBUTE_DIRECTORY) != 0,
              "deleting a nested package file does not hide its parent folder");
        check(contains(list(first_w, next_w, find_close, L"GAME:\\data\\*"), L"sub"),
              "the parent folder stays listed after deleting a nested package file");
    }

    // DeleteFile checks the visible file's read-only bit. A package file is hidden by a marker instead of being
    // passed to the host DeleteFile, but that must not bypass the same refusal. Compare each API with a real host
    // file and keep the package and overlay attributes independent.
    {
        struct DeleteCase {
            const char* name;
            DWORD package_attributes;
            DWORD overlay_attributes;  // zero means there is no overlay copy
        };
        constexpr DWORD mixed = FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_ARCHIVE;
        const DeleteCase cases[] = {
            {"package-readonly", FILE_ATTRIBUTE_READONLY, 0},
            {"package-normal", FILE_ATTRIBUTE_NORMAL, 0},
            {"package-mixed-readonly", FILE_ATTRIBUTE_READONLY | mixed, 0},
            {"package-mixed-writable", mixed, 0},
            {"readonly-package-normal-overlay", FILE_ATTRIBUTE_READONLY, FILE_ATTRIBUTE_NORMAL},
            {"normal-package-readonly-overlay", FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_READONLY},
            {"readonly-package-readonly-overlay", FILE_ATTRIBUTE_READONLY, FILE_ATTRIBUTE_READONLY},
            {"readonly-package-mixed-overlay", FILE_ATTRIBUTE_READONLY, mixed},
            {"normal-package-mixed-readonly-overlay", FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_READONLY | mixed},
        };
        for (const bool narrow : {false, true}) {
            for (const auto& item : cases) {
                const std::string leaf = std::string(narrow ? "a-" : "w-") + item.name + ".bin";
                const std::filesystem::path relative = std::filesystem::path("delete-attributes") / leaf;
                const auto package_file = package / relative, overlay_file = overlay / relative;
                const auto native_file = root / "native-delete" / leaf;
                const auto deletion_marker = overlay / ".gta6rt-deleted" / relative;
                const bool has_overlay = item.overlay_attributes != 0;
                const DWORD visible_attributes = has_overlay ? item.overlay_attributes : item.package_attributes;
                const char* visible_bytes = has_overlay ? "overlay bytes" : "package bytes";
                std::printf("DeleteFile%c case: %s\n", narrow ? 'A' : 'W', item.name);
                write_path(package_file, "package bytes");
                check(::SetFileAttributesW(package_file.c_str(), item.package_attributes) != FALSE,
                      "set the synthetic package attributes");
                const DWORD package_before = ::GetFileAttributesW(package_file.c_str());
                if (has_overlay) {
                    write_path(overlay_file, visible_bytes);
                    check(::SetFileAttributesW(overlay_file.c_str(), item.overlay_attributes) != FALSE,
                          "set the visible overlay attributes");
                }
                write_path(native_file, visible_bytes);
                check(::SetFileAttributesW(native_file.c_str(), visible_attributes) != FALSE,
                      "set the independent native file attributes");
                std::string native_a;
                if (narrow) {
                    const UINT code_page = ::AreFileApisANSI() ? CP_ACP : CP_OEMCP;
                    const int size = ::WideCharToMultiByte(code_page, 0, native_file.c_str(), -1, nullptr, 0,
                                                           nullptr, nullptr);
                    native_a.resize(size > 0 ? static_cast<size_t>(size) : 1);
                    check(size > 0 && ::WideCharToMultiByte(code_page, 0, native_file.c_str(), -1, native_a.data(),
                                                           size, nullptr, nullptr) == size,
                          "encode the native DeleteFileA control path");
                }
                SetLastError(0);
                const BOOL native_ok = narrow ? ::DeleteFileA(native_a.c_str()) : ::DeleteFileW(native_file.c_str());
                const DWORD native_error = native_ok ? 0 : GetLastError();
                const bool readonly = (visible_attributes & FILE_ATTRIBUTE_READONLY) != 0;
                check((native_ok != FALSE) == !readonly && (native_ok || native_error == ERROR_ACCESS_DENIED),
                      "native deletion rejects read-only and accepts other attribute combinations");
                const std::wstring logical = L"GAME:\\" + relative.wstring();
                const std::string logical_a = "GAME:\\delete-attributes\\" + leaf;
                SetLastError(0);
                const BOOL mapped_ok = narrow ? delete_a(logical_a.c_str()) : delete_w(logical.c_str());
                const DWORD mapped_error = mapped_ok ? 0 : GetLastError();
                check((mapped_ok != FALSE) == (native_ok != FALSE) && mapped_error == native_error,
                      "shim deletion matches its native API's result and failure error");
                check(read_path(package_file) == "package bytes" &&
                          ::GetFileAttributesW(package_file.c_str()) == package_before,
                      "deletion leaves package bytes and attributes unchanged");
                check(std::filesystem::exists(deletion_marker) == !readonly,
                      "only an accepted delete publishes a marker");
                check(std::filesystem::exists(overlay_file) == (has_overlay && readonly),
                      "a refused delete keeps its overlay and never copies the package up");
                if (readonly) {
                    // The host's answer for a real file given the same attributes, which survived its refused
                    // delete: Windows reads back exactly what was set, while Wine 9 adds ARCHIVE and keeps neither
                    // HIDDEN nor SYSTEM without its xattr support (0x21 for every read-only case here).
                    const DWORD native_attributes = ::GetFileAttributesW(native_file.c_str());
                    check(native_attributes != INVALID_FILE_ATTRIBUTES &&
                              attributes_w(logical.c_str()) == native_attributes,
                          "a refused delete retains the visible attributes");
                    const HANDLE retained = create_w(logical.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                                     OPEN_EXISTING, 0, nullptr);
                    check(retained != INVALID_HANDLE_VALUE && read_handle(retained) == visible_bytes,
                          "a refused delete leaves the same logical bytes readable");
                    if (retained != INVALID_HANDLE_VALUE) ::CloseHandle(retained);
                } else {
                    SetLastError(0);
                    check(attributes_w(logical.c_str()) == INVALID_FILE_ATTRIBUTES &&
                              GetLastError() == ERROR_FILE_NOT_FOUND,
                          "an accepted delete hides the logical file");
                }
                // Restore only these synthetic files, including on a failed assertion, for remove_all below.
                ::SetFileAttributesW(package_file.c_str(), FILE_ATTRIBUTE_NORMAL);
                if (std::filesystem::exists(overlay_file)) ::SetFileAttributesW(overlay_file.c_str(), FILE_ATTRIBUTE_NORMAL);
                if (std::filesystem::exists(native_file)) ::SetFileAttributesW(native_file.c_str(), FILE_ATTRIBUTE_NORMAL);
            }
        }
    }

    // Deleting one package file must not hide its folder. The overlay has never written to GAME:\pkgonly, so the
    // folder is only in the package, and its marker folder under .gta6rt-deleted is not a marker for the folder.
    {
        write_path(package / "pkgonly" / "a.txt", "package");
        write_path(package / "pkgonly" / "b.txt", "package");
        check(delete_w(L"GAME:\\pkgonly\\a.txt") != FALSE, "DeleteFileW hides a file in a package-only folder");
        const DWORD folder = attributes_w(L"GAME:\\pkgonly");
        check(folder != INVALID_FILE_ATTRIBUTES && (folder & FILE_ATTRIBUTE_DIRECTORY) != 0,
              "the folder of a deleted package file still resolves");
        check(contains(list(first_w, next_w, find_close, L"GAME:\\*"), L"pkgonly"),
              "and is still in its parent's listing");
        check(contains(list(first_w, next_w, find_close, L"GAME:\\pkgonly\\*"), L"b.txt"),
              "and its other files are still listed");
        const HANDLE h =
            create_w(L"GAME:\\pkgonly\\b.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "and its other files still open");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
    }

    // Folders, moves, copies and attributes: the same mapping, through the names a title imports.
    const auto mkdir_w = resolve<CreateDirectoryWFn>("api-ms-win-core-file-l1-1-0.dll", "CreateDirectoryW");
    const auto mkdir_a = resolve<CreateDirectoryAFn>("kernel32.dll", "CreateDirectoryA");
    const auto mkdir_ex_w = resolve<CreateDirectoryExWFn>("kernel32.dll", "CreateDirectoryExW");
    const auto rmdir_w = resolve<RemoveDirectoryWFn>("api-ms-win-core-file-l1-1-0.dll", "RemoveDirectoryW");
    const auto rmdir_a = resolve<RemoveDirectoryAFn>("kernelbase.dll", "RemoveDirectoryA");
    const auto move_w = resolve<MoveFileWFn>("kernel32.dll", "MoveFileW");
    const auto move_ex_w = resolve<MoveFileExWFn>("api-ms-win-core-file-l2-1-0.dll", "MoveFileExW");
    const auto move_a = resolve<MoveFileAFn>("kernel32.dll", "MoveFileA");
    const auto copy_w = resolve<CopyFileWFn>("kernel32.dll", "CopyFileW");
    const auto copy_a = resolve<CopyFileAFn>("kernel32.dll", "CopyFileA");
    const auto copy_ex_w = resolve<CopyFileExWFn>("api-ms-win-core-file-l2-1-0.dll", "CopyFileExW");
    const auto copy2 = resolve<CopyFile2Fn>("api-ms-win-core-file-l2-1-0.dll", "CopyFile2");
    const auto setattr_w = resolve<SetFileAttributesWFn>("api-ms-win-core-file-l1-1-0.dll", "SetFileAttributesW");
    const auto setattr_a = resolve<SetFileAttributesAFn>("kernel32.dll", "SetFileAttributesA");
    check(mkdir_w && mkdir_a && mkdir_ex_w && rmdir_w && rmdir_a && move_w && move_ex_w && move_a && copy_w &&
              copy_a && copy_ex_w && copy2 && setattr_w && setattr_a,
          "the folder, move, copy and attribute calls resolve through the registry");

    write_path(package / "misc" / "one.txt", "one");
    write_path(package / "misc" / "two.txt", "two");
    write_path(package / "misc" / "three.txt", "three");
    write_path(package / "misc" / "inner" / "deep.txt", "deep");
    write_path(package / "misc" / "keep.txt", "keep");

    // CreateDirectory: the new folder is in the overlay; names the title can already see are refused.
    {
        check(mkdir_w(L"GAME:\\cache", nullptr) != FALSE, "CreateDirectoryW makes GAME:\\cache");
        check(std::filesystem::is_directory(overlay / "cache"), "in the overlay");
        check(!std::filesystem::exists(package / "cache"), "and not in the package");
        const DWORD a = attributes_w(L"GAME:\\cache");
        check(a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) != 0, "the new folder resolves");
        check(contains(list(first_w, next_w, find_close, L"GAME:\\*"), L"cache"), "and is listed");
        check(mkdir_a("GAME:\\cache\\narrow", nullptr) != FALSE && std::filesystem::is_directory(overlay / "cache" / "narrow"),
              "CreateDirectoryA takes a console path");
        check(mkdir_ex_w(L"GAME:\\misc", L"GAME:\\cache\\from_template", nullptr) != FALSE &&
                  std::filesystem::is_directory(overlay / "cache" / "from_template"),
              "CreateDirectoryExW maps both the template and the new folder");
        SetLastError(0);
        check(mkdir_w(L"GAME:\\misc", nullptr) == FALSE && GetLastError() == ERROR_ALREADY_EXISTS,
              "a folder only the package has already exists");
        check(!std::filesystem::exists(overlay / "misc"), "and nothing is mirrored for it");
        SetLastError(0);
        check(mkdir_w(L"GAME:\\misc\\one.txt", nullptr) == FALSE && GetLastError() == ERROR_ALREADY_EXISTS,
              "a package file's name already exists");
        SetLastError(0);
        check(mkdir_w(L"GAME:\\", nullptr) == FALSE && GetLastError() == ERROR_ALREADY_EXISTS, "GAME:\\ already exists");
        SetLastError(0);
        check(mkdir_w(L"GAME:\\nowhere\\child", nullptr) == FALSE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "a folder whose parent does not exist reports ERROR_PATH_NOT_FOUND");
        check(!std::filesystem::exists(overlay / "nowhere"), "and leaves no parent behind");
        check(mkdir_w(L"GAME:\\misc\\made", nullptr) != FALSE && std::filesystem::is_directory(overlay / "misc" / "made"),
              "a new folder inside a package folder lands in the overlay");
    }

    // RemoveDirectory: an overlay folder goes; a package folder is hidden by a marker once it is empty.
    {
        check(rmdir_w(L"GAME:\\cache\\narrow") != FALSE && !std::filesystem::exists(overlay / "cache" / "narrow"),
              "RemoveDirectoryW removes an empty overlay folder");
        SetLastError(0);
        check(rmdir_w(L"GAME:\\cache\\narrow") == FALSE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "removing it again reports ERROR_FILE_NOT_FOUND");
        SetLastError(0);
        check(rmdir_w(L"GAME:\\misc\\inner") == FALSE && GetLastError() == ERROR_DIR_NOT_EMPTY,
              "a package folder with files is not empty");
        SetLastError(0);
        check(rmdir_w(L"GAME:\\misc\\keep.txt") == FALSE && GetLastError() == ERROR_DIRECTORY,
              "a file is not a folder");
        SetLastError(0);
        check(rmdir_w(L"GAME:\\") == FALSE && GetLastError() == ERROR_ACCESS_DENIED, "GAME:\\ cannot be removed");
        check(delete_w(L"GAME:\\misc\\inner\\deep.txt") != FALSE, "once its file is deleted");
        check(rmdir_a("GAME:\\misc\\inner") != FALSE, "RemoveDirectoryA removes the now empty package folder");
        check(std::filesystem::exists(package / "misc" / "inner" / "deep.txt"), "the package keeps the folder and file");
        SetLastError(0);
        check(attributes_w(L"GAME:\\misc\\inner") == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_FILE_NOT_FOUND,
              "the removed folder no longer resolves");
        check(!contains(list(first_w, next_w, find_close, L"GAME:\\misc\\*"), L"inner"), "and is not listed");
        const DWORD misc = attributes_w(L"GAME:\\misc");
        check(misc != INVALID_FILE_ATTRIBUTES && (misc & FILE_ATTRIBUTE_DIRECTORY) != 0, "its parent is still there");
        check(mkdir_w(L"GAME:\\misc\\inner", nullptr) != FALSE, "the removed name can be made again");
        check(list(first_w, next_w, find_close, L"GAME:\\misc\\inner\\*").size() == 2,
              "and the new folder is empty: only . and ..");
    }

    // MoveFile: the destination is in the overlay; a moved package file is deleted by a marker.
    {
        check(move_w(L"GAME:\\misc\\one.txt", L"GAME:\\misc\\moved.txt") != FALSE, "MoveFileW moves a package file");
        check(read_path(overlay / "misc" / "moved.txt") == "one", "its bytes are in the overlay under the new name");
        check(read_path(package / "misc" / "one.txt") == "one", "the package file is not modified");
        SetLastError(0);
        check(attributes_w(L"GAME:\\misc\\one.txt") == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_FILE_NOT_FOUND,
              "the old name is gone");
        SetLastError(0);
        check(move_ex_w(L"GAME:\\misc\\moved.txt", L"GAME:\\misc\\two.txt", 0) == FALSE &&
                  GetLastError() == ERROR_ALREADY_EXISTS,
              "MoveFileExW will not replace a package file without MOVEFILE_REPLACE_EXISTING");
        check(move_ex_w(L"GAME:\\misc\\moved.txt", L"GAME:\\misc\\two.txt", MOVEFILE_REPLACE_EXISTING) != FALSE,
              "and replaces it with the flag");
        HANDLE h = create_w(L"GAME:\\misc\\two.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE && read_handle(h) == "one", "the replaced name reads the moved bytes");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(read_path(package / "misc" / "two.txt") == "two", "the package's own two.txt is untouched");
        check(!std::filesystem::exists(overlay / "misc" / "moved.txt"), "the overlay source of the move is gone");
        check(move_a("GAME:\\misc\\three.txt", "GAME:\\cache\\three.txt") != FALSE &&
                  read_path(overlay / "cache" / "three.txt") == "three",
              "MoveFileA moves into another folder");
        check(move_w(L"GAME:\\cache\\from_template", L"GAME:\\cache\\renamed") != FALSE &&
                  std::filesystem::is_directory(overlay / "cache" / "renamed"),
              "an overlay folder is renamed");
        SetLastError(0);
        check(move_w(L"GAME:\\misc", L"GAME:\\misc2") == FALSE && GetLastError() == ERROR_ACCESS_DENIED,
              "a package folder is not moved");
        SetLastError(0);
        check(move_w(L"GAME:\\misc\\keep.txt", (root / "host.txt").c_str()) == FALSE &&
                  GetLastError() == ERROR_NOT_SAME_DEVICE,
              "a move between GAME: and a host path is refused as cross-volume");
        SetLastError(0);
        check(move_ex_w(L"GAME:\\misc\\keep.txt", nullptr, MOVEFILE_DELAY_UNTIL_REBOOT) == FALSE &&
                  GetLastError() == ERROR_INVALID_PARAMETER,
              "a boot-time delete of a console file is refused");
        SetLastError(0);
        check(move_w(L"GAME:\\misc\\gone.txt", L"GAME:\\misc\\x.txt") == FALSE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "moving a missing file reports ERROR_FILE_NOT_FOUND");
        SetLastError(0);
        check(move_w(L"GAME:\\misc\\keep.txt", L"GAME:\\nowhere\\keep.txt") == FALSE &&
                  GetLastError() == ERROR_PATH_NOT_FOUND,
              "moving into a missing folder reports ERROR_PATH_NOT_FOUND");
    }

    // MOVEFILE_WRITE_THROUGH on a package file: the copy and the marker are flushed before the call returns.
    {
        write_path(package / "misc" / "wt.txt", "wt");
        check(move_ex_w(L"GAME:\\misc\\wt.txt", L"GAME:\\misc\\wt_moved.txt", MOVEFILE_WRITE_THROUGH) != FALSE,
              "MoveFileExW with MOVEFILE_WRITE_THROUGH moves a package file");
        check(read_path(overlay / "misc" / "wt_moved.txt") == "wt", "its bytes are in the overlay");
        SetLastError(0);
        check(attributes_w(L"GAME:\\misc\\wt.txt") == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_FILE_NOT_FOUND,
              "and the old name is deleted");
    }

    // CopyFile: reads where a read would, writes the overlay; an existing name counts even if only the package has it.
    {
        check(copy_w(L"GAME:\\misc\\keep.txt", L"GAME:\\cache\\keep_copy.txt", TRUE) != FALSE &&
                  read_path(overlay / "cache" / "keep_copy.txt") == "keep",
              "CopyFileW copies a package file into the overlay");
        check(read_path(package / "misc" / "keep.txt") == "keep", "and the source stays");
        SetLastError(0);
        check(copy_w(L"GAME:\\misc\\keep.txt", L"GAME:\\data\\sub", FALSE) == FALSE && GetLastError() == ERROR_ACCESS_DENIED,
              "a copy onto a folder is refused");
        write_path(package / "misc" / "target.txt", "target");
        SetLastError(0);
        check(copy_w(L"GAME:\\misc\\keep.txt", L"GAME:\\misc\\target.txt", TRUE) == FALSE &&
                  GetLastError() == ERROR_FILE_EXISTS,
              "fail-if-exists sees a name only the package has");
        check(copy_a("GAME:\\misc\\keep.txt", "GAME:\\misc\\target.txt", FALSE) != FALSE &&
                  read_path(overlay / "misc" / "target.txt") == "keep" && read_path(package / "misc" / "target.txt") == "target",
              "CopyFileA overwrites through the overlay and leaves the package file alone");
        const std::filesystem::path host_copy = root / "host_copy.txt";
        check(copy_w(L"GAME:\\misc\\keep.txt", host_copy.c_str(), TRUE) != FALSE && read_path(host_copy) == "keep",
              "a copy from GAME: to a host path works");
        check(copy_ex_w(host_copy.c_str(), L"GAME:\\cache\\from_host.txt", nullptr, nullptr, nullptr, COPY_FILE_FAIL_IF_EXISTS) !=
                      FALSE &&
                  read_path(overlay / "cache" / "from_host.txt") == "keep",
              "CopyFileExW copies from a host path into GAME:");
        COPYFILE2_EXTENDED_PARAMETERS params{};
        params.dwSize = sizeof(params);
        params.dwCopyFlags = COPY_FILE_FAIL_IF_EXISTS;
        check(copy2(L"GAME:\\misc\\keep.txt", L"GAME:\\cache\\copy2.txt", &params) == S_OK &&
                  read_path(overlay / "cache" / "copy2.txt") == "keep",
              "CopyFile2 copies too");
        check(copy2(L"GAME:\\misc\\keep.txt", L"GAME:\\misc\\two.txt", &params) == HRESULT_FROM_WIN32(ERROR_FILE_EXISTS),
              "and returns its HRESULT for an existing name");

        // CopyFile2 again, through the host's CopyFile2 (where it has one) and through the CopyFileExW fallback the
        // shim uses where it has none (Wine 9), so both are checked wherever the test runs.
        for (const bool fallback : {false, true}) {
            rt::fs::force_copy_file2_fallback(fallback);
            const std::string mode = fallback ? " (fallback)" : " (host CopyFile2, or the fallback without one)";
            const std::wstring dir = fallback ? L"GAME:\\copy2_fallback" : L"GAME:\\copy2_host";
            const std::filesystem::path out = overlay / (fallback ? "copy2_fallback" : "copy2_host");
            const auto to = [&](const wchar_t* name) { return dir + L"\\" + name; };
            check(mkdir_w(dir.c_str(), nullptr) != FALSE, ("a folder for the CopyFile2 checks" + mode).c_str());

            check(copy2(L"GAME:\\misc\\keep.txt", to(L"null.txt").c_str(), nullptr) == S_OK &&
                      read_path(out / "null.txt") == "keep",
                  ("CopyFile2 without a parameter block copies" + mode).c_str());
            for (const DWORD size : {DWORD{0}, DWORD{4}, DWORD{sizeof(COPYFILE2_EXTENDED_PARAMETERS) - 1}}) {
                COPYFILE2_EXTENDED_PARAMETERS short_block{};
                short_block.dwSize = size;
                check(copy2(L"GAME:\\misc\\keep.txt", to(L"short.txt").c_str(), &short_block) == E_INVALIDARG &&
                          !std::filesystem::exists(out / "short.txt"),
                      ("a parameter block of " + std::to_string(size) +
                       " bytes is E_INVALIDARG and writes nothing" + mode)
                          .c_str());
            }
            COPYFILE2_EXTENDED_PARAMETERS block{};
            block.dwSize = sizeof(block);
            block.dwCopyFlags = COPY_FILE_FAIL_IF_EXISTS | 0x00040000 /* NO_OFFLOAD, a hint */;
            check(copy2(L"GAME:\\misc\\keep.txt", to(L"hint.txt").c_str(), &block) == S_OK &&
                      read_path(out / "hint.txt") == "keep",
                  ("fail-if-exists with a hint flag copies" + mode).c_str());
            check(copy2(L"GAME:\\misc\\keep.txt", to(L"hint.txt").c_str(), &block) ==
                      HRESULT_FROM_WIN32(ERROR_FILE_EXISTS),
                  ("and refuses an existing name" + mode).c_str());
            BOOL cancel = TRUE;
            block.dwCopyFlags = 0;
            block.pfCancel = &cancel;
            check(copy2(L"GAME:\\misc\\keep.txt", to(L"cancelled.txt").c_str(), &block) ==
                          HRESULT_FROM_WIN32(ERROR_REQUEST_ABORTED) &&
                      !std::filesystem::exists(out / "cancelled.txt"),
                  ("a cancel flag set before the copy cancels it" + mode).c_str());
            cancel = FALSE;
            check(copy2(L"GAME:\\misc\\keep.txt", to(L"cancelled.txt").c_str(), &block) == S_OK &&
                      read_path(out / "cancelled.txt") == "keep",
                  ("clearing the cancel flag allows a retry" + mode).c_str());
            block.pfCancel = nullptr;
            if (fallback) {
                // A flag that changes what the copy produces, and which CopyFileExW can't do, is refused before
                // anything is written: the fallback doesn't copy the alternate streams it was told to skip.
                block.dwCopyFlags = 0x00008000 /* COPY_FILE_SKIP_ALTERNATE_STREAMS */;
                check(copy2(L"GAME:\\misc\\keep.txt", to(L"streams.txt").c_str(), &block) ==
                              HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED) &&
                          !std::filesystem::exists(out / "streams.txt"),
                      "the fallback refuses SKIP_ALTERNATE_STREAMS and writes nothing");
                block.dwCopyFlags = 0x00000080 /* COPY_FILE_DIRECTORY */;
                check(copy2(L"GAME:\\misc\\keep.txt", to(L"directory.txt").c_str(), &block) ==
                              HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED),
                      "and COPY_FILE_DIRECTORY");
                block.dwCopyFlags = 0;
                block.dwSize = sizeof(block) + 8;  // a longer (V2-sized) block: its extra fields have no equivalent
                check(copy2(L"GAME:\\misc\\keep.txt", to(L"v2.txt").c_str(), &block) ==
                              HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED) &&
                          !std::filesystem::exists(out / "v2.txt"),
                      "and a block longer than COPYFILE2_EXTENDED_PARAMETERS");
            }
        }
        rt::fs::force_copy_file2_fallback(false);
        SetLastError(0);
        check(copy_w(L"GAME:\\misc\\one.txt", L"GAME:\\cache\\one.txt", FALSE) == FALSE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "a deleted name cannot be copied");
    }

    // SetFileAttributes: the overlay copy gets the attributes; the package keeps its own.
    {
        const DWORD before = ::GetFileAttributesW((package / "misc" / "keep.txt").c_str());
        check(setattr_w(L"GAME:\\misc\\keep.txt", FILE_ATTRIBUTE_READONLY) != FALSE, "SetFileAttributesW on a package file");
        const DWORD now = attributes_w(L"GAME:\\misc\\keep.txt");
        check(now != INVALID_FILE_ATTRIBUTES && (now & FILE_ATTRIBUTE_READONLY) != 0, "the title sees the new attributes");
        check(read_path(overlay / "misc" / "keep.txt") == "keep", "on a copy of the file in the overlay");
        check(::GetFileAttributesW((package / "misc" / "keep.txt").c_str()) == before, "the package file keeps its own");
        check(setattr_a("GAME:\\misc\\keep.txt", FILE_ATTRIBUTE_NORMAL) != FALSE &&
                  (attributes_w(L"GAME:\\misc\\keep.txt") & FILE_ATTRIBUTE_READONLY) == 0,
              "SetFileAttributesA changes the overlay copy back");
        check(setattr_w(L"GAME:\\data\\sub", FILE_ATTRIBUTE_HIDDEN) != FALSE &&
                  (attributes_w(L"GAME:\\data\\sub") & FILE_ATTRIBUTE_HIDDEN) != 0,
              "a package folder's attributes are set on an overlay mirror");
        check(count(list(first_w, next_w, find_close, L"GAME:\\data\\*"), L"sub") == 1, "which is listed once");
        SetLastError(0);
        check(setattr_w(L"GAME:\\misc\\one.txt", FILE_ATTRIBUTE_NORMAL) == FALSE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "a deleted name has no attributes to set");
    }

    // Nothing leaves the package and the overlay.
    {
        SetLastError(0);
        check(mkdir_w(L"GAME:\\..\\escape_dir", nullptr) == FALSE && GetLastError() == ERROR_INVALID_NAME &&
                  !std::filesystem::exists(root / "escape_dir"),
              "a folder above GAME: is refused");
        SetLastError(0);
        check(move_w(L"GAME:\\misc\\keep.txt", L"GAME:\\..\\escape.txt") == FALSE && GetLastError() == ERROR_INVALID_NAME,
              "a move above GAME: is refused");
        SetLastError(0);
        check(copy_w(L"GAME:\\misc\\keep.txt", L"GAME:\\C:\\x", FALSE) == FALSE && GetLastError() == ERROR_INVALID_NAME,
              "a copy to a drive inside GAME: is refused");
        SetLastError(0);
        check(rmdir_w(L"GAME:\\..") == FALSE && GetLastError() == ERROR_INVALID_NAME, "removing above GAME: is refused");
    }

    // Nothing leaves the package and the overlay.
    {
        SetLastError(0);
        const HANDLE h = create_w(L"GAME:\\..\\escape.txt", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_INVALID_NAME, "a path above GAME: is refused");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(!std::filesystem::exists(root / "escape.txt"), "and nothing was created outside");
        SetLastError(0);
        check(attributes_w(L"GAME:\\..\\x") == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_INVALID_NAME,
              "a stat above GAME: is refused");
        SetLastError(0);
        check(create_w(L"GAME:\\C:\\x", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr) == INVALID_HANDLE_VALUE &&
                  GetLastError() == ERROR_INVALID_NAME,
              "a drive inside GAME: is refused");
        SetLastError(0);
        WIN32_FIND_DATAW found{};
        check(first_w(L"GAME:\\..\\*", &found) == INVALID_HANDLE_VALUE && GetLastError() == ERROR_INVALID_NAME,
              "a listing above GAME: is refused");
        SetLastError(0);
        check(first_w(L"GAME:\\data\\*", nullptr) == INVALID_HANDLE_VALUE &&
                  GetLastError() == ERROR_INVALID_PARAMETER,
              "a null listing buffer is refused");
    }

    // CreateFile2 and FindFirstFileExA/W (used by GDK/OneCore titles and std::filesystem).
    {
        const auto create2 = resolve<CreateFile2Fn>("api-ms-win-core-file-l1-2-0.dll", "CreateFile2");
        const auto first_ex_w = resolve<FindFirstFileExWFn>("api-ms-win-core-file-l1-1-0.dll", "FindFirstFileExW");
        const auto first_ex_a = resolve<FindFirstFileExAFn>("kernelbase.dll", "FindFirstFileExA");
        check(create2 && first_ex_w && first_ex_a, "CreateFile2 and FindFirstFileExA/W are registered");

        // CreateFile2 reads an untouched package file without modifying it.
        write_path(package / "data" / "cf2_pkg.txt", "pkg-cf2");
        CREATEFILE2_EXTENDED_PARAMETERS ext{};
        ext.dwSize = sizeof(ext);
        ext.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
        const HANDLE h_read = create2(L"GAME:\\data\\cf2_pkg.txt", GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, &ext);
        check(h_read != INVALID_HANDLE_VALUE && read_handle(h_read) == "pkg-cf2",
              "CreateFile2 reads a GAME: file from the package");
        if (h_read != INVALID_HANDLE_VALUE) ::CloseHandle(h_read);

        // CreateFile2 write goes to the overlay, leaving the package untouched.
        const HANDLE h_write = create2(L"GAME:\\data\\cf2.txt", GENERIC_WRITE, 0, CREATE_ALWAYS, &ext);
        check(h_write != INVALID_HANDLE_VALUE, "CreateFile2 creates a new GAME: file in the overlay");
        if (h_write != INVALID_HANDLE_VALUE) {
            DWORD written = 0;
            ::WriteFile(h_write, "cf2", 3, &written, nullptr);
            ::CloseHandle(h_write);
        }
        check(read_path(overlay / "data" / "cf2.txt") == "cf2" && !std::filesystem::exists(package / "data" / "cf2.txt"),
              "CreateFile2 writes land in the overlay and never in the package");

        // CreateFile2 refuses path traversal.
        SetLastError(0);
        const HANDLE h_esc = create2(L"GAME:\\..\\cf2_escape.txt", GENERIC_WRITE, 0, CREATE_ALWAYS, &ext);
        check(h_esc == INVALID_HANDLE_VALUE && GetLastError() == ERROR_INVALID_NAME,
              "CreateFile2 refuses a path above GAME:");
        if (h_esc != INVALID_HANDLE_VALUE) ::CloseHandle(h_esc);

        // MAXIMUM_ALLOWED asks for every right the file grants, write included, without naming one. Such an open of
        // a package file is a write-open: it gets the overlay copy, so a write through the handle cannot reach the
        // package. CreateFileW, CreateFileA and CreateFile2 share the rule. The write is shorter than the file, so
        // the bytes after it show the overlay copy started as the package file's contents (OPEN_EXISTING keeps them).
        {
            write_path(package / "data" / "max_w.txt", "pkg-max");
            write_path(package / "data" / "max_a.txt", "pkg-max");
            write_path(package / "data" / "max_2.txt", "pkg-max");
            const auto write_through = [](HANDLE h) {
                if (h == INVALID_HANDLE_VALUE) return false;
                DWORD written = 0;
                const BOOL ok = ::WriteFile(h, "OV", 2, &written, nullptr);
                ::CloseHandle(h);
                return ok != FALSE && written == 2;
            };
            check(write_through(create_w(L"GAME:\\data\\max_w.txt", MAXIMUM_ALLOWED, 0, nullptr, OPEN_EXISTING, 0,
                                         nullptr)),
                  "CreateFileW(MAXIMUM_ALLOWED, OPEN_EXISTING) opens a package file for writing");
            check(read_path(package / "data" / "max_w.txt") == "pkg-max",
                  "a write through a MAXIMUM_ALLOWED handle does not reach the package (CreateFileW)");
            check(read_path(overlay / "data" / "max_w.txt") == "OVg-max",
                  "the write through a MAXIMUM_ALLOWED handle is in the overlay copy (CreateFileW)");

            check(write_through(create_a("GAME:\\data\\max_a.txt", MAXIMUM_ALLOWED | GENERIC_READ, 0, nullptr,
                                         OPEN_EXISTING, 0, nullptr)),
                  "CreateFileA(MAXIMUM_ALLOWED | GENERIC_READ, OPEN_EXISTING) opens a package file for writing");
            check(read_path(package / "data" / "max_a.txt") == "pkg-max" &&
                      read_path(overlay / "data" / "max_a.txt") == "OVg-max",
                  "the write goes to the overlay and not the package (CreateFileA)");

            check(write_through(create2(L"GAME:\\data\\max_2.txt", MAXIMUM_ALLOWED, 0, OPEN_EXISTING, &ext)),
                  "CreateFile2(MAXIMUM_ALLOWED, OPEN_EXISTING) opens a package file for writing");
            check(read_path(package / "data" / "max_2.txt") == "pkg-max" &&
                      read_path(overlay / "data" / "max_2.txt") == "OVg-max",
                  "the write goes to the overlay and not the package (CreateFile2)");

            // A plain read still opens the package file in place: no overlay copy appears for it.
            write_path(package / "data" / "max_read.txt", "pkg-max");
            const HANDLE h_plain = create_w(L"GAME:\\data\\max_read.txt", GENERIC_READ, FILE_SHARE_READ, nullptr,
                                            OPEN_EXISTING, 0, nullptr);
            check(h_plain != INVALID_HANDLE_VALUE && read_handle(h_plain) == "pkg-max" &&
                      !std::filesystem::exists(overlay / "data" / "max_read.txt"),
                  "a GENERIC_READ open is still a read: no overlay copy is made");
            if (h_plain != INVALID_HANDLE_VALUE) ::CloseHandle(h_plain);
        }

        // FindFirstFileExW merges package + overlay and honours FindExInfoBasic & FindExSearchLimitToDirectories.
        WIN32_FIND_DATAW fd{};
        std::vector<std::wstring> basic_names;
        bool alt_empty = true;
        HANDLE hf = first_ex_w(L"GAME:\\data\\*", FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr, 0);
        check(hf != INVALID_HANDLE_VALUE, "FindFirstFileExW opens a GAME: directory listing");
        if (hf != INVALID_HANDLE_VALUE) {
            do {
                basic_names.emplace_back(fd.cFileName);
                if (fd.cAlternateFileName[0] != L'\0') alt_empty = false;
            } while (next_w(hf, &fd));
            find_close(hf);
        }
        check(contains(basic_names, L"hello.txt") && contains(basic_names, L"cf2.txt") && !contains(basic_names, L"bye.txt"),
              "FindFirstFileExW merges overlay + package and hides deleted files");
        check(alt_empty, "FindFirstFileExW(FindExInfoBasic) leaves cAlternateFileName empty");

        std::vector<std::wstring> dir_names;
        HANDLE hdir = first_ex_w(L"GAME:\\data\\*", FindExInfoStandard, &fd, FindExSearchLimitToDirectories, nullptr, 0);
        check(hdir != INVALID_HANDLE_VALUE, "FindFirstFileExW(FindExSearchLimitToDirectories) succeeds");
        if (hdir != INVALID_HANDLE_VALUE) {
            do {
                dir_names.emplace_back(fd.cFileName);
            } while (next_w(hdir, &fd));
            find_close(hdir);
        }
        check(contains(dir_names, L"sub") && !contains(dir_names, L"hello.txt"),
              "FindFirstFileExW(FindExSearchLimitToDirectories) returns only directories");

        // FindFirstFileExA works on GAME: paths too.
        WIN32_FIND_DATAA fda{};
        HANDLE hfa = first_ex_a("GAME:\\data\\cf2.txt", FindExInfoBasic, &fda, FindExSearchNameMatch, nullptr, 0);
        check(hfa != INVALID_HANDLE_VALUE && std::string(fda.cFileName) == "cf2.txt",
              "FindFirstFileExA finds an overlay file under GAME:");
        if (hfa != INVALID_HANDLE_VALUE) find_close(hfa);
    }

    // Copy-up under two threads: both write-open the same untouched package file at once. Each open must succeed and
    // the overlay copy must hold the package's bytes; the copy goes through a private name, so neither thread can find
    // the name taken mid-copy or read a half-written file.
    {
        const std::string big(256 * 1024, 'p');  // long enough that the two copies overlap
        bool all_opened = true, all_intact = true;
        for (int round = 0; round < 50; ++round) {
            const std::wstring name = L"race" + std::to_wstring(round) + L".bin";
            write_path(package / "data" / name, big);
            const std::wstring game = L"GAME:\\data\\" + name;
            HANDLE handles[2] = {INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE};
            std::thread threads[2];
            for (int i = 0; i < 2; ++i)
                threads[i] = std::thread([&, i] {
                    handles[i] = create_w(game.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                          nullptr, OPEN_EXISTING, 0, nullptr);                });
            for (auto& t : threads) t.join();
            for (HANDLE h : handles) {
                if (h == INVALID_HANDLE_VALUE) all_opened = false;
                else ::CloseHandle(h);
            }
            if (read_path(overlay / "data" / name) != big) all_intact = false;
        }
        check(all_opened, "two threads write-opening one package file both succeed");
        check(all_intact, "the overlay copy made under two threads holds the whole package file");
        std::error_code ec;
        check(std::filesystem::is_empty(overlay / ".gta6rt-copy", ec) && !ec, "copy-up leaves no staging file behind");

        // A leftover from a process killed mid-copy stays out of the title's view.
        write_path(overlay / ".gta6rt-copy" / "123-1", "half");
        check(!contains(list(first_w, next_w, find_close, L"GAME:\\*"), L".gta6rt-copy"),
              "the copy staging folder is hidden from a GAME: listing");
    }

    // RT-218: "G:\" is the package and "T:\" a writable scratch folder in the overlay, as in the other console layers.
    // A host path under the configured folders stays a host path, and a path that climbs out of the drive is refused.
    {
        write_path(package / "drives" / "sub" / "probe.txt", "probe");
        HANDLE h = create_w(L"G:\\drives\\sub\\probe.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE && read_handle(h) == "probe", "CreateFileW opens G:\\drives\\sub\\probe.txt from the package");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        h = create_w(L"g:/drives/sub/probe.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "the drive letter is case-insensitive and accepts forward slashes");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(attributes_w(L"G:\\data\\sub") != INVALID_FILE_ATTRIBUTES, "GetFileAttributesW stats a G: folder");

        SetLastError(0);
        check(create_w(L"G:\\..\\x.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr) ==
                      INVALID_HANDLE_VALUE &&
                  GetLastError() == ERROR_INVALID_NAME,
              "G:\\.. is refused with ERROR_INVALID_NAME");

        int fresh_count = 0;
        auto fresh_overlay = [&] {
            const std::filesystem::path dir = root / ("fresh_overlay" + std::to_string(++fresh_count));
            rt::fs::configure(package, dir);
            return dir;
        };
        fresh_overlay();
        SetLastError(0);
        const DWORD t_root = attributes_w(L"T:\\");
        check(t_root != INVALID_FILE_ATTRIBUTES && (t_root & FILE_ATTRIBUTE_DIRECTORY) != 0 && GetLastError() == 0,
              "on a fresh overlay GetFileAttributesW(T:\\) is a folder, and making it leaves the last error alone");
        fresh_overlay();
        const std::vector<std::wstring> t_names = list(first_w, next_w, find_close, L"T:\\*");
        check(t_names.size() == 2 && count(t_names, L".") == 1 && count(t_names, L"..") == 1,
              "on a fresh overlay FindFirstFileW(T:\\*) lists the empty root as . and ..");
        fresh_overlay();
        check(mkdir_w(L"T:\\shadercache", nullptr) != FALSE,
              "on a fresh overlay CreateDirectoryW(T:\\shadercache) succeeds");
        std::filesystem::path fresh = fresh_overlay();
        check(copy_w(L"GAME:\\data\\hello.txt", L"T:\\hello.tmp", FALSE) != FALSE &&
                  read_path(fresh / ".gta6rt-temp" / "hello.tmp") == "package",
              "on a fresh overlay CopyFileW(GAME:\\data\\hello.txt, T:\\hello.tmp) succeeds");
        fresh = fresh_overlay();
        SetLastError(0);
        check(rmdir_w(L"T:\\") == FALSE && GetLastError() == ERROR_ACCESS_DENIED &&
                  std::filesystem::is_directory(fresh / ".gta6rt-temp"),
              "T:\\ cannot be removed, like GAME:\\");
        fresh = fresh_overlay();
        SetLastError(0);
        check(move_w(L"T:\\", L"GAME:\\moved") == FALSE && GetLastError() == ERROR_ACCESS_DENIED &&
                  attributes_w(L"GAME:\\moved") == INVALID_FILE_ATTRIBUTES &&
                  std::filesystem::is_directory(fresh / ".gta6rt-temp"),
              "T:\\ cannot be moved");
        fresh_overlay();
        attributes_w(L"T:\\");
        SetLastError(0);
        const BOOL removed_by_game_path = rmdir_w(L"GAME:\\.gta6rt-temp");
        check(removed_by_game_path != FALSE || GetLastError() != ERROR_ACCESS_DENIED,
              "only a T: path names the T:\\ root: GAME:\\.gta6rt-temp is not refused as a drive root");
        rt::fs::configure(package, overlay);

        h = create_w(L"T:\\cache\\scratch.bin", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "CreateFileW creates a file on T: (parent folders and all)");
        if (h != INVALID_HANDLE_VALUE) {
            DWORD written = 0;
            ::WriteFile(h, "scratch", 7, &written, nullptr);
            ::CloseHandle(h);
        }
        check(read_path(overlay / ".gta6rt-temp" / "cache" / "scratch.bin") == "scratch",
              "a T: file lands in the overlay's scratch folder");
        check(!std::filesystem::exists(package / "cache" / "scratch.bin") && !std::filesystem::exists(package / ".gta6rt-temp"),
              "and the package is untouched");
        h = create_w(L"t:/cache/scratch.bin", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE && read_handle(h) == "scratch", "the T: file reads back");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(attributes_w(L"GAME:\\cache\\scratch.bin") == INVALID_FILE_ATTRIBUTES,
              "T: is its own drive: GAME: does not see the scratch file");
        check(!contains(list(first_w, next_w, find_close, L"GAME:\\*"), L".gta6rt-temp"),
              "a listing of GAME:\\ does not show the scratch folder");
        SetLastError(0);
        check(create_w(L"T:\\..\\data\\hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr) ==
                      INVALID_HANDLE_VALUE &&
                  GetLastError() == ERROR_INVALID_NAME,
              "T:\\.. cannot climb out of the scratch folder into the package");

        // The overlay's own folders (.gta6rt-deleted, .gta6rt-temp, .gta6rt-copy) are hidden from GAME:\ listings and,
        // as an explicit GAME: or G: path, refused like any other invalid name, in every spelling parse_console()
        // normalises (RT-294). T:\ still names the scratch folder, and an ordinary dotted name is not reserved.
        write_path(overlay / ".gta6rt-temp" / "runtime.tmp", "runtime");
        write_path(overlay / ".gta6rt-deleted" / "data" / "gone.txt", "");
        write_path(overlay / ".gta6rt-other" / "ok.txt", "ok");
        for (const wchar_t* reserved : {L"GAME:\\.gta6rt-temp\\runtime.tmp", L"GAME:\\.GTA6RT-TEMP\\runtime.tmp",
                                        L"game:/.gta6rt-temp/runtime.tmp", L"GAME:\\.\\.gta6rt-temp\\runtime.tmp",
                                        L"GAME:\\data\\..\\.gta6rt-temp\\runtime.tmp", L"G:\\.gta6rt-temp\\runtime.tmp",
                                        L"GAME:\\.gta6rt-deleted\\data\\gone.txt", L"GAME:\\.gta6rt-copy\\x.bin",
                                        L"GAME:\\.gta6rt-temp"}) {
            SetLastError(0);
            const HANDLE h = create_w(reserved, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
            check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_INVALID_NAME,
                  "an explicit GAME: path into a reserved overlay folder is refused (CreateFileW)");
            if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
            SetLastError(0);
            check(attributes_w(reserved) == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_INVALID_NAME,
                  "and has no attributes (GetFileAttributesW)");
        }
        WIN32_FIND_DATAW marker_data{};
        SetLastError(0);
        const HANDLE markers = first_w(L"GAME:\\.gta6rt-deleted\\data\\*", &marker_data);
        check(markers == INVALID_HANDLE_VALUE && GetLastError() == ERROR_INVALID_NAME,
              "deletion markers cannot be listed through GAME:\\.gta6rt-deleted");
        if (markers != INVALID_HANDLE_VALUE) find_close(markers);
        SetLastError(0);
        const HANDLE probe = create_w(L"GAME:\\.gta6rt-temp\\probe.txt", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(probe == INVALID_HANDLE_VALUE && !std::filesystem::exists(overlay / ".gta6rt-temp" / "probe.txt"),
              "a write through GAME:\\.gta6rt-temp creates nothing in the scratch folder");
        if (probe != INVALID_HANDLE_VALUE) ::CloseHandle(probe);
        check(mkdir_w(L"GAME:\\.gta6rt-copy\\sub", nullptr) == FALSE && !std::filesystem::exists(overlay / ".gta6rt-copy" / "sub"),
              "CreateDirectoryW refuses a reserved folder");
        check(delete_w(L"GAME:\\.gta6rt-deleted\\data\\gone.txt") == FALSE &&
                  std::filesystem::exists(overlay / ".gta6rt-deleted" / "data" / "gone.txt"),
              "DeleteFileW cannot remove a deletion marker through GAME:");
        HANDLE through = create_w(L"T:\\runtime.tmp", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(through != INVALID_HANDLE_VALUE && read_handle(through) == "runtime", "T:\\ still reaches the scratch folder");
        if (through != INVALID_HANDLE_VALUE) ::CloseHandle(through);
        through = create_w(L"GAME:\\.gta6rt-other\\ok.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(through != INVALID_HANDLE_VALUE && read_handle(through) == "ok", "an ordinary dotted name under GAME: is not reserved");
        if (through != INVALID_HANDLE_VALUE) ::CloseHandle(through);

        // A package that itself sits on drive G: is reached by host paths, which must stay host paths.
        rt::fs::configure(L"G:\\leonida_missing_package", L"G:\\leonida_missing_overlay");
        SetLastError(0);
        check(attributes_w(L"G:\\leonida_missing_package\\data.txt") == INVALID_FILE_ATTRIBUTES &&
                  GetLastError() != ERROR_INVALID_NAME,
              "a path under a configured G: game folder is a host path");
        // No roots configured: nothing to map G: and T: onto, so they are host paths.
        rt::fs::configure({}, {});
        SetLastError(0);
        check(attributes_w(L"T:\\x") == INVALID_FILE_ATTRIBUTES && GetLastError() != ERROR_INVALID_NAME,
              "without configured folders G: and T: stay host drives");
        rt::fs::configure(package, overlay);
    }

    // An open that writes needs its folder to exist in the title's view, and a failed open changes nothing.
    {
        const auto overlay_has = [&](const wchar_t* name) { return std::filesystem::exists(overlay / name); };
        write_path(package / "rules" / "file.txt", "package");
        write_path(package / "rules" / "gone" / "only.txt", "package");

        SetLastError(0);
        HANDLE h = create_w(L"GAME:\\nodir\\sub\\x.dat", GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0,
                            nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "a write-open of an existing file in a missing folder fails with ERROR_PATH_NOT_FOUND");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(attributes_w(L"GAME:\\nodir") == INVALID_FILE_ATTRIBUTES && !overlay_has(L"nodir"),
              "the failed open did not create the missing folders");

        SetLastError(0);
        h = create_w(L"GAME:\\nodir2\\x.dat", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "CREATE_ALWAYS in a missing folder fails with ERROR_PATH_NOT_FOUND, as on the host");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(attributes_w(L"GAME:\\nodir2") == INVALID_FILE_ATTRIBUTES && !overlay_has(L"nodir2"),
              "and creates neither the folder nor the file");

        SetLastError(0);
        h = create_w(L"GAME:\\rules\\file.txt\\x.dat", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "CREATE_ALWAYS below a package file fails with ERROR_PATH_NOT_FOUND");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        const DWORD file_attributes = attributes_w(L"GAME:\\rules\\file.txt");
        check(file_attributes != INVALID_FILE_ATTRIBUTES && (file_attributes & FILE_ATTRIBUTE_DIRECTORY) == 0,
              "and the package file is still a file in the title's view");

        check(delete_w(L"GAME:\\rules\\gone\\only.txt") != FALSE && rmdir_w(L"GAME:\\rules\\gone") != FALSE,
              "a package folder is emptied and removed");
        SetLastError(0);
        h = create_w(L"GAME:\\rules\\gone\\new.dat", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "CREATE_ALWAYS inside a removed folder fails with ERROR_PATH_NOT_FOUND");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(attributes_w(L"GAME:\\rules\\gone") == INVALID_FILE_ATTRIBUTES,
              "and does not bring the removed folder back");

        // The folders that do exist still work: one only the package has, and one the title made.
        h = create_w(L"GAME:\\rules\\new.dat", GENERIC_WRITE, 0, nullptr, CREATE_NEW, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "CREATE_NEW in a folder only the package has still succeeds");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(overlay_has(L"rules\\new.dat") && !std::filesystem::exists(package / "rules" / "new.dat"),
              "and the file is in the overlay");
        check(mkdir_w(L"GAME:\\rules\\made", nullptr) != FALSE, "a folder the title makes");
        h = create_w(L"GAME:\\rules\\made\\x.dat", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "takes a new file");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
    }

    // A listing needs its folder to exist in the title's view: a removed or missing folder is a missing path, not an
    // empty folder and not a missing file.
    {
        write_path(package / "listing" / "old" / "a.txt", "package");
        WIN32_FIND_DATAW found{};
        check(delete_w(L"GAME:\\listing\\old\\a.txt") != FALSE && rmdir_w(L"GAME:\\listing\\old") != FALSE,
              "a package folder is emptied and removed (listing)");
        SetLastError(0);
        HANDLE h = first_w(L"GAME:\\listing\\old\\*", &found);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "FindFirstFileW inside a removed package folder fails with ERROR_PATH_NOT_FOUND");
        if (h != INVALID_HANDLE_VALUE) find_close(h);
        SetLastError(0);
        h = first_w(L"GAME:\\listing\\never\\*", &found);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "FindFirstFileW inside a folder that never existed fails with ERROR_PATH_NOT_FOUND");
        if (h != INVALID_HANDLE_VALUE) find_close(h);
        SetLastError(0);
        h = first_w(L"GAME:\\listing\\old", &found);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "FindFirstFileW of the removed folder's own name is ERROR_FILE_NOT_FOUND: its parent exists");
        if (h != INVALID_HANDLE_VALUE) find_close(h);
        SetLastError(0);
        h = first_w(L"GAME:\\listing\\nothing*.bin", &found);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "a pattern that matches nothing in an existing folder is still ERROR_FILE_NOT_FOUND");
        if (h != INVALID_HANDLE_VALUE) find_close(h);
        check(list(first_w, next_w, find_close, L"GAME:\\listing\\*").size() == 2,
              "and the emptied parent lists only . and ..");

        // A listing whose folder is a file is neither: the host answers ERROR_DIRECTORY. One level further down the
        // path is simply missing again. (Case measured on the host and reported in review by shdann.)
        write_path(package / "listing" / "file.txt", "package");
        for (const wchar_t* pattern : {L"GAME:\\listing\\file.txt\\*", L"GAME:\\listing\\file.txt\\x"}) {
            SetLastError(0);
            h = first_w(pattern, &found);
            check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_DIRECTORY,
                  "FindFirstFileW below a file fails with ERROR_DIRECTORY, as the host does");
            if (h != INVALID_HANDLE_VALUE) find_close(h);
        }
        SetLastError(0);
        h = first_w(L"GAME:\\listing\\file.txt\\sub\\*", &found);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND,
              "and two levels below a file with ERROR_PATH_NOT_FOUND");
        if (h != INVALID_HANDLE_VALUE) find_close(h);
    }

    // A move onto the same name succeeds and changes nothing; a rename that changes only the case renames.
    {
        const auto names_in = [&](const wchar_t* pattern) { return list(first_w, next_w, find_close, pattern); };
        const auto has_exact = [](const std::vector<std::wstring>& names, const wchar_t* name) {
            for (const std::wstring& n : names)
                if (n == name) return true;
            return false;
        };
        write_path(package / "moves" / "pkg.dat", "package");
        write_path(package / "moves" / "same.dat", "package");
        check(mkdir_w(L"GAME:\\moves\\dir", nullptr) != FALSE, "an overlay folder for the case rename");
        HANDLE h = create_w(L"GAME:\\moves\\save.dat", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        DWORD written = 0;
        if (h != INVALID_HANDLE_VALUE) {
            ::WriteFile(h, "save", 4, &written, nullptr);
            ::CloseHandle(h);
        }

        check(move_w(L"GAME:\\moves\\save.dat", L"GAME:\\moves\\save.dat") != FALSE,
              "MoveFileW of an overlay file onto itself succeeds");
        check(move_w(L"GAME:\\moves\\same.dat", L"GAME:\\moves\\same.dat") != FALSE,
              "MoveFileW of a package file onto itself succeeds");
        check(read_path(package / "moves" / "same.dat") == "package" && !std::filesystem::exists(overlay / "moves" / "same.dat"),
              "and a package file moved onto itself is neither copied nor hidden");
        HANDLE r = create_w(L"GAME:\\moves\\same.dat", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(r != INVALID_HANDLE_VALUE && read_handle(r) == "package", "it still reads");
        if (r != INVALID_HANDLE_VALUE) ::CloseHandle(r);

        // An overlay entry's case rename is the host's own rename, so the new spelling is only checked where the
        // host's MoveFileW changes it (not under Wine); the entry is still listed exactly once everywhere.
        const bool host_case = host_renames_case(root / "case_probe");
        if (!host_case)
            std::printf("SKIP the host's MoveFileW keeps the old case on a case-only rename (Wine): the listing's "
                        "spelling of a renamed overlay entry is not checked\n");
        check(move_w(L"GAME:\\moves\\save.dat", L"GAME:\\moves\\SAVE.DAT") != FALSE,
              "MoveFileW renames an overlay file to the same name in another case");
        std::vector<std::wstring> names = names_in(L"GAME:\\moves\\*");
        if (host_case)
            check(has_exact(names, L"SAVE.DAT") && !has_exact(names, L"save.dat") && count(names, L"save.dat") == 1,
                  "the listing shows the new case, once");
        else
            check(count(names, L"save.dat") == 1, "the listing shows the renamed file once");
        check(read_path(overlay / "moves" / "SAVE.DAT") == "save", "and the bytes are the same");

        check(move_w(L"GAME:\\moves\\pkg.dat", L"GAME:\\moves\\PKG.DAT") != FALSE,
              "MoveFileW renames a package file to the same name in another case");
        names = names_in(L"GAME:\\moves\\*");
        check(has_exact(names, L"PKG.DAT") && !has_exact(names, L"pkg.dat") && count(names, L"pkg.dat") == 1,
              "the listing shows the package file under its new case, once");
        r = create_w(L"GAME:\\moves\\pkg.dat", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(r != INVALID_HANDLE_VALUE && read_handle(r) == "package", "and it opens under either case");
        if (r != INVALID_HANDLE_VALUE) ::CloseHandle(r);
        check(read_path(package / "moves" / "pkg.dat") == "package", "the package file is untouched");

        check(move_w(L"GAME:\\moves\\dir", L"GAME:\\moves\\DIR") != FALSE,
              "MoveFileW renames an overlay folder to the same name in another case");
        names = names_in(L"GAME:\\moves\\*");
        if (host_case)
            check(has_exact(names, L"DIR"), "the listing shows the folder's new case");
        else
            check(count(names, L"dir") == 1, "the listing shows the renamed folder once");
        check(move_w(L"GAME:\\moves", L"GAME:\\moves") != FALSE, "MoveFileW of a package folder onto itself succeeds");
        SetLastError(0);
        check(move_w(L"GAME:\\moves", L"GAME:\\MOVES") == FALSE && GetLastError() == ERROR_ACCESS_DENIED,
              "a case rename of a package folder is refused like any other move of one");
    }
}

}  // namespace
