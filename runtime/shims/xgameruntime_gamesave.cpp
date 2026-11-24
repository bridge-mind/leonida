// xgameruntime: XGameSaveImpl, the class xgameruntime.lib's XGameSave* functions reach through QueryApiImpl, offline.
//
// A GDK title opens its save provider during boot (XGameSaveInitializeProvider), then reads and writes containers
// of named blobs. Each XGameSave* function in xgameruntime.lib asks QueryApiImpl for XGameSaveImpl and calls one slot
// with its own arguments after `this`. The class and interface ids and the slot order are
// xodus-gaming/xgameruntime-docs' "Layout" lists (GPL-3.0; facts only, nothing copied). Signatures, limits and error
// codes follow Microsoft's public GDK reference (learn.microsoft.com/gaming/gdk: the XGameSave function pages and
// "Error codes"). No GDK header or library code is used.
//
// Offline the saves are files, in the layout tools/saves.py imports into (docs/saves.md):
//   <root>\local-user\<container>\<blob>      root = LEONIDA_SAVE_DIR, else %LOCALAPPDATA%\Leonida\saves
// Container and blob names are %-encoded: [A-Za-z0-9_-] and '.' (except as the first character) stay, every other
// UTF-8 byte becomes %XX. A container's display name is kept in "<container>\.display", which no encoded blob name
// can collide with (an encoded name never starts with '.'). Names ending in '~' are temporary files of an update in
// progress; an encoded name never ends in '~' either.
//
// An update buffers its writes and deletes; SubmitUpdate writes every new blob to "<blob>~", then renames each over
// its blob and removes the deleted ones, so a failed write leaves the container as it was.
//
// ASSUMPTION (no executable yet): the quota is a fixed 1 GiB (the GDK says the per-user quota "is not a fixed size");
// a blob name is at most 256 bytes (GS_MAX_BLOB_NAME_SIZE is not on the public pages); GetContainerInfo of a
// container that does not exist returns S_OK without calling the callback; ReadBlobData with blobNames = nullptr
// reads every blob of the container. syncOnDemand is accepted and ignored: nothing syncs offline.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

namespace {

namespace fs = std::filesystem;

// ---- Public GDK types (XGameSave.h, as the GDK reference documents them) ----

using XUserHandle = void*;
struct XGameSaveProvider;
struct XGameSaveContainer;
struct XGameSaveUpdate;
using XGameSaveProviderHandle = XGameSaveProvider*;
using XGameSaveContainerHandle = XGameSaveContainer*;
using XGameSaveUpdateHandle = XGameSaveUpdate*;

struct XGameSaveContainerInfo {
    const char* name;
    const char* displayName;
    uint32_t blobCount;
    uint64_t totalSize;
    time_t lastModifiedTime;
    bool needsSync;
};
struct XGameSaveBlobInfo {
    const char* name;
    uint32_t size;
};
struct XGameSaveBlob {
    XGameSaveBlobInfo info;
    uint8_t* data;
};
using ContainerInfoCallback = bool(CALLBACK*)(const XGameSaveContainerInfo* info, void* context);
using BlobInfoCallback = bool(CALLBACK*)(const XGameSaveBlobInfo* info, void* context);

// XGameErr.h values, from the GDK "Error codes" page (Game save).
constexpr HRESULT E_GS_INVALID_CONTAINER_NAME = static_cast<HRESULT>(0x80830001);
constexpr HRESULT E_GS_OUT_OF_LOCAL_STORAGE = static_cast<HRESULT>(0x80830003);
constexpr HRESULT E_GS_UPDATE_TOO_BIG = static_cast<HRESULT>(0x80830005);
constexpr HRESULT E_GS_QUOTA_EXCEEDED = static_cast<HRESULT>(0x80830006);
constexpr HRESULT E_GS_PROVIDED_BUFFER_TOO_SMALL = static_cast<HRESULT>(0x80830007);
constexpr HRESULT E_GS_BLOB_NOT_FOUND = static_cast<HRESULT>(0x80830008);
constexpr HRESULT E_GS_HANDLE_EXPIRED = static_cast<HRESULT>(0x8083000D);

constexpr uint64_t kMaxUpdateBytes = 16ull * 1024 * 1024;   // GS_MAX_BLOB_SIZE, 16 MB: one blob and one update
constexpr uint64_t kQuotaBytes = 1ull << 30;                 // ASSUMPTION, see the header
constexpr size_t kMaxContainerName = 256;
constexpr size_t kMaxBlobName = 256;                         // ASSUMPTION, see the header

// ---- Names ----

std::string encode_name(std::string_view name) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    for (const char ch : name) {
        const auto c = static_cast<unsigned char>(ch);
        const bool keep = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
                          c == '-' || (c == '.' && !out.empty());
        if (keep) {
            out += ch;
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

// The inverse; false for a file name encode_name could not have produced (temporary files, .display, strays).
bool decode_name(std::string_view encoded, std::string& out) {
    out.clear();
    if (encoded.empty() || encoded.front() == '.' || encoded.back() == '~') return false;
    auto digit = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] != '%') {
            out += encoded[i];
            continue;
        }
        if (i + 2 >= encoded.size()) return false;
        const int hi = digit(encoded[i + 1]), lo = digit(encoded[i + 2]);
        if (hi < 0 || lo < 0) return false;
        out += static_cast<char>(hi << 4 | lo);
        i += 2;
    }
    return encode_name(out) == encoded;   // one spelling per name
}

// XGameSaveCreateContainer's rule: at most 256 characters; a path part (up to the last '/') of [A-Za-z0-9_/], then a
// non-empty file part of [A-Za-z0-9_.-] that does not end in '.' nor hold "..".
bool valid_container_name(const char* name) {
    if (!name) return false;
    const std::string_view n(name);
    if (n.empty() || n.size() > kMaxContainerName) return false;
    const size_t slash = n.rfind('/');
    const std::string_view path = slash == std::string_view::npos ? std::string_view{} : n.substr(0, slash + 1);
    const std::string_view file = slash == std::string_view::npos ? n : n.substr(slash + 1);
    auto alnum = [](char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); };
    for (const char c : path)
        if (!alnum(c) && c != '_' && c != '/') return false;
    if (file.empty() || file.back() == '.' || file.find("..") != std::string_view::npos) return false;
    for (const char c : file)
        if (!alnum(c) && c != '_' && c != '.' && c != '-') return false;
    return true;
}

bool valid_blob_name(const char* name) { return name && name[0] && std::strlen(name) <= kMaxBlobName; }

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}
std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

fs::path save_root() {
    wchar_t buffer[MAX_PATH * 2];
    DWORD n = GetEnvironmentVariableW(L"LEONIDA_SAVE_DIR", buffer, static_cast<DWORD>(std::size(buffer)));
    if (n > 0 && n < std::size(buffer)) return fs::path(buffer);
    n = GetEnvironmentVariableW(L"LOCALAPPDATA", buffer, static_cast<DWORD>(std::size(buffer)));
    if (n > 0 && n < std::size(buffer)) return fs::path(buffer) / L"Leonida" / L"saves";
    return {};
}

// ---- Handles ----

// One mutex for the handle sets and the files: saves are small and rare, so nothing is gained by finer locking.
std::recursive_mutex g_mutex;

struct XGameSaveProvider {
    fs::path user_dir;   // <root>\local-user
    std::string config_id;
    uint64_t serial = 0;
};
struct XGameSaveContainer {
    fs::path user_dir;   // the provider's, copied: a container outlives a closed provider safely
    std::string name;
    uint64_t serial = 0;
    fs::path dir() const { return user_dir / widen(encode_name(name)); }
};
struct Op {
    bool write;
    std::vector<uint8_t> data;
};
struct XGameSaveUpdate {
    XGameSaveContainer* container;
    uint64_t container_serial;       // the container's serial when the update was made
    std::string display_name;
    bool has_display_name;
    std::map<std::string, Op> ops;   // blob name -> its one write or delete
    uint64_t bytes;
    uint64_t serial = 0;
};

std::unordered_set<const void*> g_providers, g_containers, g_updates;

bool live(const std::unordered_set<const void*>& set, const void* h) { return h && set.count(h) != 0; }

// Every handle gets a serial when it is made, and no two get the same one. The heap may hand a closed handle's address
// to the next handle of its kind, so whatever keeps a handle past the call that gave it (an update its container, an
// async job its provider, container or update) keeps the serial as well, and the handle counts as the same one only
// while both match. A reused address is then refused as expired instead of acting on another save. Under g_mutex.
uint64_t g_next_serial = 1;

template <class T> bool live_as(const std::unordered_set<const void*>& set, const T* h, uint64_t serial) {
    return live(set, h) && h->serial == serial;
}

// The serial of a live handle; 0, which no handle has, for one that isn't, so a job made from it fails as expired.
template <class T> uint64_t serial_of(const std::unordered_set<const void*>& set, const T* h) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return live(set, h) ? h->serial : 0;
}

// ---- Files ----

struct BlobFile {
    std::string name;
    uint64_t size;
    fs::path path;
};

std::vector<BlobFile> list_blobs(const fs::path& dir) {
    std::vector<BlobFile> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        std::string name;
        if (!decode_name(narrow(e.path().filename().wstring()), name)) continue;
        out.push_back({name, static_cast<uint64_t>(e.file_size(ec)), e.path()});
    }
    std::sort(out.begin(), out.end(), [](const BlobFile& a, const BlobFile& b) { return a.name < b.name; });
    return out;
}

time_t modified_time(const fs::path& p) {
    WIN32_FILE_ATTRIBUTE_DATA data{};
    if (!GetFileAttributesExW(p.c_str(), GetFileExInfoStandard, &data)) return 0;
    const uint64_t t = (static_cast<uint64_t>(data.ftLastWriteTime.dwHighDateTime) << 32) | data.ftLastWriteTime.dwLowDateTime;
    return t < 116444736000000000ull ? 0 : static_cast<time_t>((t - 116444736000000000ull) / 10000000ull);
}

std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return f ? std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()) : std::string{};
}

bool write_file(const fs::path& p, const void* data, size_t size) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    f.close();
    return !f.fail();
}

struct ContainerRecord {
    std::string name, display_name;
    uint32_t blob_count = 0;
    uint64_t total_size = 0;
    time_t modified = 0;
};

ContainerRecord describe(const std::string& name, const fs::path& dir) {
    ContainerRecord r;
    r.name = name;
    r.display_name = read_text(dir / L".display");
    for (const BlobFile& b : list_blobs(dir)) {
        ++r.blob_count;
        r.total_size += b.size;
        const time_t t = modified_time(b.path);
        if (t > r.modified) r.modified = t;
    }
    return r;
}

std::vector<ContainerRecord> list_containers(const fs::path& user_dir, std::string_view prefix) {
    std::vector<ContainerRecord> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(user_dir, ec)) {
        if (!e.is_directory(ec)) continue;
        std::string name;
        if (!decode_name(narrow(e.path().filename().wstring()), name)) continue;
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        out.push_back(describe(name, e.path()));
    }
    std::sort(out.begin(), out.end(), [](const ContainerRecord& a, const ContainerRecord& b) { return a.name < b.name; });
    return out;
}

uint64_t used_bytes(const fs::path& user_dir) {
    uint64_t used = 0;
    for (const ContainerRecord& c : list_containers(user_dir, {})) used += c.total_size;
    return used;
}

void enumerate_containers(const fs::path& user_dir, std::string_view prefix, void* context, ContainerInfoCallback cb) {
    for (const ContainerRecord& c : list_containers(user_dir, prefix)) {
        const XGameSaveContainerInfo info{c.name.c_str(), c.display_name.c_str(), c.blob_count, c.total_size,
                                          c.modified, false};
        if (!cb(&info, context)) break;
    }
}

// ---- Reading blobs into the title's buffer ----
//
// The buffer is laid out as the GDK samples read it: an array of XGameSaveBlob, one per blob, whose info.name and
// data point further into the same buffer (each blob's bytes 8-aligned, then its name).

struct BlobRead {
    std::string name;
    std::vector<uint8_t> data;
};

size_t align8(size_t n) { return (n + 7) & ~size_t{7}; }

size_t packed_size(const std::vector<BlobRead>& blobs) {
    size_t n = align8(blobs.size() * sizeof(XGameSaveBlob));
    for (const BlobRead& b : blobs) n += align8(b.data.size()) + align8(b.name.size() + 1);
    return n;
}

void pack(const std::vector<BlobRead>& blobs, void* buffer) {
    auto* base = static_cast<uint8_t*>(buffer);
    auto* entries = static_cast<XGameSaveBlob*>(buffer);
    size_t at = align8(blobs.size() * sizeof(XGameSaveBlob));
    for (size_t i = 0; i < blobs.size(); ++i) {
        uint8_t* data = base + at;
        if (!blobs[i].data.empty()) std::memcpy(data, blobs[i].data.data(), blobs[i].data.size());
        at += align8(blobs[i].data.size());
        char* name = reinterpret_cast<char*>(base + at);
        std::memcpy(name, blobs[i].name.c_str(), blobs[i].name.size() + 1);
        at += align8(blobs[i].name.size() + 1);
        entries[i].info.name = name;
        entries[i].info.size = static_cast<uint32_t>(blobs[i].data.size());
        entries[i].data = data;
    }
}

// Reads the named blobs (every blob when names is null) of a live container.
HRESULT read_blobs(const XGameSaveContainer* c, const char* const* names, uint32_t count, std::vector<BlobRead>& out) {
    out.clear();
    const fs::path dir = c->dir();
    std::vector<std::string> wanted;
    if (names) {
        for (uint32_t i = 0; i < count; ++i) {
            if (!valid_blob_name(names[i])) return E_INVALIDARG;
            wanted.emplace_back(names[i]);
        }
    } else {
        for (const BlobFile& b : list_blobs(dir)) wanted.push_back(b.name);
    }
    for (const std::string& name : wanted) {
        std::ifstream f(dir / widen(encode_name(name)), std::ios::binary);
        if (!f) return E_GS_BLOB_NOT_FOUND;
        out.push_back({name, std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>())});
    }
    return S_OK;
}

// ---- Committing an update ----

HRESULT commit(XGameSaveUpdate* u) {
    XGameSaveContainer* c = u->container;
    const fs::path dir = c->dir();
    // The quota: what the container holds after the update, plus every other container.
    uint64_t after = used_bytes(c->user_dir);
    for (const auto& [name, op] : u->ops) {
        std::error_code ec;
        const fs::path p = dir / widen(encode_name(name));
        const uint64_t old = fs::is_regular_file(p, ec) ? static_cast<uint64_t>(fs::file_size(p, ec)) : 0;
        after = after - old + (op.write ? op.data.size() : 0);
    }
    if (after > kQuotaBytes) return E_GS_QUOTA_EXCEEDED;

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return E_GS_OUT_OF_LOCAL_STORAGE;
    std::vector<std::pair<fs::path, fs::path>> renames;   // temporary -> final
    for (const auto& [name, op] : u->ops) {
        if (!op.write) continue;
        const fs::path final_path = dir / widen(encode_name(name));
        fs::path temp = final_path;
        temp += L"~";
        if (!write_file(temp, op.data.data(), op.data.size())) {
            for (const auto& r : renames) fs::remove(r.first, ec);
            fs::remove(temp, ec);
            return E_GS_OUT_OF_LOCAL_STORAGE;
        }
        renames.emplace_back(temp, final_path);
    }
    for (const auto& [temp, final_path] : renames)
        if (!MoveFileExW(temp.c_str(), final_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            return E_GS_OUT_OF_LOCAL_STORAGE;
    for (const auto& [name, op] : u->ops)
        if (!op.write) fs::remove(dir / widen(encode_name(name)), ec);
    if (u->has_display_name) write_file(dir / L".display", u->display_name.data(), u->display_name.size());
    return S_OK;
}


// ---- Operations shared by the sync slots and the async work ----

HRESULT open_provider(XUserHandle user, const char* config_id, XGameSaveProviderHandle* out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!user || !config_id) return E_INVALIDARG;
    const fs::path root = save_root();
    if (root.empty()) return E_GS_OUT_OF_LOCAL_STORAGE;
    const fs::path user_dir = root / L"local-user";
    std::error_code ec;
    fs::create_directories(user_dir, ec);
    if (ec) return E_GS_OUT_OF_LOCAL_STORAGE;
    auto* p = new XGameSaveProvider{user_dir, config_id};
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    p->serial = g_next_serial++;
    g_providers.insert(p);
    *out = p;
    static std::atomic<int> s_logged{0};
    if (s_logged.fetch_add(1) == 0)
        std::fprintf(stderr, "[rt] xgamesave: offline provider for \"%s\" at %ls\n", config_id, user_dir.c_str());
    return S_OK;
}

HRESULT remaining_quota(XGameSaveProviderHandle p, int64_t* remaining) {
    if (!remaining) return E_POINTER;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_providers, p)) return E_GS_HANDLE_EXPIRED;
    const uint64_t used = used_bytes(p->user_dir);
    *remaining = used >= kQuotaBytes ? 0 : static_cast<int64_t>(kQuotaBytes - used);
    return S_OK;
}

HRESULT delete_container(XGameSaveProviderHandle p, const char* name) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_providers, p)) return E_GS_HANDLE_EXPIRED;
    if (!valid_container_name(name)) return E_GS_INVALID_CONTAINER_NAME;
    std::error_code ec;
    fs::remove_all(p->user_dir / widen(encode_name(name)), ec);
    return ec ? E_FAIL : S_OK;
}

HRESULT submit_update(XGameSaveUpdateHandle u) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_updates, u) || !live_as(g_containers, u->container, u->container_serial)) return E_GS_HANDLE_EXPIRED;
    return commit(u);
}

// ---- Async: each *Async slot runs its work on the title's queue ----

struct Job {
    const void* identity;
    XUserHandle user = nullptr;
    std::string text;   // configId
    XGameSaveProviderHandle provider = nullptr;
    XGameSaveContainerHandle container = nullptr;
    XGameSaveUpdateHandle update = nullptr;
    uint64_t serial = 0;   // the serial of the provider, container or update above, when the job was made
    std::vector<std::string> names;
    bool all_blobs = false;
    // results
    int64_t quota = 0;
    std::vector<BlobRead> blobs;
    std::string folder;
};

const char kInitIdentity = 0, kQuotaIdentity = 0, kDeleteIdentity = 0, kReadIdentity = 0, kSubmitIdentity = 0,
           kFolderIdentity = 0;

// The blob count of the last ReadBlobDataAsync result collected on this thread (GetResult runs on the caller's).
thread_local uint32_t t_read_count = 0;

HRESULT do_work(Job& job, size_t& size) {
    size = 0;
    if (job.identity == &kInitIdentity) {
        // The handle is made when the title collects the result, so an uncollected result leaks nothing; here only
        // the folder is checked.
        if (!job.user) return E_INVALIDARG;
        const fs::path root = save_root();
        std::error_code ec;
        if (root.empty() || (fs::create_directories(root / L"local-user", ec), ec)) return E_GS_OUT_OF_LOCAL_STORAGE;
        size = sizeof(XGameSaveProviderHandle);
        return S_OK;
    }
    if (job.identity == &kQuotaIdentity) {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!live_as(g_providers, job.provider, job.serial)) return E_GS_HANDLE_EXPIRED;
        const HRESULT hr = remaining_quota(job.provider, &job.quota);
        if (SUCCEEDED(hr)) size = sizeof(int64_t);
        return hr;
    }
    if (job.identity == &kDeleteIdentity) {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!live_as(g_providers, job.provider, job.serial)) return E_GS_HANDLE_EXPIRED;
        return delete_container(job.provider, job.text.c_str());
    }
    if (job.identity == &kSubmitIdentity) {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!live_as(g_updates, job.update, job.serial)) return E_GS_HANDLE_EXPIRED;
        return submit_update(job.update);
    }
    if (job.identity == &kReadIdentity) {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!live_as(g_containers, job.container, job.serial)) return E_GS_HANDLE_EXPIRED;
        std::vector<const char*> names;
        for (const std::string& n : job.names) names.push_back(n.c_str());
        const HRESULT hr = read_blobs(job.container, job.all_blobs ? nullptr : names.data(),
                                      static_cast<uint32_t>(names.size()), job.blobs);
        if (SUCCEEDED(hr)) size = packed_size(job.blobs);
        return hr;
    }
    if (job.identity == &kFolderIdentity) {
        const fs::path root = save_root();
        if (root.empty()) return E_GS_OUT_OF_LOCAL_STORAGE;
        const fs::path dir = root / L"files" / widen(encode_name(job.text));
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) return E_GS_OUT_OF_LOCAL_STORAGE;
        job.folder = narrow(dir.wstring());
        size = job.folder.size() + 1;
        return S_OK;
    }
    return E_UNEXPECTED;
}

HRESULT CALLBACK job_provider(XAsyncOp op, const XAsyncProviderData* data) {
    auto* job = static_cast<Job*>(data->context);
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork: {
        size_t size = 0;
        const HRESULT hr = do_work(*job, size);
        XAsyncComplete(data->async, hr, SUCCEEDED(hr) ? size : 0);
        return S_OK;
    }
    case XAsyncOp::GetResult:
        if (!data->buffer) return E_INVALIDARG;
        if (job->identity == &kInitIdentity) {
            if (data->bufferSize < sizeof(XGameSaveProviderHandle)) return E_INVALIDARG;
            return open_provider(job->user, job->text.c_str(), static_cast<XGameSaveProviderHandle*>(data->buffer));
        }
        if (job->identity == &kQuotaIdentity) {
            if (data->bufferSize < sizeof(int64_t)) return E_INVALIDARG;
            *static_cast<int64_t*>(data->buffer) = job->quota;
            return S_OK;
        }
        if (job->identity == &kReadIdentity) {
            if (data->bufferSize < packed_size(job->blobs)) return E_GS_PROVIDED_BUFFER_TOO_SMALL;
            pack(job->blobs, data->buffer);
            t_read_count = static_cast<uint32_t>(job->blobs.size());
            return S_OK;
        }
        if (job->identity == &kFolderIdentity) {
            if (data->bufferSize < job->folder.size() + 1) return E_GS_PROVIDED_BUFFER_TOO_SMALL;
            std::memcpy(data->buffer, job->folder.c_str(), job->folder.size() + 1);
            return S_OK;
        }
        return S_OK;
    case XAsyncOp::Cleanup:
        delete job;
        return S_OK;
    default:
        return S_OK;
    }
}

HRESULT begin(XAsyncBlock* async, Job* job, const char* name) {
    if (!async) {
        delete job;
        return E_POINTER;
    }
    const HRESULT hr = XAsyncBegin(async, job, job->identity, name, job_provider);
    if (FAILED(hr)) delete job;
    return hr;
}

// ---- IXGameSaveImpl (slots 3-29), IXGameSaveImpl2 (30-32) ----

// 3: HRESULT XGameSaveInitializeProvider(XUserHandle requestingUser, const char* configurationId, bool syncOnDemand,
//                                         XGameSaveProviderHandle* provider)
HRESULT STDMETHODCALLTYPE save_init(void*, XUserHandle user, const char* config_id, bool, XGameSaveProviderHandle* out) {
    return open_provider(user, config_id, out);
}

// 4: HRESULT XGameSaveInitializeProviderAsync(XUserHandle, const char* configurationId, bool syncOnDemand, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE save_init_async(void*, XUserHandle user, const char* config_id, bool, XAsyncBlock* async) {
    if (!user || !config_id) return E_INVALIDARG;
    auto* job = new Job{&kInitIdentity};
    job->user = user;
    job->text = config_id;
    return begin(async, job, "XGameSaveInitializeProviderAsync");
}

// 5: HRESULT XGameSaveInitializeProviderResult(XAsyncBlock* async, XGameSaveProviderHandle* provider)
HRESULT STDMETHODCALLTYPE save_init_result(void*, XAsyncBlock* async, XGameSaveProviderHandle* out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    return XAsyncGetResult(async, &kInitIdentity, sizeof(*out), out, nullptr);
}

// 6: void XGameSaveCloseProvider(XGameSaveProviderHandle provider)
void STDMETHODCALLTYPE save_close_provider(void*, XGameSaveProviderHandle p) {
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!p || g_providers.erase(p) == 0) return;
    }
    delete p;
}

// 7: HRESULT XGameSaveGetRemainingQuota(XGameSaveProviderHandle provider, int64_t* remainingQuota)
HRESULT STDMETHODCALLTYPE save_quota(void*, XGameSaveProviderHandle p, int64_t* remaining) {
    return remaining_quota(p, remaining);
}

// 8: HRESULT XGameSaveGetRemainingQuotaAsync(XGameSaveProviderHandle provider, XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE save_quota_async(void*, XGameSaveProviderHandle p, XAsyncBlock* async) {
    auto* job = new Job{&kQuotaIdentity};
    job->provider = p;
    job->serial = serial_of(g_providers, p);
    return begin(async, job, "XGameSaveGetRemainingQuotaAsync");
}

// 9: HRESULT XGameSaveGetRemainingQuotaResult(XAsyncBlock* async, int64_t* remainingQuota)
HRESULT STDMETHODCALLTYPE save_quota_result(void*, XAsyncBlock* async, int64_t* remaining) {
    if (!remaining) return E_POINTER;
    return XAsyncGetResult(async, &kQuotaIdentity, sizeof(*remaining), remaining, nullptr);
}

// 10: HRESULT XGameSaveDeleteContainer(XGameSaveProviderHandle provider, const char* containerName)
HRESULT STDMETHODCALLTYPE save_delete(void*, XGameSaveProviderHandle p, const char* name) {
    return delete_container(p, name);
}

// 11: HRESULT XGameSaveDeleteContainerAsync(XGameSaveProviderHandle provider, const char* containerName, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE save_delete_async(void*, XGameSaveProviderHandle p, const char* name, XAsyncBlock* async) {
    if (!valid_container_name(name)) return E_GS_INVALID_CONTAINER_NAME;
    auto* job = new Job{&kDeleteIdentity};
    job->provider = p;
    job->serial = serial_of(g_providers, p);
    job->text = name;
    return begin(async, job, "XGameSaveDeleteContainerAsync");
}

// 12: HRESULT XGameSaveDeleteContainerResult(XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE save_delete_result(void*, XAsyncBlock* async) {
    return XAsyncGetResult(async, &kDeleteIdentity, 0, nullptr, nullptr);
}

// 13: HRESULT XGameSaveGetContainerInfo(XGameSaveProviderHandle, const char* containerName, void* context,
//                                       XGameSaveContainerInfoCallback* callback)
HRESULT STDMETHODCALLTYPE save_container_info(void*, XGameSaveProviderHandle p, const char* name, void* context,
                                              ContainerInfoCallback cb) {
    if (!cb) return E_INVALIDARG;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_providers, p)) return E_GS_HANDLE_EXPIRED;
    if (!valid_container_name(name)) return E_GS_INVALID_CONTAINER_NAME;
    const fs::path dir = p->user_dir / widen(encode_name(name));
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return S_OK;
    const ContainerRecord c = describe(name, dir);
    const XGameSaveContainerInfo info{c.name.c_str(), c.display_name.c_str(), c.blob_count, c.total_size, c.modified,
                                      false};
    cb(&info, context);
    return S_OK;
}

// 14: HRESULT XGameSaveEnumerateContainerInfo(XGameSaveProviderHandle, void* context, XGameSaveContainerInfoCallback*)
HRESULT STDMETHODCALLTYPE save_enum_containers(void*, XGameSaveProviderHandle p, void* context, ContainerInfoCallback cb) {
    if (!cb) return E_INVALIDARG;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_providers, p)) return E_GS_HANDLE_EXPIRED;
    enumerate_containers(p->user_dir, {}, context, cb);
    return S_OK;
}

// 15: HRESULT XGameSaveEnumerateContainerInfoByName(XGameSaveProviderHandle, const char* containerNamePrefix,
//                                                   void* context, XGameSaveContainerInfoCallback*)
HRESULT STDMETHODCALLTYPE save_enum_containers_by_name(void*, XGameSaveProviderHandle p, const char* prefix,
                                                       void* context, ContainerInfoCallback cb) {
    if (!cb) return E_INVALIDARG;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_providers, p)) return E_GS_HANDLE_EXPIRED;
    enumerate_containers(p->user_dir, prefix ? std::string_view(prefix) : std::string_view{}, context, cb);
    return S_OK;
}

// 16: HRESULT XGameSaveCreateContainer(XGameSaveProviderHandle, const char* containerName, XGameSaveContainerHandle*).
// The folder appears when the first update is submitted.
HRESULT STDMETHODCALLTYPE save_create_container(void*, XGameSaveProviderHandle p, const char* name,
                                                XGameSaveContainerHandle* out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_providers, p)) return E_GS_HANDLE_EXPIRED;
    if (!valid_container_name(name)) return E_GS_INVALID_CONTAINER_NAME;
    auto* c = new XGameSaveContainer{p->user_dir, name};
    c->serial = g_next_serial++;
    g_containers.insert(c);
    *out = c;
    return S_OK;
}

// 17: void XGameSaveCloseContainer(XGameSaveContainerHandle context)
void STDMETHODCALLTYPE save_close_container(void*, XGameSaveContainerHandle c) {
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!c || g_containers.erase(c) == 0) return;
    }
    delete c;
}

HRESULT enumerate_blobs(XGameSaveContainerHandle c, std::string_view prefix, void* context, BlobInfoCallback cb) {
    if (!cb) return E_INVALIDARG;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_containers, c)) return E_GS_HANDLE_EXPIRED;
    for (const BlobFile& b : list_blobs(c->dir())) {
        if (b.name.compare(0, prefix.size(), prefix) != 0) continue;
        const XGameSaveBlobInfo info{b.name.c_str(), static_cast<uint32_t>(b.size)};
        if (!cb(&info, context)) break;
    }
    return S_OK;
}

// 18: HRESULT XGameSaveEnumerateBlobInfo(XGameSaveContainerHandle, void* context, XGameSaveBlobInfoCallback*)
HRESULT STDMETHODCALLTYPE save_enum_blobs(void*, XGameSaveContainerHandle c, void* context, BlobInfoCallback cb) {
    return enumerate_blobs(c, {}, context, cb);
}

// 19: HRESULT XGameSaveEnumerateBlobInfoByName(XGameSaveContainerHandle, const char* blobNamePrefix, void* context,
//                                              XGameSaveBlobInfoCallback*)
HRESULT STDMETHODCALLTYPE save_enum_blobs_by_name(void*, XGameSaveContainerHandle c, const char* prefix, void* context,
                                                  BlobInfoCallback cb) {
    return enumerate_blobs(c, prefix ? std::string_view(prefix) : std::string_view{}, context, cb);
}

// 20: HRESULT XGameSaveReadBlobData(XGameSaveContainerHandle, const char** blobNames, uint32_t* countOfBlobs,
//                                   size_t blobsSize, XGameSaveBlob* blobData)
HRESULT STDMETHODCALLTYPE save_read(void*, XGameSaveContainerHandle c, const char** names, uint32_t* count,
                                    size_t size, XGameSaveBlob* out) {
    if (!count || !out) return E_POINTER;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_containers, c)) return E_GS_HANDLE_EXPIRED;
    std::vector<BlobRead> blobs;
    const HRESULT hr = read_blobs(c, names, names ? *count : 0, blobs);
    if (FAILED(hr)) return hr;
    if (size < packed_size(blobs)) return E_GS_PROVIDED_BUFFER_TOO_SMALL;
    pack(blobs, out);
    *count = static_cast<uint32_t>(blobs.size());
    return S_OK;
}

// 21: HRESULT XGameSaveReadBlobDataAsync(XGameSaveContainerHandle, const char** blobNames, uint32_t countOfBlobs,
//                                        XAsyncBlock* async). The names are copied: the title may free them at once.
HRESULT STDMETHODCALLTYPE save_read_async(void*, XGameSaveContainerHandle c, const char** names, uint32_t count,
                                          XAsyncBlock* async) {
    auto* job = new Job{&kReadIdentity};
    job->container = c;
    job->serial = serial_of(g_containers, c);
    job->all_blobs = names == nullptr;
    for (uint32_t i = 0; names && i < count; ++i) {
        if (!valid_blob_name(names[i])) {
            delete job;
            return E_INVALIDARG;
        }
        job->names.emplace_back(names[i]);
    }
    return begin(async, job, "XGameSaveReadBlobDataAsync");
}

// 22: HRESULT XGameSaveReadBlobDataResult(XAsyncBlock* async, size_t blobsSize, XGameSaveBlob* blobData,
//                                         uint32_t* countOfBlobs). blobsSize: XAsyncGetResultSize's answer.
HRESULT STDMETHODCALLTYPE save_read_result(void*, XAsyncBlock* async, size_t size, XGameSaveBlob* out, uint32_t* count) {
    if (!out || !count) return E_POINTER;
    *count = 0;
    t_read_count = 0;
    HRESULT hr = XAsyncGetResult(async, &kReadIdentity, size, out, nullptr);
    if (hr == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) || hr == E_NOT_SUFFICIENT_BUFFER)
        hr = E_GS_PROVIDED_BUFFER_TOO_SMALL;
    if (SUCCEEDED(hr)) *count = t_read_count;
    return hr;
}

// 23: HRESULT XGameSaveCreateUpdate(XGameSaveContainerHandle, const char* containerDisplayName, XGameSaveUpdateHandle*)
HRESULT STDMETHODCALLTYPE save_create_update(void*, XGameSaveContainerHandle c, const char* display_name,
                                             XGameSaveUpdateHandle* out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_containers, c)) return E_GS_HANDLE_EXPIRED;
    auto* u = new XGameSaveUpdate{c, c->serial, display_name ? display_name : "", display_name != nullptr, {}, 0};
    u->serial = g_next_serial++;
    g_updates.insert(u);
    *out = u;
    return S_OK;
}

// 24: void XGameSaveCloseUpdate(XGameSaveUpdateHandle context)
void STDMETHODCALLTYPE save_close_update(void*, XGameSaveUpdateHandle u) {
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!u || g_updates.erase(u) == 0) return;
    }
    delete u;
}

// 25: HRESULT XGameSaveSubmitBlobWrite(XGameSaveUpdateHandle, const char* blobName, const uint8_t* data, size_t size).
// One write or delete per blob per update; the update (and so each blob) stays within GS_MAX_BLOB_SIZE.
HRESULT STDMETHODCALLTYPE save_write(void*, XGameSaveUpdateHandle u, const char* name, const uint8_t* data, size_t size) {
    if (!valid_blob_name(name) || (!data && size)) return E_INVALIDARG;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_updates, u)) return E_GS_HANDLE_EXPIRED;
    if (u->ops.count(name)) return E_INVALIDARG;
    if (size > kMaxUpdateBytes || u->bytes + size > kMaxUpdateBytes) return E_GS_UPDATE_TOO_BIG;
    u->ops.emplace(name, Op{true, std::vector<uint8_t>(data, data + size)});
    u->bytes += size;
    return S_OK;
}

// 26: HRESULT XGameSaveSubmitBlobDelete(XGameSaveUpdateHandle, const char* blobName)
HRESULT STDMETHODCALLTYPE save_delete_blob(void*, XGameSaveUpdateHandle u, const char* name) {
    if (!valid_blob_name(name)) return E_INVALIDARG;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!live(g_updates, u)) return E_GS_HANDLE_EXPIRED;
    if (u->ops.count(name)) return E_INVALIDARG;
    u->ops.emplace(name, Op{false, {}});
    return S_OK;
}

// 27: HRESULT XGameSaveSubmitUpdate(XGameSaveUpdateHandle update)
HRESULT STDMETHODCALLTYPE save_submit(void*, XGameSaveUpdateHandle u) { return submit_update(u); }

// 28: HRESULT XGameSaveSubmitUpdateAsync(XGameSaveUpdateHandle update, XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE save_submit_async(void*, XGameSaveUpdateHandle u, XAsyncBlock* async) {
    auto* job = new Job{&kSubmitIdentity};
    job->update = u;
    job->serial = serial_of(g_updates, u);
    return begin(async, job, "XGameSaveSubmitUpdateAsync");
}

// 29: HRESULT XGameSaveSubmitUpdateResult(XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE save_submit_result(void*, XAsyncBlock* async) {
    return XAsyncGetResult(async, &kSubmitIdentity, 0, nullptr, nullptr);
}

// 30: HRESULT XGameSaveFilesGetFolderWithUiAsync(XUserHandle requestingUser, const char* configurationId,
//                                                XAsyncBlock* async): a folder of its own, <root>\files\<configId>.
HRESULT STDMETHODCALLTYPE files_folder_async(void*, XUserHandle user, const char* config_id, XAsyncBlock* async) {
    if (!user || !config_id) return E_INVALIDARG;
    auto* job = new Job{&kFolderIdentity};
    job->user = user;
    job->text = config_id;
    return begin(async, job, "XGameSaveFilesGetFolderWithUiAsync");
}

// 31: HRESULT XGameSaveFilesGetFolderWithUiResult(XAsyncBlock* async, size_t folderSize, char* folderResult)
HRESULT STDMETHODCALLTYPE files_folder_result(void*, XAsyncBlock* async, size_t size, char* folder) {
    if (!folder) return E_POINTER;
    HRESULT hr = XAsyncGetResult(async, &kFolderIdentity, size, folder, nullptr);
    if (hr == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) || hr == E_NOT_SUFFICIENT_BUFFER)
        hr = E_GS_PROVIDED_BUFFER_TOO_SMALL;
    return hr;
}

// 32: HRESULT XGameSaveFilesGetRemainingQuota(XUserHandle requestingUser, const char* configurationId,
//                                             int64_t* remainingQuota)
HRESULT STDMETHODCALLTYPE files_quota(void*, XUserHandle user, const char* config_id, int64_t* remaining) {
    if (!remaining) return E_POINTER;
    if (!user || !config_id) return E_INVALIDARG;
    const fs::path root = save_root();
    if (root.empty()) return E_GS_OUT_OF_LOCAL_STORAGE;
    uint64_t used = 0;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(root / L"files" / widen(encode_name(config_id)), ec))
        if (e.is_regular_file(ec)) used += static_cast<uint64_t>(e.file_size(ec));
    *remaining = used >= kQuotaBytes ? 0 : static_cast<int64_t>(kQuotaBytes - used);
    return S_OK;
}

// ---- The object ----

constexpr GUID kSaveClass = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}};
const GUID kSaveInterfaces[] = {
    kSaveClass,                                                                             // IXGameSaveImpl
    {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6e, 0xe2}},         // IXGameSaveImpl2
    {0x1bfff3af, 0xf14a, 0x40a3, {0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9}},         // IXGameSaveImpl3 (no methods)
};
constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

struct SaveObject {
    const void* const* vtable;
};
SaveObject* save_object();

HRESULT query(REFIID iid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (iid == kIUnknown) {
        *out = save_object();
        return S_OK;
    }
    for (const GUID& id : kSaveInterfaces)
        if (iid == id) {
            *out = save_object();
            return S_OK;
        }
    return E_NOINTERFACE;
}

// A process-wide singleton, as the other runtime classes: AddRef and Release don't free.
HRESULT STDMETHODCALLTYPE object_query_interface(void*, REFIID iid, void** out) { return query(iid, out); }
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

#define FN(f) reinterpret_cast<const void*>(&f)

const void* const kSaveVtable[] = {
    FN(object_query_interface), FN(object_add_ref), FN(object_release),   // 0-2  IUnknown
    FN(save_init),                      // 3  XGameSaveInitializeProvider
    FN(save_init_async),                // 4  XGameSaveInitializeProviderAsync
    FN(save_init_result),               // 5  XGameSaveInitializeProviderResult
    FN(save_close_provider),            // 6  XGameSaveCloseProvider
    FN(save_quota),                     // 7  XGameSaveGetRemainingQuota
    FN(save_quota_async),               // 8  XGameSaveGetRemainingQuotaAsync
    FN(save_quota_result),              // 9  XGameSaveGetRemainingQuotaResult
    FN(save_delete),                    // 10 XGameSaveDeleteContainer
    FN(save_delete_async),              // 11 XGameSaveDeleteContainerAsync
    FN(save_delete_result),             // 12 XGameSaveDeleteContainerResult
    FN(save_container_info),            // 13 XGameSaveGetContainerInfo
    FN(save_enum_containers),           // 14 XGameSaveEnumerateContainerInfo
    FN(save_enum_containers_by_name),   // 15 XGameSaveEnumerateContainerInfoByName
    FN(save_create_container),          // 16 XGameSaveCreateContainer
    FN(save_close_container),           // 17 XGameSaveCloseContainer
    FN(save_enum_blobs),                // 18 XGameSaveEnumerateBlobInfo
    FN(save_enum_blobs_by_name),        // 19 XGameSaveEnumerateBlobInfoByName
    FN(save_read),                      // 20 XGameSaveReadBlobData
    FN(save_read_async),                // 21 XGameSaveReadBlobDataAsync
    FN(save_read_result),               // 22 XGameSaveReadBlobDataResult
    FN(save_create_update),             // 23 XGameSaveCreateUpdate
    FN(save_close_update),              // 24 XGameSaveCloseUpdate
    FN(save_write),                     // 25 XGameSaveSubmitBlobWrite
    FN(save_delete_blob),               // 26 XGameSaveSubmitBlobDelete
    FN(save_submit),                    // 27 XGameSaveSubmitUpdate
    FN(save_submit_async),              // 28 XGameSaveSubmitUpdateAsync
    FN(save_submit_result),             // 29 XGameSaveSubmitUpdateResult
    FN(files_folder_async),             // 30 XGameSaveFilesGetFolderWithUiAsync (2)
    FN(files_folder_result),            // 31 XGameSaveFilesGetFolderWithUiResult (2)
    FN(files_quota),                    // 32 XGameSaveFilesGetRemainingQuota (2)
};
static_assert(std::size(kSaveVtable) == 33, "IUnknown + IXGameSaveImpl's 27 methods + IXGameSaveImpl2's 3");

#undef FN

SaveObject* save_object() {
    static SaveObject o{kSaveVtable};
    return &o;
}

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is XGameSaveImpl, with *hr set.
bool xgameruntime_gamesave_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id != kSaveClass) return false;
    *hr = query(interface_id, out);
    return true;
}

// For tests: the provider, container and update handles a title holds open.
long xgameruntime_gamesave_open_handles() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return static_cast<long>(g_providers.size() + g_containers.size() + g_updates.size());
}

}  // namespace rt::shims
