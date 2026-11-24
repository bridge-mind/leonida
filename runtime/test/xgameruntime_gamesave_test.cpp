// XGameSaveImpl through QueryApiImpl, called the way xgameruntime.lib's XGameSave* thunks call it: QueryApiImpl with
// the class id (or a later interface id), one vtable slot with `this` first. Slots follow xgameruntime-docs' layout;
// arguments, limits and error codes follow the public GDK reference. Saves go to a temporary LEONIDA_SAVE_DIR; no
// game data, nothing online.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "shims.h"

namespace rt::shims {
long xgameruntime_gamesave_open_handles();
}

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

using Handle = void*;
struct ContainerInfo {
    const char* name;
    const char* displayName;
    uint32_t blobCount;
    uint64_t totalSize;
    time_t lastModifiedTime;
    bool needsSync;
};
struct BlobInfo {
    const char* name;
    uint32_t size;
};
struct Blob {
    BlobInfo info;
    uint8_t* data;
};

constexpr HRESULT E_GS_INVALID_CONTAINER_NAME = static_cast<HRESULT>(0x80830001);
constexpr HRESULT E_GS_UPDATE_TOO_BIG = static_cast<HRESULT>(0x80830005);
constexpr HRESULT E_GS_PROVIDED_BUFFER_TOO_SMALL = static_cast<HRESULT>(0x80830007);
constexpr HRESULT E_GS_BLOB_NOT_FOUND = static_cast<HRESULT>(0x80830008);
constexpr HRESULT E_GS_HANDLE_EXPIRED = static_cast<HRESULT>(0x8083000D);

constexpr GUID kSaveClass = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}};
constexpr GUID kSaveImpl2 = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6e, 0xe2}};
constexpr GUID kSaveImpl3 = {0x1bfff3af, 0xf14a, 0x40a3, {0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9}};
constexpr GUID kOther = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};  // XGameImpl

using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
using Fn = void*;
Fn slot(void* obj, int index) { return (*static_cast<Fn* const*>(obj))[index]; }
template <class F> F at(void* obj, int index) { return reinterpret_cast<F>(slot(obj, index)); }

bool finished(XAsyncBlock* async) { return XAsyncGetStatus(async, true) != E_PENDING; }

using Init = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, bool, Handle*);
using InitAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, bool, XAsyncBlock*);
using ResultHandle = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, Handle*);
using Close = void(STDMETHODCALLTYPE*)(void*, Handle);
using Quota = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, int64_t*);
using WithAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, XAsyncBlock*);
using QuotaResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, int64_t*);
using ByName = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*);
using ByNameAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, XAsyncBlock*);
using AsyncOnly = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*);
using ContainerCb = bool(CALLBACK*)(const ContainerInfo*, void*);
using BlobCb = bool(CALLBACK*)(const BlobInfo*, void*);
using GetInfo = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, void*, ContainerCb);
using EnumContainers = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, void*, ContainerCb);
using EnumBlobs = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, void*, BlobCb);
using EnumBlobsByName = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, void*, BlobCb);
using Create = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, Handle*);
using Read = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char**, uint32_t*, size_t, Blob*);
using ReadAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char**, uint32_t, XAsyncBlock*);
using ReadResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, Blob*, uint32_t*);
using Write = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, const uint8_t*, size_t);
using Submit = HRESULT(STDMETHODCALLTYPE*)(void*, Handle);
using FolderAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, XAsyncBlock*);
using FolderResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, char*);
using FilesQuota = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, int64_t*);

struct Seen {
    std::vector<std::string> names, displays;
    std::vector<uint32_t> counts;
    std::vector<uint64_t> sizes;
    int stop_after = 1000;
};
bool CALLBACK on_container(const ContainerInfo* info, void* context) {
    auto* seen = static_cast<Seen*>(context);
    seen->names.emplace_back(info->name);
    seen->displays.emplace_back(info->displayName ? info->displayName : "");
    seen->counts.push_back(info->blobCount);
    seen->sizes.push_back(info->totalSize);
    return static_cast<int>(seen->names.size()) < seen->stop_after;
}
bool CALLBACK on_blob(const BlobInfo* info, void* context) {
    auto* seen = static_cast<Seen*>(context);
    seen->names.emplace_back(info->name);
    seen->sizes.push_back(info->size);
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a crash still shows the checks before it
    std::printf("xgameruntime_gamesave_test: XGameSaveImpl through QueryApiImpl\n");
    const std::filesystem::path root = std::filesystem::temp_directory_path() / L"leonida_gamesave_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    SetEnvironmentVariableW(L"LEONIDA_SAVE_DIR", root.c_str());

    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    void* save = nullptr;
    check(query(&kSaveClass, kSaveClass, &save) == S_OK && save, "QueryApiImpl hands out XGameSaveImpl for its class id");
    if (!save) return 1;
    void* save2 = nullptr;
    void* save3 = nullptr;
    check(query(&kSaveClass, kSaveImpl2, &save2) == S_OK && save2 == save &&
              query(&kSaveClass, kSaveImpl3, &save3) == S_OK && save3 == save,
          "IXGameSaveImpl2 and IXGameSaveImpl3 answer with the same object");
    void* none = reinterpret_cast<void*>(1);
    check(query(&kSaveClass, kOther, &none) == E_NOINTERFACE && none == nullptr, "another interface id is refused");

    Handle user = reinterpret_cast<Handle>(0x1000);   // XUserHandle: opaque here, only checked for null
    Handle provider = nullptr;
    check(at<Init>(save, 3)(save, nullptr, "SCID", false, &provider) == E_INVALIDARG && !provider,
          "InitializeProvider without a user is refused");
    check(at<Init>(save, 3)(save, user, "SCID", true, &provider) == S_OK && provider,
          "InitializeProvider opens the offline provider (syncOnDemand is accepted)");
    check(std::filesystem::is_directory(root / L"local-user"), "the provider's folder is <LEONIDA_SAVE_DIR>\\local-user");

    int64_t quota = 0;
    check(at<Quota>(save, 7)(save, provider, &quota) == S_OK && quota == (int64_t{1} << 30),
          "GetRemainingQuota of an empty provider is the whole 1 GiB");

    // ---- containers and names ----
    Handle container = nullptr;
    check(at<Create>(save, 16)(save, provider, "bad name", &container) == E_GS_INVALID_CONTAINER_NAME && !container,
          "a container name with a space is refused");
    for (const char* bad : {"a..b", "dir/", "x.", "a/b-c/d", ""})
        check(at<Create>(save, 16)(save, provider, bad, &container) == E_GS_INVALID_CONTAINER_NAME,
              "a container name against the documented rule is refused");
    check(at<Create>(save, 16)(save, provider, std::string(257, 'a').c_str(), &container) == E_GS_INVALID_CONTAINER_NAME,
          "a container name over 256 characters is refused");
    check(at<Create>(save, 16)(save, provider, "slots/SAVE_00.sav-1", &container) == S_OK && container,
          "a name with a path part and . - in the file part is accepted");
    check(!std::filesystem::exists(root / L"local-user" / L"slots%2FSAVE_00.sav-1"),
          "a container's folder appears on its first update, not on create");

    // ---- an update ----
    Handle update = nullptr;
    check(at<Create>(save, 23)(save, container, "Story 1", &update) == S_OK && update, "CreateUpdate");
    const std::string story(5000, 'S'), profile = "profile bytes";
    check(at<Write>(save, 25)(save, update, "story", reinterpret_cast<const uint8_t*>(story.data()), story.size()) == S_OK,
          "SubmitBlobWrite buffers a blob");
    check(at<Write>(save, 25)(save, update, "story", reinterpret_cast<const uint8_t*>("x"), 1) == E_INVALIDARG,
          "a second write of the same blob in one update is refused");
    check(at<Write>(save, 25)(save, update, "profile/main", reinterpret_cast<const uint8_t*>(profile.data()),
                              profile.size()) == S_OK,
          "a blob name with a slash is accepted (it is %-encoded on disk)");
    std::vector<uint8_t> big(16u * 1024 * 1024, 1);
    check(at<Write>(save, 25)(save, update, "big", big.data(), big.size()) == E_GS_UPDATE_TOO_BIG,
          "an update past 16 MB is refused");
    check(at<Submit>(save, 27)(save, update) == S_OK, "SubmitUpdate commits");
    const auto dir = root / L"local-user" / L"slots%2FSAVE_00.sav-1";
    check(std::filesystem::file_size(dir / L"story") == story.size() && std::filesystem::exists(dir / L"profile%2Fmain"),
          "the blobs are files, in tools/saves.py's layout and encoding");
    at<Close>(save, 24)(save, update);

    // ---- enumeration ----
    Seen containers;
    check(at<EnumContainers>(save, 14)(save, provider, &containers, on_container) == S_OK &&
              containers.names.size() == 1 && containers.names[0] == "slots/SAVE_00.sav-1" &&
              containers.displays[0] == "Story 1" && containers.counts[0] == 2 &&
              containers.sizes[0] == story.size() + profile.size(),
          "EnumerateContainerInfo: name, display name, blob count and size");
    Seen info;
    check(at<GetInfo>(save, 13)(save, provider, "slots/SAVE_00.sav-1", &info, on_container) == S_OK &&
              info.names.size() == 1,
          "GetContainerInfo calls back for a container that exists");
    Seen missing;
    check(at<GetInfo>(save, 13)(save, provider, "nothing", &missing, on_container) == S_OK && missing.names.empty(),
          "GetContainerInfo of a missing container calls nothing");
    Seen blobs;
    check(at<EnumBlobs>(save, 18)(save, container, &blobs, on_blob) == S_OK && blobs.names.size() == 2 &&
              blobs.names[0] == "profile/main" && blobs.names[1] == "story" && blobs.sizes[1] == story.size(),
          "EnumerateBlobInfo lists both blobs with their decoded names and sizes");
    Seen prefixed;
    check(at<EnumBlobsByName>(save, 19)(save, container, "sto", &prefixed, on_blob) == S_OK &&
              prefixed.names.size() == 1 && prefixed.names[0] == "story",
          "EnumerateBlobInfoByName filters by prefix");

    // ---- reading ----
    const char* names[] = {"story", "profile/main"};
    uint32_t count = 2;
    std::vector<uint8_t> small(64);
    check(at<Read>(save, 20)(save, container, names, &count, small.size(), reinterpret_cast<Blob*>(small.data())) ==
              E_GS_PROVIDED_BUFFER_TOO_SMALL,
          "ReadBlobData into a buffer too small for the blobs is refused");
    std::vector<uint64_t> buffer(1024);
    count = 2;
    auto* out = reinterpret_cast<Blob*>(buffer.data());
    check(at<Read>(save, 20)(save, container, names, &count, buffer.size() * 8, out) == S_OK && count == 2 &&
              std::strcmp(out[0].info.name, "story") == 0 && out[0].info.size == story.size() &&
              std::memcmp(out[0].data, story.data(), story.size()) == 0 &&
              std::strcmp(out[1].info.name, "profile/main") == 0 &&
              std::memcmp(out[1].data, profile.data(), profile.size()) == 0,
          "ReadBlobData returns both blobs, byte for byte, pointing into the buffer");
    const char* absent[] = {"nope"};
    count = 1;
    check(at<Read>(save, 20)(save, container, absent, &count, buffer.size() * 8, out) == E_GS_BLOB_NOT_FOUND,
          "a blob that is not there is E_GS_BLOB_NOT_FOUND");
    count = 0;
    check(at<Read>(save, 20)(save, container, nullptr, &count, buffer.size() * 8, out) == S_OK && count == 2,
          "blobNames = nullptr reads every blob");

    XAsyncBlock async{};
    check(at<ReadAsync>(save, 21)(save, container, names, 1, &async) == S_OK && finished(&async),
          "ReadBlobDataAsync runs");
    size_t needed = 0;
    check(XAsyncGetResultSize(&async, &needed) == S_OK && needed > story.size(),
          "its result size is what the buffer needs (XAsyncGetResultSize)");
    std::vector<uint64_t> abuf((needed + 7) / 8);
    uint32_t acount = 0;
    auto* aout = reinterpret_cast<Blob*>(abuf.data());
    check(at<ReadResult>(save, 22)(save, &async, needed, aout, &acount) == S_OK && acount == 1 &&
              std::strcmp(aout[0].info.name, "story") == 0 && std::memcmp(aout[0].data, story.data(), story.size()) == 0,
          "ReadBlobDataResult fills the title's buffer with one blob");

    // ---- delete a blob, async submit ----
    check(at<Create>(save, 23)(save, container, nullptr, &update) == S_OK, "a second update");
    check(at<ByName>(save, 26)(save, update, "profile/main") == S_OK, "SubmitBlobDelete");
    async = XAsyncBlock{};
    check(at<WithAsync>(save, 28)(save, update, &async) == S_OK && finished(&async) &&
              at<AsyncOnly>(save, 29)(save, &async) == S_OK,
          "SubmitUpdateAsync commits the delete");
    check(!std::filesystem::exists(dir / L"profile%2Fmain") && std::filesystem::exists(dir / L"story"),
          "the deleted blob is gone, the other stays");
    Seen after;
    at<EnumContainers>(save, 14)(save, provider, &after, on_container);
    check(after.displays.size() == 1 && after.displays[0] == "Story 1", "an update without a display name keeps it");
    at<Close>(save, 24)(save, update);
    check(at<Submit>(save, 27)(save, update) == E_GS_HANDLE_EXPIRED, "a closed update is refused");

    // ---- quota ----
    async = XAsyncBlock{};
    int64_t aquota = 0;
    check(at<WithAsync>(save, 8)(save, provider, &async) == S_OK && finished(&async) &&
              at<QuotaResult>(save, 9)(save, &async, &aquota) == S_OK &&
              aquota == (int64_t{1} << 30) - static_cast<int64_t>(story.size()),
          "GetRemainingQuotaAsync subtracts what the saves hold");

    // ---- delete the container ----
    async = XAsyncBlock{};
    check(at<ByNameAsync>(save, 11)(save, provider, "slots/SAVE_00.sav-1", &async) == S_OK && finished(&async) &&
              at<AsyncOnly>(save, 12)(save, &async) == S_OK && !std::filesystem::exists(dir),
          "DeleteContainerAsync removes the container");
    at<Close>(save, 17)(save, container);

    // ---- async provider, XGameSaveFiles ----
    async = XAsyncBlock{};
    Handle provider2 = nullptr;
    check(at<InitAsync>(save, 4)(save, user, "SCID", false, &async) == S_OK && finished(&async) &&
              at<ResultHandle>(save, 5)(save, &async, &provider2) == S_OK && provider2 && provider2 != provider,
          "InitializeProviderAsync hands out a second provider");
    async = XAsyncBlock{};
    check(at<FolderAsync>(save, 30)(save, user, "SCID", &async) == S_OK && finished(&async), "FilesGetFolderWithUiAsync");
    size_t folder_size = 0;
    XAsyncGetResultSize(&async, &folder_size);
    std::string folder(folder_size, '\0');
    check(folder_size > 0 && at<FolderResult>(save, 31)(save, &async, folder.size(), folder.data()) == S_OK &&
              std::filesystem::is_directory(std::filesystem::u8path(folder.c_str())),
          "FilesGetFolderWithUiResult returns a folder that exists");
    int64_t fquota = 0;
    check(at<FilesQuota>(save, 32)(save, user, "SCID", &fquota) == S_OK && fquota == (int64_t{1} << 30),
          "FilesGetRemainingQuota of an empty folder");

    // ---- a handle that outlives its container, after the heap hands the address out again ----
    // An update and an async read keep the container handle they were made from. Once that container is closed, the
    // next container may get the same address; the stale update and read must still be refused, not write into or
    // read from the newer container (another save).
    {
        Handle first = nullptr, stale_update = nullptr;
        check(at<Create>(save, 16)(save, provider, "first", &first) == S_OK &&
                  at<Create>(save, 23)(save, first, nullptr, &stale_update) == S_OK &&
                  at<Write>(save, 25)(save, stale_update, "mine", reinterpret_cast<const uint8_t*>("first's"), 7) == S_OK,
              "an update of container \"first\" buffers a blob");
        XTaskQueueHandle manual = nullptr;
        XTaskQueueCreate(XTaskQueueDispatchMode::Manual, XTaskQueueDispatchMode::Manual, &manual);
        XAsyncBlock stale_read{};
        stale_read.queue = manual;
        check(manual && at<ReadAsync>(save, 21)(save, first, nullptr, 0, &stale_read) == S_OK,
              "a ReadBlobDataAsync of \"first\" waits on a manual queue");
        at<Close>(save, 17)(save, first);

        // Containers are made until one lands at the closed container's address (most heaps hand it back at once).
        Handle second = nullptr;
        std::vector<Handle> others;
        for (int i = 0; i < 4096 && !second; ++i) {
            Handle c = nullptr;
            if (at<Create>(save, 16)(save, provider, "second", &c) != S_OK) break;
            if (c == first) second = c;
            else others.push_back(c);
        }
        for (Handle c : others) at<Close>(save, 17)(save, c);
        if (!second) {
            std::printf("SKIP the heap never reused a closed container's address; the stale-handle checks need it\n");
        } else {
            Handle theirs = nullptr;
            check(at<Create>(save, 23)(save, second, nullptr, &theirs) == S_OK &&
                      at<Write>(save, 25)(save, theirs, "theirs", reinterpret_cast<const uint8_t*>("second's"), 8) ==
                          S_OK &&
                      at<Submit>(save, 27)(save, theirs) == S_OK,
                  "container \"second\", at the closed container's address, gets a blob of its own");
            at<Close>(save, 24)(save, theirs);
            check(at<Submit>(save, 27)(save, stale_update) == E_GS_HANDLE_EXPIRED,
                  "SubmitUpdate of an update whose container was closed is refused, though a new container has its "
                  "address");
            check(!std::filesystem::exists(root / L"local-user" / L"second" / L"mine"),
                  "the stale update wrote nothing into the container that took its container's address");
        }
        while (XTaskQueueDispatch(manual, XTaskQueuePort::Work, 0) ||
               XTaskQueueDispatch(manual, XTaskQueuePort::Completion, 0)) {}
        if (second) {
            size_t stale_size = 0;
            const HRESULT read_hr = XAsyncGetStatus(&stale_read, false);
            XAsyncGetResultSize(&stale_read, &stale_size);
            check(read_hr == E_GS_HANDLE_EXPIRED && stale_size == 0,
                  "ReadBlobDataAsync of a closed container completes with E_GS_HANDLE_EXPIRED, not with the blobs of "
                  "the container that took its address");
            at<Close>(save, 17)(save, second);
        }
        if (XAsyncGetStatus(&stale_read, false) == S_OK) {   // collect a result so the async state is released
            std::vector<uint64_t> drain(1024);
            uint32_t drained = 0;
            at<ReadResult>(save, 22)(save, &stale_read, drain.size() * 8, reinterpret_cast<Blob*>(drain.data()), &drained);
        }
        at<Close>(save, 24)(save, stale_update);
        if (manual) XTaskQueueCloseHandle(manual);
        std::filesystem::remove_all(root / L"local-user" / L"first", ec);
        std::filesystem::remove_all(root / L"local-user" / L"second", ec);
    }

    at<Close>(save, 6)(save, provider);
    at<Close>(save, 6)(save, provider2);
    at<Close>(save, 6)(save, provider);   // twice: ignored
    check(at<Quota>(save, 7)(save, provider, &quota) == E_GS_HANDLE_EXPIRED, "a closed provider is refused");
    check(rt::shims::xgameruntime_gamesave_open_handles() == 0, "every handle the test opened is closed");

    std::filesystem::remove_all(root, ec);
    std::printf("xgameruntime_gamesave_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
