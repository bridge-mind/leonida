// Script the Vulkan size and data calls without loading a driver.
#include "pipeline_cache_snapshot.h"

#include <cstdio>
#include <cstring>

#define CHECK(condition) do {                                            \
    if (!(condition)) {                                                  \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);      \
        return 1;                                                        \
    }                                                                    \
} while (0)

namespace {

// A cache that workers keep growing: every call (size query or read) first adds `growth` bytes, or doubles the
// cache when `growth` is 0. A read into a buffer smaller than the cache writes a prefix and returns VK_INCOMPLETE;
// otherwise it writes the whole cache and reports its size, as vkGetPipelineCacheData does.
struct BusyCache {
    size_t bytes, growth;
    int queries = 0, reads = 0;

    VkResult operator()(size_t* size, void* data) {
        bytes += growth ? growth : bytes;
        if (!data) {
            ++queries;
            *size = bytes;
            return VK_SUCCESS;
        }
        ++reads;
        if (*size < bytes) {
            std::memset(data, 'p', *size);
            return VK_INCOMPLETE;
        }
        *size = bytes;
        std::memset(data, 's', *size);
        return VK_SUCCESS;
    }
};

// The snapshot, or an empty vector if it threw, so a failing case is reported by CHECK instead of ending the test.
std::vector<char> snapshot(BusyCache& cache) {
    try {
        return pipeline_cache_snapshot([&](size_t* size, void* d) { return cache(size, d); });
    } catch (const std::exception&) {
        return {};
    }
}

}  // namespace

int main() {
    // A busy cache that grows less than 1/16 between the size query and the read: one attempt is enough. Reading
    // into exactly the queried size returned VK_INCOMPLETE here on every attempt.
    {
        BusyCache cache{4096, 128};
        CHECK(snapshot(cache) == std::vector<char>(4096 + 2 * 128, 's'));
        CHECK(cache.queries == 1 && cache.reads == 1);
    }

    // Growth past the first attempt's headroom: the retry asks for more room and succeeds.
    {
        BusyCache cache{4096, 400};
        CHECK(snapshot(cache) == std::vector<char>(4096 + 4 * 400, 's'));
        CHECK(cache.queries == 2 && cache.reads == 2);
    }

    // A cache that doubles between every call outgrows even the last attempt's 1/2 headroom: the four-attempt error.
    {
        BusyCache cache{4096, 0};
        CHECK(snapshot(cache).empty() && cache.queries == 4 && cache.reads == 4);
    }

    // A successful read can return less than the size query.
    {
        int queries = 0, reads = 0;
        const auto shrinking = [&](size_t* size, void* data) {
            if (!data) {
                ++queries;
                *size = 64;
                return VK_SUCCESS;
            }
            ++reads;
            *size = 48;
            std::memset(data, 's', *size);
            return VK_SUCCESS;
        };
        CHECK(pipeline_cache_snapshot(shrinking) == std::vector<char>(48, 's'));
        CHECK(queries == 1 && reads == 1);
    }

    auto rejected = [](VkResult query_result, size_t available, VkResult read_result, size_t written) {
        int calls = 0;
        auto get = [&](size_t* size, void* data) {
            ++calls;
            *size = data ? written : available;
            return data ? read_result : query_result;
        };
        try {
            pipeline_cache_snapshot(get);
        } catch (const std::exception&) {
            return calls;
        }
        return 0;
    };
    CHECK(rejected(VK_ERROR_DEVICE_LOST, 32, VK_SUCCESS, 32) == 1);
    CHECK(rejected(VK_SUCCESS, 0, VK_SUCCESS, 0) == 1);
    CHECK(rejected(VK_SUCCESS, 16, VK_SUCCESS, 16) == 1);
    CHECK(rejected(VK_SUCCESS, 32, VK_ERROR_DEVICE_LOST, 32) == 2);
    CHECK(rejected(VK_SUCCESS, 32, VK_SUCCESS, 0) == 2);
    CHECK(rejected(VK_SUCCESS, 32, VK_SUCCESS, 16) == 2);
    CHECK(rejected(VK_SUCCESS, 32, VK_SUCCESS, 32 + 32 / 16 + 1) == 2);  // more than the buffer it was given
    CHECK(rejected(VK_SUCCESS, 32, VK_INCOMPLETE, 32) == 8);
    return 0;
}
