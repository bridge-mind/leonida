// vkGetPipelineCacheData can return a prefix if workers grow the cache between calls.
#pragma once

#include <vulkan/vulkan_core.h>

#include <stdexcept>
#include <string>
#include <vector>

template<class GetData>
std::vector<char> pipeline_cache_snapshot(GetData get) {
    for (int attempt = 0; attempt < 4; ++attempt) {
        size_t size = 0;
        VkResult result = get(&size, nullptr);
        if (result != VK_SUCCESS)
            throw std::runtime_error("vkGetPipelineCacheData size query returned " + std::to_string(result));
        if (size < sizeof(VkPipelineCacheHeaderVersionOne))
            throw std::runtime_error("cache size is smaller than its header");

        // Workers keep adding pipelines between the size query and the read, so read into a buffer with room to
        // spare: 1/16 of the queried size on the first attempt, doubling on each retry (1/8, 1/4, 1/2). A buffer
        // larger than the data is fine; the driver writes the whole cache and reports the bytes it wrote.
        const size_t capacity = size + (size >> (4 - attempt));
        // Destroy each incomplete snapshot before allocating its replacement.
        std::vector<char> data(capacity);
        size = capacity;
        result = get(&size, data.data());
        if (result == VK_INCOMPLETE) continue;
        if (result != VK_SUCCESS)
            throw std::runtime_error("vkGetPipelineCacheData read returned " + std::to_string(result));
        if (size < sizeof(VkPipelineCacheHeaderVersionOne) || size > data.size())
            throw std::runtime_error("invalid pipeline cache byte count");
        data.resize(size);
        return data;
    }
    throw std::runtime_error("vkGetPipelineCacheData returned VK_INCOMPLETE on four attempts");
}
