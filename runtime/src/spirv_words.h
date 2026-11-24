#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <vector>

namespace leonida {

// Checks the binary framing used by the translator and returns words aligned for Vulkan.
// The caller still relies on the driver or spirv-val for instruction and capability validation.
inline std::vector<std::uint32_t> read_spirv_words(std::istream& input, std::streamoff byte_count) {
    constexpr std::streamoff word_bytes = sizeof(std::uint32_t);
    constexpr std::streamoff header_bytes = 5 * word_bytes;
    if (byte_count < header_bytes || byte_count % word_bytes || !input.seekg(0)) return {};

    std::uint32_t magic = 0;
    if (!input.read(reinterpret_cast<char*>(&magic), word_bytes) || magic != 0x07230203) return {};

    std::vector<std::uint32_t> words(static_cast<std::size_t>(byte_count / word_bytes));
    words.front() = magic;
    if (!input.read(reinterpret_cast<char*>(words.data() + 1), byte_count - word_bytes)) return {};
    return words;
}

}  // namespace leonida
