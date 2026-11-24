// Synthetic GPU test shaders are packaged under generated/ in runtime-windows artifacts.
// Run downloaded tests from the artifact directory, as CTest does for a local build.
#pragma once

#include <filesystem>

namespace vktest {
inline std::filesystem::path shader_fixture_path(const std::filesystem::path& compiled_path) {
    if (std::filesystem::is_regular_file(compiled_path)) return compiled_path;
    const auto packaged = std::filesystem::path("generated") / compiled_path.filename();
    if (std::filesystem::is_regular_file(packaged)) return packaged;
    // Preserve the original path in the caller's missing-file diagnostic.
    return compiled_path;
}
}  // namespace vktest
