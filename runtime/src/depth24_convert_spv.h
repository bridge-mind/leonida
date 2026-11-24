#pragma once
#include <cstdint>

namespace leonida {

// depth24_convert.comp compiled with: glslc --target-env=vulkan1.1 -O -mfmt=c depth24_convert.comp -o depth24_convert.inc
inline constexpr std::uint32_t depth24_convert_spirv[] =
#include "depth24_convert.inc"
;

} // namespace leonida
