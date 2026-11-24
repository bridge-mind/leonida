#pragma once
#include <cstdint>

namespace leonida {

// present_compose.comp compiled with: glslc --target-env=vulkan1.1 -O -mfmt=c present_compose.comp -o present_compose.inc
inline constexpr std::uint32_t present_compose_spirv[] =
#include "present_compose.inc"
;

} // namespace leonida
