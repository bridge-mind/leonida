#include "../src/conservative_raster.h"
#include <cstdio>
#include <limits>

int main() {
    using leonida::conservative_raster::tier1;
    unsigned failures = 0;
    const auto check = [&](bool value) { if (!value) ++failures; };
    check(tier1(true, 8, 0.0f));
    check(tier1(true, 8, 1.0f / 256.0f));
    check(tier1(true, 8, 0.5f - 1.0f / 512.0f));
    check(!tier1(false, 8, 0.0f));
    check(!tier1(true, 7, 0.0f));
    check(!tier1(true, 8, 0.5f));
    check(!tier1(true, 8, -0.001f));
    check(!tier1(true, 8, std::numeric_limits<float>::infinity()));
    check(!tier1(true, 8, std::numeric_limits<float>::quiet_NaN()));
    using leonida::conservative_raster::tier;
    check(tier(false, 8, 0, true) == 0);
    check(tier(true, 7, 0, true) == 0);
    check(tier(true, 8, 0, true) == 2);
    check(tier(true, 8, 0, false) == 1);
    check(tier(true, 8, 1.0f / 512.0f, true) == 2);
    check(tier(true, 8, std::nextafter(1.0f / 512.0f, 1.0f), true) == 1);
    check(tier(true, 8, 1.0f / 256.0f, true) == 1);
    check(tier(true, 16, 0.003f, true) == 2);
    check(tier(true, 16, 0.003f, false) == 1);
    check(tier(true, 8, std::numeric_limits<float>::quiet_NaN(), true) == 0);
    check(tier(true, 8, 0, true, true, true) == 3);
    check(tier(true, 8, 0, true, false, true) == 2);
    check(tier(true, 8, 0, true, true, false) == 2);
    check(tier(true, 8, 0, false, true, true) == 1);
    check(tier(false, 8, 0, true, true, true) == 0);
    check(tier(true, 8, 1.0f / 256.0f, true, true, true) == 1);
    using leonida::conservative_raster::inner_coverage_requirements;
    const std::uint32_t module[] = {0x07230203, 0x00010500, 0, 2, 0, (2u << 16) | 17, 5265,
                                  (3u << 16) | 16, 1, 4446};
    const auto requirements = inner_coverage_requirements(module);
    check(requirements.valid && requirements.fully_covered && requirements.post_depth);
    check(!inner_coverage_requirements(std::span(module).first(9)).valid);
    const std::uint32_t zero_count[] = {0x07230203, 0x00010500, 0, 2, 0, 0};
    check(!inner_coverage_requirements(zero_count).valid);
    const std::uint32_t ordinary[] = {0x07230203, 0x00010500, 0, 2, 0, (2u << 16) | 17, 1};
    const auto regular = inner_coverage_requirements(ordinary);
    check(regular.valid && !regular.fully_covered && !regular.post_depth);
    if (failures) std::fprintf(stderr, "conservative raster capability checks: %u failures\n", failures);
    else std::puts("conservative raster: enabled extension, precision and uncertainty bounds checked");
    return failures ? 1 : 0;
}
