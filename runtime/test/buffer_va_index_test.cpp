// BufferVaIndex (runtime/src/buffer_va_index.h): the VA -> buffer lookup behind VulkanContext::buffer_at. CPU only.
//
// Checks fixed cases (edges, aliasing placed buffers, re-registration), then random tables against a linear scan
// with the same rule, and prints how long a lookup takes against the linear scan over every resource it replaces.
// `--bench` times larger tables.

#include "../src/buffer_va_index.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

using leonida::BufferVaIndex;
using Index = BufferVaIndex<std::uintptr_t>;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

// The id that holds `va` (preferring one that holds `bytes` from it), 0 for none.
std::uintptr_t at(const Index& index, std::uint64_t va, std::uint64_t bytes = 1) {
    Index::Hit hit{};
    return index.find(va, hit, bytes) ? hit.id : 0;
}

void fixed_cases() {
    Index index;
    check(at(index, 0x1000) == 0, "an empty index holds nothing");

    index.insert(0x1000, 0x100, 1);  // [0x1000, 0x1100)
    index.insert(0x2000, 0x10, 2);   // [0x2000, 0x2010)
    check(at(index, 0x1000) == 1, "the first byte");
    check(at(index, 0x10ff) == 1, "the last byte");
    check(at(index, 0x1100) == 0, "one past the end");
    check(at(index, 0x0fff) == 0, "one before the start");
    check(at(index, 0x1800) == 0, "a gap between two buffers");
    check(at(index, 0x200f) == 2, "the second buffer's last byte");
    check(at(index, ~std::uint64_t(0)) == 0, "the highest VA");

    Index::Hit hit{};
    check(index.find(0x1080, hit) && hit.start == 0x1000 && hit.size == 0x100, "a hit names its range");

    index.insert(0, 0x100, 3);
    index.insert(0x3000, 0, 4);
    check(index.size() == 2 && at(index, 0) == 0 && at(index, 0x3000) == 0, "a null VA or an empty range is not indexed");

    // Aliasing placed buffers: a long one with two short ones inside it. The nearest start that holds the VA wins.
    Index alias;
    alias.insert(0x10000, 0x10000, 10);  // [0x10000, 0x20000)
    alias.insert(0x11000, 0x100, 11);    // inside it
    alias.insert(0x12000, 0x100, 12);    // inside it
    check(at(alias, 0x11080) == 11, "a short buffer inside a long one");
    check(at(alias, 0x11100) == 10, "past the short buffer, the long one still holds the VA");
    check(at(alias, 0x1f000) == 10, "past every short buffer");
    check(at(alias, 0x20000) == 0, "past the long buffer");

    // A view that starts in a short buffer and runs past its end: the long buffer around it holds all of it.
    check(at(alias, 0x11080, 0x80) == 11, "a view that fits in the short buffer stays in it");
    check(at(alias, 0x11080, 0x81) == 10, "a view past the short buffer's end goes to the long one around it");
    check(at(alias, 0x1ff00, 0x200) == 10, "a view no range holds: the closest that holds its start");
    check(at(alias, 0x11080, 0x10000) == 11, "a view past the long buffer too: the closest start, as without a size");
    check(at(alias, 0x11080, 0) == 11, "a size of 0 counts as one byte");

    // Two resources placed at the same offset: the larger ID, either way they were added.
    alias.insert(0x30000, 0x100, 21);
    alias.insert(0x30000, 0x200, 20);
    check(at(alias, 0x30080) == 21, "the same start: the larger ID");
    check(at(alias, 0x30180) == 20, "the same start: past the shorter one");

    // Erasing and re-registering.
    alias.erase(0x11000, 11);
    check(at(alias, 0x11080) == 10, "an erased buffer leaves the one around it");
    alias.erase(0x11000, 11);  // twice: nothing
    alias.erase(0x10000, 99);  // a wrong ID: nothing
    check(at(alias, 0x10000) == 10, "an erase with the wrong ID keeps the range");
    alias.erase(0x10000, 10);
    alias.erase(0x12000, 12);
    alias.erase(0x30000, 20);
    alias.erase(0x30000, 21);
    check(alias.size() == 0 && at(alias, 0x10000) == 0, "everything erased");
    alias.insert(0x40000, 0x10, 30);  // the bound restarts from an empty index
    check(at(alias, 0x40008) == 30 && at(alias, 0x40010) == 0, "an index refilled after it emptied");

    // The bound follows the live ranges: freeing the longest one brings it down to the next longest.
    Index bound;
    bound.insert(0x10000, 0x100, 1);
    bound.insert(0x20000, 0x100, 2);  // the same size twice: dropping one keeps the other's
    bound.insert(0x30000, 0x40000000, 3);
    check(bound.longest() == 0x40000000, "the bound is the longest range");
    bound.erase(0x30000, 3);
    check(bound.longest() == 0x100, "freeing the longest range lowers the bound");
    bound.erase(0x10000, 1);
    check(bound.longest() == 0x100 && at(bound, 0x20080) == 2, "a range of the same size keeps the bound");
    bound.insert(0x20000, 0x40, 2);  // re-registered with a smaller size: the old size goes
    check(bound.longest() == 0x40 && at(bound, 0x20040) == 0, "re-registering replaces the size");
    bound.erase(0x20000, 2);
    check(bound.longest() == 0 && bound.size() == 0, "an empty index has no bound");
}

// The rule BufferVaIndex follows, by a scan: among the ranges that hold all of [va, va + bytes), the greatest
// (start, id); when none does, the greatest among those that hold va.
struct Range {
    std::uint64_t start, size;
    std::uintptr_t id;
};

std::uintptr_t scan(const std::vector<Range>& ranges, std::uint64_t va, std::uint64_t bytes = 1) {
    const Range* best = nullptr;
    const Range* whole = nullptr;
    const auto greater = [](const Range& a, const Range* b) {
        return !b || a.start > b->start || (a.start == b->start && a.id > b->id);
    };
    for (const Range& r : ranges) {
        if (va < r.start || va - r.start >= r.size) continue;
        if (greater(r, best)) best = &r;
        if (bytes <= r.size - (va - r.start) && greater(r, whole)) whole = &r;
    }
    return whole ? whole->id : best ? best->id : 0;
}

void random_tables() {
    std::mt19937_64 rng(0x1e0a1da);
    for (int round = 0; round < 200; ++round) {
        Index index;
        std::vector<Range> ranges;
        const bool aliasing = round % 2;  // half the tables pack buffers apart, half overlap them
        std::uint64_t next = 0x10000;
        const int count = 1 + int(rng() % 64);
        for (int i = 0; i < count; ++i) {
            const std::uint64_t size = 1 + rng() % 0x4000;
            const std::uint64_t start = aliasing ? 0x10000 + rng() % 0x20000 : next;
            next += size + rng() % 0x1000;
            const Range r{start, size, std::uintptr_t(i + 1)};
            ranges.push_back(r);
            index.insert(r.start, r.size, r.id);
        }
        // Drop a few, as remove_resource does.
        for (int i = 0; i < count / 4; ++i) {
            const std::size_t k = rng() % ranges.size();
            index.erase(ranges[k].start, ranges[k].id);
            ranges.erase(ranges.begin() + std::ptrdiff_t(k));
            if (ranges.empty()) break;
        }
        std::uint64_t longest = 0;
        for (const Range& r : ranges) longest = r.size > longest ? r.size : longest;
        if (index.longest() != longest) {
            std::printf("FAIL: round %d: bound %llu, longest live range %llu\n", round,
                        (unsigned long long)index.longest(), (unsigned long long)longest);
            ++failures;
            return;
        }
        for (int q = 0; q < 2000; ++q) {
            const std::uint64_t va = 0x8000 + rng() % (next + 0x20000);
            // A quarter of the lookups name one byte, the rest a view of up to twice the largest buffer.
            const std::uint64_t bytes = q % 4 ? 1 + rng() % 0x8000 : 1;
            if (at(index, va, bytes) != scan(ranges, va, bytes)) {
                std::printf("FAIL: round %d va 0x%llx bytes 0x%llx: index %llu, scan %llu\n", round,
                            (unsigned long long)va, (unsigned long long)bytes,
                            (unsigned long long)at(index, va, bytes), (unsigned long long)scan(ranges, va, bytes));
                ++failures;
                return;
            }
        }
    }
}

// What buffer_at did before: a walk over every registered resource, images included, in an unordered_map.
struct Binding {
    bool image;
    std::uint64_t va, size;
};

void time_lookups(std::size_t resources) {
    std::unordered_map<std::uintptr_t, Binding> all;
    Index index;
    std::vector<std::uint64_t> vas;
    std::mt19937_64 rng(resources);
    std::uint64_t next = 0x10000;
    for (std::size_t i = 1; i <= resources; ++i) {
        if (i % 2) {  // half images, as in a frame's working set
            all[i] = {true, 0, 0};
            continue;
        }
        const std::uint64_t size = 256 + rng() % (1 << 20);
        all[i] = {false, next, size};
        index.insert(next, size, i);
        vas.push_back(next + rng() % size);
        next += (size + 255) / 256 * 256;
    }
    constexpr int lookups = 20000;
    using clock = std::chrono::steady_clock;
    std::uint64_t sink = 0;

    auto t0 = clock::now();
    for (int i = 0; i < lookups; ++i) {
        const std::uint64_t va = vas[std::size_t(i) % vas.size()];
        for (const auto& [id, b] : all) {
            if (b.image || va < b.va || va - b.va >= b.size) continue;
            sink += id;
            break;
        }
    }
    const double linear = std::chrono::duration<double, std::nano>(clock::now() - t0).count() / lookups;

    t0 = clock::now();
    for (int i = 0; i < lookups; ++i) sink += at(index, vas[std::size_t(i) % vas.size()]);
    const double indexed = std::chrono::duration<double, std::nano>(clock::now() - t0).count() / lookups;

    std::printf("%7zu resources: linear scan %10.1f ns/lookup, index %6.1f ns/lookup (x%.0f)%s\n", resources, linear,
                indexed, linear / (indexed > 0 ? indexed : 1), sink ? "" : " ");
}

// A miss past the last of many small buffers, once a large buffer has come and gone (found in review of #2823): the
// index that held it must be as fast as one that never did.
void time_miss_after_free() {
    constexpr std::uint64_t count = 20000, stride = 256;
    Index fresh, churned;
    for (std::uint64_t i = 0; i < count; ++i) {
        fresh.insert(0x10000 + i * stride, stride, i + 1);
        churned.insert(0x10000 + i * stride, stride, i + 1);
    }
    churned.insert(0x10000, std::uint64_t(1) << 30, count + 1);  // 1 GiB over all of them
    churned.erase(0x10000, count + 1);
    check(churned.longest() == stride, "the bound comes back down after a large buffer is freed");

    const std::uint64_t miss = 0x10000 + count * stride + stride;
    constexpr int lookups = 20000;
    using clock = std::chrono::steady_clock;
    std::uint64_t sink = 0;
    double ns[2];
    const Index* indexes[2] = {&fresh, &churned};
    for (int k = 0; k < 2; ++k) {
        const auto t0 = clock::now();
        for (int i = 0; i < lookups; ++i) sink += at(*indexes[k], miss + std::uint64_t(i % 64));
        ns[k] = std::chrono::duration<double, std::nano>(clock::now() - t0).count() / lookups;
    }
    std::printf("miss past %llu buffers: fresh index %.1f ns, after a freed 1 GiB buffer %.1f ns%s\n",
                (unsigned long long)count, ns[0], ns[1], sink ? " (hit?)" : "");
}

} // namespace

int main(int argc, char** argv) {
    fixed_cases();
    random_tables();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    time_miss_after_free();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::puts("BufferVaIndex: fixed cases and 200 random tables agree with the scan");
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    for (const std::size_t n : {std::size_t(1000), std::size_t(10000)}) time_lookups(n);
    if (bench)
        for (const std::size_t n : {std::size_t(50000), std::size_t(100000)}) time_lookups(n);
    return 0;
}
