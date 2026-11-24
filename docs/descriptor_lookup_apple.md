# Descriptor lookup cost on Apple M3 Max

Divergent misses showed the largest change with table size. On the GPU-only memory type, one million logical lane lookups took 49 µs at 1,000 live descriptors, 156 µs at 10,000 and 479 µs at 100,000. Divergent texture hits took 65, 68 and 74 µs. These are synthetic compute results from one M3 Max on AC power, running macOS 27.3 and MoltenVK 1.4.0. The 40-core GPU ran the benchmark unchanged at `f165a13`. [D3D-56 benchmark PR #639](https://github.com/ItsNotRexed/Leonida/pull/639) · [Measured revision](https://github.com/ItsNotRexed/Leonida/commit/f165a13ff8bc12d2516e399bb87f923553fa107e)

## Measurements

Each cell is gross GPU time in **µs per million logical lane lookups**, including key reads and result accumulation. The benchmark also prints a figure with a key-read baseline subtracted. The table uses its gross column.

| Live descriptors | Memory type | Uniform T# hit | Uniform S# hit | Divergent T# hit | Divergent T#+S# pair | Divergent miss |
|---:|:---|---:|---:|---:|---:|---:|
| 1,000 | GPU-only | 20.8 | 23.2 | 65.0 | 64.0 | 49.3 |
| 1,000 | Host-visible | 20.8 | 23.2 | 65.1 | 64.0 | 49.3 |
| 10,000 | GPU-only | 21.3 | 20.8 | 67.6 | 67.4 | 155.9 |
| 10,000 | Host-visible | 21.3 | 20.8 | 67.7 | 67.4 | 155.8 |
| 100,000 | GPU-only | 20.8 | 20.5 | 73.9 | 74.7 | 479.3 |
| 100,000 | Host-visible | 20.9 | 20.5 | 73.8 | 74.7 | 482.7 |

At 100,000 entries on type 0, a divergent T# dispatch took 1.24 ms and a T#+S# dispatch took 2.51 ms. Each paired iteration resolves two descriptors.

Both Vulkan memory types use the same 128 GiB unified heap. Type 0 is device-local. Type 1 is device-local, host-visible, coherent and cached. At 100,000 entries, the 12 round medians for divergent misses ranged from 467–525 µs per million on type 0 and 471–505 on type 1.

The tables contain 90% T# texture keys and 10% S# sampler keys. Their allocations were approximately 0.156, 2.50 and 80 MiB. At 100,000 entries the ordinary uniform hit runs covered all 90,000 textures or all 10,000 samplers. Each workgroup shared a key for each iteration. The separate hot-key control reused one texture. Divergent misses exhausted the eight-probe search and visited 1,024, 16,384 and 131,072 distinct absent keys across the three sizes. Those counts describe the query schedule. Cache residency and memory traffic were unmeasured.

## Method and replay

Each dispatch used 4,096 groups of 64 lanes and 64 iterations. The M3 Max reported 32-lane subgroups. The denominator was 16,777,216 logical lookups per single-key dispatch and 33,554,432 per paired dispatch, counting the texture and sampler separately. Vulkan timestamps bracketed one dispatch. The queue advertised 64 valid bits and a 1 ns tick period.

Each cell is the median of 12 process medians, each from 25 timed dispatches after five warmups. The 11 scenarios ran in the benchmark's fixed order. Across both memory types and three sizes there were 19,800 timed dispatches. A separate pre-timing dispatch checked 5,042 of 262,144 lane accumulators against the CPU oracle for each case. The run used the default macOS loader with validation layers off.

From a Leonida checkout with Vulkan headers, CMake, Ninja and MoltenVK installed:

```sh
git fetch https://github.com/ItsNotRexed/Leonida.git f165a13ff8bc12d2516e399bb87f923553fa107e
git worktree add --detach ../leonida-639 FETCH_HEAD
cd ../leonida-639
cmake -S runtime -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target vk_descriptor_bench
build/vk_descriptor_bench --self-test
build/vk_descriptor_bench --list-devices
for memory in device host; do
  build/vk_descriptor_bench --device 0 --memory "$memory" --sizes 1000,10000,100000 \
    --groups 4096 --iters 64 --warmup 5 --reps 25 --max-dispatch-ms 0
done
```

These commands predate `--warmup-ms` (TEST-277, default 300 ms of extra warm-up). With a current build add `--warmup-ms 0` to reproduce them as run.

Device 0 was the M3 Max. For the full 12-round series, let A, B and C denote 1k, 10k and 100k. The first six size orders were ABC, CBA, BAC, CAB, ACB and BCA, starting with device, host, device, device, host and host respectively. The next six reversed the sequence of rounds and swapped the starting memory type in each.
