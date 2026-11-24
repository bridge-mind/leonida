# Descriptor lookup cost on NVIDIA RTX 4070 SUPER (Ada)

Divergent misses showed the largest change with table size. On the GPU-only memory type, one million logical lane lookups took 150 µs at 1,000 live descriptors, 657 µs at 10,000 and 998 µs at 100,000. Divergent texture hits took 39.7, 75.0 and 119.0 µs. The ReBAR type (device-local and host-visible) stayed within 0.5% of GPU-only in every cell. Host-visible memory outside VRAM was within 8% of them at 1,000 entries, and at 100,000 it was 2.5× slower for divergent misses and 6.0× slower for uniform texture hits. These are synthetic compute results from one GeForce RTX 4070 SUPER on Windows 11 with NVIDIA driver 621.01. The benchmark ran unchanged at `dcd8ecf`. [D3D-56 benchmark PR #639](https://github.com/ItsNotRexed/Leonida/pull/639) · [Measured revision](https://github.com/ItsNotRexed/Leonida/commit/dcd8ecf10b138566d567020562b7afd853a89024)

## Measurements

Each cell is gross GPU time in **µs per million logical lane lookups**, including key reads and result accumulation. The benchmark also prints a figure with a key-read baseline subtracted. The table uses its gross column.

| Live descriptors | Memory type | Uniform T# hit | Uniform S# hit | Divergent T# hit | Divergent T#+S# pair | Divergent miss |
|---:|:---|---:|---:|---:|---:|---:|
| 1,000 | GPU-only | 8.6 | 8.3 | 39.7 | 40.3 | 150.0 |
| 1,000 | Host-visible | 9.2 | 9.0 | 39.9 | 40.4 | 150.6 |
| 1,000 | ReBAR | 8.6 | 8.3 | 39.6 | 40.3 | 149.9 |
| 10,000 | GPU-only | 8.7 | 8.5 | 75.0 | 89.4 | 657.4 |
| 10,000 | Host-visible | 11.5 | 10.1 | 90.0 | 106.3 | 886.8* |
| 10,000 | ReBAR | 8.7 | 8.5 | 75.2 | 89.3 | 658.8 |
| 100,000 | GPU-only | 9.1 | 9.1 | 119.0 | 167.1 | 998.4 |
| 100,000 | Host-visible | 54.4 | 13.9 | 172.4 | 236.6* | 2,460.9* |
| 100,000 | ReBAR | 9.1 | 9.1 | 119.3 | 166.8 | 1,002.1 |

\* Shortened by the benchmark's Windows timeout guard in 11 or 12 of the 12 rounds (see Method).

At 100,000 entries on type 1, a divergent T# dispatch took 2.00 ms and a T#+S# dispatch took 5.61 ms. Each paired iteration resolves two descriptors.

GPU-only is type 1, `DEVICE_LOCAL`, on heap 0 (11,997 MiB). Host-visible is type 2, `HOST_VISIBLE | HOST_COHERENT` without `HOST_CACHED`, on heap 1 (16,243 MiB, not device-local). ReBAR is type 4, `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, on heap 0. Every allocation on type 4 succeeded, up to the 20 MiB table. Resizable BAR was enabled: `nvidia-smi` reported a 16,384 MiB BAR1 aperture. On a similar card where the ReBAR type is missing or refuses the allocation, the benchmark skips that placement and the ReBAR rows are absent.

At 100,000 entries, the 12 round medians for GPU-only and ReBAR split into two groups. The four rounds whose process ran the 10,000-entry table first (rounds 3, 6, 7 and 10) measured 293–299 µs per million for divergent misses and 59.6–60.2 for divergent T# hits. The other eight rounds measured 992–1,015 and 118.7–142.9. The table medians follow the eight. The cause was not measured. Host-visible round medians for divergent misses ranged from 1,870–2,810 µs per million, and the spread did not follow the size order. At 1,000 and 10,000 entries, the round medians for each table cell differed by at most 10% on GPU-only and ReBAR, and by up to 37% on host-visible.

The tables contain 90% T# texture keys and 10% S# sampler keys. Their allocations were 163,844, 2,621,444 and 20,971,524 bytes (0.156, 2.50 and 20.0 MiB). The capacities were 4,096, 65,536 and 524,288 slots, at 24.4%, 15.3% and 19.1% load. At 100,000 entries the ordinary uniform hit runs covered all 90,000 textures or all 10,000 samplers. Each workgroup shared a key for each iteration. The separate hot-key control reused one texture and measured 7.2 µs per million on all three types and sizes. Divergent misses visited 1,024, 16,384 and 131,072 distinct absent keys across the three sizes. Those counts describe the query schedule. Cache residency and memory traffic were unmeasured.

## Method and replay

Each dispatch used 4,096 groups of 64 lanes and 64 iterations. The RTX 4070 SUPER reported 32-lane subgroups. The denominator was 16,777,216 logical lookups per single-key dispatch and 33,554,432 per paired dispatch, counting the texture and sampler separately. Vulkan timestamps bracketed one dispatch. The queue advertised 64 valid bits and a 1 ns tick period.

Each cell is the median of 12 process medians, each from 25 timed dispatches after five warmups. The 11 scenarios ran in the benchmark's fixed order. There were 36 processes: three memory types in each of 12 rounds. Across them there were 28,948 timed dispatches. Every process exited 0, wrote nothing to stderr and reported `timing_status` `ok` on all 1,188 rows. The run used the default Windows Vulkan loader with validation layers off.

Unlike the [M3 Max run](descriptor_lookup_apple.md), this run kept the benchmark's default `--max-dispatch-ms 120`, so that a slow dispatch cannot reach the Windows 2 s GPU timeout (TDR). Before each scenario, a 64-group, four-iteration probe estimates the full dispatch time. If the estimate exceeds 120 ms, the benchmark lowers the iteration count to aim at 60 ms and uses nine timed dispatches after two warmups. The figure stays per million lookups. Only host-visible cells were shortened. The 10,000-entry divergent miss ran with 17–20 iterations in 11 rounds. At 100,000 entries, divergent misses ran with 13–15 iterations in all 12 rounds, and T#+S# pairs with 25–31 in 11 rounds. The table marks these cells. Shortened dispatches had medians of 0.49–9.84 ms, far below the 120 ms limit.

From a Leonida checkout on Windows, with Visual Studio 2022 (C++ desktop workload, C++ CMake tools) and the Vulkan SDK installed:

```powershell
git fetch https://github.com/ItsNotRexed/Leonida.git dcd8ecf10b138566d567020562b7afd853a89024
git worktree add --detach ..\leonida-ada FETCH_HEAD
cd ..\leonida-ada
powershell -ExecutionPolicy Bypass -File scripts\build_runtime.ps1
runtime\build\vk_descriptor_bench.exe --list-devices
runtime\build\vk_descriptor_bench.exe --self-test
foreach ($memory in 'device', 'host', 'rebar') {
  runtime\build\vk_descriptor_bench.exe --device 0 --memory $memory --sizes 1000,10000,100000 `
    --groups 4096 --iters 64 --warmup 5 --reps 25 --csv
}
```

These commands predate `--warmup-ms` (TEST-277, default 300 ms of extra warm-up). With a current build add `--warmup-ms 0` to reproduce them as run.

Device 0 was the RTX 4070 SUPER, the only Vulkan device. For the full 12-round series, let A, B and C denote 1k, 10k and 100k. The rounds used the same size orders and starting memory types as the M3 Max run. The first six size orders were ABC, CBA, BAC, CAB, ACB and BCA, starting with device, host, device, device, host and host respectively. The next six reversed the sequence of rounds and swapped the starting memory type in each. Each round ran the other of device and host second, and ReBAR third.

## System

| | |
|:---|:---|
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB (discrete) |
| PCI id | `0x10de:0x2783`, subsystem `0x89521043` |
| Link | PCIe 4.0 x16 (maximum reported by `nvidia-smi`) |
| Resizable BAR | On (BAR1 aperture 16,384 MiB, `nvidia-smi -q -d MEMORY`) |
| Subgroup size | 32 |
| Driver | NVIDIA 621.01 (`nvidia-smi`), Vulkan API 1.4.357 |
| OS | Windows 11 Home 25H2, build 26200.9974 |
| Vulkan SDK | 1.4.363.1 (headers for the build) |
| Compiler | Visual Studio 2022 Build Tools, `RelWithDebInfo` via `scripts\build_runtime.ps1` |
