# Descriptor lookup cost on Intel UHD Graphics 770

@kapitche ran the D3D-56 `vk_descriptor_bench` synthetic compute benchmark on an Intel
UHD Graphics 770 (Xe-LP integrated GPU) under Windows 11 with Intel driver `101.6129`.
Vulkan reported a subgroup size of 32 and a timestamp period of 52.083 ns. The run used
`main` at [`8df277f`](https://github.com/ItsNotRexed/Leonida/commit/8df277fb035b384971497f6f9790f7fc4b9ec1d1).
[Original report by @kapitche](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7073751039)

## Measurements

The benchmark used its defaults of 4,096 workgroups of 64 invocations, 64 iterations
per invocation and a median of 25 timed dispatches per case. Each dispatch counted
16,777,216 logical lane lookups for single-key cases and 33,554,432 for paired cases,
where each iteration resolves one texture (T#) and one sampler (S#).

All values are **net µs per million logical lane lookups**, after the benchmark
subtracts the matching key-read baseline scaled to the number of lookups per iteration.
The lookup layout and eight-probe limit are described in
[the Vulkan design, §5.1](d3d12_vulkan.md#51-descriptor-hash-table-details).

| Live entries | Placement | Uniform T# hit | Uniform T#+S# pair | Divergent T# hit | Divergent miss |
|---:|:---|---:|---:|---:|---:|
| 1,000 | `device_local` | 263.0 | 300.5 | 1,350 | 6,249 |
| 1,000 | `host_visible` | 263.3 | 300.3 | 1,350 | 6,249 |
| 10,000 | `device_local` | 264.2 | 296.2 | 1,364 | 6,413 |
| 10,000 | `host_visible` | 263.9 | 296.3 | 1,363 | 6,402 |
| 100,000 | `device_local` | 280.5 | 316.0 | 1,483 | 8,263 |
| 100,000 | `host_visible` | 280.9 | 316.5 | 1,481 | 8,217 |

## Interpretation

For each size and scenario, the two placements differ by less than 0.6% relative
to `device_local`. For this run, Vulkan memory type 0 (`device_local`) and type 1
(`host_visible`, with `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) both used
shared system RAM. The small differences apply to these two memory types on this setup.

From 1,000 to 100,000 entries, the hit timings rose by about 5–10% across the two
placements, while divergent misses, which exhaust the eight-probe search, rose by
about 31–32%.
