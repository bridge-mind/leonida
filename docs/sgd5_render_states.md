# SGD5 pass render-state indices

In the Xbox 1.0.2.1 shader archive, bytes 6, 7 and 8 of a 24-byte pass select rows in the effect's rasterizer, depth/stencil and blend tables. These byte positions and state indices are zero-based. The six bytes immediately before them use one-based shader program selectors, with zero meaning unused. A renderer looking up a stored pass can therefore identify the three explicit state records alongside its shader programs. [Program lookup and effect boundaries](https://github.com/ItsNotRexed/Leonida/pull/2519)

| Pass byte | Effect-local table | Bytes per row | Stored rows across both databases | Effects with passes and at least two rows | Effects in which the pass byte selects every row |
|---:|---|---:|---:|---:|---:|
| 6 | Rasterizer | 16 | 64 | 4 | 4 |
| 7 | Depth/stencil | 24 | 124 | 33 | 33 |
| 8 | Blend | 68 | 398 | 7 | 7 |

All **28,798 passes** in the two databases satisfy the same rule. For a nonempty table, the byte is strictly less than the table's row count. If the count is zero, the byte is zero and there is no stored row to read. The state the runtime supplies in that case is unknown. An explicit row at index zero and an empty table can be distinguished by retaining the table count.

## Reproduce on a local package

Use the `shaders.rpf` listed in the [Xbox 1.0.2.1 manifest](../game/xbox-1.0.2.1.json). The two SGD5 databases start at `0x9000` and `0x10287000`. They contain 910 and 22 effects. Use each effect's own six program lists and its own state tables. [Named-pass reader and package test](https://github.com/ItsNotRexed/Leonida/pull/3036)

1. From the end of the six program ordinal arrays, skip 80 bytes. For each row width in `(16, 24, 68, 36)`, read one little-endian u32 count, record the offset of the first row, and advance by `count * width`. The fourth table is needed to locate the techniques.
2. Read the u32 technique count and skip four bytes per technique. The next u32 is the pass count. For every 24-byte pass, check offsets 6, 7 and 8 against counts 0, 1 and 2 respectively. For a nonempty table the selected row starts at `first_row + pass_byte * width`.
3. For an independent check of the byte positions, repeat the test for **each whole byte** at offsets 6 through 23. For each table, require every pass to satisfy the bounds above. Within each effect with passes and at least two rows, also require the set of byte values across its passes to equal the set of row indices. Exactly one whole-byte position per table satisfies both conditions across the archive. The table above gives those positions and the number of effects that distinguish them.

## Interpretation of the rows

The state names are supported by comparison with the public SGD2 reader and measurements of the Xbox rows. The 16-byte rows have fill mode 3, cull mode in `1..3` and a one-byte front-face flag. In all 124 depth/stencil rows the enable flags and write mask are in `0..1`, and the comparison function is in `1..8`. All 398 blend rows have a four-byte prefix and eight eight-byte render-target entries. Their enable flag is in `0..1` and their write mask is in `0..15`. Individual enum values do not establish the rest of the Xbox state layout. [SGD2 format comparison](https://github.com/Kalmarv/rage-engine-formats/blob/91e5abb68476709fac80f256bc64f928c6fc2d19/src/sgd2_effect.rs)

The measurements identify the indices stored in the package. Runtime selection of a technique, application of overrides, and the resulting image still require execution evidence.
