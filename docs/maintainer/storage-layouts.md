# NInfer Persistent Storage Layouts

This reference records the persistent tensor layouts and resource encoding used by current
`.ninfer` artifacts, including alignment, byte order, padding, encoded-size rules, and logical
decode. Numeric semantics come from [`tensor-formats.md`](tensor-formats.md); framing and object
ranges come from [`artifact-container.md`](artifact-container.md).

## 1. Registered identities

The storage registry contains exactly these identities:

| Identity | Kind | Compatible numeric formats | Logical shape | Object alignment |
|---|---|---|---|---:|
| `contiguous_le_v1` | tensor layout | `bf16`, `fp32`, `int32` | rank `0..16` | 256 bytes |
| `row_split_k128_v1` | tensor layout | `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | rank 2 `[N,K]` | 256 bytes |
| `block_scale_k16_m128x4_v1` | tensor layout | `nvfp4` | rank 2 `[N,K]`, `N % 128 == 0`, `K % 64 == 0` | 256 bytes |
| `row_scale_v1` | tensor layout | `fp8_e4m3fn_row_bf16` | rank 2 `[N,K]` | 256 bytes |
| `ggml_blocks_v1` | tensor layout | the nine `ggml_*` block formats | rank `1..16`, `K % values_per_block == 0` | 256 bytes |
| `ggml_rows_page4k_v1` | tensor layout | the nine `ggml_*` block formats | rank 2 `[N,K]`, `K % values_per_block == 0`, row ≤ 4096 bytes | 4096 bytes |
| `raw_bytes_v1` | resource encoding | not applicable | nonempty byte string | 1 byte |

These format/layout pairs define the current codec support. Native consumer requirements are
covered separately in Section 10. The two GGML layouts are tooling-only for now, like their
formats: the C++ layout registry does not accept them yet.

Object alignment applies to the object's payload-relative `offset` in the `.ninfer` JSON. Internal
plane offsets and padding belong to the selected layout. Inter-object padding belongs to the
container and is not included in an object's `bytes`.

The helper used below is:

```text
align_up(x, a) = ceil_div(x, a) * a
```

All formula inputs and intermediate results are nonnegative integers. The container implementation
must reject a shape or calculation that cannot be represented by its file-offset and size types.

## 2. `contiguous_le_v1`

### 2.1 Logical traversal

`contiguous_le_v1` stores direct logical words in C order: the last logical dimension varies
fastest. For shape `[D0, D1, ..., D(r-1)]`, coordinate `[i0, i1, ..., i(r-1)]` has linear index:

```text
index = (((i0 * D1 + i1) * D2 + i2) ... ) * D(r-1) + i(r-1)
```

A rank-zero shape `[]` contains one scalar word. For every other legal shape:

```text
elements = product(shape)
```

The layout performs no reshape, transpose, type conversion, or numerical canonicalization. Those
operations, when required, occur before layout encoding under the model-specific conversion recipe.

### 2.2 Word bytes

Words are serialized least-significant byte first:

| Format | Bytes per element | Stored word |
|---|---:|---|
| `bf16` | 2 | the exact 16-bit bfloat16 logical word, little-endian |
| `fp32` | 4 | the exact 32-bit IEEE-754 binary32 logical word, little-endian |
| `int32` | 4 | the exact 32-bit two's-complement logical word, little-endian |

Signed zero, subnormal, infinity, NaN payload, and integer-word behavior are determined by the
numeric-format contract. The layout only preserves the word bits.

### 2.3 Encoded size

There is no internal prefix, stride table, per-row padding, or trailing padding:

```text
payload_bytes = elements * bytes_per_element(format)
```

The tensor object's JSON `bytes` must equal `payload_bytes` exactly.

## 3. `row_split_k128_v1`

### 3.1 Logical and physical geometry

The logical tensor is a positive rank-two matrix `[N,K]`. The format supplies code width `b` and
group size `G`:

| Format | `b` | `G` | Base bytes per group `B` | High bytes per group `H` |
|---|---:|---:|---:|---:|
| `q4_g64_fp16` | 4 | 64 | 32 | 0 |
| `q5_g64_fp16` | 5 | 64 | 32 | 8 |
| `q6_g64_fp16` | 6 | 64 | 32 | 16 |
| `q8_g32_fp16` | 8 | 32 | 32 | 0 |

The layout extends the last axis to a multiple of 128:

```text
K_pad              = align_up(K, 128)
groups_per_row     = K_pad / G
logical_groups     = ceil_div(K, G)
physical_group_cnt = N * groups_per_row
```

`K_pad` is physical geometry and is not added to the JSON `shape`. Because both registered group
sizes divide 128, `groups_per_row` is integral.

For the final partially logical group, lanes whose column is at least `K` have signed code zero. Its
scale remains the scale of the logical group defined by the numeric-format contract. Any complete
physical group after `logical_groups` has scale word `0x0000` and all codes zero. Physical padding
therefore cannot change a decoded logical value.

### 3.2 Payload planes

The payload contains three conceptual planes in this order:

```text
base-code plane
zero padding to a 256-byte boundary
optional high-bit plane
zero padding to a 256-byte boundary
binary16 scale plane
```

Q4 and Q8 have no high-bit bytes. They still place the scale plane at the first 256-byte boundary
after the base-code plane. There is no padding after the scale plane inside the object.

Within every plane, traversal order is:

```text
row 0 group 0, row 0 group 1, ..., row 1 group 0, ...
```

Bytes belonging to one group are adjacent.

### 3.3 Base-code plane

For Q4, Q5, and Q6, let `q[i]` be lane `i`'s signed code and let:

```text
u[i] = q[i] modulo 2^b
```

be its unsigned `b`-bit two's-complement word. Each consecutive lane pair occupies one base byte:

```text
base[j] = (u[2*j] & 0x0f) | ((u[2*j + 1] & 0x0f) << 4)
```

Thus the even lane is in the low nibble and the odd lane is in the high nibble. Every G64 group
occupies 32 base bytes.

For Q8, each lane occupies one byte containing its exact 8-bit two's-complement word. Lane `i`
occupies byte `i`, so every G32 group occupies 32 base bytes. The numeric-format restriction that
excludes code `-128` remains in force.

The complete base plane is the concatenation of these per-group byte sequences in plane traversal
order.

### 3.4 High-bit plane

Only Q5 and Q6 have a high-bit plane. For each lane:

```text
high[i] = (u[i] >> 4) & ((1 << (b - 4)) - 1)
```

The high-bit stream is lane-major. Within one lane, bit 4 is emitted first, followed by bit 5 for
Q6. Stream bit `t` is stored in bit `(t mod 8)` of byte `floor(t / 8)`; bit zero is therefore the
first bit of every byte.

Equivalently:

- Q5 byte 0 contains the high bit of lanes 0 through 7 in byte bits 0 through 7;
- Q6 byte 0 contains lane 0 bits 4 and 5 in byte bits 0 and 1, lane 1 bits 4 and 5 in byte bits 2
  and 3, and so on.

One Q5 G64 group occupies 8 high bytes. One Q6 G64 group occupies 16 high bytes.

### 3.5 Scale plane

Every physical group owns one 16-bit scale word. Scale words follow the same row/group traversal as
the code planes and are stored little-endian:

```text
scale_index(row, group) = row * groups_per_row + group
```

For a logical group, the word is exactly the binary16 multiplier defined by its numeric format. The
padding rule in Section 3.1 defines the scale words for wholly physical groups.

### 3.6 Plane offsets and encoded size

Let:

```text
base_bytes  = N * groups_per_row * B
high_bytes  = N * groups_per_row * H
scale_bytes = N * groups_per_row * 2

base_offset  = 0
high_offset  = align_up(base_bytes, 256)
scale_offset = high_offset + align_up(high_bytes, 256)

payload_bytes = scale_offset + scale_bytes
```

The byte ranges are:

```text
base  = [base_offset,  base_offset  + base_bytes)
high  = [high_offset,  high_offset  + high_bytes)
scale = [scale_offset, scale_offset + scale_bytes)
```

When `H = 0`, `high_bytes = 0` and `scale_offset = high_offset`. Bytes between the end of the base
plane and `high_offset`, and between the end of a nonempty high plane and `scale_offset`, are zero.

The tensor object's JSON `bytes` must equal `payload_bytes`; it does not include the gap needed to
align the following object.

For example, a Q5 tensor with shape `[2,130]` has `K_pad=256`, four groups per row,
`base_bytes=256`, `high_bytes=64`, `scale_bytes=16`, `high_offset=256`, `scale_offset=512`, and
`payload_bytes=528`.

### 3.7 Row views and row slices

Row addressability is an intrinsic property of this layout. Define:

```text
base_row_bytes  = groups_per_row * B
high_row_bytes  = groups_per_row * H
scale_row_bytes = groups_per_row * 2
```

A non-owning logical view of consecutive rows `[row_begin, row_begin + row_count)` uses:

```text
base_view  = base_offset  + row_begin * base_row_bytes
high_view  = high_offset  + row_begin * high_row_bytes   # absent when H = 0
scale_view = scale_offset + row_begin * scale_row_bytes
```

and spans `row_count * base_row_bytes`, `row_count * high_row_bytes`, and
`row_count * scale_row_bytes` in the respective planes. Its logical shape is `[row_count,K]` and it
retains the parent object's `K_pad` and `groups_per_row`. A row view is therefore three plane spans,
not one assumed-contiguous payload range.

If those rows are materialized as a standalone payload, their three row spans are concatenated in
the same plane order, with the plane offsets and zero padding recomputed from Section 3.6 using
`N=row_count`. This produces another valid `row_split_k128_v1` tensor without decoding or repacking
individual codes.

## 4. `block_scale_k16_m128x4_v1`

This layout stores only rank-two `nvfp4` matrices `[N,K]` satisfying:

```text
N > 0
K > 0
N % 128 == 0
K % 64 == 0
```

It adds no logical padding. Let:

```text
code_plane_bytes      = N * K / 2
scale_plane_offset    = align_up(code_plane_bytes, 256)
scale_plane_bytes     = N * K / 16
weight_divisor_offset = scale_plane_offset + scale_plane_bytes
payload_bytes         = weight_divisor_offset + 4
```

The payload is a row-major E2M1 packed-code plane, zero padding to `scale_plane_offset`, a
swizzled E4M3FN scale plane, and the little-endian FP32 weight-divisor word. Within each packed code
byte, the low nibble is the smaller K coordinate and the high nibble is the next coordinate.

For logical row `n`, scale-group coordinate `g=floor(k/16)`, and `K_tiles=K/64`, define:

```text
row_tile   = floor(n / 128)
row_inner  = n % 128
scale_tile = floor(g / 4)
scale_lane = g % 4
```

The scale word's byte offset within the scale plane is:

```text
(row_tile * K_tiles + scale_tile) * 512
+ (row_inner % 32) * 16
+ floor(row_inner / 32) * 4
+ scale_lane
```

Layout decoding must recover the original packed E2M1 words, natural `[N,K/16]` E4M3FN scale-word
matrix, and exact divisor word. It never decodes and re-encodes either floating-point format.

## 5. `row_scale_v1`

`row_scale_v1` stores only rank-two `fp8_e4m3fn_row_bf16` matrices `[N,K]` with positive
dimensions. It adds no logical or physical matrix padding. Let:

```text
code_plane_bytes   = N * K
scale_plane_offset = align_up(code_plane_bytes, 256)
scale_plane_bytes  = N * 2
payload_bytes      = scale_plane_offset + scale_plane_bytes
```

The payload begins with one row-major E4M3FN byte per logical weight. Coordinate `[n,k]` is stored
at code-plane offset `n * K + k`. Zero bytes fill the interval from `code_plane_bytes` to
`scale_plane_offset`.

The scale plane contains one little-endian BF16 word per logical row in increasing `n` order. Scale
word `n` begins at `scale_plane_offset + 2 * n`. The layout neither converts the BF16 multiplier nor
combines it with its E4M3FN row. Code and scale validity and represented-weight reconstruction are
defined by `fp8_e4m3fn_row_bf16` in [`tensor-formats.md`](tensor-formats.md).

A logical row view consists of its `K` consecutive code bytes and its one BF16 scale word; those two
spans are not one assumed-contiguous payload range. A standalone consecutive slice or row gather is
encoded by concatenating the selected code rows, recomputing the scale-plane alignment for the new
row count, and appending the selected scale words in the same row order. It does not decode or
requantize either plane.

## 6. `ggml_blocks_v1`

`ggml_blocks_v1` stores a GGML block format of shape `[..., K]` (rank 1 through 16) with no
padding. Let `B` and `S` be the format's values and bytes per block, and `rows` the product of the
leading dimensions (1 for rank 1):

```text
row_bytes     = (K / B) * S
payload_bytes = rows * row_bytes
```

A row is its `K / B` blocks in increasing K order, each block's bytes unchanged, and rows follow in
C order of the leading coordinates. Logical element `[r, k]` of the `[rows, K]` view lives in block
`k / B` of row `r`, at payload offset `r * row_bytes + (k / B) * S`.

A GGUF tensor with dimensions `(ne0 = K, ne1, ne2, ...)` therefore has the logical shape
`[..., ne2, ne1, K]` and identical bytes. Consecutive complete rows form one contiguous byte range,
so a row slice, an expert of an expert bank, or a group of parameters packed by concatenating
complete rows needs no repacking. A converter that interleaves parameters (for example each
expert's gate rows followed by its up rows) does so by the order of the rows it writes; the layout
itself is unchanged.

## 7. `ggml_rows_page4k_v1`

`ggml_rows_page4k_v1` stores a rank-two GGML block matrix `[N,K]` whose rows are read one at a time
by direct 4 KiB I/O, such as Qwen3.8-Flash-Next's n-gram embedding table. With `row_bytes` as in
Section 6, which must be at most 4096:

```text
rows_per_page = floor(4096 / row_bytes)
pages         = ceil(N / rows_per_page)
payload_bytes = pages * 4096
offset(row n) = (n / rows_per_page) * 4096 + (n % rows_per_page) * row_bytes
```

Page `p` holds rows `p * rows_per_page` onward, each row's bytes unchanged and in order. The bytes
from `rows_per_page * row_bytes` to the end of every page, and the slots of the missing rows in the
last page, are zero; a reader rejects other contents. No row crosses a page boundary. The object
alignment of 4096 bytes, with the container's 4096-byte payload alignment, puts every page on a
4096-byte boundary of its file.

For the 90-byte IQ4_NL rows of a 160-wide table, 45 rows fill 4050 bytes of each page and 46 bytes
are zero; 320,001,536 rows take 7,111,146 pages (29,127,254,016 bytes, 1.1% more than the unpaged
rows).

## 8. `raw_bytes_v1`

`raw_bytes_v1` is a resource encoding, not a tensor layout. Its enclosing object payload is
the resource byte string itself:

```text
payload_bytes = resource_length
alignment     = 1
```

It has no embedded length, header, terminator, filename, character encoding, compression, or
trailing padding. The resource object's JSON `bytes` is its exact nonzero length, and a reader
returns the complete span unchanged. A model contract assigns a resource name and interprets those
bytes; the common encoding does not infer that meaning from the name.

## 9. Decode boundary

Layout decoding yields only persistent logical words:

- `contiguous_le_v1` yields the direct BF16, FP32, or I32 words in logical coordinate order;
- `row_split_k128_v1` yields the grouped signed codes and binary16 scales for logical columns
  `0..K-1`, discarding physical columns `K..K_pad-1`;
- `block_scale_k16_m128x4_v1` yields the packed E2M1 words, natural E4M3FN group-scale words, and
  matrix-level FP32 weight divisor;
- `row_scale_v1` yields the natural row-major E4M3FN code words and one BF16 multiplier per logical
  row;
- `ggml_blocks_v1` and `ggml_rows_page4k_v1` yield each row's GGML blocks unchanged;
- `raw_bytes_v1` yields the enclosing resource bytes.

Dequantized values follow the reconstruction rule in `tensor-formats.md`. This document does
not select a quantization encoder, output dtype, accumulation dtype, kernel, runtime device layout,
or model consumer.

## 10. Logical views and native operands

Bindings address C-order logical element ranges of a parent object. The parent retains its full
geometry and backing allocation, so a view can locate code and scale planes using the original
matrix dimensions. Materialization uploads each required parent once and binds non-owning views.

[`weight_view.cpp`](../../src/core/weight_view.cpp) provides plane addressing and the bridge to
native operands. Direct tensors can use a contiguous element range. Grouped integer matrices can
use consecutive complete rows with unchanged K, using independent code, high-bit and scale pointers.

The current native `Weight` bridge requires a complete parent for FP8 and NVFP4. Their consumers
use the complete matrix geometry for plane addressing; a row slice cannot be passed as though its
payload were a newly packed smaller matrix. Adjacent logical projections can still share one
parent: when the chosen fused implementation consumes their complete union, it receives that
parent as one native weight.

Offline codecs can produce a standalone slice with its own plane offsets. The loader does not
perform that transformation. An execution implementation that accepts additional view forms must
consume the original parent geometry correctly.
