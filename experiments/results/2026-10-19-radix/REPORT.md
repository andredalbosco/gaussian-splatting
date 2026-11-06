# GPU radix sorting in the main renderer

Historical snapshot before dispatch fusion and source reorganization. The original
source hashes and measurements below describe that snapshot. The radix implementation now
lives in `src/radix_sort.cpp`; see [the subsequent fusion results](../2026-10-19-radix-fusion/report.md)
for the four-pipeline, 32-dispatch version. Current shader validators use the standalone
`experiments/radix_validation.hpp` harness rather than including the app source.

The checkpoint before GPU-sort integration is `d036288`. The main quad renderer now
generates depth keys and sorts indices on the GPU, before its render pass. The production
implementation is `src/radix_sort.cpp` plus compute entry points in `src/shaders.wgsl`.
`Gaussian` and `Camera` keep their existing names. The tiled renderer remains a separate
experiment and is not linked into the main app.

## Pipeline

1. Read the Gaussian positions and camera forward vector; emit `(descending_depth_key, source_index)`.
2. For each of eight 4-bit digits, form workgroup histograms and stable local ranks using
   atomic membership bitmaps and `countOneBits`.
3. Prefix-scan histogram blocks in parallel, then scan their much smaller totals across bins.
4. Scatter into the alternate key/index buffer. Every pass is stable.
5. Extract source indices into the render pipeline's `order` storage buffer.

All 34 dispatches are recorded in one compute pass on the frame's command encoder.
The render pass follows and reads `order` without a CPU readback or index upload.
The sort runs only when the view direction changes or new Gaussians are loaded. Translation
adds a common depth offset, so it does not change mathematical relative ordering.

The signed-float key mapping reverses numeric depth order, including negative depths;
positive and negative zero have the same key. Initial indices are ascending, so equal-depth
ties retain source-index order, matching the previous CPU comparator. NaN inputs are not
part of the controlled-scene contract. GPU and CPU floating-point dot products can differ
slightly for rotated cameras; ordering is exact with respect to the GPU-generated keys.

Scratch storage costs two 8-byte/item key/index buffers plus a 4-byte/item rank buffer,
with roughly 0.25 bytes/item of histogram storage and small block totals/uniforms. The
existing 4-byte/item render order buffer is separate. Workgroups use 256 invocations;
no subgroup features or workgroup scheduling assumptions are required.

## Validation

`validate_radix_sort` checks 25 cases against a CPU reference, sorting twice per case:
empty/singleton inputs, signed and widely separated depths, signed zero, all-equal depths,
ascending/descending inputs, repeated keys, rotated cameras, and counts straddling
256-item workgroups and 65,536-item histogram blocks. The largest is 1,048,577 items,
which crosses an additional hierarchical prefix boundary.
There are zero ordering/key-generation errors. Each rebuild binds a new set of buffers,
covering changing scene sizes. Key generation is checked separately; rotated dot products
use an error bound scaled by the sum of absolute products to account for cancellation.

`validate_splats` now invokes the production GPU sort before rendering, rather than uploading
CPU indices. All six finite-difference CPU image-reference cases pass, including two-layer
blending and a rotated camera. Maximum channel error is 1/255.

The native and Emscripten builds pass. Live Chrome rendering, orbit, and zoom were
visually checked on the Moonlight desktop; no WebGPU warnings/errors were logged
(the only console error was a missing favicon).

## Initial timings

RTX 4090, native Dawn/Vulkan, Clang -O3. Units are milliseconds, lower is better.

| Gaussians | CPU depth + std::sort | GPU depth + radix + index extraction | GPU p90 |
|---|---:|---:|---:|
| 10,000 | 0.629 | 0.073 | 0.074 |
| 100,000 | 8.501 | 0.099 | 0.103 |
| 1,000,000 | 91.664 | 0.254 | 0.268 |

CPU numbers are the median of five runs, including depth calculation and sorting but
excluding index upload. GPU numbers are pass timestamps from 32 operations after four
warmups, including depth-key generation, all radix passes, and final index extraction.
Positions were resident on the GPU before measurement. CPU command recording, submission,
synchronization, scene upload, presentation, and readback are excluded from the GPU numbers.
This is a processing-time comparison, not an end-to-end frame-rate or latency speedup.
The GPU was shared with other applications; these are initial measurements rather than a
multi-trial controlled performance study. Timestamp quantization was disabled.

## Reproduce

```sh
cmake --build build --target gaussian_splatting validate_splats validate_radix_sort -j 4
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/validate_splats
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/validate_radix_sort experiments/results/2026-10-19-radix
```

Omit the output directory to run correctness checks without timings.
