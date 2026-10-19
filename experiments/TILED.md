# Tiled GPU prototype

The standalone `tiled_viewer` experiment defaults to the tiled renderer; its
`GPU tiles` checkbox selects the quad renderer for comparison. The main app uses
only the quad renderer. All tiled sources and shaders live in `experiments/`.
No spherical harmonics were added.

```sh
cmake --build build --target tiled_viewer validate_tiled -j 4
./build/tiled_viewer
./build/validate_tiled
```

## Pipeline

1. Project once per Gaussian; reject invisible/negligible splats.
2. Atomically compact 32-byte projected records.
3. Generate 16-byte `(depth, tile, compact index, reverse source ID)` records,
   using opacity-aware bounds and an ellipse–tile intersection test.
4. Stable four-bit GPU radix passes: reverse ID, depth, tile.
5. Identify each tile's contiguous range.
6. Composite front-to-back in 16×16 workgroups with shared batches and early exit.
7. Present the RGBA8 storage texture with a fullscreen triangle.

Input Gaussians use 48 bytes versus 64 in the quad path. RGB is quantized to RGB8;
position, rotation, scale, and opacity remain f32. Projected records use 32 bytes.
There are two tile-record buffers for sorting, plus histograms/ranges. The comparison
app keeps the quad buffers too; compact records do not mean lower total memory than
the quad-only renderer. Initial tile-record capacity is 1,048,576 and grows on demand.

Counters are copied asynchronously for UI statistics and overflow recovery only.
Rendering never waits for a CPU-visible count. Overflow preserves the previous
complete image while buffers grow; the UI reports when a device limit is reached.

## Validation

`validate_tiled` compares against a GPU quad reference rendering into RGBA16Float.
The tiled result is RGBA8 with RGB8 source colors and early transmittance termination.
The per-channel acceptance tolerance is 4/255, including reference half-float blend
rounding. It also checks lexicographic ordering of every generated radix record,
compact-index bounds, overflow flags, and automatic capacity growth.

All 11 cases passed: empty scene, single Gaussian, two layers, equal-depth tie,
rotated anisotropic Gaussian, behind-camera rejection, random/cropped/low-opacity
splats, 10,000 random Gaussians, 400 overlapping Gaussians, Luigi, and forced overflow.
The largest observed channel difference was 0.014299 for Luigi. No channels exceeded
the tolerance. Original five finite-difference projection tests remain available
through `validate_splats`.

## Initial timings: RTX 4090, native Vulkan, Clang -O3

These are a first diagnostic, not a speedup claim. The quad path is faster here.
The timestamps are quantized to 65.536 microseconds; zero means below that resolution.

| Scene | Resolution | Tile records | Quad GPU ms | Tiled GPU ms | Quad completed-frame ms | Tiled completed-frame ms |
|---|---:|---:|---:|---:|---:|---:|
| 10,000 random | 640×360 | 18,903 | <0.066 | 0.459 | 0.188 | 15.258 |
| 400 overlapping | 640×360 | 368,000 | 0.262 | 4.915 | 1.880 | 13.747 |
| Luigi, 14,526 | 1280×720 | 147,620 | 0.066 | 2.556 | 0.128 | 16.999 |

Both timing paths output RGBA8. Quad GPU time covers its render pass. Tiled GPU time
covers projection, compaction, tile generation, sorting, range identification, and
compute compositing, excluding fullscreen presentation. Completed-frame time includes
CPU encoding, uniform writes for tiles, command-buffer finish, submission and waiting
for completion. It excludes initialization, scene upload, image readback, ImGui,
presentation, and CPU depth sorting. Cameras are fixed and quad indices are pre-sorted;
this is the cached-sort case, not a moving-camera end-to-end comparison.

Each method has two warmups followed by seven samples, reporting medians. Native
Dawn validation is enabled and Chrome/the desktop were also active; treat these as
exploratory results. The log includes CPU recording and command-buffer finish times.
The main next targets are reducing dispatch/submission overhead and replacing the
portable shared-memory partition/prefix stages with a faster scalable radix sorter.
GPU rendering is not inherently faster merely because sorting moved off the CPU.

[Full results](results/2026-10-19-tiled/validation.txt)

Chrome on the Moonlight desktop also passed rendering, camera interaction, mode
switching and window resizing at 3× device scale, with no console errors or warnings.
