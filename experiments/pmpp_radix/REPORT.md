# WGSL radix comparison on Gaussian assets — 2026-11-09

Measured on NVIDIA GeForce RTX 4090, driver 595.71.05, native Dawn/Vulkan (adapter backend enum 6), Release build. These are GPU timestamp measurements, not CPU timings or frame times.

## Results

Each cell is the median of 15 trial medians (three camera orientations, five trials per orientation). Each trial has 64 timestamp samples, each spanning four complete sorts and divided by four. Eight warmup sorts precede each trial. Method order alternates between trials. Timestamp quantization is requested disabled; recorded intervals still have finite resolution, reduced by batching four sorts.

| Asset | Gaussians | Current masks, ms | PMPP-style local sorting, ms | Local / masks |
|---|---:|---:|---:|---:|
| luigi | 14526 | 0.047360 | 0.068608 | 1.45x |
| black_jacket | 41470 | 0.049920 | 0.073216 | 1.47x |

Both methods produced exactly the CPU reference stable order for both assets and all three orientations (12 method/scene/view checks, zero mismatches). Reference ordering uses the GPU-generated keys, with original source index as the tie breaker, avoiding CPU/GPU dot-product rounding differences. This validates ordering on the tested assets; it is not a new full edge-case suite or rendering validation.

## Compared implementations

- Masks: unchanged production `src/shaders.wgsl` and `RadixSort` host code.
- PMPP-style: experimental WGSL port of the optimized four-bit approach in PMPP fifth edition §§14.6–14.7. Four stable binary shared-memory partitions form 16 ordered local buckets. Local bucket boundaries produce counts and ranks; the existing global histogram scan stages then locate buckets, and scatter reads locally sorted pairs in lane order. This is an implementation of the book's approach, not code supplied by the book or an optimized library implementation.
- Both use 256 invocations/workgroup, eight four-bit passes, the same hierarchical global scan code, 32 dispatches, identical depth-key generation, and final Gaussian-index output. All Gaussian records are resident on the GPU before timestamps begin. Uploads, allocations, pipeline compilation, CPU command recording/submission, readback, and rendering are excluded.
- The portable local partition uses the existing 17-barrier Kogge–Stone helper four times per pass, plus redistribution/bucket-boundary barriers. It uses no subgroup features.
- The local-sort variant persists a sorted key/index pair and rank (12 bytes/key) in the scratch buffer between histogram/local-sort and scatter dispatches; masks persist only rank (4 bytes/key). This is a real implementation cost included in the comparison, not an isolated measurement of mask instructions versus partition instructions. More optimized variants could fuse/restructure these phases and use different scans or subgroup operations.
- The scatter's conditional `radix_input` reference in the experimental shader retains the production automatically generated bind-group layout; its count-zero branch never executes for dispatched keys.

The result establishes that the current mask sorter beats this straightforward PMPP-style WGSL port on these two assets. It does not establish that masks always beat local sorting, explain the time difference causally, or compare the basic 32-pass binary PMPP version. The assets are 14.5K and 41.5K keys, so both have only one histogram scan chunk per digit; this run does not establish large-array scaling. The GPU remained attached to the active desktop with other applications resident; background load and clock variation were not eliminated. No other benchmark was launched concurrently by this experiment.

## Reproduce

From the project root:

```sh
cmake --build build --target pmpp_radix_benchmark -j 8
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/pmpp_radix_benchmark experiments/pmpp_radix/timings.csv
```

`timings.csv` contains trial median/p90 values, `timings.csv.samples.csv` contains all 3,840 final timestamp samples (milliseconds per sort), and `run.log` records validation and trial output. `pilot_*` files preserve the initial single-sort timestamp run; the table above uses the final four-sort batches only. Production renderer source was not changed; the experiment adds one CMake target.

## Input/source SHA-256

- `assets/luigi.ply`: `6971debf967bac924a4c8bd4457b354db66326792d1a300fd996c7eefede04aa`
- `assets/black_jacket.ply`: `128f64140b357488920ed8c3fe35553d78f74c0deb8af8eefdd11b2398b827c9`
- `src/shaders.wgsl`: `feced2d27c5ed6370fb3b6b091c8576d18f7a6660b016295716f8eee8f29e575`
- `src/radix_sort.cpp`: `57afed68836c36bdb12ad542293b46815e6db4c015df29766e11bfa1285f75bd`
- `experiments/pmpp_radix/local_sort.wgsl`: `0e5d61cae1f648dfd06e23a1e6f2e2ab7b0a4f2ef2210ac32ec0b36c5268594a`
- `experiments/pmpp_radix/benchmark.cpp`: `9d062180b3823f3f5662c3677ed9387f4dbb8f277c57c9dc13dad300341ea1d2`
