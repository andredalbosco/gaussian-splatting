# Gaussian projection benchmark

Measured on the local RTX 4090 through Dawn/Vulkan. Units below are **microseconds**, lower is better. This compares WGSL projection algorithms in a common harness, not complete upstream engines.

## Result

There is **no compelling projection-math winner** in this workload. At 10,000 Gaussians the normal vertex-rendering medians cluster near 14 us. Compute-only projection medians are all 6.144 us; at 100,000 they cluster near 11–12 us. The arithmetic reductions did not translate into a robust large improvement.

**Keep projection in the vertex path for this quad renderer.** Computing covariance once and writing it to an intermediate buffer was slower at both measured counts. For the current formula, 10k small-splat rendering was 13.78 us directly versus 17.41 us with compute preparation (26.4% slower). At 100k it was 65.73 versus 75.78 us (15.3% slower).

This does **not** answer whether a tiled renderer that already needs a projected buffer should use compute. Here the extra pass exists solely to avoid repeating covariance projection across quad vertices. Nor does it establish behavior at millions of Gaussians.

## Full rendering

These times include attachment clearing, projection or preparation plus its dependency, quad rasterization, Gaussian fragment evaluation, blending, and attachment storage. They exclude presentation, UI, sorting, CPU preparation, uploads, command recording, and readback. Render target: 1920 x 1080 RGBA8Unorm, four-vertex triangle-strip instances, premultiplied alpha, no depth attachment.

| Method | 10k small, vertex | 10k small, compute + render | 10k large, vertex | 10k large, compute + render | 100k small, vertex | 100k small, compute + render |
|---|---:|---:|---:|---:|---:|---:|
| Current projected axes | 13.78 | 17.41 | 48.43 | 52.22 | 65.73 | 75.78 |
| Shared Jacobian + rotation matrix | 13.78 | 17.41 | 48.80 | 52.22 | 66.66 | 76.80 |
| Shared Jacobian + inverse quaternion | 13.87 | 17.41 | 48.66 | 51.71 | 65.76 | 75.78 |
| Sparse camera-space axes | 14.24 | 17.41 | 50.75 | 52.22 | 66.37 | 76.80 |
| gsplat-style, constructed covariance | 13.87 | 17.41 | 50.61 | 52.22 | 70.90 | 78.85 |
| gsplat-style, precomputed covariance | 14.08 | 17.41 | 49.58 | 52.22 | 66.70 | 76.80 |
| PlayCanvas-style, constructed covariance | 13.74 | 18.43 | 48.51 | 52.22 | 68.24 | 75.78 |
| PlayCanvas-style, precomputed covariance | 13.79 | 17.41 | 49.78 | 53.76 | 68.59 | 74.75 |
| Symmetric quadratic form, constructed | 13.63 | 17.41 | 50.48 | 54.27 | 66.34 | 77.82 |
| Symmetric quadratic form, precomputed | 13.89 | 21.50 | 48.61 | 52.22 | 66.26 | 76.80 |
| Sparse camera covariance, precomputed | 13.70 | 17.41 | 48.38 | 57.34 | 66.40 | 76.80 |

The small arithmetic differences are less than, or comparable to, interference between trials. Do not choose a formula based on tenths of a microsecond here. Precomputed covariance did not establish a repeatable rendering advantage.

## Projection-only compute and vertex diagnostic

Compute-only runs perform one projection per Gaussian and write the same 48-byte record (clip center, covariance, color). They include dispatch and storage costs. The vertex diagnostic uses four vertices per Gaussian and covariance-dependent positions outside the clip volume; it executes no fragments but still includes vertex processing, primitive processing, clipping, and common pass clear/store overhead. It is not a pure FLOP or shader instruction benchmark.

| Method | 10k compute only | 100k compute only | 10k off-screen vertex | 100k off-screen vertex |
|---|---:|---:|---:|---:|
| Current projected axes | 6.14 | 12.29 | 7.46 | 55.57 |
| Shared Jacobian + rotation matrix | 6.14 | 12.29 | 7.66 | 54.50 |
| Shared Jacobian + inverse quaternion | 6.14 | 11.26 | 7.54 | 56.70 |
| Sparse camera-space axes | 6.14 | 11.26 | 7.49 | 55.84 |
| gsplat-style, constructed covariance | 6.14 | 12.29 | 7.71 | 56.00 |
| gsplat-style, precomputed covariance | 6.14 | 11.26 | 7.54 | 54.61 |
| PlayCanvas-style, constructed covariance | 6.14 | 11.26 | 7.58 | 56.72 |
| PlayCanvas-style, precomputed covariance | 6.14 | 11.26 | 7.68 | 55.71 |
| Symmetric quadratic form, constructed | 6.14 | 11.78 | 7.46 | 58.10 |
| Symmetric quadratic form, precomputed | 6.14 | 11.26 | 7.46 | 54.46 |
| Sparse camera covariance, precomputed | 6.14 | 12.29 | 7.65 | 54.14 |

Shared rows with an explicit rotation matrix and precomputed symmetric covariance can slightly improve the 100k vertex diagnostic, but the results do not establish a consistent normal-rendering improvement. The inverse-quaternion row version did not beat the baseline reliably. Explicit source-level matrix expansion is not evidence of fewer executed GPU instructions: no ISA or hardware instruction counters were collected.

## Trial variation

Each cell above is the median of 96 samples: three shuffled trials of 32 samples, with eight warmup operations per batch. Below are the individual trial medians for the normal 100k vertex-rendering workload.

| Method | Trial 1 | Trial 2 | Trial 3 |
|---|---:|---:|---:|
| Current projected axes | 66.51 | 65.87 | 65.57 |
| Shared Jacobian + rotation matrix | 65.63 | 69.50 | 69.04 |
| Shared Jacobian + inverse quaternion | 65.60 | 65.47 | 68.66 |
| Sparse camera-space axes | 65.57 | 66.75 | 67.47 |
| gsplat-style, constructed covariance | 76.10 | 71.92 | 65.82 |
| gsplat-style, precomputed covariance | 66.78 | 65.81 | 73.36 |
| PlayCanvas-style, constructed covariance | 65.71 | 69.65 | 75.01 |
| PlayCanvas-style, precomputed covariance | 65.57 | 74.42 | 68.59 |
| Symmetric quadratic form, constructed | 68.14 | 66.19 | 66.03 |
| Symmetric quadratic form, precomputed | 77.57 | 65.95 | 65.74 |
| Sparse camera covariance, precomputed | 66.54 | 68.58 | 66.05 |

Additional trial medians, p10, and p90 are in `summary.csv`; every raw timestamp duration is in `samples.csv`. Timestamps use GPU pass boundaries. Compute + render starts at the beginning of the compute pass and ends at the end of the render pass, so the inter-pass dependency is included. Dawn's `timestamp_quantization` toggle is disabled. Nevertheless compute-boundary measurements in this environment frequently occur in 1.024 us increments; the observed timing granularity limits sub-microsecond comparisons. Native vertex-rendering measurements have finer increments.

The GPU was shared with existing graphics and inference applications. Before the first run it showed roughly 28–46% utilization. See `gpu-load.csv` for sampled clocks/load during the final run. Large scheduling outliers are retained. These are warm repeated batches; their small working sets can benefit from cache, so these results do not establish a DRAM-bandwidth bottleneck or single interactive-frame latency. No unrelated applications were stopped.

## Compared algorithms

| Variant | Work performed per invocation |
|---|---|
| Current projected axes | Rotate three scaled axes with quaternion cross products; apply full view-projection quotient derivative to each; form three covariance entries. |
| Shared rows + matrix | Construct world-to-pixel Jacobian rows once; multiply both by quaternion-derived rotation matrix; apply scale; take three dot products. |
| Shared rows + quaternion | Construct the same rows; rotate each by the inverse quaternion; apply scale; take three dot products. |
| Sparse camera axes | Construct camera-space rotated/scaled axes; exploit sparse perspective derivatives; move focal scaling after covariance dot products. |
| gsplat-style | Construct or load world covariance; transform covariance into camera space; rectangular J C J-transpose; retain gsplat's centered-camera off-axis clamp. |
| PlayCanvas-style | Construct six covariance entries with row dot products or load them; form combined camera/Jacobian transform T; compute T-transpose C T. |
| Symmetric form | Construct or load world covariance; combine camera and Jacobian; evaluate only the three symmetric quadratic-form outputs. |
| Sparse covariance | Load world covariance; transform to camera space; explicitly expand sparse perspective projection with shared xz/yz terms. |

The source shader contains separate constant-specialized functions/branches, and each variant has its own pipeline. The camera is rotated and translated: an identity view is not used to make camera-space variants artificially cheap.

**gsplat adaptation:** its covariance construction, camera-space covariance transform, rectangular perspective Jacobian, and clamp are ported to WGSL. Input quaternions are normalized once, as in our loader, so its per-call quaternion renormalization is omitted. Timing scenes lie within the frustum and its clamp is inactive. The off-axis validation tests the clamp separately. Principal point is centered, making the clamp 1.3 times the half-FOV tangent. This is not CUDA/PyTorch performance.

**PlayCanvas adaptation:** its six-entry covariance and combined-transform projection are used with FP32 data and our camera convention. Half precision, source decompression, eigenvector-oriented quads, minimum pixel-size culling, optional AA opacity compensation, and other camera models are excluded to compare the projection expressions under the same render workload. The shared renderer uses our axis-aligned 3-sigma quads. This is not full PlayCanvas performance.

**Precomputed variants:** CPU scene preparation replaces scale/quaternion with six world-space covariance values in two vec4 fields. Both input formats occupy 64 bytes/Gaussian, including center and color. No extra covariance buffer is fetched. Construction/upload happens outside the timed region, appropriate for a static scene. For deforming Gaussians that cost must be included. An intermediate projection buffer holds 48 bytes/Gaussian; compute writes it, then vertices read it. Conic inversion and quad extent calculation remain in the common vertex stage, so a more compact or fully prepared intermediate is a different experiment.

## Data and correctness

Random seed: 20261019. Exactly 10,000 or 100,000 anisotropic Gaussians, uniformly random orientations, camera-space depths 2–20, centers within 85% of the horizontal/vertical frustum extent. Each camera-axis scale is drawn relative to z/focal: 0.4–2.4 pixels for the small workload and 1.6–9.6 pixels for the large workload, before orientation effects and filtering. Inputs are CPU sorted back-to-front before uploading. All methods use identical +0.3 diagonal filtering, 3-sigma support, alpha cap 0.99, and 1/255 alpha rejection.

- Double-precision CPU reference: 10,000 ordinary plus 10,000 off-axis Gaussians per method, comparing all three covariance entries. Scaled tolerance: 2e-5, denominator 1 + sqrt(Cxx Cyy).
- Failures: 0. Maximum scaled error: 3.65e-06. Includes camera rotation, translation, anisotropy, and gsplat clamping.
- Rendered image comparison: all eleven direct and prepared paths at 1080p, 10,000 larger splats. Maximum channel difference: 2/255; channels above 2/255: 0. Comparison is against the current projection formula with the shared rasterizer, rather than against each upstream engine's different quad/AA conventions.
- These are numerical/image readback checks; no interactive browser or manual visual inspection was performed for this experiment.

## VRAM limitation and retained runs

The attempted million-Gaussian workload failed with `VK_ERROR_OUT_OF_DEVICE_MEMORY`. A contemporaneous `nvidia-smi` query showed 23,970 MiB used and only 109 MiB free. Source + precomputed source + projected output would require about 176 MB before allocator overhead. This is an environment limitation, not a measured performance result. Its completed 10k/100k projection samples and failure log are retained as `failed-1m-*`. The first successful timing pass, before adding the off-screen diagnostic, is retained as `first-pass-*`; its normal rendering results likewise showed no large formula advantage and slower compute preparation. Main tables use only the final complete run.

## Reproduce

```bash
cmake --build build --target projection_benchmark -j 4
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/projection_benchmark experiments/results/2026-10-19-projection
python3 experiments/analyze_projection.py experiments/results/2026-10-19-projection
```

The target uses the repository's Clang/Release/-O3 CMake + Ninja build. `--validate-only` skips timing; `--math-only` runs the off-screen diagnostics after correctness checks. Benchmark files live under `experiments/`; the app and tiled renderer are unchanged. Predictions were written before timing in `PREDICTIONS.md`.

## Pinned primary sources

- [gsplat, commit 512d366b6707](https://raw.githubusercontent.com/nerfstudio-project/gsplat/512d366b67073d77ca099ede742683c165dfc23b/gsplat/cuda/include/Utils.cuh); local snapshot in `sources/`. SHA-256: `f481e9efa784de4cd49cba153d14d4a6337733cc72b207d6e78fc024fec4f00f`.
- [playcanvas, commit 40854828f51d](https://raw.githubusercontent.com/playcanvas/engine/40854828f51deaf53221093403114e9490e8da04/src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatCorner.js); local snapshot in `sources/`. SHA-256: `a2e3189af7800fe8eb675f3219b1325941a9d3e17857285582e90059fdd66995`.
- [WebGPU Fundamentals: GPU timing](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html).
