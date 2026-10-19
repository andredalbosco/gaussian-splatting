# Pixel-space versus clip-space Gaussian splatting

## Result

**No meaningful performance advantage emerged from changing coordinate units.** On the local RTX 4090 with Dawn/Vulkan, one-million-Gaussian full rendering medians differed by less than 0.2%, and the off-screen vertex diagnostic differed by less than 0.03%. Smaller workloads had somewhat larger variation, with overlapping per-trial results. These observations do not establish exact equivalence on every GPU or camera/workload.

Keep the pixel-space covariance for clarity: focal lengths, covariance, the 0.3 filter, and quad bounds retain direct pixel meanings. Emitting homogeneous clip positions is also reasonable, but the measured speed difference does not justify a change by itself. The production renderer was not changed by this experiment.

## Complete rendering

GPU pass durations in **microseconds**, lower is better. Target: 1920×1080 RGBA8Unorm; four vertices/Gaussian, back-to-front sorted inputs, common premultiplied blending and Gaussian fragment shader.

| Formulation | 10k small | 10k large | 100k small | 1M small |
|---|---:|---:|---:|---:|
| Pixel covariance → NDC output | 14.544 | 52.112 | 66.048 | 580.448 |
| Pixel covariance → clip output | 14.464 | 51.104 | 65.632 | 580.192 |
| NDC covariance → clip output | 14.464 | 52.080 | 67.376 | 580.800 |
| Clip covariance → clip output | 14.592 | 51.232 | 66.240 | 580.704 |

## Vertex diagnostic

The vertex source calculates covariance, conic, and quad position; the shader then shifts clip x outside the clip volume using `abs(x) + 2*w`, so no fragments are shaded. This includes vertex, primitive, clipping, and pass clear/store work. It is not an isolated instruction-throughput measurement. Compilers may eliminate outputs that no longer affect the diagnostic position; therefore treat this as supporting evidence rather than a replacement for the normal rendering results.

| Formulation | 10k | 100k | 1M |
|---|---:|---:|---:|
| Pixel covariance → NDC output | 7.376 | 54.464 | 517.024 |
| Pixel covariance → clip output | 7.456 | 54.416 | 517.024 |
| NDC covariance → clip output | 7.488 | 54.896 | 516.928 |
| Clip covariance → clip output | 7.456 | 54.400 | 516.896 |

## Exactly what differs

Let D = diag(2/width, 2/height), Cpx be pixel covariance, and w be the Gaussian center's clip w.

| Method | Covariance and fragment-offset units | Vertex position |
|---|---|---|
| Pixel → NDC | Pixels | `(clip.xy/w + D*offset_px, clip.z/w, 1)` |
| Pixel → clip | Pixels | `(clip.xy + w*D*offset_px, clip.z, w)` |
| NDC → clip | NDC; Cndc = D Cpx D | `(clip.xy + w*offset_ndc, clip.z, w)` |
| Clip → clip | Center-w-scaled NDC; Cclip = w² D Cpx D | `(clip.xy + offset_clip, clip.z, w)` |

All methods use the explicit perspective Jacobian and `B = J * W * R * S`, constructed from normalized quaternion/scale. The pixel variants use pixel focal lengths. The NDC and clip variants receive the projection diagonal directly in the uniform, so they do not pay for reconstructing it from pixel focal lengths in each vertex.

For the clip-covariance variant, Jclip = w Jndc; with our positive-Z perspective camera, w = view_z, so its first two Jacobian columns do not contain a 1/z scale. Its depth derivative still contains view_xy/view_z. This is the footprint on the quad's constant-w plane, **not** the covariance of raw homogeneous xy with the perspective depth derivative omitted.

The 0.3 pixel² blur is transformed consistently:

- Pixel covariance: add diag(0.3, 0.3).
- NDC covariance: add diag(0.3 * (2/viewport)²).
- Clip covariance: add diag(0.3 * (2*w/viewport)²).

Each fragment receives offset and inverse covariance in the same units, keeping `q = offsetᵀ * inverse_covariance * offset` invariant. The fragment shader source is identical for all variants. All corners of one quad have the same w, so homogeneous output preserves the intended within-quad interpolation mathematically.

## Correctness

- 10,000 ordinary plus 10,000 off-axis anisotropic Gaussians per method, rotated/translated camera, double-precision CPU covariance reference.
- GPU covariance converted back into pixel units for comparison. Scaled tolerance: 2e-5, denominator 1 + sqrt(Cxx*Cyy).
- Covariance failures: 0; maximum scaled error: 2.96e-06.
- Rendered 1080p image comparisons against the pixel/NDC baseline using 10,000 larger splats: maximum channel difference 2/255; channels exceeding 2/255: 0.
- Numerical and image-readback checks only; no browser/manual visual validation is claimed.

## Measurement contract

Each table cell is the pooled median of **448 samples: seven shuffled trials × 64 samples**, with eight warmup passes per batch. Raw timings, p10/p90, and every trial median are in `samples.csv` and `summary.csv`. Pass timestamps include attachment clear/store, projection, rasterization and, for normal rendering, fragments/blending. They exclude CPU generation/sorting, upload, command recording, readback, presentation, UI, and compute preprocessing. All timed paths are vertex shaders.

Seed 20261019; uniformly random orientations; camera-space depths 2–20; centers inside 85% of the frustum half-extents. Scales are generated relative to z/focal: 0.4–2.4 pixel-axis standard deviations for small splats and 1.6–9.6 for large splats before rotation/filtering. Identical 3-sigma support, 0.99 alpha cap, and 1/255 alpha rejection.

Source Gaussian input is 64 bytes per Gaussian. The same uniform layout and bind groups are used throughout. Data is warm between repeated draws; this is not a cold-cache or interactive-frame benchmark. Timestamp quantization is disabled. Other existing GPU applications remain active; `gpu-load.csv` records GPU memory, clocks, load, temperature, and power. Background activity and clock variation limit interpretation of small differences. No shader ISA or hardware instruction counts were collected.

One million Gaussians fit in this run: approximately 549 MiB was free before the run, versus only 109 MiB during the earlier projection-algorithm experiment. The source buffer also reuses the unused packed-input slot rather than allocating a second Gaussian copy.

The first measurement, before supplying projection diagonals directly, is retained under `derived-scales-*`; it also found near-identical timings. The final tables use only the complete direct-uniform run. Predictions were recorded before either run in `PREDICTIONS.md`.

## Reproduction

```bash
cmake --build build --target space_benchmark -j 4
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/space_benchmark experiments/results/2026-10-19-spaces
python3 experiments/analyze_spaces.py experiments/results/2026-10-19-spaces
```

Native Clang, -O3, CMake + Ninja. Pass `--validate-only` to run correctness checks without timing, or `--math-only` to time only the off-screen diagnostics after correctness checks. Source files: `experiments/space_benchmark.cpp` and `experiments/space_benchmark.wgsl`.

Projection and timing semantics were checked against [WebGPU Fundamentals: perspective projection](https://webgpufundamentals.org/webgpu/lessons/webgpu-perspective-projection.html) and [GPU timing](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html).
