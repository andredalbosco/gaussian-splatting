# Golus comparison

## Apple M1 Pro · Dawn/Metal · 2026-10-18

Two isolated runs, with the live viewer paused. Numbers below are uncapped offscreen batch wall time per pass in milliseconds; lower is better. They exclude ImGui, wireframes, presentation, camera uploads, and command encoding.

| Scene | Resolution | Exact | Golus | Exact + object rays |
|---|---|---:|---:|---:|
| Luigi framed | 1280×720 | 0.738 | 0.625 | 0.469 |
| Luigi close | 1280×720 | 1.809 | 1.499 | 1.200 |
| Luigi minimum distance | 1280×720 | 5.025 | 4.731 | 3.031 |
| Test ellipsoid | 1280×720 | 0.049 | 0.051 | 0.048 |
| Test sphere | 1280×720 | 0.051 | 0.052 | 0.050 |
| Luigi framed | 2560×1440 | 1.667 | 1.450 | 1.044 |
| Luigi close | 2560×1440 | 5.033 | 4.139 | 2.962 |
| Luigi minimum distance | 2560×1440 | 17.825 | 15.437 | 11.176 |
| Test ellipsoid | 2560×1440 | 0.081 | 0.097 | 0.082 |
| Test sphere | 2560×1440 | 0.104 | 0.102 | 0.098 |

Each call warms up with 32 passes, records 128 timestamp-instrumented passes, and separately measures an uninstrumented batch of 128 passes. Both methods use the same ellipsoid frustum culling, input data, opaque shading, and depth settings. The control variant keeps exact bounds but moves ray transforms to the vertex stage, as in Golus mode.

The Golus adaptation uses a camera-facing square sized to the largest semi-axis and interpolates object-space rays. Exact bounds are tighter for anisotropic ellipsoids. The control shows that vertex-stage ray transforms help Luigi without giving up those tighter bounds.

Compared with the exact method, fewer than 0.017% of pixels differ by more than 5/255 per RGB channel in the tested views; most control/test-object views match exactly. This is a numeric image comparison, not an exhaustive correctness proof.

GPU timestamp medians and p90 values are also emitted, but differed sharply from uninstrumented throughput on this Metal run. The table uses batch wall time. These are rendering measurements, not interactive FPS or a universal ranking. See [WebGPU Fundamentals on timing](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html).

Reproduce with `./build/gaussian_splatting --benchmark`. The benchmark prints adapter metadata and CSV rows. The table preserves the reported Mac measurements; the old `build-mac-reference/` directory has been removed.

## Sources and adaptation

[Ben Golus: Rendering a Sphere on a Quad](https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c) and [the complete Unity shader](https://gist.github.com/bgolus/1188cd89968b977d5c468bf7bbb3250b). This comparison adapts the perspective billboard and object-space rays. It does not implement the article’s textures, shadows, orthographic projection, or conservative-depth semantics. A camera-inside-bounding-sphere fallback is added.
