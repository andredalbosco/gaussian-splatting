#!/usr/bin/env python3
"""Summarize the raw GPU samples without requiring third-party packages."""
import collections
import csv
import hashlib
import json
import pathlib
import statistics
import sys

root = pathlib.Path(__file__).resolve().parent.parent
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'experiments/results/2026-10-19-projection'
methods = [
    ('baseline_axes', 'Current projected axes'),
    ('shared_rows_matrix', 'Shared Jacobian + rotation matrix'),
    ('shared_rows_quaternion', 'Shared Jacobian + inverse quaternion'),
    ('sparse_camera_axes', 'Sparse camera-space axes'),
    ('gsplat_constructed', 'gsplat-style, constructed covariance'),
    ('gsplat_precomputed', 'gsplat-style, precomputed covariance'),
    ('playcanvas_constructed', 'PlayCanvas-style, constructed covariance'),
    ('playcanvas_precomputed', 'PlayCanvas-style, precomputed covariance'),
    ('symmetric_constructed', 'Symmetric quadratic form, constructed'),
    ('symmetric_precomputed', 'Symmetric quadratic form, precomputed'),
    ('sparse_covariance_precomputed', 'Sparse camera covariance, precomputed'),
]
groups = collections.defaultdict(list)
trials = collections.defaultdict(list)
for row in csv.DictReader((out / 'samples.csv').open()):
    key = row['workload'], row['method'], row['path']
    value = float(row['gpu_us'])
    groups[key].append(value)
    trials[key + (int(row['trial']),)].append(value)

def median(work, method, path):
    return statistics.median(groups[work, method, path])

with (out / 'summary.csv').open('w') as file:
    writer = csv.writer(file)
    writer.writerow(['workload', 'method', 'path', 'samples', 'median_us', 'p10_us', 'p90_us', 'trial0_median_us', 'trial1_median_us', 'trial2_median_us'])
    for key, values in groups.items():
        values = sorted(values)
        writer.writerow([*key, len(values), statistics.median(values), values[len(values)//10], values[len(values)*9//10],
                         *(statistics.median(trials[key+(i,)]) for i in range(3))])

def table(columns):
    lines = ['| Method | ' + ' | '.join(c[0] for c in columns) + ' |', '|---|' + '---:|'*len(columns)]
    for method, label in methods:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{median(work,method,path):.2f}' for _,work,path in columns) + ' |')
    return '\n'.join(lines)

def ranges(work, path):
    lines = ['| Method | Trial 1 | Trial 2 | Trial 3 |', '|---|---:|---:|---:|']
    for method, label in methods:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{statistics.median(trials[work,method,path,i]):.2f}' for i in range(3)) + ' |')
    return '\n'.join(lines)

validation = list(csv.DictReader((out/'validation.csv').open()))
images = list(csv.DictReader((out/'images.csv').open()))
provenance = json.loads((out/'sources/provenance.json').read_text())
report = f'''# Gaussian projection benchmark

Measured on the local RTX 4090 through Dawn/Vulkan. Units below are **microseconds**, lower is better. This compares WGSL projection algorithms in a common harness, not complete upstream engines.

## Result

There is **no compelling projection-math winner** in this workload. At 10,000 Gaussians the normal vertex-rendering medians cluster near 14 us. Compute-only projection medians are all 6.144 us; at 100,000 they cluster near 11–12 us. The arithmetic reductions did not translate into a robust large improvement.

**Keep projection in the vertex path for this quad renderer.** Computing covariance once and writing it to an intermediate buffer was slower at both measured counts. For the current formula, 10k small-splat rendering was {median('render_10k_small','baseline_axes','vertex_render'):.2f} us directly versus {median('render_10k_small','baseline_axes','compute_render'):.2f} us with compute preparation ({100*(median('render_10k_small','baseline_axes','compute_render')/median('render_10k_small','baseline_axes','vertex_render')-1):.1f}% slower). At 100k it was {median('render_100k_small','baseline_axes','vertex_render'):.2f} versus {median('render_100k_small','baseline_axes','compute_render'):.2f} us ({100*(median('render_100k_small','baseline_axes','compute_render')/median('render_100k_small','baseline_axes','vertex_render')-1):.1f}% slower).

This does **not** answer whether a tiled renderer that already needs a projected buffer should use compute. Here the extra pass exists solely to avoid repeating covariance projection across quad vertices. Nor does it establish behavior at millions of Gaussians.

## Full rendering

These times include attachment clearing, projection or preparation plus its dependency, quad rasterization, Gaussian fragment evaluation, blending, and attachment storage. They exclude presentation, UI, sorting, CPU preparation, uploads, command recording, and readback. Render target: 1920 x 1080 RGBA8Unorm, four-vertex triangle-strip instances, premultiplied alpha, no depth attachment.

{table([('10k small, vertex','render_10k_small','vertex_render'),('10k small, compute + render','render_10k_small','compute_render'),('10k large, vertex','render_10k_large','vertex_render'),('10k large, compute + render','render_10k_large','compute_render'),('100k small, vertex','render_100k_small','vertex_render'),('100k small, compute + render','render_100k_small','compute_render')])}

The small arithmetic differences are less than, or comparable to, interference between trials. Do not choose a formula based on tenths of a microsecond here. Precomputed covariance did not establish a repeatable rendering advantage.

## Projection-only compute and vertex diagnostic

Compute-only runs perform one projection per Gaussian and write the same 48-byte record (clip center, covariance, color). They include dispatch and storage costs. The vertex diagnostic uses four vertices per Gaussian and covariance-dependent positions outside the clip volume; it executes no fragments but still includes vertex processing, primitive processing, clipping, and common pass clear/store overhead. It is not a pure FLOP or shader instruction benchmark.

{table([('10k compute only','projection_10k','compute_only'),('100k compute only','projection_100k','compute_only'),('10k off-screen vertex','vertex_math_10k','vertex_render'),('100k off-screen vertex','vertex_math_100k','vertex_render')])}

Shared rows with an explicit rotation matrix and precomputed symmetric covariance can slightly improve the 100k vertex diagnostic, but the results do not establish a consistent normal-rendering improvement. The inverse-quaternion row version did not beat the baseline reliably. Explicit source-level matrix expansion is not evidence of fewer executed GPU instructions: no ISA or hardware instruction counters were collected.

## Trial variation

Each cell above is the median of 96 samples: three shuffled trials of 32 samples, with eight warmup operations per batch. Below are the individual trial medians for the normal 100k vertex-rendering workload.

{ranges('render_100k_small','vertex_render')}

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
- Failures: {sum(int(r['failures']) for r in validation)}. Maximum scaled error: {max(float(r['max_scaled_error']) for r in validation):.3g}. Includes camera rotation, translation, anisotropy, and gsplat clamping.
- Rendered image comparison: all eleven direct and prepared paths at 1080p, 10,000 larger splats. Maximum channel difference: {max(int(r['max_channel_error']) for r in images)}/255; channels above 2/255: {sum(int(r['channels_over_2']) for r in images)}. Comparison is against the current projection formula with the shared rasterizer, rather than against each upstream engine's different quad/AA conventions.
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

'''
for source in provenance:
    report += f"- [{source['name']}, commit {source['commit'][:12]}]({source['url']}); local snapshot in `sources/`. SHA-256: `{source['sha256']}`.\n"
report += '- [WebGPU Fundamentals: GPU timing](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html).\n'
(out/'REPORT.md').write_text(report)
files=['experiments/projection_benchmark.cpp','experiments/projection_benchmark.wgsl','experiments/analyze_projection.py']
(out/'benchmark-sha256.json').write_text(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},indent=2)+'\n')
print(report)
