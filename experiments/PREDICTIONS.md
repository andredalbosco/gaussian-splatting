# Ellipsoid benchmark predictions — recorded before measurements

2026-10-18. No timings from this experiment have been run when writing this file.
These are source-level operation estimates and falsifiable predictions, not GPU instruction counts.
A scalar add/subtract/multiply counts as one FLOP; a fused multiply-add counts as two.
Division, sqrt, reciprocal sqrt, comparisons, interpolation, memory traffic, and raster/depth work
are recorded separately rather than treated as equally priced FLOPs.

## Combinations

Four coverage proxies: oriented box (36 vertices), projected bounds of its eight corners
(4 vertices), exact perspective ellipsoid bounds (4 vertices), and a Golus bounding-sphere
billboard (4 vertices). Each is crossed with vertex/fragment object-space ray transforms
and general/normalized local-ray intersection: 16 exact-surface combinations.
Four additional box variants enable front-face culling, matching the main renderer,
with the same two transform stages and two intersections (20 exact variants total).
Two extra controls use the existing projected ellipse and Gaussian opacity; their images
and depth behavior differ, so they are not substitutes in the exact-surface ranking.

The Golus variant adapts his bounds and object-space-ray idea. It does not reproduce Unity
textures, shadows, orthographic rendering, or conservative-depth semantics.

## Arithmetic model

Quaternion-vector rotation: 30 ordinary scalar operations (two 9-op cross products,
three scalar-vector products/additions). A general mat4-vector product: 28 ordinary operations.
Transforming both ray origin and direction into ellipsoid coordinates: 63 ordinary operations
plus six scalar divisions. This is moved between stages without changing the surface.

Approximate coverage construction cost per vertex, before object-space ray transforms:

| Proxy | Vertex count | Ordinary operations per vertex | Important extra operations |
|---|---:|---:|---|
| Oriented box | 36 | ~67 | Rasterizes box faces; no face culling in the initial controlled comparison |
| Eight-corner projected box quad | 4 | ~550–650 | 16 perspective divisions, min/max reductions; camera-plane fallback |
| Exact ellipsoid quad | 4 | ~400–450 | Bounds divisions, two square roots, inverse projection; camera-plane fallback |
| Golus sphere billboard | 4 | ~90–130 | Normalizations, perspective expansion sqrt/division; inside-sphere fallback |

These are approximate scalarized expressions: the compiler removes zero matrix terms,
folds constants, shares reciprocals, and may eliminate unused varyings. Bounds work is
repeated per vertex in this implementation; no compute prepass is included.

For V vertices, F fragment invocations, and H hits, the relevant model is:

    cost ≈ V × bounds_cost + ray_transform_cost × (V or F)
           + F × intersection_cost + H × shading_and_depth_cost
           + rasterization/interpolation/memory/depth costs.

Moving transforms to vertices replaces about F × (63 ordinary + 6 divisions) with
V × that cost, at the expense of interpolated local rays. For the box V=36, for quads V=4.
The general intersection computes dot(d,d) and divides by it twice; normalization instead
adds a reciprocal sqrt and scaling and can remove those divisions. Both retain the same
closest-point formulation to avoid conflating normalization with quadratic cancellation.
Hit reconstruction accounts for any changed ray parameter. Common shading/depth work stays fixed.

## Predictions

1. Vertex-transformed rays should win for large footprints and overlap. For very tiny
   primitives, their added vertex work/interpolants can offset the saved fragment arithmetic.
2. Golus should have the lowest quad setup cost away from its fallback. It should be
   competitive for spheres and small footprints, where its enclosure is tight or fill is cheap.
3. Exact ellipsoid bounds should overtake Golus as anisotropy and projected size increase:
   extra vertex arithmetic buys fewer fragment invocations, misses, and ray transforms.
   At 4× pixel count, this benefit should generally increase.
4. The eight-corner box quad should usually lose to exact ellipsoid bounds: it repeats eight
   corner projections per vertex and encloses extra empty pixels. It is a simple conservative control.
5. The box should have high geometry cost (36 vertices versus 4) and potentially duplicate
   fragment work with culling disabled. It may remain competitive for extremely elongated shapes
   because its projected faces can cover less area than an axis-aligned rectangle. No universal loser predicted.
6. Normalizing the local ray has no confident winner: reciprocal sqrt/scaling compete with
   divisions the compiler may already lower to a shared reciprocal. Expect a smaller effect
   than coverage or transform placement; test both rather than assuming fewer written divisions wins.
7. A hard projected ellipse should have low fragment cost (a 2D quadratic plus cutoff) and
   no surface-depth reconstruction. It should be faster when fill-bound, but is a different image.
8. The Gaussian evaluates exp and blends over a 3-sigma footprint: approximately 9× the
   1-sigma area before clipping. It can cost more than the hard ellipse despite sharing its
   projected covariance. Multi-object opacity rendering has no sorting in this experiment,
   so those timings measure only the supplied draw order, not a complete 3DGS renderer.
9. Single-object passes may be dominated by attachment clear/store and launch overhead.
   Differences there may be too small to rank reliably; many-object and close-up cases matter.
10. General/normalized and vertex/fragment variants should agree within small floating-point
    differences. Exact proxies should agree in tested outside-camera views. Large discrepancies
    invalidate an apparent speedup and must be investigated before ranking that variant.

11. Front-face culling on the box should reduce duplicate face rasterization and fragment
    work without changing its 36 vertex invocations. It should improve the box especially
    at large screen footprints; both culling modes must be reported to avoid penalizing
    the box solely for a pipeline-state choice.

## Planned measurement

Use the local RTX 4090 through Dawn/Vulkan. Record adapter/backend and driver/load context.
Test single spheres, anisotropic ellipsoids (framed, close, and off-axis), a many-object grid,
and Luigi (framed and close), at 1280×720 and 2560×1440. Same camera, source data,
opaque shading and depth states for all exact methods; no UI, wireframes, presentation,
asset loading, uploads, sorting, command encoding, or pipeline compilation in the timed pass.
Measure warmed GPU pass timestamps and a separate uninstrumented submitted batch. Include
an empty-pass baseline. Repeat with shuffled variant order and retain raw samples/CSV.
Compare RGB and depth with exact-quad/vertex/general reference; report differences rather
than demand bit equality. Retain representative rendered images for manual inspection.

Timing reference: https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html
