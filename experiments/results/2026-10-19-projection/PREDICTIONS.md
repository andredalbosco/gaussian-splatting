# Predictions recorded before timing

1. Sharing the quotient-rule Jacobian across two rows should reduce arithmetic relative to rotating three axes and projecting each separately. Quaternion inverse rotation may be competitive with an explicit rotation matrix; the compiler determines whether common terms survive.
2. Covariance-first construction should cost more than the factored route when every invocation starts from quaternion/scale. This prediction concerns the complete path, not the previously quoted projection-only arithmetic counts.
3. Precomputing six world covariance values at scene load should make the covariance paths competitive for static Gaussians. This excludes one-time CPU construction/upload and changes the input representation; it is not a free per-frame optimization for deforming data.
4. The handwritten symmetric forms should reduce source-level arithmetic, but may match matrix expressions after compiler dead-code elimination and FMA contraction.
5. Projection-only dispatches at 10,000 Gaussians are likely to be dominated by dispatch, memory, and scheduling; differences should be clearer at 100,000 and 1,000,000.
6. Compute preprocessing plus rendering adds a dispatch, a 48-byte/Gaussian intermediate write, a dependency, and vertex reads. It may lose at 10,000 despite avoiding repeated projection across four vertices; it may win at larger counts.
7. Larger splats should hide projection improvements behind fragment evaluation and blending.

All methods use FP32, normalized input quaternions, identical 0.3 diagonal filtering, identical axis-aligned 3-sigma quads, common premultiplied blending, and sorted inputs. gsplat's clamp is retained but inactive in the timing scenes; off-axis correctness is checked separately with a clamped reference. PlayCanvas's half source data, eigenvector quad, optional AA compensation, and full engine are not timed. gsplat is ported from CUDA to WGSL; its full rasterizer is not timed.

The GPU was busy before testing (about 46% utilization). Do not interpret small timing differences as decisive. Trials are shuffled and per-trial medians/ranges will be reported.
