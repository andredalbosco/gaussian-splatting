# 10,000 random Gaussians — predictions before timing

2026-10-18. Keep the original predictions and results unchanged.
The source-level arithmetic model is in PREDICTIONS.md; only the scene changes here.

Use seed 10000, 10,000 centers uniform in [-2,2]^3; independent log-uniform semi-axes
in [0.01,0.12]; uniform rotations from normalized four-dimensional isotropic Gaussian
samples; independent RGB in [0.2,0.95] and opacity in [0.1,0.9]. Render the same dataset
from camera distances 7.5 and 3.8, centered at the origin, at both existing resolutions.

- Each box pass submits 360,000 vertices, each quad pass 40,000. Per-vertex coverage
  and ray work is otherwise unchanged from the original FLOP estimates.
- The 3D volume creates irregular overlap and depth occlusion, unlike the planar grid.
  Coverage/overdraw should matter more in the close view and at higher resolution.
- Exact bounds should beat the looser enclosing-sphere Golus bounds on this anisotropic
  mixture, especially close up; the winner for small distant footprints is less certain.
- Front-face culling should again improve the box. Predict a substantial cost for
  two-sided boxes but no universal ordering between culled boxes and loose quads.
- Based on the first sweep, do not predict a strong vertex/fragment-transform or
  normalization winner merely from FLOP counts. Keep both combinations in the sweep.
- Gaussian opacity remains a separate unsorted rendering control. Its 3-sigma support
  and lack of opaque depth writes should increase overlap cost; it is not a complete 3DGS benchmark.
