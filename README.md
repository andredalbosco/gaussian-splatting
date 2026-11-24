# gaussian-splatting
This is an experimental implementation of a Gaussian Splatting renderer.

Main features:
- 3D Gaussians projected into screen-space as 2D ellipsoid (a.k.a splatted), then drawn as instanced quads.
- GPU radix sort for back-to-front alpha rendering.
- View dependent color through spherical harmonics.
- Interactive orbit and zoom.

One notable feature is that we use hardware rasterization, instead of rendering in a compute shader/kernel (and thus no need for tiling). The compute renderer is very useful when you want it to be differentiable, which we don't need since we implement only a viewer.

## References
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al. 2023)](https://arxiv.org/abs/2308.04079)
- [EWA Volume Splatting (Zwicker et al. 2001)](https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization), [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- Programming Massively Parallel Processors, 5th ed. (Radix Sort)
