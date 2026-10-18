# Ben Golus sphere impostor

- Article: https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c
- Shader: https://gist.github.com/bgolus/1188cd89968b977d5c468bf7bbb3250b
- `CompleteSphereImpostor.shader` is an unmodified reference downloaded on 2026-10-18.
- The WGSL implementation adapts the perspective camera-facing billboard, maximum-scale sphere bounds, and object-space rays. Our sphere radius is one before applying scale.
- Shared ellipsoid frustum culling and an exact-bounds fallback when the camera is inside the bounding sphere are additions. WGSL exposes ordinary fragment depth, so the Unity conservative-depth optimization is not included.
