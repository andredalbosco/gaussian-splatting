# AGENTS

## Coding Style

Keep the code as lean as possible.

- Avoid error checking, defensive programming, and other safeguards that clutter the code.
- Assume the app structure and runtime environment are controlled unless told otherwise.
- Prefer the shortest working implementation for demo code.
- Always strive toward minimality, and take any clean opportunity to eliminate existing code.
- Adhere to the existing style of the file or surrounding code instead of introducing a new one.
- Avoid unnecessary abstractions, helper functions, and comments.

## References

- Reference materials may be stored under `/references` and consulted while working.
- If a referenced paper or resource is not available locally, download it into `/references` using the links listed in `README.md`.
- Consult `https://webgpufundamentals.org/` whenever WebGPU API usage, WGSL structure, pipeline setup, canvas resizing, textures, buffers, compute, camera/math, lighting, or debugging details are needed.
- WebGPU Fundamentals contents map:
  - Basics: fundamentals, inter-stage variables, uniforms, storage buffers, vertex buffers, textures, constants, memory layout, transparency, bind group layouts, copying data, limits/features, timing, WGSL, how it works, compatibility mode.
  - 3D math: translation, rotation, scale, matrix math, orthographic projection, perspective projection, cameras, matrix stacks, scene graphs.
  - Lighting: directional, point, spot.
  - Techniques: large clip-space triangle, environment maps, skyboxes, post-processing, LUTs, highlighting, camera controls, picking.
  - Compute: compute shader basics, histogram examples.
  - Misc: resizing the canvas, multiple canvases, points, WebGPU from WebGL, optimization, debugging, resources, WGSL function reference.
