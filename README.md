# Gaussian Splatting

Gaussian splatting using WebGPU through Dawn.
The math and rendering stages are explained in [docs/gaussian_splatting.md](docs/gaussian_splatting.md).
`src/main.cpp` loads Luigi and renders Gaussian quads using `src/shaders.wgsl`.
Experimental renderers and benchmarks live in `experiments/`.
`GSplatApp` derives from `App` in `src/app.hpp` and `src/app.cpp`, which handles window/device setup,
the frame loop, UI frames, and presentation. GLFW, ImGui, and WebGPU helpers live in
`src/glfw_utils.cpp`, `src/imgui_utils.cpp`, and `src/wgpu_utils.cpp`; general helpers stay in `src/utils.cpp`.
Orbit camera and view/projection math live in `src/camera.hpp` and `src/camera.cpp`.
Eigen 5.0.1 supplies vector, quaternion, and camera-matrix math.
Native development targets blacknoir on Linux. The Emscripten build uses browser WebGPU.

```sh
git submodule update --init
cmake --preset default
cmake --build --preset default --target gaussian_splatting -j 8
./build/gaussian_splatting
```

The default preset uses Clang, Ninja, a Release build, and `build/`.
`compile_commands.json` is exposed at the repository root for clangd.

The first configure downloads Dawn's pinned dependencies, including GLFW.

On blacknoir, Dawn also needs the Xlib/XCB development header (`sudo apt install libx11-xcb-dev`).
Build for X11:

```sh
cmake --preset default
cmake --build --preset default --target gaussian_splatting -j 16
DISPLAY=:101 XAUTHORITY="$HOME/.Xauthority" ./build/gaussian_splatting
```

Drag with the left mouse button to orbit, and scroll to zoom.
The window can be resized; close it to exit.
Dear ImGui 1.93.0 provides a live camera panel with FOV, clipping planes, orbit/zoom sensitivity,
angles, explicit world-space distance limits, and a reset button. UI interaction captures the mouse.
Scene bounds and initial framing are computed by `GSplatApp`; `OrbitCamera` only handles camera state and controls.
`--capture` renders one frame and exits.

## Browser build

With Emscripten activated, use a fresh `build/` when switching between native and browser toolchains:

```sh
emcmake cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gaussian_splatting -j 8
python3 -m http.server 8080 --bind 127.0.0.1 --directory build
```

Open `http://localhost:8080/` in a WebGPU-enabled browser.
The browser build uses Pongasoft GLFW through `--use-port=contrib.glfw3`.
`MakeCanvasResizable` fills the browser window and the port handles HiDPI scaling.
The custom `src/index.html` contains only a canvas that fills the page.
The build packages `src/shaders.wgsl` and `assets/luigi.ply` into the browser filesystem; rebuild after asset edits.
For remote hosting, serve over HTTPS.

## Sample scene

`assets/luigi.ply` contains 14,526 Gaussians. Source details are in `assets/README.md`.
`readPlyFile` in `src/gaussian.cpp` reads binary little-endian Gaussian PLY files with float properties,
including position, SH DC color, opacity, logarithmic scale, and quaternion rotation.

Each Gaussian is drawn with a four-vertex triangle strip. The vertex shader projects its
three scaled, rotated axes through the view-projection differential, then forms the 2D
covariance as `B * transpose(B)`. Covariance is measured in pixels, with 0.3 pixel² added
to each diagonal to filter tiny footprints. An axis-aligned quad encloses the 3-sigma ellipse.
The fragment shader evaluates `opacity * exp(-0.5 * q)` using the inverse covariance,
truncates at 3 sigma, caps alpha at 0.99, and discards contributions below 1/255.

The CPU sorts Gaussian centers by camera-space depth, far to near, when the view direction
changes. Only the sorted index buffer is uploaded; Gaussian data stays on the GPU.
Premultiplied-alpha blending composites the splats without depth writes or a depth attachment.
Center-depth ordering is an approximation for overlapping extended Gaussians.
Colors are fixed RGB decoded from the PLY's DC coefficients; no view-dependent spherical
harmonics are evaluated. Projection is the local perspective approximation discussed in
[the ellipsoid experiments](experiments/BENCHMARKS.md).

The native shader validation compares GPU images with an independent CPU reference using
finite differences for the projection differential. It checks opacity falloff, an analytically
known two-layer blend, anisotropic rotation, an off-axis center, viewport resizing, and
near/far/behind-camera rejection:

```sh
cmake --build build --target validate_splats -j 12
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/validate_splats
```

The five cases pass with at most 1/255 channel error. Native builds and live rendering were
verified on the RTX 4090; the browser build was not exercised for this change.

## Ellipsoid experiment

`experiments/ellipsoid.cpp` and `experiments/ellipsoid.wgsl` provide a separate executable:

```sh
cmake --build build --target ellipsoid -j 16
DISPLAY=:101 XAUTHORITY="$HOME/.Xauthority" ./build/ellipsoid
```

One window compares four views of the same ellipsoid, from left to right, top to bottom:

1. Oriented box cage with exact ray–ellipsoid intersections.
2. Exact screen-space bounding quad with the same intersections.
3. Hard-edged 2D ellipse from the local perspective covariance approximation (one sigma).
4. Gaussian opacity using that covariance, truncated at three sigma.

Drag any view to orbit all four; scroll to zoom. Semi-axes, rotation, FOV, opacity,
and proxy outlines are shared controls. The first two silhouettes should agree;
the projected ellipse approximates perspective locally and can differ at close distances.
The Gaussian has wider support because its cutoff is three sigma rather than one.
The experiment uses premultiplied alpha blending; see
[WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-transparency.html).

For the browser build, build target `ellipsoid` in `build` and open `ellipsoid.html`.
The combination benchmark and predictions are in [experiments/BENCHMARKS.md](experiments/BENCHMARKS.md)
and [experiments/PREDICTIONS.md](experiments/PREDICTIONS.md). Build target `ellipsoid_benchmark`
for the native offscreen sweep of coverage, ray-transform stage, and intersection combinations.

The previous Golus comparison measurements remain in [experiments/GOLUS_BENCHMARKS.md](experiments/GOLUS_BENCHMARKS.md);
that report describes the earlier comparison renderer. Its benchmark helper files are also under `experiments/`.

## References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl, Kopanas, Leimkühler, and Drettakis, SIGGRAPH 2023; [official differentiable rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization).
- [gsplat](https://github.com/nerfstudio-project/gsplat): [covariance and projection helpers](https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/include/Utils.cuh), and [projection, filtering, and bounding radii](https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu).
- [PlayCanvas](https://github.com/playcanvas/engine): [covariance projection and bounding quads](https://github.com/playcanvas/engine/blob/main/src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatCorner.js), and [Gaussian opacity evaluation](https://github.com/playcanvas/engine/blob/main/src/scene/shader-lib/wgsl/chunks/gsplat/frag/gsplat.js).
- [Ben Golus: Rendering a Sphere on a Quad](https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c)
- [Golus's complete Unity shader](https://gist.github.com/bgolus/1188cd89968b977d5c468bf7bbb3250b), saved under `references/`

- [WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-fundamentals.html)
- [Dawn](https://dawn.googlesource.com/dawn)

## Tiled GPU renderer

`experiments/tiled.cpp` and `experiments/tiled.wgsl` project and compact visible Gaussians, generate
ellipse-tested 16×16 tile records, radix-sort them by tile and front-to-back depth,
and composite pixels in compute. A fullscreen triangle presents the output.
No CPU sorting or visible-count readback is needed to render; a small asynchronous
readback provides UI counters and grows the tile-record storage after overflow.

Source Gaussians use 48 bytes (including RGB8 color) and compact projected records use
32 bytes. Tile records use 16 bytes and have a second radix scratch buffer. The retained
quad comparison mode also keeps its original source/order buffers resident.

This first portable radix implementation is experimental and slower than the quad
renderer on the tested RTX 4090 workloads. See [the measured comparison](experiments/TILED.md).

```sh
cmake --build --preset default --target tiled_viewer validate_tiled -j 16
./build/tiled_viewer
./build/validate_tiled
```
