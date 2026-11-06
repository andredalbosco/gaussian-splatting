# Gaussian Splatting

Gaussian splatting using WebGPU through Dawn.
The math and rendering stages are explained in [docs/gaussian_splatting.md](docs/gaussian_splatting.md).
`src/main.cpp` defines the app and renders Gaussian quads
using `src/shaders.wgsl`; GPU radix setup and recording live in `src/radix_sort.cpp`.
`GaussianSplattingApp` owns the render pipeline, scene buffers, camera uploads, and `RadixSort` in `main.cpp`.
Experimental renderers and benchmarks live in `experiments/`.
`GaussianSplattingApp` is defined in `src/main.cpp` and derives from `App` in `src/app.hpp` and `src/app.cpp`, which handles window/device setup,
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

The default preset uses Clang, ccache, Ninja, a Release build, and `build/`.
The browser build also uses ccache through `EM_COMPILER_WRAPPER`.
The watcher uses `watchexec` for native filesystem events, debouncing, and process-group restarts.
CMake runs for the first watcher configuration or a changed preset; Ninja regenerates when CMake files change.
Inspect cache statistics with `ccache --show-stats`.
`build/compile_commands.json` is the compilation database; `.clangd` points to `build/`.
`SRC_DIR` points to `src/`; `ASSETS_DIR` points to `assets/`, including in the browser filesystem.

The root `CMakeLists.txt` defines the app and shared libraries.
`experiments/CMakeLists.txt` defines optional experiment targets, excluded from the default build.
Build an experiment explicitly, for example `ninja -C build ellipsoid`.
Shell scripts live in `scripts/`; `./scripts/watch.sh` rebuilds and restarts the native app.
Install `watchexec` in your PATH (for example, `cargo install watchexec-cli`).
An edit stops the app, builds with Ninja, and launches the new executable if the build succeeds.
Failed builds wait for the next edit or manual restart. Exiting the watcher stops its process group.
Press **r** to rebuild/relaunch, **s** to stop the app, or **q** to quit. Running `watch.sh` again asks the existing watcher to
relaunch instead of starting a second one. It inherits your display; for Moonlight on blacknoir,
use `DISPLAY=:101 XAUTHORITY="$HOME/.Xauthority" ./scripts/watch.sh`.
`./scripts/deploy-web.sh` publishes browser artifacts to the `web` branch.

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
Dear ImGui 1.93.0 displays Gaussian count, frame timing, and a FOV control. UI interaction captures the mouse.
Scene bounds and initial framing are computed by `GaussianSplattingApp`; `OrbitCamera` only handles camera state and controls.
`--capture` renders one frame and exits.

## Browser build

With Emscripten activated, build separately in `build-web/`:

```sh
./scripts/build-web.sh
python3 -m http.server 8080 --bind 127.0.0.1 --directory build-web
```

Open `http://localhost:8080/` in a WebGPU-enabled browser.
The browser build uses Pongasoft GLFW through `--use-port=contrib.glfw3`.
`MakeCanvasResizable` fills the browser window and the port handles HiDPI scaling.
The custom `index.html` contains only a canvas that fills the page.
The build packages `src/shaders.wgsl`, Black Jacket, and Luigi into the browser filesystem; rebuild after asset edits.
For remote hosting, serve over HTTPS.

## Sample scene

The default scene, `assets/black_jacket.ply`, contains 41,470 Gaussians with degree-3 SH.
Choose Black Jacket or Luigi from the ImGui **Asset** dropdown. Switching reloads buffers,
resets camera framing, and rebuilds the sorted order.
Source details and attribution are in `assets/README.md`.
`readPlyFile` in `src/gaussian.cpp` reads binary little-endian Gaussian PLY files with float properties,
including position, SH DC color, opacity, logarithmic scale, and quaternion rotation.

Each Gaussian is drawn with a four-vertex triangle strip. The vertex shader projects its
three scaled, rotated axes with an explicit perspective Jacobian, then forms the 2D
covariance as `B * transpose(B)` in clip-space footprint units. The `cov_blur` filter
adds 0.3 pixel² of diagonal variance, converted to clip units. An axis-aligned quad encloses
the 3-sigma ellipse, and its offsets are added directly to the clip-space center.
The fragment shader evaluates `opacity * exp(-0.5 * q)` using the inverse covariance,
truncates at 3 sigma, caps alpha at 0.99, and discards contributions below 1/255.

Compute shaders sort Gaussian indices by signed center depth, far to near, when the view
direction changes. `RadixSort::recordCommands` in `src/radix_sort.cpp` records eight stable four-bit radix passes before the render pass.
The first histogram generates depth keys; the final scatter writes sorted indices directly. Equal depths retain
source-index order. The GPU writes the `order` buffer directly; there is no CPU sort,
sorted-index upload, or readback in the viewer.
Premultiplied-alpha blending composites the splats without depth writes or a depth attachment.
Center-depth ordering is an approximation for overlapping extended Gaussians.
Colors use degree 0–3 spherical harmonics decoded from `f_dc_*` and `f_rest_*` PLY properties.
The vertex shader evaluates them along the world-space camera-to-Gaussian direction.
Missing bands are zero, and colors are clamped only below zero after evaluation.
`vs_main` evaluates SH for every rendered Gaussian. DC-only experiment scenes supply only coefficient zero. Projection is the local perspective approximation discussed in
[the ellipsoid experiments](experiments/BENCHMARKS.md).

The native shader validation compares GPU images with an independent CPU reference using
finite differences for the projection differential. It checks opacity falloff, an analytically
known two-layer blend, anisotropic rotation, an off-axis center, viewport resizing, and
near/far/behind-camera rejection:

```sh
cmake --build build --target validate_splats -j 12
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/validate_splats
```

The six cases use GPU-generated order and pass with at most 1/255 channel error.
The validators exercise the production `RadixSort` implementation directly. Native validation and a one-frame main-app run pass
on the RTX 4090; the Emscripten build also passes. Browser runtime was not rechecked
after the radix fusion.

## GPU radix-sort validation

The compute implementation uses workgroup bitmaps for stable local ranks and hierarchical
histogram prefixes; it does not require subgroups. Signed float depth keys preserve
far-to-near ordering, and positive/negative zero share a key. Scene reloads rebuild the
sort buffers and invalidate the cached order.

```sh
cmake --build build --target validate_radix_sort -j 8
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/validate_radix_sort
```

Set `RADIX_TEST_MAX_COUNT=100000` for validation when GPU memory is occupied; larger cases are reported as skipped.
The validation checks CPU-reference index order, equal-depth stability, signed depths,
rotated cameras, repeated sorting, and counts crossing workgroup/prefix boundaries up to
1,048,577 Gaussians. Optional timing results are written when an output directory is supplied;
see [the initial sort comparison](experiments/results/2026-10-19-radix/REPORT.md)
and [the fusion comparison](experiments/results/2026-10-19-radix-fusion/report.md).

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

## Spherical harmonics

Each `Gaussian` stores position, opacity, scale, rotation, and 16 SH RGB coefficients
padded to `vec4f`. The combined record is 304 bytes. There is no redundant RGB color or separate SH buffer.
Older color-based experiments retain a separate `ColoredGaussian` layout. The PLY reader converts channel-major `f_rest_*` values to coefficient-major RGB.
Luigi has only DC; `assets/black_jacket.ply` has full degree-3 SH (41,470 Gaussians, 10.3 MB).
See [asset attribution](assets/README.md).

```sh
ninja -C build validate_sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/validate_sh
```

The validator checks degree 0–3 PLY loading, all 16 GPU basis terms from five camera directions,
negative/HDR colors, and coefficients loaded from Black Jacket. Its CPU reference uses an
associated-Legendre recurrence rather than the shader polynomials.
The basis, direction, and lower-only color clamp follow [original 3DGS](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L17-L64).
