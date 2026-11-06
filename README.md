# Gaussian Splatting

A starting point for Gaussian splatting using WebGPU through Dawn.
`src/main.cpp` loads Luigi and renders its Gaussians as ellipsoids through screen-space bounding quads using `src/shaders.wgsl`.
Window and Dawn initialization live in `src/utils.cpp`.
Orbit camera and view/projection math live in `src/camera.hpp` and `src/camera.cpp`.
Eigen 5.0.1 supplies vector, quaternion, and camera-matrix math.
On macOS, Dawn uses Metal. The Emscripten build uses browser WebGPU.

```sh
git submodule update --init
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gaussian_splatting -j 8
./build/gaussian_splatting
```

The first configure downloads Dawn's pinned dependencies, including GLFW.
Drag with the left mouse button to orbit, and scroll to zoom.
The window can be resized; close it to exit.
Dear ImGui 1.93.0 provides a live camera panel with FOV, clipping planes, orbit/zoom sensitivity,
angles, explicit world-space distance limits, and a reset button. UI interaction captures the mouse.
The Scene selector switches between Luigi and a single shaded test ellipsoid with editable scale and rotation in degrees.
The Method selector compares exact ellipsoid bounds with a Golus camera-facing billboard and an exact-bounds/object-rays control.
Rendering toggles show the ellipsoid, cyan quad edges (including the triangle diagonal), and orange oriented box edges.
Wireframes are drawn over the surface so all edges remain visible. The test scene enables both wireframes by default;
the Sphere and Ellipsoid preset buttons make it easy to compare shapes.
Scene bounds and initial framing are computed by `GSplatApp`; `OrbitCamera` only handles camera state and controls.
On macOS, its display, position, and size are saved to `.window-state` when moved or resized,
and restored between launches, including `watch.sh` restarts.

## Metal capture

```sh
mkdir -p build/captures
MTL_CAPTURE_ENABLED=1 DAWN_TRACE_FILE_BASE="$PWD/build/captures/luigi" ./build/gaussian_splatting --capture
```

`--capture` renders one frame and exits. Dawn saves a timestamped `.gputrace` under `build/captures/`;
open it in Xcode and select Replay to inspect the draw, buffers, shaders, and attachments.
The capture includes initialization, so resource-creation insights are not steady-state performance measurements.

## Browser build

With Emscripten activated (`source /Users/dbsc/emsdk/emsdk_env.sh` on this Mac):

```sh
emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release
cmake --build build-web --target gaussian_splatting -j 8
python3 -m http.server 8080 --bind 127.0.0.1 --directory build-web
```

Open `http://localhost:8080/` in a WebGPU-enabled browser.
The browser build uses Pongasoft GLFW through `--use-port=contrib.glfw3`.
`MakeCanvasResizable` fills the browser window and the port handles HiDPI scaling.
The custom `index.html` contains only a canvas that fills the page.
The build packages `src/shaders.wgsl` and `assets/luigi.ply` into the browser filesystem; rebuild after asset edits.
For remote hosting, serve over HTTPS.

## Sample scene

`assets/luigi.ply` contains 14,526 Gaussians. Source details are in `assets/README.md`.
`readPlyFile` in `src/utils.cpp` reads binary little-endian Gaussian PLY files with float properties,
including position, SH DC color, opacity, logarithmic scale, and quaternion rotation.

This stage draws opaque ellipsoids through screen-space bounding quads.
The decoded Gaussian scales are the ellipsoid semi-axes (one standard deviation); its quaternion sets the orientation.
Both methods cull ellipsoids against the same viewing frustum and use four vertices in a triangle strip.
The exact method computes the perspective bounds of the ellipsoid and reconstructs rays with the inverse view-projection.
The Golus method faces a world-space quad toward the camera, using a sphere radius equal to the largest semi-axis.
Its quad half-size is `r * d / sqrt(d*d - r*r)`, where `d` is the camera-to-center distance.
It transforms rays into unit-sphere space in the vertex shader, avoiding those transforms per fragment.
The local ray stays unnormalized so the hit parameter also reconstructs the world-space hit.
When the camera is inside the bounding sphere, Golus mode falls back to exact bounds.
Visible ellipsoids crossing the camera plane use a full-screen quad in the exact path.
Both paths write ordinary fragment depth; the Unity shader's conservative-depth semantic is unavailable in WGSL.
The fragment shader inverse-rotates and inverse-scales each ray, intersects a unit sphere, and writes the surface depth.
The local ray direction stays unnormalized so the hit distance remains valid in world space.
It decodes opacity but does not use it for rendering yet. Gaussian density, transparency,
depth sorting, and higher-order spherical harmonics are still to come.

## Comparing rendering speed

```sh
./build/gaussian_splatting --benchmark
```

The native benchmark uses the three shader variants on the same scene, camera, render target, and depth settings.
It tests Luigi at three distances, a single ellipsoid, and a sphere at 1280×720 and 2560×1440.
Wireframes, ImGui, presentation, and camera uploads are excluded. Each measurement warms up with 32 passes,
records 128 GPU pass timings, and separately measures a batch of 128 passes without timestamp instrumentation.
`throughput_ms` is the batch wall time divided by 128; it is uncapped rendering throughput, not interactive frame time.
`changed_pixels` counts RGB differences greater than 5/255 against the exact method.
Pause other copies of the viewer while timing. Results and scope are in [BENCHMARKS.md](BENCHMARKS.md).

## References

- [Ben Golus: Rendering a Sphere on a Quad](https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c)
- [Golus's complete Unity shader](https://gist.github.com/bgolus/1188cd89968b977d5c468bf7bbb3250b), saved under `references/`

- [WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-fundamentals.html)
- [Dawn](https://dawn.googlesource.com/dawn)
