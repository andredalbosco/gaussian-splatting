# Gaussian Splatting

A starting point for Gaussian splatting using WebGPU through Dawn.
`src/main.cpp` loads Luigi and renders its Gaussians as ellipsoids through oriented 3D box cages using `src/shaders.wgsl`.
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

This stage draws opaque ellipsoids through oriented 3D box cages.
The decoded Gaussian scales are the ellipsoid semi-axes (one standard deviation); its quaternion sets the orientation.
The fragment shader inverse-rotates and inverse-scales each ray, intersects a unit sphere, and writes the surface depth.
The local ray direction stays unnormalized so the hit distance remains valid in world space.
It decodes opacity but does not use it for rendering yet. Gaussian density, transparency,
depth sorting, and higher-order spherical harmonics are still to come.

## References

- [WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-fundamentals.html)
- [Dawn](https://dawn.googlesource.com/dawn)
