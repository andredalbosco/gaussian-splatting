# Gaussian Splatting

A starting point for Gaussian splatting using WebGPU through Dawn.
For now, `main.cpp` only prints `Hello, world!`.

Dawn is pinned as a Git submodule in `external/dawn`. It is not built or linked yet.

```sh
git submodule update --init
cmake -S . -B build
cmake --build build
./build/gaussian_splatting
```
