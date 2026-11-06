#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export EM_COMPILER_WRAPPER=ccache

cmake --preset web
ninja -C build-web -j 8 gaussian_splatting
