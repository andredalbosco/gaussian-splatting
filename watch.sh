#!/usr/bin/env bash
cd "$(dirname "$0")"

app_pid=
previous=

stop_app() {
    if [[ -n "$app_pid" ]]; then
        kill "$app_pid" 2>/dev/null
        wait "$app_pid" 2>/dev/null
        app_pid=
    fi
}

trap stop_app EXIT
trap 'exit 0' INT TERM

echo "Watching src/, assets/, and CMake files. Ctrl+C to stop."

while true; do
    current=$(find src assets -type f -exec cksum {} +; cksum CMakeLists.txt CMakePresets.json)
    if [[ "$current" != "$previous" ]]; then
        previous=$current
        if cmake --preset default &&
            cmake --build --preset default --target gaussian_splatting -j 8; then
            stop_app
            ./build/gaussian_splatting &
            app_pid=$!
        fi
    fi
    sleep 0.5
done
