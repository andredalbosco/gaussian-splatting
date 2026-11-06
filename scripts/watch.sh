#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1

mkdir -p build
exec 9>build/.watch.lock
if ! flock -n 9; then
    printf '%s\n' "$EPOCHREALTIME" > build/.watch-restart
    echo "Asked the running watcher to rebuild and relaunch the app."
    exit 0
fi
printf '%s\n' "$EPOCHREALTIME" > build/.watch-restart

interactive=()
[[ -t 0 ]] && interactive=(--interactive)

echo "Watching with watchexec. Press r to relaunch, q or Ctrl+C to stop. DISPLAY=${DISPLAY:-unset}"
exec watchexec "${interactive[@]}" --restart --debounce 250ms --fs-events modify,create,remove,rename --stop-timeout 2s \
    --no-vcs-ignore --shell bash \
    --watch src --watch assets --watch CMakeLists.txt \
    --watch experiments/CMakeLists.txt --watch CMakePresets.json --watch index.html \
    --watch build/.watch-restart \
    --filter '**/src/**' --filter '**/assets/**' --filter '**/CMakeLists.txt' \
    --filter '**/CMakePresets.json' --filter '**/index.html' \
    --filter '**/experiments/CMakeLists.txt' --filter '**/build/.watch-restart' -- '
        if [[ ! -f build/build.ninja ]] || ! cmp -s CMakePresets.json build/.watch-preset; then
            cmake --preset default && cp CMakePresets.json build/.watch-preset || exit 1
        fi
        ninja -C build -j 8 gaussian_splatting && exec ./build/gaussian_splatting
    '
