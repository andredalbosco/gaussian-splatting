#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

./scripts/build-web.sh

remote=$(git remote get-url origin)
author_name=$(git config user.name)
author_email=$(git config user.email)
publish_dir=$(mktemp -d)
trap 'rm -rf "$publish_dir"' EXIT

git -C "$publish_dir" init --quiet --initial-branch=web
git -C "$publish_dir" remote add origin "$remote"
if git ls-remote --exit-code origin refs/heads/web > /dev/null; then
    git -C "$publish_dir" fetch --quiet --depth=1 origin web
    git -C "$publish_dir" reset --quiet --hard FETCH_HEAD
fi

cp build-web/index.html build-web/index.js build-web/index.wasm build-web/index.data "$publish_dir/"
touch "$publish_dir/.nojekyll"
git -C "$publish_dir" add index.html index.js index.wasm index.data .nojekyll
if ! git -C "$publish_dir" diff --cached --quiet; then
    git -C "$publish_dir" -c user.name="$author_name" -c user.email="$author_email" \
        commit --quiet -m "Deploy Gaussian splatting browser build"
fi
git -C "$publish_dir" push origin HEAD:refs/heads/web
