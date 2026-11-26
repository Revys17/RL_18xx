#!/usr/bin/env bash
# Publish the site package for packages B and C (docs/SITE_ENGINE_RS.md §9.1, contract K2):
#   $RUST1/pkg/<commit>/{engine_rs.js, engine_rs_bg.wasm, engine_rs.d.ts, BUILD.json, site_engine}
# then moves $RUST1/pkg/current to it (ln -sfn, after the directory is complete). Refuses a dirty engine-rs tree
# unless ALLOW_DIRTY=1 (the directory is then <commit>-dirty-<time>).
set -euo pipefail
RUST1="${RUST1:-/tmp/claude-1000/-home-revys-workspace/252285c2-5545-52b0-bcda-165a3e4e9ceb/scratchpad/rust1}"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
COMMIT="$(git -C "$WT" rev-parse --short=12 HEAD)"
NAME="$COMMIT"
if [ -n "$(git -C "$WT" status --porcelain -- engine-rs)" ]; then
  [ "${ALLOW_DIRTY:-0}" = 1 ] || { echo "publish: engine-rs has uncommitted changes (ALLOW_DIRTY=1 to publish anyway)" >&2; exit 1; }
  NAME="$COMMIT-dirty-$(date -u +%H%M%S)"
fi
BUILD="$RUST1/A/build-$NAME"
rm -rf "$BUILD"
"$WT/engine-rs/site/build-site.sh" "$BUILD"
DEST="$RUST1/pkg/$NAME"
rm -rf "$DEST.tmp" && mkdir -p "$DEST.tmp"
cp "$BUILD"/pkg/engine_rs.js "$BUILD"/pkg/engine_rs_bg.wasm "$BUILD"/pkg/engine_rs.d.ts "$BUILD"/pkg/BUILD.json "$DEST.tmp"/
cp "$BUILD"/bin/site_engine "$DEST.tmp"/
rm -rf "$DEST" && mv "$DEST.tmp" "$DEST"
ln -sfn "$NAME" "$RUST1/pkg/current"
echo "publish: $DEST (current -> $NAME)"
