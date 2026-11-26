#!/usr/bin/env bash
# engine-rs's site build (docs/SITE_ENGINE_RS.md §5.1, contract K5): the server's CLI and the browser's wasm package.
#
#   site/build-site.sh <out>
#     <out>/bin/site_engine        native, `--features site` (the server's shadow; the corpus tools)
#     <out>/pkg/engine_rs.js       wasm-bindgen --target no-modules glue (defines `wasm_bindgen`)
#     <out>/pkg/engine_rs_bg.wasm  `--features wasm,only-1830`, profile wasm-release, wasm-opt -Oz when available
#     <out>/pkg/engine_rs.d.ts     the WasmGame API's types
#     <out>/pkg/BUILD.json         {engine, commit, api, key, titles, features, built, sizes}
#
# Needs: cargo (Rust >= the host's), the wasm32-unknown-unknown target, wasm-bindgen-cli equal to the crate's
# wasm-bindgen (Cargo.lock; 0.2.123), wasm-opt (binaryen; optional). It runs in the site's Docker images as is (B
# calls it from its engine stage) and on the workstation (where it runs under `taskset -c 24-31,56-63 nice -n 19`
# when taskset exists). The commit: $ENGINE_RS_COMMIT, else git, else engine-rs/PINNED (the vendored copy), else
# "unknown"; compiled into both outputs (`version`).
set -euo pipefail
OUT="${1:?usage: site/build-site.sh <out dir>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$HERE"

if [ -z "${ENGINE_RS_COMMIT:-}" ]; then
  if git -C "$HERE" rev-parse --git-dir >/dev/null 2>&1; then
    ENGINE_RS_COMMIT="$(git -C "$HERE" rev-parse --short=12 HEAD)"
    if [ -n "$(git -C "$HERE" status --porcelain -- . 2>/dev/null)" ]; then
      ENGINE_RS_COMMIT="${ENGINE_RS_COMMIT}-dirty"
    fi
  elif [ -f "$HERE/PINNED" ]; then
    ENGINE_RS_COMMIT="$(grep -oE '\b[0-9a-f]{12,40}\b' "$HERE/PINNED" | head -1 | cut -c1-12)"
  fi
fi
export ENGINE_RS_COMMIT="${ENGINE_RS_COMMIT:-unknown}"

RUN=()
if command -v taskset >/dev/null 2>&1 && taskset -c 24-31,56-63 true 2>/dev/null; then
  RUN=(taskset -c 24-31,56-63 nice -n 19)
elif command -v nice >/dev/null 2>&1; then
  RUN=(nice -n 19)
fi

BINDGEN="${WASM_BINDGEN:-wasm-bindgen}"
want=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p' | head -1)
have=$("$BINDGEN" --version | awk '{print $2}')
if [ "$want" != "$have" ]; then
  echo "build-site: wasm-bindgen CLI $have does not match the crate's $want" >&2
  exit 1
fi

TARGET_DIR="${CARGO_TARGET_DIR:-$HERE/target}"
echo "build-site: engine-rs $ENGINE_RS_COMMIT -> $OUT"
"${RUN[@]}" cargo build --release --no-default-features --features site --bin site_engine
mkdir -p "$OUT/bin"
cp "$TARGET_DIR/release/site_engine" "$OUT/bin/site_engine"

"${RUN[@]}" cargo build --profile wasm-release --target wasm32-unknown-unknown --no-default-features \
  --features wasm,only-1830 --lib
rm -rf "$OUT/pkg" && mkdir -p "$OUT/pkg"
"${RUN[@]}" "$BINDGEN" --target no-modules --out-dir "$OUT/pkg" --out-name engine_rs \
  "$TARGET_DIR/wasm32-unknown-unknown/wasm-release/engine_rs.wasm"
rm -f "$OUT/pkg/engine_rs_bg.wasm.d.ts"
if command -v wasm-opt >/dev/null 2>&1; then
  "${RUN[@]}" wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-mutable-globals "$OUT/pkg/engine_rs_bg.wasm" -o "$OUT/pkg/engine_rs_bg.opt.wasm"
  mv "$OUT/pkg/engine_rs_bg.opt.wasm" "$OUT/pkg/engine_rs_bg.wasm"
else
  echo "build-site: wasm-opt not found: the wasm is not size-optimised" >&2
fi

size() { wc -c < "$1" | tr -d ' '; }
gz() { if command -v gzip >/dev/null 2>&1; then gzip -9c "$1" | wc -c | tr -d ' '; else echo null; fi; }
br() { if command -v brotli >/dev/null 2>&1; then brotli -q 11 -c "$1" | wc -c | tr -d ' '; else echo null; fi; }
API=$("$OUT/bin/site_engine" version | sed -n 's/.*"api":\([0-9]*\).*/\1/p')
KEY=$("$OUT/bin/site_engine" version | sed -n 's/.*"key":\([0-9]*\).*/\1/p')
cat > "$OUT/pkg/BUILD.json" <<JSON
{"engine": "engine-rs",
 "commit": "$ENGINE_RS_COMMIT",
 "api": $API,
 "key": $KEY,
 "titles": ["1830"],
 "features": {"pkg": ["site", "wasm", "only-1830"], "site_engine": ["site"]},
 "built": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "wasm_bindgen": "$have",
 "sizes": {"engine_rs_bg.wasm": {"raw": $(size "$OUT/pkg/engine_rs_bg.wasm"), "gzip": $(gz "$OUT/pkg/engine_rs_bg.wasm"), "brotli": $(br "$OUT/pkg/engine_rs_bg.wasm")},
           "engine_rs.js": {"raw": $(size "$OUT/pkg/engine_rs.js"), "gzip": $(gz "$OUT/pkg/engine_rs.js"), "brotli": $(br "$OUT/pkg/engine_rs.js")},
           "site_engine": {"raw": $(size "$OUT/bin/site_engine")}}}
JSON
echo "build-site: done"
cat "$OUT/pkg/BUILD.json"
