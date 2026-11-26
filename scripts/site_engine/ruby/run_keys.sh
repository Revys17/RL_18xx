#!/bin/bash
# The reference Ruby shadow keys (keys_reference.rb: A's twin of §5.3, not B's lib/engine_shadow.rb) of every game of
# $RUST1/corpus/<subdir>, in throwaway 18xx_rack:dev containers (no network, the site's source read-only), then
# compare with engine-rs's (keys_corpus.py --out) with compare_keys.py.
#   scripts/site_engine/ruby/run_keys.sh <corpus subdir> <out dir> [jobs]
RUST1=${RUST1:-/tmp/claude-1000/-home-revys-workspace/252285c2-5545-52b0-bcda-165a3e4e9ceb/scratchpad/rust1}
SITE=${SITE:-$HOME/workspace/18xx-next}
RB=$(cd "$(dirname "$0")" && pwd)
SUB=$1; OUT=$2; J=${3:-6}
mkdir -p "$OUT"
export RUST1 SITE RB SUB OUT
ls "$RUST1/corpus/$SUB"/*.json | xargs -n 40 | xargs -P "$J" -I{} bash -c '
  args=(); for f in {}; do args+=("/corpus/$SUB/$(basename "$f")"); done
  timeout 3600 taskset -c 24-31,56-63 nice -n 19 docker run --rm --network none --cpuset-cpus 24-31,56-63 \
    -e LOG_LEVEL=info -v "$SITE":/src:ro -v "$RUST1/corpus":/corpus:ro -v "$RB":/rb:ro -v "$OUT":/out -w /src \
    18xx_rack:dev ruby /rb/keys_reference.rb /out "${args[@]}" > /dev/null 2>&1'
echo done
