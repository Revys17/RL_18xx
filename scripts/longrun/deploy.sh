#!/bin/bash
# Frozen deployments for long runs (docs/longrun_runbook.md).
#
# A deployment is a standalone git clone checked out (detached) at a tagged commit, with its own Python
# interpreter (a copy of uv's managed one, so `uv python upgrade` can't move it), its own .venv built by
# `uv sync --locked` (which compiles engine-rs; packages copied, not hardlinked, out of uv's cache), a record of
# the commit and environment (DEPLOYMENT.json, deployment-packages.txt), and every file made read-only. A run
# imports only from its deployment, so a merge, rebase or edit elsewhere can't reach it (az1 died on 2026-10-18
# when a merge put conflict markers into the worktree it imported from).
#
#   deploy.sh create <tag> <dest> [--cpus LIST] [--allow-untagged] [--repo DIR]
#   deploy.sh verify <dest>        exit 0: the tree, the interpreter and the packages are as deployed
#   deploy.sh in-use <dest>        exit 0 (and list them) if processes use the deployment
#   deploy.sh update <tag> <dest> [--cpus LIST] [--allow-untagged]   refused while anything uses it
#
# --cpus pins the build (taskset -c LIST, nice 19). Put deployments outside .claude/worktrees: the desktop app
# cleans up worktrees there.
set -euo pipefail

die() { echo "deploy.sh: $*" >&2; exit 1; }
say() { echo "deploy.sh: $*" >&2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }

# ----------------------------------------------------------------------------- checks

# Processes whose command line, working directory or mapped files (the venv's libraries, engine_rs) are
# under the deployment. Prints "pid cmdline" lines.
users_of() {
    local dest=$1 pid cmd cwd
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        [ "$pid" = "$$" ] && continue
        cmd=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null) || continue
        [ -n "$cmd" ] || continue
        [[ "$cmd" == *scripts/longrun/deploy.sh* ]] && continue  # this script (and its subshells)
        cwd=$(readlink "$proc/cwd" 2>/dev/null) || cwd=""
        if [[ "$cmd" == *"$dest/"* ]] || [ "$cwd" = "$dest" ] || [[ "$cwd" == "$dest/"* ]] \
            || grep -qF "$dest/" "$proc/maps" 2>/dev/null; then
            echo "$pid ${cmd:0:400}"
        fi
    done
}

in_use() {
    local dest=$1 users
    users=$(users_of "$dest")
    # The launcher holds a shared lock on .longrun.lock for the run's lifetime (inherited by main.py).
    if [ -e "$dest/.longrun.lock" ] && ! flock -n -x "$dest/.longrun.lock" true 2>/dev/null; then
        users="${users}${users:+$'\n'}(lock) $dest/.longrun.lock is held"
    fi
    [ -n "$users" ] || return 1
    echo "$users"
}

verify() {
    local dest=$1 recorded head dirty py
    [ -f "$dest/DEPLOYMENT.json" ] || { echo "no DEPLOYMENT.json in $dest"; return 1; }
    recorded=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$dest/DEPLOYMENT.json")
    head=$(GIT_OPTIONAL_LOCKS=0 git -C "$dest" rev-parse HEAD)
    [ "$head" = "$recorded" ] || { echo "HEAD is $head, deployed $recorded"; return 1; }
    dirty=$(GIT_OPTIONAL_LOCKS=0 git -C "$dest" status --porcelain --untracked-files=no)
    [ -z "$dirty" ] || { echo "tracked files changed since deployment:"; echo "$dirty" | head -20; return 1; }
    py=$(readlink -f "$dest/.venv/bin/python")
    [[ "$py" == "$dest/.python/"* ]] || { echo ".venv/bin/python resolves to $py, outside the deployment"; return 1; }
    "$dest/.venv/bin/python" -I "$SCRIPT_DIR/deploy_record.py" packages "$dest/.venv" \
        | diff -q - "$dest/deployment-packages.txt" > /dev/null \
        || { echo "installed packages differ from deployment-packages.txt"; return 1; }
    (cd "$dest" && sha256sum --quiet -c deployment-sha256.txt) > /dev/null 2>&1 \
        || { echo "built files differ from deployment-sha256.txt"; return 1; }
    return 0
}

# ----------------------------------------------------------------------------- build

resolve_commit() {
    local repo=$1 ref=$2 allow_untagged=$3 commit tags
    commit=$(git -C "$repo" rev-parse --verify --quiet "$ref^{commit}") || die "no commit $ref in $repo"
    tags=$(git -C "$repo" tag --points-at "$commit" | tr '\n' ' ')
    if [ -z "$tags" ] && [ "$allow_untagged" != 1 ]; then
        die "$ref ($commit) has no tag; tag it (git tag <name> $commit) or pass --allow-untagged"
    fi
    echo "$commit"
}

build() {  # build <dest> <commit> <ref> <repo> <cpus>
    local dest=$1 commit=$2 ref=$3 repo=$4 cpus=$5 pin=() src_py src_root
    [ -n "$cpus" ] && pin=(taskset -c "$cpus" nice -n 19)
    git -C "$dest" -c advice.detachedHead=false checkout --quiet --detach "$commit"
    # Agents opening the deployment read this first (untracked, ignored by verify).
    cat > "$dest/CLAUDE.local.md" <<EOF
# FROZEN DEPLOYMENT -- DO NOT EDIT

This directory is a frozen deployment of $commit ($ref) that a long run imports from
(docs/longrun_runbook.md). Never edit, merge, check out, rebase, \`uv sync\` or build here: a live run
reads these files lazily and would crash or silently change. Work in another worktree; a new version
is a new deployment (scripts/longrun/deploy.sh create).
EOF
    printf 'CLAUDE.local.md\n.longrun.lock\nDEPLOYMENT.json\ndeployment-*.txt\n.python/\n.previous/\n' >> "$dest/.git/info/exclude"

    # A private copy of uv's managed interpreter: the venv's python is a symlink to it, so upgrading or
    # removing uv's pythons can't change the run's.
    src_py=$(readlink -f "$(cd "$dest" && env -u VIRTUAL_ENV uv python find --no-project --python-preference only-managed "$(cat .python-version)")")
    src_root=$(dirname "$(dirname "$src_py")")
    [ -x "$src_py" ] || die "no uv-managed interpreter for $(cat "$dest/.python-version")"
    mkdir -p "$dest/.python"
    cp -a "$src_root" "$dest/.python/"
    local py="$dest/.python/$(basename "$src_root")/bin/$(basename "$src_py")"
    say "interpreter $("$py" -V) copied from $src_root"

    (
        cd "$dest"
        export UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never MATURIN_PEP517_ARGS="--locked"
        uv venv --quiet --python "$py" .venv
        say "uv sync --locked (compiles engine-rs; a few minutes)"
        "${pin[@]}" uv sync --locked --python "$py" --reinstall-package rl18xx
        "${pin[@]}" .venv/bin/python -m compileall -q -j 0 rl18xx main.py scripts > /dev/null || true
        # engine-rs/target (cargo's build tree) is not needed at run time.
        rm -rf engine-rs/target
        .venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); import engine_rs, torch, rl18xx.agent.alphazero.search_learning"
    ) || die "build failed"

    "$dest/.venv/bin/python" -I "$SCRIPT_DIR/deploy_record.py" packages "$dest/.venv" > "$dest/deployment-packages.txt"
    (cd "$dest" && find .venv -name 'engine_rs*.so' -print0 | xargs -0 sha256sum; sha256sum uv.lock engine-rs/Cargo.lock "$(realpath --relative-to=. "$py")") \
        > "$dest/deployment-sha256.txt"
    "$dest/.venv/bin/python" -I "$SCRIPT_DIR/deploy_record.py" record "$dest" "$commit" "$ref" "$repo" > "$dest/DEPLOYMENT.json.new"
    mv "$dest/DEPLOYMENT.json.new" "$dest/DEPLOYMENT.json"
    touch "$dest/.longrun.lock"
    freeze "$dest"
    verify "$dest" > /dev/null || die "verify failed right after the build: $(verify "$dest")"
    say "deployed $commit ($ref) at $dest"
}

# Read-only: every file and directory loses its write bits (git can't check out, editors can't save,
# `uv sync` can't touch the venv). Symlinks are not followed.
freeze() { chmod -R a-w "$1"; }
thaw() { chmod -R u+w "$1"; }

# ----------------------------------------------------------------------------- commands

cmd=${1:-}; shift || true
cpus="" allow_untagged=0 repo=""
args=()
while [ $# -gt 0 ]; do
    case $1 in
        --cpus) cpus=$2; shift 2 ;;
        --allow-untagged) allow_untagged=1; shift ;;
        --repo) repo=$2; shift 2 ;;
        -h|--help) usage ;;
        *) args+=("$1"); shift ;;
    esac
done
# The source repository: the one this script belongs to (its main checkout, not a worktree).
if [ -z "$repo" ]; then
    repo=$(dirname "$(cd "$SCRIPT_DIR" && git rev-parse --path-format=absolute --git-common-dir)")
fi

case $cmd in
    create)
        [ ${#args[@]} -eq 2 ] || usage
        ref=${args[0]}; dest=$(realpath -m "${args[1]}")
        [ -e "$dest" ] && die "$dest exists (deploy.sh update to change it)"
        [[ "$dest" == */.claude/worktrees/* ]] && die "put deployments outside .claude/worktrees (the app cleans them up)"
        commit=$(resolve_commit "$repo" "$ref" "$allow_untagged")
        mkdir -p "$(dirname "$dest")"
        git clone --quiet --no-checkout --no-hardlinks "$repo" "$dest"
        build "$dest" "$commit" "$ref" "$repo" "$cpus"
        ;;
    verify)
        [ ${#args[@]} -eq 1 ] || usage
        dest=$(realpath "${args[0]}")
        if out=$(verify "$dest"); then echo "ok: $dest is as deployed"; else echo "MODIFIED: $out" >&2; exit 1; fi
        ;;
    in-use)
        [ ${#args[@]} -eq 1 ] || usage
        dest=$(realpath "${args[0]}")
        in_use "$dest"
        ;;
    update)
        [ ${#args[@]} -eq 2 ] || usage
        ref=${args[0]}; dest=$(realpath "${args[1]}")
        [ -f "$dest/DEPLOYMENT.json" ] || die "$dest is not a deployment"
        if users=$(in_use "$dest"); then
            die "refusing to update $dest while it is in use:"$'\n'"$users"
        fi
        commit=$(resolve_commit "$repo" "$ref" "$allow_untagged")
        thaw "$dest"
        mkdir -p "$dest/.previous"
        mv "$dest/DEPLOYMENT.json" "$dest/.previous/DEPLOYMENT.$(date +%Y%m%dT%H%M%S).json"
        rm -rf "$dest/.venv" "$dest/.python" "$dest/deployment-"*.txt
        git -C "$dest" fetch --quiet --tags origin
        build "$dest" "$commit" "$ref" "$repo" "$cpus"
        ;;
    *) usage ;;
esac
