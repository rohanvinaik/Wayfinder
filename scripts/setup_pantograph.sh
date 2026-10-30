#!/usr/bin/env bash
# The Lean kernel Wayfinder's search acts through: PyPantograph v0.3.14, whose Pantograph is the
# release cut for Lean 4.28.0 (the version of the soundness gate and every saved proof), with
# upstream's v0.3.15 Options fix backported — v0.3.14 as released does not build on Lean 4.28.0.
# Then `uv sync --extra dev --extra lean --inexact` builds and installs it into .venv.
set -euo pipefail
WF="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$WF/data/vendor/PyPantograph-v0.3.14-lean428"
if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --recurse-submodules --branch v0.3.14 https://github.com/stanford-centaur/PyPantograph "$DEST"
fi
cd "$DEST/src"
if git apply --check "$WF/patches/pantograph-v0.3.14-lean4.28.patch" 2>/dev/null; then
  git apply "$WF/patches/pantograph-v0.3.14-lean4.28.patch"
fi
grep -q "opts.set key val.toString" Pantograph/Library.lean  # the backport is in place
echo "PyPantograph v0.3.14 (+ Lean 4.28 backport) ready at $DEST"
