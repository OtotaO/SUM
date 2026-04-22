#!/usr/bin/env bash
# Install the repo's git hooks for this clone.
#
# What this does: points `core.hooksPath` at scripts/hooks/, so every
# commit runs the hooks in-tree. This keeps hooks versioned with the
# code — pulling a new hook is the same `git pull` as pulling a
# feature change.
#
# What this does NOT do: modify any global git config. Only this
# repo's .git/config is touched.
#
# Uninstall:    git config --unset core.hooksPath
# Inspect:      git config --get core.hooksPath
# Skip a hook:  git commit --no-verify   (one-shot escape valve)

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/scripts/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
  echo "install-hooks: $HOOKS_DIR not found — are you in the SUM repo?" >&2
  exit 1
fi

chmod +x "$HOOKS_DIR"/*
git config core.hooksPath scripts/hooks

echo "install-hooks: git hooks enabled — core.hooksPath = scripts/hooks"
echo "install-hooks: currently installed:"
for h in "$HOOKS_DIR"/*; do
  [ -f "$h" ] && echo "  $(basename "$h")"
done
