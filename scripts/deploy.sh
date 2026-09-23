#!/usr/bin/env sh
# Thin wrapper so `bash scripts/deploy.sh [local|status|down] [flags]` works
# from bash/zsh/Git Bash. All logic lives in scripts/deploy.py -- one implementation for
# Windows, macOS, Linux and CI. See `scripts/deploy.sh --help`.
set -e
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed -- https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
exec uv run "$(dirname "$0")/deploy.py" "$@"
