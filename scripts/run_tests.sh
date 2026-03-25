#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Running tests..."

if command -v uv &> /dev/null; then
    uv run pytest tests/ -v --tb=short "$@"
else
    python -m pytest tests/ -v --tb=short "$@"
fi

echo "Tests complete."
