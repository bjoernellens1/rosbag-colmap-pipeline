#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

BAG_PATH="${1:-data/raw/session01.bag}"
CONFIG="${2:-configs/default.yaml}"
WORKSPACE="${3:-data/workspaces/session01}"

echo "Running full pipeline..."
echo "  Bag: $BAG_PATH"
echo "  Config: $CONFIG"
echo "  Workspace: $WORKSPACE"

gttool full "$BAG_PATH" --config "$CONFIG" --workspace "$WORKSPACE"

echo "Done. Output:"
echo "  $WORKSPACE/outputs/trajectory_metric_tum.txt"
