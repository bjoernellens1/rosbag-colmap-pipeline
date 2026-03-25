#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Building development Docker image..."

docker build -t colmap-rgbd-gt:dev -f docker/Dockerfile.dev .

echo "Development image built: colmap-rgbd-gt:dev"
echo ""
echo "To run:"
echo "  docker run -it --rm -v \$(pwd):/app colmap-rgbd-gt:dev"
