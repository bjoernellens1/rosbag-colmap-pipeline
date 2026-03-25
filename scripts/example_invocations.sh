#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Example invocations for gttool"
echo "================================"
echo ""

echo "1. Inspect a bag file:"
echo "   gttool inspect-bag data/raw/session01.bag"
echo ""

echo "2. Extract RGB, depth, and camera info:"
echo "   gttool extract data/raw/session01.bag --config configs/realsense.yaml"
echo ""

echo "3. Run COLMAP reconstruction:"
echo "   gttool run-colmap data/workspaces/session01"
echo ""

echo "4. Estimate metric scale:"
echo "   gttool scale-depth data/workspaces/session01 --method umeyama"
echo ""

echo "5. Export trajectory in TUM format:"
echo "   gttool export-tum data/workspaces/session01"
echo ""

echo "6. Run complete pipeline (most common):"
echo "   gttool full data/raw/session01.bag --config configs/default.yaml"
echo ""

echo "7. With explicit topic overrides:"
echo "   gttool full data/raw/session01.db3 \\"
echo "     --rgb /camera/color/image_raw \\"
echo "     --depth /camera/aligned_depth_to_color/image_raw \\"
echo "     --camera-info /camera/color/camera_info"
echo ""

echo "8. Using Docker:"
echo "   docker run -it --rm \\"
echo "     -v \$(pwd)/data:/app/data \\"
echo "     colmap-rgbd-gt:dev \\"
echo "     full /app/data/raw/session01.bag --config /app/configs/default.yaml"
echo ""

echo "9. Docker Compose (development):"
echo "   docker compose -f docker/docker-compose.yml run dev"
echo ""
