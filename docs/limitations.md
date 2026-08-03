# Limitations

- COLMAP runs CPU-only (not GPU-accelerated)
- Requires sufficient texture for COLMAP
- Scale accuracy depends on depth quality
- Not suitable for real-time operation
- Produces *pseudo-ground-truth*, not true ground truth — the trajectories are
  depth-scaled COLMAP reconstructions suitable as reference trajectories for
  SLAM/VO evaluation, not a metrology-grade ground truth source
