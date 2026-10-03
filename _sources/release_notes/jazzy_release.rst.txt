ROS 2 Jazzy v2.2.0 Release
******************************************************************

Release Target
==================================================================

This page describes the ``v2.2.0`` release from the ``ros2`` branch.

* ROS distribution: Jazzy
* Packages:

  * ``elevation_map_msgs`` 2.2.0
  * ``elevation_mapping_cupy`` 2.2.0
  * ``semantic_sensor`` 2.2.0

Highlights
==================================================================

Correctness
-----------

* Replaced the incorrect dilation distance and row-wrapping behavior with an
  exact Euclidean implementation.
* Restored float32 point geometry and added early NaN/Inf rejection.
* Removed point-order-dependent fusion and visibility races using immutable
  snapshots, proposal buffers, and deterministic per-cell finalization.
* Replaced fixed-step visibility sampling with exact 2D grid DDA traversal.
* Fixed min-filter buffer aliasing and PointCloud2 row-padding handling.

Throughput
----------

* Reused mapping buffers and removed redundant reductions, map reads, and host
  synchronizations.
* Cached plugin results by map generation and used ping-pong min-filter
  buffers.
* Reduced GridMap message construction overhead by encoding contiguous float32
  bytes directly.
* Skipped map construction for publishers without subscribers.
* Switched to CuPy's device memory pool and made periodic trimming opt-in.
* Pinned GridMap 2.2.2 as a CI source dependency so the GPU test image builds
  message support against its own ROS 2 middleware ABI.

Measured Results
==================================================================

Measurements used an NVIDIA RTX 4090, 30 warmups, 200 measured iterations, and
three independent processes. Tables report the median per-run p95.

* Core mapping callback p95 improved by 55.3--64.2% for deterministic
  10k--100k-point clouds.
* The in-process ROS callback plus six-layer filtered GridMap preparation
  improved p95 by 26.3--65.8%.
* Large-radius 602 x 602 dilation improved from 8.704 ms to 0.590 ms p95.
* A 600 x 600 single-layer GridMap message improved from 10.80 ms to 0.90 ms
  p95.

An equivalent deterministic finalizer was benchmarked in CuPy RawKernel and
NVIDIA Warp. Warp did not provide a repeatable p95 benefit, so this release
does not add Warp as a dependency. Full methodology and machine-readable
results are in ``docs/development/elevation_mapping_gpu_optimization.md`` and
``benchmarks/results/rtx4090_20260720_summary.json``.

Validation
==================================================================

Validation in the isolated ROS 2 Jazzy GPU workspace completed with:

* 126 direct GPU unit tests passed.
* 77 colcon tests passed with no errors, failures, or skips.
* TF/GridMap integration, save/load services, and the synthetic demo launch
  passed.

The full CI script was also run locally on ``starship`` in the exact pinned
GPU container image used by ``jazzy-docker-tests.yml``
(``sha256:2bef0b5f33c844b6851a06027a1e8f05aae07d36e6962f8d93d129d7d2646963``).
It built 10 packages, passed both semantic runtime smokes, and passed all 77
tests. The GitHub documentation workflow passed. The remote ROS 2 workflow was
cancelled while queued because this repository does not currently have a
registered self-hosted runner; the equivalent container command above is the
release gate used for this tag.

Deployment Note
==================================================================

The available workspace did not contain a representative raw Ouster
PointCloud2 bag. Before robot deployment, replay a representative recorded
cloud, compare map output, and measure executor, serialization, DDS latency,
and missed publication deadlines on the target system.
