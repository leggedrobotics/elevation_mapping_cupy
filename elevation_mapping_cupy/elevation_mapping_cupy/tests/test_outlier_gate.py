import cupy as cp

from elevation_mapping_cupy.kernels.custom_kernels import add_points_kernel, finalize_map_kernel

SIZE = 8
ROW = COL = 4
CELL_VARIANCE = 0.0017
OUTLIER_VARIANCE = 0.01


def _kernel_inputs():
    previous = cp.zeros((7, SIZE, SIZE), dtype=cp.float32)
    previous[1].fill(10.0)
    previous[0, ROW, COL] = 0.0
    previous[1, ROW, COL] = CELL_VARIANCE
    previous[2, ROW, COL] = 1.0
    newmap = cp.zeros_like(previous)
    visibility = cp.zeros((3, SIZE, SIZE), dtype=cp.float32)
    visibility[2].fill(cp.inf)
    normals = cp.zeros((3, SIZE, SIZE), dtype=cp.float32)
    output = cp.empty_like(previous)
    return previous, newmap, visibility, normals, output


def _fuse_point(raw_z):
    previous, newmap, visibility, normals, output = _kernel_inputs()
    add_points = add_points_kernel(
        resolution=0.04,
        width=SIZE,
        height=SIZE,
        sensor_noise_factor=0.01,
        mahalanobis_thresh=2.0,
        outlier_variance=OUTLIER_VARIANCE,
        wall_num_thresh=10,
        max_ray_length=10.0,
        cleanup_step=0.1,
        min_valid_distance=0.1,
        max_height_range=10.5,
        cleanup_cos_thresh=0.5,
        ramped_height_range_a=0.3,
        ramped_height_range_b=1.0,
        ramped_height_range_c=0.2,
        enable_edge_shaped=False,
        enable_visibility_cleanup=False,
    )
    # Sensor sits 1m above the map origin; raw z is in the sensor frame.
    points = cp.asarray([0.02, 0.02, raw_z], dtype=cp.float32)
    add_points(
        cp.asarray([0.0], dtype=cp.float32),
        cp.asarray([0.0], dtype=cp.float32),
        cp.eye(3, dtype=cp.float32).ravel(),
        cp.asarray([0.0, 0.0, 1.0], dtype=cp.float32),
        previous,
        normals,
        points,
        newmap,
        visibility,
        size=1,
    )
    finalize = finalize_map_kernel(SIZE, SIZE, 1.0, 10.0, OUTLIER_VARIANCE)
    finalize(previous, newmap, visibility, output, size=SIZE * SIZE)
    return newmap, output


def test_consistent_residual_on_confident_cell_is_not_an_outlier():
    # Point noise is ~0.01 m^2; a 1 cm residual is well within
    # 2 * sqrt(map_v + v) ~= 21.6 cm of the cell estimate.
    newmap, output = _fuse_point(raw_z=-0.99)

    assert float(newmap[5, ROW, COL]) == 0.0
    assert float(newmap[2, ROW, COL]) == 1.0
    assert float(output[1, ROW, COL]) < CELL_VARIANCE


def test_inconsistent_residual_is_still_rejected_as_outlier():
    newmap, output = _fuse_point(raw_z=-0.4)

    assert float(newmap[5, ROW, COL]) == 1.0
    assert float(newmap[2, ROW, COL]) == 0.0
    assert cp.isclose(output[1, ROW, COL], CELL_VARIANCE + OUTLIER_VARIANCE)
