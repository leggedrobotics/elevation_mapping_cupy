import pytest
from elevation_mapping_cupy import parameter, elevation_mapping
import cupy as cp

from elevation_mapping_cupy.parameter import Parameter

from elevation_mapping_cupy.kernels import add_points_kernel
from elevation_mapping_cupy.kernels import (
    average_kernel,
    class_average_kernel,
    alpha_kernel,
    bayesian_inference_kernel,
    add_color_kernel,
    color_average_kernel,
    sum_compact_kernel,
    sum_max_kernel,
    sum_kernel,
)


# to check output run: pytest -rP test_semantic_kernels.py


# only kernel where we test only one layer
def test_color_kernel():
    # params
    cell_n = 4
    pcl_ids, layer_ids = cp.array([3], dtype=cp.int32), cp.array([0], dtype=cp.int32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    points_all = cp.array([[0.1, 0.1, 0.1, 0.3], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=cp.float32)
    semantic_map = cp.zeros((1, 4, 4), dtype=cp.float32)

    # compile kernel
    add_color_kernel_ = add_color_kernel(cell_n, cell_n,)
    color_average_kernel_ = color_average_kernel(cell_n, cell_n)

    # updatelayer
    color_map = cp.zeros((1 + 3 * layer_ids.shape[0], cell_n, cell_n), dtype=cp.uint32,)

    points_all = points_all.astype(cp.float32)
    add_color_kernel_(
        points_all,
        R,
        t,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        color_map,
        size=(points_all.shape[0] * pcl_ids.shape[0]),
    )
    color_average_kernel_(
        color_map,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        semantic_map,
        size=(cell_n * cell_n * pcl_ids.shape[0]),
    )
    print(color_map)


@pytest.mark.parametrize(
    "map_shape, points_all,pcl_ids, layer_ids",
    [
        (
            (4, 4, 4),
            cp.array(
                [[0.1, 0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1, 0.2], [0.1, 0.1, 0.1, 0.1, 0.2]], dtype=cp.float32
            ),
            cp.array([3, 4], dtype=cp.int32),
            cp.array([1, 2], dtype=cp.int32),
        ),
    ],
)
def test_sum_kernel(map_shape, points_all, pcl_ids, layer_ids):
    # create points
    resolution = 0.9
    points = points_all[:, :3]
    # arguments for kernel
    semantic_map = cp.zeros(map_shape, dtype=cp.float32)
    new_map = cp.zeros(map_shape, dtype=cp.float32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    # compile kernel
    sum_kernel_ = sum_kernel(0.9, 4, 4,)
    # simulating adding the points
    print("idx, valid, inside, values")
    points_all[:, 0] = 1
    points_all[:, 1:3] = 1.0
    print("points all after ", points_all)
    # run the kernel
    sum_kernel_(
        points_all,
        R,
        t,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        semantic_map,
        new_map,
        size=(points_all.shape[0] * pcl_ids.shape[0]),
    )
    print("semantic_map", semantic_map)
    print("new_map", new_map)


@pytest.mark.parametrize(
    "map_shape, points_all,pcl_ids, layer_ids",
    [
        (
            (4, 4, 4),
            cp.array(
                [[0.1, 0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1, 0.2], [0.1, 0.1, 0.1, 0.1, 0.2]], dtype=cp.float32
            ),
            cp.array([3, 4], dtype=cp.int32),
            cp.array([1, 2], dtype=cp.int32),
        ),
    ],
)
def test_average_kernel(map_shape, points_all, pcl_ids, layer_ids):
    elevation_map = cp.zeros((3, 4, 4), dtype=cp.float32)
    elevation_map[2] = points_all.shape[0]
    semantic_map = cp.zeros(map_shape, dtype=cp.float32)
    new_map = cp.ones(map_shape, dtype=cp.float32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    # compile kernel
    average_kernel_ = average_kernel(4, 4,)
    cell_n = 4
    print("new_map", new_map)
    print("semantic_map", semantic_map)
    print("elevation_map", elevation_map)

    average_kernel_(
        new_map,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        elevation_map,
        semantic_map,
        size=(cell_n * cell_n * pcl_ids.shape[0]),
    )
    print("semantic_map", semantic_map)


@pytest.mark.parametrize(
    "map_shape, points_all,pcl_ids, layer_ids",
    [
        (
            (3, 4, 4),
            cp.array(
                [[0.1, 0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1, 0.2], [0.1, 0.1, 0.1, 0.1, 0.2]], dtype=cp.float32
            ),
            cp.array([3], dtype=cp.int32),
            cp.array([0], dtype=cp.int32),
        ),
    ],
)
def test_bayesian_inference_kernel(map_shape, points_all, pcl_ids, layer_ids):
    # params
    cell_n = 4
    resolution = 0.9
    elevation_map = cp.zeros((3, 4, 4), dtype=cp.float32)
    elevation_map[2] = points_all.shape[0]
    semantic_map = cp.zeros(map_shape, dtype=cp.float32)
    new_map = cp.ones(map_shape, dtype=cp.float32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)

    # compile kernel
    sum_mean = cp.ones((pcl_ids.shape[0], cell_n, cell_n,), cp.float32,)
    sum_compact_kernel_ = sum_compact_kernel(resolution, cell_n, cell_n,)
    bayesian_inference_kernel_ = bayesian_inference_kernel(cell_n, cell_n,)
    # updatelayer
    sum_mean *= 0
    sum_compact_kernel_(
        points_all,
        R,
        t,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        sum_mean,
        size=(points_all.shape[0] * pcl_ids.shape[0]),
    )
    bayesian_inference_kernel_(
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        elevation_map,
        new_map,
        sum_mean,
        semantic_map,
        size=(cell_n * cell_n * pcl_ids.shape[0]),
    )
    print("semantic_map", semantic_map)


@pytest.mark.parametrize(
    "map_shape, points_all,pcl_ids, layer_ids",
    [
        (
            (4, 4, 4),
            cp.array(
                [[0.1, 0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1, 0.2], [0.1, 0.1, 0.1, 0.1, 0.2]], dtype=cp.float32
            ),
            cp.array([3, 4], dtype=cp.int32),
            cp.array([1, 2], dtype=cp.int32),
        ),
    ],
)
def test_class_average_kernel(map_shape, points_all, pcl_ids, layer_ids):
    # params
    cell_n = 4
    resolution = 0.9
    average_weight = 0.5
    elevation_map = cp.zeros((3, 4, 4), dtype=cp.float32)
    elevation_map[2] = 3
    semantic_map = cp.zeros(map_shape, dtype=cp.float32)
    new_map = cp.zeros(map_shape, dtype=cp.float32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    # compile kernel
    sum_kernel_ = sum_kernel(0.9, 4, 4,)
    class_average_kernel_ = class_average_kernel(cell_n, cell_n, average_weight,)
    print("x,y,z,class")
    print("points all after ", points_all)

    # simulating adding the points
    print("idx, valid, inside, values")
    points_all[:, 0] = 1
    points_all[:, 1:3] = 1.0
    print("points all after ", points_all)
    # run the kernel
    sum_kernel_(
        points_all,
        R,
        t,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        semantic_map,
        new_map,
        size=(points_all.shape[0] * pcl_ids.shape[0]),
    )
    print("new_map bef", new_map)
    print("pcl_ids.shape[0]", pcl_ids.shape[0])
    class_average_kernel_(
        new_map,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        elevation_map,
        semantic_map,
        size=(cell_n * cell_n * pcl_ids.shape[0]),
    )
    print("semantic_map", semantic_map)
    print("new_map", new_map)


@pytest.mark.parametrize(
    "map_shape, points_all,pcl_ids, layer_ids",
    [
        (
            (4, 4, 4),
            cp.array(
                [[0.1, 0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1, 0.2], [0.1, 0.1, 0.1, 0.1, 0.2]], dtype=cp.float32
            ),
            cp.array([3, 4], dtype=cp.int32),
            cp.array([1, 2], dtype=cp.int32),
        ),
    ],
)
def test_class_bayesian_inference_fct(map_shape, points_all, pcl_ids, layer_ids):
    # params
    cell_n = 4
    resolution = 0.9
    elevation_map = cp.zeros((3, 4, 4), dtype=cp.float32)
    elevation_map[2] = points_all.shape[0]
    semantic_map = cp.zeros(map_shape, dtype=cp.float32)
    new_map = cp.zeros(map_shape, dtype=cp.float32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    # compile kernel
    alpha_kernel_ = alpha_kernel(resolution, cell_n, cell_n,)
    # simulating adding the points
    print("idx, valid, inside, values")
    points_all[:, 0] = 1
    points_all[:, 1:3] = 1.0
    print("points all after ", points_all)
    # run the kernel
    alpha_kernel_(
        points_all,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        new_map,
        size=(points_all.shape[0] * pcl_ids.shape[0]),
    )
    # calculate new thetas
    sum_alpha = cp.sum(new_map[layer_ids], axis=0)
    # do not divide by zero
    sum_alpha[sum_alpha == 0] = 1
    semantic_map[layer_ids] = new_map[layer_ids] / cp.expand_dims(sum_alpha, axis=0)
    print("semantic_map", semantic_map)
    print("new_map", new_map)
