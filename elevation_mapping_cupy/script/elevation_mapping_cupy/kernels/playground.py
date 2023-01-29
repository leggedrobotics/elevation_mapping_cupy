import cupy as cp

from elevation_mapping_cupy.parameter import Parameter

from elevation_mapping_cupy.kernels import (
    add_points_kernel,

)
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

def add_points_to_map(points_all):
    """Add points to map.
    TODO fine tune the parameters to make it work"""
    param = Parameter()
    param.ramped_height_range_a = 10
    param.ramped_height_range_b = 0.001
    param.ramped_height_range_c = 10
    param.min_valid_distance = 0.001
    param.max_height_range = 10.0
    add_points_kernel_ = add_points_kernel(
        0.2,
        4,
        4,
        param.sensor_noise_factor,
        param.mahalanobis_thresh,
        param.outlier_variance,
        param.wall_num_thresh,
        param.max_ray_length,
        param.cleanup_step,
        param.min_valid_distance,
        param.max_height_range,
        param.cleanup_cos_thresh,
        param.ramped_height_range_a,
        param.ramped_height_range_b,
        param.ramped_height_range_c,
        param.enable_edge_sharpen,
        param.enable_visibility_cleanup,
    )
    # create points
    points_all = cp.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=cp.float32)
    points = points_all[:, :3]
    elevation_map = cp.zeros((3, 3, 3), dtype=cp.float32)
    normal_map = cp.zeros((3, 3, 3), dtype=cp.float32)
    new_map = cp.zeros((3, 3, 3), dtype=cp.float32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)

    add_points_kernel_(
        cp.array([0.0], dtype=cp.float32),
        cp.array([0.0], dtype=cp.float32),
        R,
        t,
        normal_map,
        points,
        elevation_map,
        new_map,
        size=(points.shape[0]),
    )
    print("idx, valid, inside, values")
    points_all[:, 0] = 1
    points_all[:, 1:3] = 1.0
    print("points all after", points_all)


def sum_kernel_fct():
    # create points
    points_all = cp.array([[0.1, 0.1, 0.1, 0.3], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=cp.float32)
    points = points_all[:, :3]
    # arguments for kernel
    semantic_map = cp.zeros((1, 3, 3), dtype=cp.float32)
    new_map = cp.zeros((1, 3, 3), dtype=cp.float32)
    pcl_ids, layer_ids = cp.array([3], dtype=cp.int32), cp.array([0], dtype=cp.int32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    # compile kernel
    sum_kernel_ = sum_kernel(
        0.9,
        4,
        4,
    )
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
        size=(points_all.shape[0]*pcl_ids.shape[0]),
    )
    print("semantic_map", semantic_map)
    print("new_map", new_map)


def average_kernel_fct():
    points_all = cp.array([[0.1, 0.1, 0.1, 0.3], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=cp.float32)
    elevation_map = cp.zeros((3, 4, 4), dtype=cp.float32)
    elevation_map[2] = 12
    semantic_map = cp.zeros((1,4, 4), dtype=cp.float32)
    new_map = cp.ones((1, 4, 4), dtype=cp.float32)
    pcl_ids, layer_ids = cp.array([3], dtype=cp.int32), cp.array([0], dtype=cp.int32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    # compile kernel
    average_kernel_ = average_kernel(
        4,
        4,
    )
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
        size=(cell_n * cell_n*pcl_ids.shape[0]),
    )
    print("semantic_map", semantic_map)

def color_kernel():
    # params
    cell_n = 4
    pcl_ids, layer_ids = cp.array([3], dtype=cp.int32), cp.array([0], dtype=cp.int32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    points_all = cp.array([[0.1, 0.1, 0.1, 0.3], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=cp.float32)
    semantic_map = cp.zeros((1,4, 4), dtype=cp.float32)

    # compile kernel
    add_color_kernel_ = add_color_kernel(
        cell_n,
        cell_n,
    )
    color_average_kernel_ = color_average_kernel(cell_n, cell_n)

    # updatelayer
    color_map = cp.zeros(
        (1 + 3 * layer_ids.shape[0], cell_n, cell_n),
        dtype=cp.uint32,
    )

    points_all = points_all.astype(cp.float32)
    add_color_kernel_(
        points_all,
        R,
        t,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        color_map,
        size=(points_all.shape[0]*pcl_ids.shape[0]),
    )
    color_average_kernel_(
        color_map,
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        semantic_map,
        size=(cell_n * cell_n*pcl_ids.shape[0]),
    )
    print(color_map)

def bayesian_inference_kernel_fct():
    # params
    cell_n = 4
    resolution = 0.9
    elevation_map = cp.zeros((3, 4, 4), dtype=cp.float32)
    elevation_map[2] = 12
    semantic_map = cp.zeros((1,4, 4), dtype=cp.float32)
    new_map = cp.ones((1, 4, 4), dtype=cp.float32)
    pcl_ids, layer_ids = cp.array([3], dtype=cp.int32), cp.array([0], dtype=cp.int32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    points_all = cp.array([[0.1, 0.1, 0.1, 0.3], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=cp.float32)
    semantic_map = cp.zeros((1,4, 4), dtype=cp.float32)

    # compile kernel
    sum_mean = cp.ones(
        (
            pcl_ids.shape[0],
            cell_n,
            cell_n,
        ),
        cp.float32,
    )
    # TODO initialize the variance with a value different than 0
    sum_compact_kernel_ = sum_compact_kernel(
        resolution,
        cell_n,
        cell_n,
    )
    bayesian_inference_kernel_ = bayesian_inference_kernel(
        cell_n,
        cell_n,
    )

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
        size=(points_all.shape[0]*pcl_ids.shape[0]),
    )
    bayesian_inference_kernel_(
        pcl_ids,
        layer_ids,
        cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
        elevation_map,
        new_map,
        sum_mean,
        semantic_map,
        size=(cell_n * cell_n*pcl_ids.shape[0]),
    )
    print("semantic_map", semantic_map)


def class_average_kernel_fct():
    # params
    cell_n = 4
    resolution = 0.9
    average_weight = 0.5
    map_shape = (4, 4, 4)
    elevation_map = cp.zeros((3,4,4), dtype=cp.float32)
    elevation_map[2] = 3
    semantic_map = cp.zeros(map_shape, dtype=cp.float32)
    new_map = cp.zeros(map_shape, dtype=cp.float32)
    pcl_ids, layer_ids = cp.array([3,4], dtype=cp.int32), cp.array([1,2], dtype=cp.int32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    points_all = cp.array([[0.1, 0.1, 0.1, 0.3,0.3], [0.1, 0.1, 0.1, 0.1,0.2], [0.1, 0.1, 0.1, 0.1,0.2]], dtype=cp.float32)
    # compile kernel
    sum_kernel_ = sum_kernel(
        0.9,
        4,
        4,
    )
    class_average_kernel_ = class_average_kernel(
        cell_n,
        cell_n,
        average_weight,
    )
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
        size=(points_all.shape[0]*pcl_ids.shape[0]),
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
        size=(cell_n * cell_n*pcl_ids.shape[0]),
    )
    print("semantic_map", semantic_map)
    print("new_map", new_map)

def class_bayesian_inference_fct():
    # params
    cell_n = 4
    resolution = 0.9
    average_weight = 0.5
    elevation_map = cp.zeros((3, 4, 4), dtype=cp.float32)
    elevation_map[2] = 3
    semantic_map = cp.zeros((1,4, 4), dtype=cp.float32)
    new_map = cp.zeros((1, 4, 4), dtype=cp.float32)
    pcl_ids, layer_ids = cp.array([3], dtype=cp.int32), cp.array([0], dtype=cp.int32)
    R = cp.eye(3, dtype=cp.float32)
    t = cp.array([0, 0, 0], dtype=cp.float32)
    points_all = cp.array([[0.1, 0.1, 0.1, 0.3], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=cp.float32)
    semantic_map = cp.zeros((1,4, 4), dtype=cp.float32)
    # compile kernel
    alpha_kernel_ = alpha_kernel(
        resolution,
        cell_n,
        cell_n,
    )
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
        size=(points_all.shape[0]*pcl_ids.shape[0]),
    )
    # calculate new thetas
    sum_alpha = cp.sum(new_map[layer_ids], axis=0)
    # do not divide by zero
    sum_alpha[sum_alpha == 0] = 1
    semantic_map[layer_ids] = new_map[layer_ids] / cp.expand_dims(sum_alpha, axis=0)
    print("semantic_map", semantic_map)
    print("new_map", new_map)




if __name__ == "__main__":
    # average
    sum_kernel_fct()
    average_kernel_fct()
    # color kernel
    color_kernel()
    # bayesian inference
    bayesian_inference_kernel_fct()
    # class average
    class_average_kernel_fct()
    # class bayesian inference
    class_bayesian_inference_fct()

