import numpy as np
import scipy as nsp
import scipy.ndimage

from traversability_filter import TraversabilityFilter
from parameter import Parameter
from custom_kernels import add_points_kernel
from custom_kernels import error_counting_kernel
from custom_kernels import average_map_kernel
from custom_kernels import dilation_filter_kernel

import cupy as cp
import cupyx.scipy as csp
import cupyx.scipy.ndimage

xp = cp
sp = csp
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)


class ElevationMap(object):
    def __init__(self, param):
        self.resolution = param.resolution
        self.center = xp.array([0, 0], dtype=float)
        self.map_length = param.map_length
        # +2 is a border for outside map
        self.cell_n = int(self.map_length / self.resolution) + 2

        # 'mean' or 'max'
        self.gather_mode = param.gather_mode

        self.sensor_noise_factor = param.sensor_noise_factor
        self.mahalanobis_thresh = param.mahalanobis_thresh
        self.outlier_variance = param.outlier_variance
        self.time_variance = param.time_variance

        self.max_variance = param.max_variance
        self.dilation_size = param.dilation_size
        self.traversability_inlier = param.traversability_inlier
        self.wall_num_thresh = param.wall_num_thresh
        self.min_height_drift_cnt = param.min_height_drift_cnt

        self.enable_edge_sharpen = param.enable_edge_sharpen

        # layers: elevation, variance, is_valid, traversability
        self.elevation_map = xp.zeros((4, self.cell_n, self.cell_n))
        # Initial variance
        self.initial_variance = param.initial_variance
        self.elevation_map[1] += self.initial_variance

        self.compile_kernels()

        self.traversability_filter = TraversabilityFilter(param.w1,
                                                          param.w2,
                                                          param.w3,
                                                          param.w_out)
        self.traversability_filter.to_gpu()

    def clear(self):
        self.elevation_map *= 0.0
        # Initial variance
        self.elevation_map[1] += self.initial_variance

    def get_position(self, position):
        position[0][:] = xp.asnumpy(self.center)

    def move(self, delta_position):
        delta_position = xp.asarray(delta_position)
        delta_pixel = xp.round(delta_position / self.resolution)
        delta_position = delta_pixel * self.resolution
        self.center += xp.asarray(delta_position)
        self.shift_map(delta_pixel)

    def move_to(self, position):
        position = xp.asarray(position)
        delta = position - self.center
        delta_pixel = xp.around(delta / self.resolution)
        delta = delta_pixel * self.resolution
        self.center += delta
        self.shift_map(-delta_pixel)

    def shift_map(self, delta_pixel):
        shift_value = delta_pixel
        shift_fn = sp.ndimage.interpolation.shift
        # elevation
        self.elevation_map[0] = shift_fn(self.elevation_map[0], shift_value,
                                         cval=0.0)
        # variance
        self.elevation_map[1] = shift_fn(self.elevation_map[1], shift_value,
                                         cval=self.initial_variance)
        # is valid (1 is valid 0 is not valid)
        self.elevation_map[2] = shift_fn(self.elevation_map[2], shift_value,
                                         cval=0)

    def compile_kernels(self):
        self.new_map = cp.zeros((4, self.cell_n, self.cell_n))
        self.traversability_input = cp.zeros((self.cell_n, self.cell_n))
        self.add_points_kernel = add_points_kernel(self.resolution,
                                                   self.cell_n,
                                                   self.cell_n,
                                                   self.sensor_noise_factor,
                                                   self.mahalanobis_thresh,
                                                   self.outlier_variance,
                                                   self.wall_num_thresh,
                                                   self.enable_edge_sharpen)
        self.error_counting_kernel = error_counting_kernel(self.resolution,
                                                           self.cell_n,
                                                           self.cell_n,
                                                           self.sensor_noise_factor,
                                                           self.mahalanobis_thresh,
                                                           self.outlier_variance,
                                                           self.traversability_inlier)

        self.average_map_kernel = average_map_kernel(self.cell_n, self.cell_n,
                                                     self.max_variance, self.initial_variance)

        self.dilation_filter_kernel = dilation_filter_kernel(self.cell_n, self.cell_n, self.dilation_size)

    def update_map_with_kernel(self, points, R, t):
        self.new_map *= 0.0
        error = xp.array([0.0], dtype=xp.float32)
        error_cnt = xp.array([0], dtype=xp.float32)
        self.error_counting_kernel(self.elevation_map, points,
                                   self.center[0], self.center[1], R, t,
                                   self.new_map, error, error_cnt,
                                   size=(points.shape[0]))
        if error_cnt > self.min_height_drift_cnt:
            mean_error = error / error_cnt
            self.elevation_map[0] += mean_error
        self.add_points_kernel(points, self.center[0], self.center[1], R, t,
                               self.elevation_map, self.new_map,
                               size=(points.shape[0]))
        self.average_map_kernel(self.new_map, self.elevation_map,
                                size=(self.cell_n * self.cell_n))

        # dilation before traversability_filter
        self.traversability_input *= 0.0
        self.dilation_filter_kernel(self.elevation_map[0],
                                    self.elevation_map[2],
                                    self.traversability_input,
                                    size=(self.cell_n * self.cell_n))
        # calculate traversability
        traversability = self.traversability_filter(self.traversability_input)
        self.elevation_map[3][3:-3, 3:-3] = traversability.reshape((traversability.shape[2], traversability.shape[3]))

    def update_variance(self):
        self.elevation_map[1] += self.time_variance * self.elevation_map[2]

    def input(self, raw_points, R, t):
        raw_points = xp.asarray(raw_points)
        raw_points = raw_points[~xp.isnan(raw_points).any(axis=1)]
        self.update_map_with_kernel(raw_points, xp.asarray(R), xp.asarray(t))

    def get_maps(self):
        elevation = xp.where(self.elevation_map[2] > 0.5,
                             self.elevation_map[0].copy(), xp.nan)
        variance = self.elevation_map[1].copy()
        traversability = self.elevation_map[3].copy()
        elevation = elevation[1:-1, 1:-1]
        variance = variance[1:-1, 1:-1]
        traversability = traversability[1:-1, 1:-1]

        maps = xp.stack([elevation, variance, traversability], axis=0)
        # maps = xp.transpose(maps, axes=(0, 2, 1))
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        maps = xp.asnumpy(maps)
        return maps

    def get_maps_ref(self, elevation_data, variance_data, traversability_data):
        maps = self.get_maps()
        elevation_data[...] = xp.asnumpy(maps[0])
        stream = cp.cuda.Stream(non_blocking=True)
        # elevation_data[...] = xp.asnumpy(maps[0], stream=stream)
        variance_data[...] = xp.asnumpy(maps[1], stream=stream)
        traversability_data[...] = xp.asnumpy(maps[2], stream=stream)


if __name__ == '__main__':
    #  Test script for profiling.
    #  $ python -m cProfile -o profile.stats elevation_mapping.py
    #  $ snakeviz profile.stats
    xp.random.seed(123)
    points = xp.random.rand(100000, 3)
    R = xp.random.rand(3, 3)
    t = xp.random.rand(3)
    print(R, t)
    param = Parameter()
    param.load_weights('../config/weights.yaml')
    elevation = ElevationMap(param)
    for i in range(200):
        elevation.input(points, R, t)
        print(i)
