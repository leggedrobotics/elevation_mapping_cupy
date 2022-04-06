import os
import numpy as np
import scipy as nsp
import scipy.ndimage
import threading

from traversability_filter import get_filter_chainer, get_filter_torch
from parameter import Parameter
from custom_kernels import add_points_kernel
from custom_kernels import error_counting_kernel
from custom_kernels import average_map_kernel
from custom_kernels import dilation_filter_kernel
from custom_kernels import min_filter_kernel
from custom_kernels import normal_filter_kernel
from custom_kernels import polygon_mask_kernel
from map_initializer import MapInitializer
from plugins.plugin_manager import PluginManger

from traversability_polygon import get_masked_traversability, is_traversable, calculate_area, transform_to_map_position, transform_to_map_index

import cupy as cp
import cupyx.scipy as csp
import cupyx.scipy.ndimage

import time

xp = cp
sp = csp
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)


class ElevationMap(object):
    def __init__(self, param: Parameter):
        self.param = param

        self.resolution = param.resolution
        self.center = xp.array([0, 0, 0], dtype=float)
        self.map_length = param.map_length
        # +2 is a border for outside map
        self.cell_n = int(round(self.map_length / self.resolution)) + 2

        self.map_lock = threading.Lock()

        # layers: elevation, variance, is_valid, traversability, time, upper_bound, is_upper_bound
        self.elevation_map = xp.zeros((7, self.cell_n, self.cell_n))
        self.layer_names = ["elevation", "variance", "is_valid", "traversability", "time", "upper_bound", "is_upper_bound"]
        self.traversability_data = xp.full((self.cell_n, self.cell_n), xp.nan)
        self.normal_map = xp.zeros((3, self.cell_n, self.cell_n))
        # Initial variance
        self.initial_variance = param.initial_variance
        self.elevation_map[1] += self.initial_variance
        self.elevation_map[3] += 1.0

        # Initial mean_error
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

        self.compile_kernels()

        if param.use_chainer:
            self.traversability_filter = get_filter_chainer(param.w1, param.w2, param.w3, param.w_out)
        else:
            self.traversability_filter = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)
        self.untraversable_polygon = xp.zeros((1, 2))

        # Plugins
        self.plugin_manager = PluginManger(cell_n=self.cell_n)
        plugin_config_file = os.path.join(param.package_dir, param.plugin_config_file)
        self.plugin_manager.load_plugin_settings(plugin_config_file)

        self.map_initializer = MapInitializer(self.initial_variance, param.initialized_variance,
                                              xp=cp, method='points')

    def clear(self):
        with self.map_lock:
            self.elevation_map *= 0.0
            # Initial variance
            self.elevation_map[1] += self.initial_variance
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

    def get_position(self, position):
        position[0][:] = xp.asnumpy(self.center)

    def move(self, delta_position):
        delta_position = xp.asarray(delta_position)
        delta_pixel = xp.round(delta_position[:2] / self.resolution)
        delta_position_xy = delta_pixel * self.resolution
        self.center[:2] += xp.asarray(delta_position_xy)
        self.center[2] += xp.asarray(delta_position[2])
        self.shift_map_xy(delta_pixel)
        self.shift_map_z(-delta_position[2])

    def move_to(self, position):
        position = xp.asarray(position)
        delta = position - self.center
        delta_pixel = xp.around(delta[:2] / self.resolution)
        delta_xy = delta_pixel * self.resolution
        self.center[:2] += delta_xy
        self.center[2] += delta[2]
        self.shift_map_xy(-delta_pixel)
        self.shift_map_z(-delta[2])

    def shift_map_xy(self, delta_pixel):
        shift_value = delta_pixel
        shift_fn = sp.ndimage.interpolation.shift
        with self.map_lock:
            # elevation
            self.elevation_map[0] = shift_fn(self.elevation_map[0], shift_value,
                                             cval=0.0)
            # variance
            self.elevation_map[1] = shift_fn(self.elevation_map[1], shift_value,
                                             cval=self.initial_variance)
            # is valid (1 is valid 0 is not valid)
            self.elevation_map[2] = shift_fn(self.elevation_map[2], shift_value,
                                             cval=0)
            self.elevation_map[5] = shift_fn(self.elevation_map[5], shift_value,
                                             cval=0)
            self.elevation_map[6] = shift_fn(self.elevation_map[6], shift_value,
                                             cval=0)

    def shift_map_z(self, delta_z):
        with self.map_lock:
            self.elevation_map[0] += delta_z
            self.elevation_map[5] += delta_z

    def compile_kernels(self):
        self.new_map = cp.zeros((7, self.cell_n, self.cell_n))
        self.traversability_input = cp.zeros((self.cell_n, self.cell_n))
        self.traversability_mask_dummy = cp.zeros((self.cell_n, self.cell_n))
        self.min_filtered = cp.zeros((self.cell_n, self.cell_n))
        self.min_filtered_mask = cp.zeros((self.cell_n, self.cell_n))
        self.mask = cp.zeros((self.cell_n, self.cell_n))
        self.add_points_kernel = add_points_kernel(self.resolution,
                                                   self.cell_n,
                                                   self.cell_n,
                                                   self.param.sensor_noise_factor,
                                                   self.param.mahalanobis_thresh,
                                                   self.param.outlier_variance,
                                                   self.param.wall_num_thresh,
                                                   self.param.max_ray_length,
                                                   self.param.cleanup_step,
                                                   self.param.min_valid_distance,
                                                   self.param.max_height_range,
                                                   self.param.cleanup_cos_thresh,
                                                   self.param.ramped_height_range_a,
                                                   self.param.ramped_height_range_b,
                                                   self.param.ramped_height_range_c,
                                                   self.param.enable_edge_sharpen,
                                                   self.param.enable_visibility_cleanup)
        self.error_counting_kernel = error_counting_kernel(self.resolution,
                                                           self.cell_n,
                                                           self.cell_n,
                                                           self.param.sensor_noise_factor,
                                                           self.param.mahalanobis_thresh,
                                                           self.param.drift_compensation_variance_inlier,
                                                           self.param.traversability_inlier,
                                                           self.param.min_valid_distance,
                                                           self.param.max_height_range,
                                                           self.param.ramped_height_range_a,
                                                           self.param.ramped_height_range_b,
                                                           self.param.ramped_height_range_c,
                                                           )
        self.average_map_kernel = average_map_kernel(self.cell_n, self.cell_n,
                                                     self.param.max_variance, self.initial_variance)

        self.dilation_filter_kernel = dilation_filter_kernel(self.cell_n, self.cell_n, self.param.dilation_size)
        self.dilation_filter_kernel_initializer = dilation_filter_kernel(self.cell_n, self.cell_n, self.param.dilation_size_initialize)
        self.min_filter_kernel = min_filter_kernel(self.cell_n, self.cell_n, self.param.min_filter_size)
        self.polygon_mask_kernel = polygon_mask_kernel(self.cell_n, self.cell_n, self.resolution)
        self.normal_filter_kernel = normal_filter_kernel(self.cell_n, self.cell_n, self.resolution)

    def shift_translation_to_map_center(self, t):
        t -= self.center

    def update_map_with_kernel(self, points, R, t, position_noise, orientation_noise):
        self.new_map *= 0.0
        error = xp.array([0.0], dtype=xp.float32)
        error_cnt = xp.array([0], dtype=xp.float32)
        with self.map_lock:
            self.shift_translation_to_map_center(t)
            self.error_counting_kernel(self.elevation_map, points,
                                       cp.array([0.]), cp.array([0.]), R, t,
                                       self.new_map, error, error_cnt,
                                       size=(points.shape[0]))
            if (self.param.enable_drift_compensation
                    and error_cnt > self.param.min_height_drift_cnt
                    and (position_noise > self.param.position_noise_thresh
                         or orientation_noise > self.param.orientation_noise_thresh)):
                self.mean_error = error / error_cnt
                self.additive_mean_error += self.mean_error
                if np.abs(self.mean_error) < self.param.max_drift:
                    self.elevation_map[0] += self.mean_error * self.param.drift_compensation_alpha
            self.add_points_kernel(points, cp.array([0.]), cp.array([0.]), R, t, self.normal_map,
                                   self.elevation_map, self.new_map,
                                   size=(points.shape[0]))
            self.average_map_kernel(self.new_map, self.elevation_map,
                                    size=(self.cell_n * self.cell_n))

            if self.param.enable_overlap_clearance:
                self.clear_overlap_map(t)

            # dilation before traversability_filter
            self.traversability_input *= 0.0
            self.dilation_filter_kernel(self.elevation_map[5],
                                        self.elevation_map[2]+self.elevation_map[6],
                                        self.traversability_input,
                                        self.traversability_mask_dummy,
                                        size=(self.cell_n * self.cell_n))
            # calculate traversability
            traversability = self.traversability_filter(self.traversability_input)
            self.elevation_map[3][3:-3, 3:-3] = traversability.reshape((traversability.shape[2], traversability.shape[3]))

        # calculate normal vectors
        self.update_normal(self.traversability_input)

    def clear_overlap_map(self, t):
        cell_range = int(self.param.overlap_clear_range_xy / self.resolution)
        cell_range = np.clip(cell_range, 0, self.cell_n)
        cell_min = self.cell_n // 2 - cell_range // 2
        cell_max = self.cell_n // 2 + cell_range // 2
        height_min = t[2] - self.param.overlap_clear_range_z
        height_max = t[2] + self.param.overlap_clear_range_z
        near_map = self.elevation_map[:, cell_min:cell_max, cell_min:cell_max]
        clear_idx = cp.logical_or(near_map[0] < height_min, near_map[0] > height_max)
        near_map[0][clear_idx] = 0.0
        near_map[1][clear_idx] = self.initial_variance
        near_map[2][clear_idx] = 0.0
        clear_idx = cp.logical_or(near_map[5] < height_min, near_map[5] > height_max)
        near_map[5][clear_idx] = 0.0
        near_map[6][clear_idx] = 0.0
        self.elevation_map[:, cell_min:cell_max, cell_min:cell_max] = near_map

    def get_additive_mean_error(self):
        return self.additive_mean_error

    def update_variance(self):
        self.elevation_map[1] += self.param.time_variance * self.elevation_map[2]

    def update_time(self):
        self.elevation_map[4] += self.param.time_interval

    def update_upper_bound_with_valid_elevation(self):
        mask = self.elevation_map[2] > 0.5
        self.elevation_map[5] = xp.where(mask, self.elevation_map[0], self.elevation_map[5])
        self.elevation_map[6] = xp.where(mask, 0.0, self.elevation_map[6])

    def input(self, raw_points, R, t, position_noise, orientation_noise):
        raw_points = xp.asarray(raw_points)
        raw_points = raw_points[~xp.isnan(raw_points).any(axis=1)]
        self.update_map_with_kernel(raw_points, xp.asarray(R), xp.asarray(t), position_noise, orientation_noise)

    def get_min_filtered(self):
        self.min_filtered *= 0.0
        self.min_filtered_mask *= 0.0
        self.min_filter_kernel(self.elevation_map[0],
                               self.elevation_map[2],
                               self.min_filtered,
                               self.min_filtered_mask,
                               size=(self.cell_n * self.cell_n))
        if self.param.min_filter_iteration > 1:
            for i in range(self.param.min_filter_iteration - 1):
                self.min_filter_kernel(self.min_filtered,
                                       self.min_filtered_mask,
                                       self.min_filtered,
                                       self.min_filtered_mask,
                                       size=(self.cell_n * self.cell_n))
        min_filtered = xp.where(self.min_filtered_mask > 0.5,
                                self.min_filtered.copy(), xp.nan)
        return min_filtered

    def update_normal(self, dilated_map):
        with self.map_lock:
            self.normal_map *= 0.0
            self.normal_filter_kernel(dilated_map, self.elevation_map[2], self.normal_map, size=(self.cell_n * self.cell_n))

    def get_elevation(self):
        elevation = xp.where(self.elevation_map[2] > 0.5,
                             self.elevation_map[0].copy(), xp.nan)
        elevation = elevation[1:-1, 1:-1] + self.center[2]
        return elevation

    def get_variance(self):
        variance = self.elevation_map[1].copy()
        variance = variance[1:-1, 1:-1]
        return variance

    def get_traversability(self):
        traversability = xp.where((self.elevation_map[2]+self.elevation_map[6]) > 0.5,
                                  self.elevation_map[3].copy(), xp.nan)
        self.traversability_data[3:-3, 3: -3] = traversability[3:-3, 3:-3]
        traversability = self.traversability_data[1:-1, 1:-1]
        return traversability

    def get_min_filtered_map(self):
        min_filtered = self.get_min_filtered()
        min_filtered = min_filtered[1:-1, 1:-1] + self.center[2]
        return min_filtered

    def get_maps(self, selection):
        map_list = []
        with self.map_lock:
            if 0 in selection:
                map_list.append(self.get_elevation())
            if 1 in selection:
                map_list.append(self.get_variance())
            if 2 in selection:
                map_list.append(self.get_traversability())
            if 3 in selection:
                map_list.append(self.get_min_filtered_map())
            if 4 in selection:
                time_layer = self.elevation_map[4].copy()
                time_layer = time_layer[1:-1, 1:-1]
                map_list.append(time_layer)
            if 5 in selection:
                upper_bound = xp.where(xp.logical_or(xp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5), self.elevation_map[2] > 0.5),
                                       self.elevation_map[5].copy(), xp.nan)
                upper_bound = upper_bound[1:-1, 1:-1] + self.center[2]
                map_list.append(upper_bound)
            if 6 in selection:
                is_upper_bound = xp.where(xp.logical_or(xp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5), self.elevation_map[2] > 0.5),
                                          self.elevation_map[6].copy(), xp.nan)
                is_upper_bound = is_upper_bound[1:-1, 1:-1]
                map_list.append(is_upper_bound)

        maps = xp.stack(map_list, axis=0)
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        maps = xp.asnumpy(maps)
        return maps

    def get_normal_maps(self):
        normal = self.normal_map.copy()
        normal_x = normal[0, 1:-1, 1:-1]
        normal_y = normal[1, 1:-1, 1:-1]
        normal_z = normal[2, 1:-1, 1:-1]
        maps = xp.stack([normal_x, normal_y, normal_z], axis=0)
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        maps = xp.asnumpy(maps)
        return maps

    def get_plugin_maps(self):
        self.plugin_manager.update_sync(self.elevation_map, self.layer_names)
        return self.plugin_manager.get_synced_maps()

    def get_maps_ref(self,
                     selection,  # list of numbers to get ex. [0, 1, 3]
                     elevation_data,
                     variance_data,
                     traversability_data,
                     min_filtered_data,
                     time_data,
                     upper_bound,
                     is_upper_bound,
                     normal_x_data, normal_y_data, normal_z_data, normal=False):
        maps = self.get_maps(selection)
        idx = 0
        # somehow elevation_data copy in non_blocking mode does not work.
        if 0 in selection:
            elevation_data[...] = xp.asnumpy(maps[idx])
            idx += 1
        self.stream = cp.cuda.Stream(non_blocking=True)
        if 1 in selection:
            variance_data[...] = xp.asnumpy(maps[idx], stream=self.stream)
            idx += 1
        if 2 in selection:
            traversability_data[...] = xp.asnumpy(maps[idx], stream=self.stream)
            idx += 1
        if 3 in selection:
            min_filtered_data[...] = xp.asnumpy(maps[idx], stream=self.stream)
            idx += 1
        if 4 in selection:
            time_data[...] = xp.asnumpy(maps[idx], stream=self.stream)
            idx += 1
        if 5 in selection:
            upper_bound[...] = xp.asnumpy(maps[idx], stream=self.stream)
            idx += 1
        if 6 in selection:
            is_upper_bound[...] = xp.asnumpy(maps[idx], stream=self.stream)
            idx += 1
        if normal:
            normal_maps = self.get_normal_maps()
            normal_x_data[...] = xp.asnumpy(normal_maps[0], stream=self.stream)
            normal_y_data[...] = xp.asnumpy(normal_maps[1], stream=self.stream)
            normal_z_data[...] = xp.asnumpy(normal_maps[2], stream=self.stream)

    def get_normal_ref(self, normal_x_data, normal_y_data, normal_z_data):
        maps = self.get_normal_maps()
        self.stream = cp.cuda.Stream(non_blocking=True)
        normal_x_data[...] = xp.asnumpy(maps[0], stream=self.stream)
        normal_y_data[...] = xp.asnumpy(maps[1], stream=self.stream)
        normal_z_data[...] = xp.asnumpy(maps[2], stream=self.stream)

    def get_polygon_traversability(self, polygon, result):
        polygon = xp.asarray(polygon)
        area = calculate_area(polygon)
        pmin = self.center[:2] - self.map_length / 2 + self.resolution
        pmax = self.center[:2] + self.map_length / 2 - self.resolution
        polygon[:, 0] = polygon[:, 0].clip(pmin[0], pmax[0])
        polygon[:, 1] = polygon[:, 1].clip(pmin[1], pmax[1])
        polygon_min = polygon.min(axis=0)
        polygon_max = polygon.max(axis=0)
        polygon_bbox = cp.concatenate([polygon_min, polygon_max]).flatten()
        polygon_n = polygon.shape[0]
        clipped_area = calculate_area(polygon)
        self.polygon_mask_kernel(polygon, self.center[0], self.center[1],
                                 polygon_n, polygon_bbox, self.mask,
                                 size=(self.cell_n * self.cell_n))
        masked, masked_isvalid = get_masked_traversability(self.elevation_map,
                                                           self.mask)
        if masked_isvalid.sum() > 0:
            t = masked.sum() / masked_isvalid.sum()
        else:
            t = 0.0
        is_safe, un_polygon = is_traversable(masked,
                                             self.param.safe_thresh,
                                             self.param.safe_min_thresh,
                                             self.param.max_unsafe_n)
        untraversable_polygon_num = 0
        if un_polygon is not None:
            un_polygon = transform_to_map_position(un_polygon,
                                                   self.center[:2],
                                                   self.cell_n,
                                                   self.resolution)
            untraversable_polygon_num = un_polygon.shape[0]
        if clipped_area < 0.001:
            is_safe = False
            print('requested polygon is outside of the map')
        result[...] = np.array([is_safe, t, area])
        self.untraversable_polygon = un_polygon
        return untraversable_polygon_num

    def get_untraversable_polygon(self, untraversable_polygon):
        untraversable_polygon[...] = xp.asnumpy(self.untraversable_polygon)

    def initialize_map(self, points, method='cubic'):
        self.clear()
        with self.map_lock:
            points = cp.asarray(points)
            indices = transform_to_map_index(points[:, :2],
                                             self.center[:2],
                                             self.cell_n,
                                             self.resolution)
            points[:, :2] = indices.astype(points.dtype)
            points[:, 2] -= self.center[2]
            self.map_initializer(self.elevation_map, points, method)
            if self.param.dilation_size_initialize > 0:
                for i in range(2):
                    self.dilation_filter_kernel_initializer(self.elevation_map[0],
                                                            self.elevation_map[2],
                                                            self.elevation_map[0],
                                                            self.elevation_map[2],
                                                            size=(self.cell_n * self.cell_n))
            self.update_upper_bound_with_valid_elevation()


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
    param.load_weights('../config/weights.dat')
    elevation = ElevationMap(param)
    for i in range(500):
        elevation.input(points, R, t, 0, 0)
        elevation.update_normal(elevation.elevation_map[0])
        print(i)
