#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import math
import os
import threading
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Union, Optional

import numpy as np

from elevation_mapping_cupy.traversability_filter import (
    get_filter_chainer,
    get_filter_torch,
)
from elevation_mapping_cupy.parameter import Parameter

from elevation_mapping_cupy.kernels import (
    add_points_kernel,
)

from elevation_mapping_cupy.kernels import sum_kernel
from elevation_mapping_cupy.kernels import error_counting_kernel
from elevation_mapping_cupy.kernels import finalize_map_kernel
from elevation_mapping_cupy.kernels import dilation_filter_kernel
from elevation_mapping_cupy.kernels import normal_filter_kernel
from elevation_mapping_cupy.kernels import polygon_mask_kernel
from elevation_mapping_cupy.kernels import image_to_map_correspondence_kernel

from elevation_mapping_cupy.map_initializer import MapInitializer
from elevation_mapping_cupy.plugins.plugin_manager import PluginManager
from elevation_mapping_cupy.semantic_map import SemanticMap
from elevation_mapping_cupy.traversability_polygon import (
    get_masked_traversability,
    is_traversable,
    calculate_area,
    transform_to_map_position,
    transform_to_map_index,
)

import cupy as cp

xp = cp
pool = cp.get_default_memory_pool()
cp.cuda.set_allocator(pool.malloc)


@dataclass
class GridGeometry:
    """Lightweight holder describing a GridMap's geometry."""

    length_x: float
    length_y: float
    resolution: float
    center: np.ndarray
    orientation: np.ndarray

    @property
    def bounds_x(self) -> Tuple[float, float]:
        half = self.length_x / 2.0
        return self.center[0] - half, self.center[0] + half

    @property
    def bounds_y(self) -> Tuple[float, float]:
        half = self.length_y / 2.0
        return self.center[1] - half, self.center[1] + half

    @property
    def shape(self) -> Tuple[int, int]:
        cols = int(round(self.length_x / self.resolution))
        rows = int(round(self.length_y / self.resolution))
        return rows, cols


BASE_LAYER_TO_INDEX = {
    "elevation": 0,
    "variance": 1,
    "is_valid": 2,
    "traversability": 3,
    "time": 4,
    "upper_bound": 5,
    "is_upper_bound": 6,
}

NORMAL_LAYER_TO_INDEX = {
    "normal_x": 0,
    "normal_y": 1,
    "normal_z": 2,
}


class ElevationMap:
    """Core elevation mapping class."""

    def __init__(self, param: Parameter):
        """

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """

        self.param = param
        self.data_type = self.param.data_type
        self.resolution = param.resolution
        self.center = xp.array([0, 0, 0], dtype=self.data_type)
        self.base_rotation = xp.eye(3, dtype=self.data_type)
        self.map_length = param.map_length
        self.cell_n = param.cell_n

        self.map_lock = threading.Lock()
        self.semantic_map = SemanticMap(self.param)
        self.elevation_map = xp.zeros((7, self.cell_n, self.cell_n), dtype=self.data_type)
        self.layer_names = [
            "elevation",
            "variance",
            "is_valid",
            "traversability",
            "time",
            "upper_bound",
            "is_upper_bound",
        ]

        # buffers
        self.traversability_buffer = xp.full((self.cell_n, self.cell_n), xp.nan)
        self.normal_map = xp.zeros((3, self.cell_n, self.cell_n), dtype=self.data_type)
        # Initial variance
        self.initial_variance = param.initial_variance
        self.elevation_map[1] += self.initial_variance
        self.elevation_map[3] += 1.0

        # overlap clearance
        cell_range = int(self.param.overlap_clear_range_xy / self.resolution)
        cell_range = np.clip(cell_range, 0, self.cell_n)
        self.cell_min = self.cell_n // 2 - cell_range // 2
        self.cell_max = self.cell_n // 2 + cell_range // 2

        # Initial mean_error
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

        self.compile_kernels()

        self.compile_image_kernels()

        self.semantic_map.initialize_fusion()
        self.preallocate_semantic_layers()

        # No shell substitutions in research code: param.weight_file is expected to be a real path.
        param.load_weights(param.weight_file)

        if param.use_chainer:
            self.traversability_filter = get_filter_chainer(param.w1, param.w2, param.w3, param.w_out)
        else:
            self.traversability_filter = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)
        self.untraversable_polygon = xp.zeros((1, 2))

        # Plugins
        self.plugin_manager = PluginManager(cell_n=self.cell_n, resolution=self.resolution)
        self.plugin_manager.load_plugin_settings(param.plugin_config_file)

        self.map_initializer = MapInitializer(self.initial_variance, param.initialized_variance, xp=cp, method="points")

    def clear(self):
        """Reset all the layers of the elevation & the semantic map."""
        with self.map_lock:
            self.elevation_map *= 0.0
            # Initial variance
            self.elevation_map[1] += self.initial_variance
            self.semantic_map.clear()
            self.plugin_manager.reset_layers()

        self.mean_error = 0.0
        self.additive_mean_error = 0.0

    def get_center_position(self, position):
        """Return the position of the map center.

        Args:
            position (numpy.ndarray):

        """
        position[0][:] = xp.asnumpy(self.center)

    def move(self, delta_position):
        """Shift the map along all three axes according to the input.

        Args:
            delta_position (numpy.ndarray):
        """
        # Shift map using delta position.
        delta_position = xp.asarray(delta_position)
        delta_pixel = xp.round(delta_position[:2] / self.resolution)
        delta_position_xy = delta_pixel * self.resolution
        self.center[:2] += xp.asarray(delta_position_xy)
        self.center[2] += xp.asarray(delta_position[2])
        self.shift_map_xy(delta_pixel)
        self.shift_map_z(-delta_position[2])

    def move_to(self, position, R):
        """Shift the map to an absolute position and update the rotation of the robot.

        Args:
            position (numpy.ndarray):
            R (cupy._core.core.ndarray):
        """
        # Shift map to the center of robot.
        self.base_rotation = xp.asarray(R, dtype=self.data_type)
        position = xp.asarray(position)
        delta = position - self.center
        delta_pixel = xp.around(delta[:2] / self.resolution)
        delta_xy = delta_pixel * self.resolution
        self.center[:2] += delta_xy
        self.center[2] += delta[2]
        self.shift_map_xy(-delta_pixel)
        self.shift_map_z(-delta[2])

    def pad_value(self, x, shift_value, idx=None, value=0.0):
        """Create a padding of the map along x,y-axis according to amount that has shifted.

        Args:
            x (cupy._core.core.ndarray):
            shift_value (cupy._core.core.ndarray):
            idx (Union[None, int, None, None]):
            value (float):
        """
        if idx is None:
            if shift_value[0] > 0:
                x[:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:, :, shift_value[1] :] = value
        else:
            if shift_value[0] > 0:
                x[idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[idx, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[idx, :, shift_value[1] :] = value

    def shift_map_xy(self, delta_pixel):
        """Shift the map along the horizontal axes according to the input.

        Args:
            delta_pixel (cupy._core.core.ndarray): Shift in [x, y] order (world coordinates).
                x corresponds to columns (axis 2), y corresponds to rows (axis 1).

        Note:
            The map array has shape (layers, height, width) = (layers, rows, cols).
            In row-major convention: axis 1 = rows = Y, axis 2 = cols = X.
            cp.roll with axis=(1, 2) expects [row_shift, col_shift] = [y_shift, x_shift].
            Since delta_pixel is [x, y], we swap to [y, x] for correct axis mapping.
        """
        # Swap [x, y] to [y, x] to match axis=(1, 2) = (rows=Y, cols=X)
        shift_value = cp.array([delta_pixel[1], delta_pixel[0]], dtype=cp.int32)
        if cp.abs(shift_value).sum() == 0:
            return
        with self.map_lock:
            self.elevation_map = cp.roll(self.elevation_map, shift_value, axis=(1, 2))
            self.pad_value(self.elevation_map, shift_value, value=0.0)
            self.pad_value(self.elevation_map, shift_value, idx=1, value=self.initial_variance)
            # Semantic layers live on the same grid and must follow the same [y, x] roll.
            self.semantic_map.shift_map_xy(shift_value)
            # Plugin layers are computed on-demand; invalidate cache when shifting.
            self.plugin_manager.reset_layers()

    def shift_map_z(self, delta_z):
        """Shift the relevant layers along the vertical axis.

        Args:
            delta_z (cupy._core.core.ndarray):
        """
        with self.map_lock:
            # elevation
            self.elevation_map[0] += delta_z
            # upper bound
            self.elevation_map[5] += delta_z

    def compile_kernels(self):
        """Compile all kernels belonging to the elevation map."""

        self.new_map = cp.zeros(
            (self.elevation_map.shape[0], self.cell_n, self.cell_n),
            dtype=self.data_type,
        )
        self.error = cp.zeros(1, dtype=cp.float32)
        self.error_cnt = cp.zeros(1, dtype=cp.float32)
        self.map_snapshot = cp.empty_like(self.elevation_map)
        self.visibility_map = cp.empty(
            (3, self.cell_n, self.cell_n),
            dtype=self.data_type,
        )
        self.traversability_input = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.traversability_mask_dummy = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.min_filtered = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.min_filtered_mask = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.mask = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)

        self.add_points_kernel = add_points_kernel(
            self.resolution,
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
            self.param.enable_visibility_cleanup,
        )
        self.error_counting_kernel = error_counting_kernel(
            self.resolution,
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
        self.finalize_map_kernel = finalize_map_kernel(
            self.cell_n,
            self.cell_n,
            self.param.max_variance,
            self.initial_variance,
            self.param.outlier_variance,
        )

        self.dilation_filter_kernel = dilation_filter_kernel(self.cell_n, self.cell_n, self.param.dilation_size)
        self.dilation_filter_kernel_initializer = dilation_filter_kernel(
            self.cell_n, self.cell_n, self.param.dilation_size_initialize
        )
        self.polygon_mask_kernel = polygon_mask_kernel(self.cell_n, self.cell_n, self.resolution)
        self.normal_filter_kernel = normal_filter_kernel(self.cell_n, self.cell_n, self.resolution)

    def preallocate_semantic_layers(self):
        """Create the semantic layers declared by the configured subscribers up front.

        Upstream creates semantic layers lazily, on the first message that carries a channel.
        That makes `exists_layer`/`get_map_with_name_ref` fail for a layer that is configured
        but has not been observed yet, which in turn makes a publisher listing that layer throw
        until the first semantic frame arrives. Creating them eagerly (as zeros) keeps the
        published layer set stable from startup and is otherwise a no-op: the first real message
        finds the layer already present.
        """
        for config in self.param.subscriber_cfg.values():
            channels = list(config.get("channels", []) or [])
            if not channels:
                continue
            if config.get("data_type") == "image":
                channel_fusions = self.param.image_channel_fusions
                layer_specs = self.semantic_map.layer_specs_image
            elif config.get("data_type") == "pointcloud":
                channel_fusions = self.param.pointcloud_channel_fusions
                layer_specs = self.semantic_map.layer_specs_points
            else:
                continue
            process_channels, _ = self.semantic_map.get_fusion(channels, channel_fusions, layer_specs)
            for channel in process_channels:
                self.semantic_map.add_layer(channel)

    def compile_image_kernels(self):
        """Compile kernels related to processing image messages.

        Only allocated when at least one subscriber is configured with ``data_type: image``.
        Otherwise the attributes stay None and ``input_image`` raises a clear error instead of
        the AttributeError the unported ros2 branch would have produced.
        """
        self.valid_correspondence = None
        self.uv_correspondence = None
        self.image_to_map_correspondence_kernel = None

        for config in self.param.subscriber_cfg.values():
            if config.get("data_type") == "image":
                self.valid_correspondence = cp.asarray(
                    np.zeros((self.cell_n, self.cell_n), dtype=np.bool_), dtype=np.bool_
                )
                self.uv_correspondence = cp.asarray(
                    np.zeros((2, self.cell_n, self.cell_n), dtype=np.float32),
                    dtype=np.float32,
                )
                self.image_to_map_correspondence_kernel = image_to_map_correspondence_kernel(
                    resolution=self.resolution,
                    width=self.cell_n,
                    height=self.cell_n,
                    tolerance_z_collision=self.param.image_z_collision_tolerance,
                )
                break

    def shift_translation_to_map_center(self, t):
        """Deduct the map center to get the translation of a point w.r.t. the map center.

        Args:
            t (cupy._core.core.ndarray): Absolute point position
        """
        t -= self.center

    def update_map_with_kernel(self, points_all, channels, R, t, position_noise, orientation_noise):
        """Update map with new measurement.

        Args:
            points_all (cupy._core.core.ndarray):
            channels (List[str]):
            R (cupy._core.core.ndarray):
            t (cupy._core.core.ndarray):
            position_noise (float):
            orientation_noise (float):
        """
        points = cp.ascontiguousarray(points_all[:, :3])

        with self.map_lock:
            self.new_map.fill(0.0)
            self.error.fill(0.0)
            self.error_cnt.fill(0.0)
            self.map_snapshot[...] = self.elevation_map
            self.visibility_map.fill(0.0)
            self.visibility_map[2].fill(cp.inf)
            t = t.copy()
            t[2] -= self.center[2]
            self.error_counting_kernel(
                self.map_snapshot,
                points,
                self.center[:1],
                self.center[1:2],
                R,
                t,
                self.new_map,
                self.error,
                self.error_cnt,
                size=(points.shape[0]),
            )
            if (
                self.param.enable_drift_compensation
                and self.error_cnt > self.param.min_height_drift_cnt
                and (
                    position_noise > self.param.position_noise_thresh
                    or orientation_noise > self.param.orientation_noise_thresh
                )
            ):
                self.mean_error = self.error / self.error_cnt
                self.additive_mean_error += self.mean_error
                if np.abs(self.mean_error) < self.param.max_drift:
                    correction = self.mean_error * self.param.drift_compensation_alpha
                    self.map_snapshot[0] += correction
            self.add_points_kernel(
                self.center[:1],
                self.center[1:2],
                R,
                t,
                self.map_snapshot,
                self.normal_map,
                points,
                self.new_map,
                self.visibility_map,
                size=(points.shape[0]),
            )

            self.finalize_map_kernel(
                self.map_snapshot,
                self.new_map,
                self.visibility_map,
                self.elevation_map,
                size=(self.cell_n * self.cell_n),
            )

            # Fuse any extra (non-xyz) pointcloud channels into the semantic map. `self.new_map`
            # is passed as the per-cell hit-count source (layer 2), which `add_points_kernel`
            # above still accumulates on this branch, and `finalize_map_kernel` only reads.
            self.semantic_map.update_layers_pointcloud(points_all, channels, R, t, self.new_map)

            if self.param.enable_overlap_clearance:
                self.clear_overlap_map(t)

            self.traversability_input *= 0.0
            self.dilation_filter_kernel(
                self.elevation_map[5],
                self.elevation_map[2] + self.elevation_map[6],
                self.traversability_input,
                self.traversability_mask_dummy,
                size=(self.cell_n * self.cell_n),
            )

            traversability = self.traversability_filter(self.traversability_input)
            self.elevation_map[3][3:-3, 3:-3] = traversability.reshape(
                (traversability.shape[2], traversability.shape[3])
            )
            self.plugin_manager.reset_layers()

        self.update_normal(self.traversability_input)

    def clear_overlap_map(self, t):
        """Clear overlapping areas around the map center.

        Args:
            t (cupy._core.core.ndarray): Absolute point position
        """

        height_min = t[2] - self.param.overlap_clear_range_z
        height_max = t[2] + self.param.overlap_clear_range_z
        near_map = self.elevation_map[:, self.cell_min : self.cell_max, self.cell_min : self.cell_max]
        valid_idx = ~cp.logical_or(near_map[0] < height_min, near_map[0] > height_max)
        near_map[0] = cp.where(valid_idx, near_map[0], 0.0)
        near_map[1] = cp.where(valid_idx, near_map[1], self.initial_variance)
        near_map[2] = cp.where(valid_idx, near_map[2], 0.0)
        valid_idx = ~cp.logical_or(near_map[5] < height_min, near_map[5] > height_max)
        near_map[5] = cp.where(valid_idx, near_map[5], 0.0)
        near_map[6] = cp.where(valid_idx, near_map[6], 0.0)
        self.elevation_map[:, self.cell_min : self.cell_max, self.cell_min : self.cell_max] = near_map

    def get_additive_mean_error(self):
        """Returns the additive mean error.

        Returns:

        """
        return self.additive_mean_error

    def update_variance(self):
        """Adds the time variacne to the valid cells."""
        self.elevation_map[1] += self.param.time_variance * self.elevation_map[2]

    def update_time(self):
        """adds the time interval to the time layer."""
        self.elevation_map[4] += self.param.time_interval

    def update_upper_bound_with_valid_elevation(self):
        """Filters all invalid cell's upper_bound and is_upper_bound layers."""
        mask = self.elevation_map[2] > 0.5
        self.elevation_map[5] = cp.where(mask, self.elevation_map[0], self.elevation_map[5])
        self.elevation_map[6] = cp.where(mask, 0.0, self.elevation_map[6])

    def input_pointcloud(
        self,
        raw_points: cp._core.core.ndarray,
        channels: List[str],
        R: cp._core.core.ndarray,
        t: cp._core.core.ndarray,
        position_noise: float,
        orientation_noise: float,
    ):
        """Input the point cloud and fuse the new measurements to update the elevation map.

        Args:
            raw_points (cupy._core.core.ndarray):
            channels (List[str]):
            R  (cupy._core.core.ndarray):
            t (cupy._core.core.ndarray):
            position_noise (float):
            orientation_noise (float):

        Returns:
            None:
        """
        raw_points = cp.asarray(raw_points, dtype=self.data_type)
        additional_channels = channels[3:]
        self.update_map_with_kernel(
            raw_points,
            additional_channels,
            cp.asarray(R, dtype=self.data_type),
            cp.asarray(t, dtype=self.data_type),
            position_noise,
            orientation_noise,
        )

    def input_image(
        self,
        image: List[cp._core.core.ndarray],
        channels: List[str],
        R: cp._core.core.ndarray,
        t: cp._core.core.ndarray,
        K: cp._core.core.ndarray,
        D: cp._core.core.ndarray,
        distortion_model: str,
        image_height: int,
        image_width: int,
    ):
        """Project image channels into the map using the camera calibration and pose."""

        if self.image_to_map_correspondence_kernel is None:
            raise RuntimeError(
                "input_image() called but no image kernels were compiled. "
                "At least one entry in `subscribers` must have `data_type: image`."
            )

        image = np.stack(image, axis=0)
        if len(image.shape) == 2:
            image = image[None]

        image = cp.asarray(image, dtype=self.data_type)
        K = cp.asarray(K, dtype=self.data_type)
        R = cp.asarray(R, dtype=self.data_type)
        t = cp.asarray(t, dtype=self.data_type)
        D = cp.asarray(D, dtype=self.data_type)
        image_height = cp.float32(image_height)
        image_width = cp.float32(image_width)

        if len(D) < 4:
            D = cp.zeros(5, dtype=self.data_type)
        elif len(D) == 4:
            D = cp.concatenate([D, cp.zeros(1, dtype=self.data_type)])
        else:
            D = D[:5]

        if distortion_model == "radtan":
            pass
        elif distortion_model in {"equidistant", "plumb_bob"}:
            D *= 0
        else:
            D *= 0

        P = cp.asarray(K @ cp.concatenate([R, t[:, None]], 1), dtype=np.float32)
        t_cam_map = -R.T @ t - self.center
        t_cam_map = t_cam_map.get()
        x1 = cp.uint32((self.cell_n / 2) + (t_cam_map[0] / self.resolution))
        y1 = cp.uint32((self.cell_n / 2) + (t_cam_map[1] / self.resolution))
        z1 = cp.float32(t_cam_map[2])

        self.uv_correspondence *= 0
        self.valid_correspondence[:, :] = False

        with self.map_lock:
            self.image_to_map_correspondence_kernel(
                self.elevation_map,
                x1,
                y1,
                z1,
                P.reshape(-1),
                K.reshape(-1),
                D.reshape(-1),
                image_height,
                image_width,
                self.center,
                self.uv_correspondence,
                self.valid_correspondence,
                size=int(self.cell_n * self.cell_n),
            )
            self.semantic_map.update_layers_image(
                image,
                channels,
                self.uv_correspondence,
                self.valid_correspondence,
                image_height,
                image_width,
            )
            # Plugin layers are generation-cached on this branch; semantic input is a map
            # change, so the cache has to be invalidated or semantic-derived plugin layers
            # would be computed once and then frozen.
            self.plugin_manager.reset_layers()

    def update_normal(self, dilated_map):
        """Clear the normal map and then apply the normal kernel with dilated map as input.

        Args:
            dilated_map (cupy._core.core.ndarray):
        """
        with self.map_lock:
            self.normal_map *= 0.0
            self.normal_filter_kernel(
                dilated_map,
                self.elevation_map[2],
                self.normal_map,
                size=(self.cell_n * self.cell_n),
            )

    def process_map_for_publish(self, input_map, fill_nan=False, add_z=False, xp=cp):
        """Process the input_map according to the fill_nan and add_z flags.

        Args:
            input_map (cupy._core.core.ndarray):
            fill_nan (bool):
            add_z (bool):
            xp (module):

        Returns:
            cupy._core.core.ndarray:
        """
        m = input_map.copy()
        if fill_nan:
            m = xp.where(self.elevation_map[2] > 0.5, m, xp.nan)
        if add_z:
            m = m + self.center[2]
        return m[1:-1, 1:-1]

    def get_elevation(self):
        """Get the elevation layer.

        Returns:
            elevation layer

        """
        return self.process_map_for_publish(self.elevation_map[0], fill_nan=True, add_z=True)

    def get_variance(self):
        """Get the variance layer.

        Returns:
            variance layer
        """
        return self.process_map_for_publish(self.elevation_map[1], fill_nan=False, add_z=False)

    def get_traversability(self):
        """Get the traversability layer.

        Returns:
            traversability layer
        """
        traversability = cp.where(
            (self.elevation_map[2] + self.elevation_map[6]) > 0.5,
            self.elevation_map[3].copy(),
            cp.nan,
        )
        self.traversability_buffer[3:-3, 3:-3] = traversability[3:-3, 3:-3]
        traversability = self.traversability_buffer[1:-1, 1:-1]
        return traversability

    def get_time(self):
        """Get the time layer.

        Returns:
            time layer
        """
        return self.process_map_for_publish(self.elevation_map[4], fill_nan=False, add_z=False)

    def get_upper_bound(self):
        """Get the upper bound layer.

        Returns:
            upper_bound: upper bound layer
        """
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5),
                self.elevation_map[2] > 0.5,
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        upper_bound = cp.where(valid, self.elevation_map[5].copy(), cp.nan)
        upper_bound = upper_bound[1:-1, 1:-1] + self.center[2]
        return upper_bound

    def get_is_upper_bound(self):
        """Get the is upper bound layer.

        Returns:
            is_upper_bound: layer
        """
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5),
                self.elevation_map[2] > 0.5,
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        is_upper_bound = cp.where(valid, self.elevation_map[6].copy(), cp.nan)
        is_upper_bound = is_upper_bound[1:-1, 1:-1]
        return is_upper_bound

    def xp_of_array(self, array):
        """Indicate which library is used for xp.

        Args:
            array (cupy._core.core.ndarray):

        Returns:
            module: either np or cp
        """
        if type(array) == cp.ndarray:
            return cp
        elif type(array) == np.ndarray:
            return np

    def copy_to_cpu(self, array, data, stream=None):
        """Transforms the data to float32 and if on gpu loads it to cpu.

        Args:
            array (cupy._core.core.ndarray):
            data (numpy.ndarray):
            stream (Union[None, cupy.cuda.stream.Stream, None, None, None, None, None, None, None]):
        """
        if type(array) == np.ndarray:
            data[...] = array.astype(np.float32, copy=False)
        elif type(array) == cp.ndarray:
            source = array.astype(np.float32, copy=False)
            if stream is not None:
                cp.asnumpy(source, stream=stream, out=data)
            else:
                cp.asnumpy(source, out=data)

    def trim_memory_pool(self):
        """Release cached CuPy allocator blocks that are not currently in use."""
        pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def exists_layer(self, name):
        """Check if the layer exists in elevation map or in the semantic map.

        Args:
            name (str): Layer name

        Returns:
            bool: Indicates if layer exists.
        """
        if name in self.layer_names:
            return True
        elif name in self.semantic_map.layer_names:
            return True
        elif name in self.plugin_manager.layer_names:
            return True
        else:
            return False

    def get_map_with_name_ref(self, name, data):
        """Load a layer according to the name input to the data input.

        Args:
            name (str): Name of the layer.
            data (numpy.ndarray): Data structure that contains layer.

        """
        use_stream = True
        xp = cp
        with self.map_lock:
            if name == "elevation":
                m = self.get_elevation()
                use_stream = False
            elif name == "variance":
                m = self.get_variance()
            elif name == "is_valid":
                m = self.elevation_map[2].copy()[1:-1, 1:-1]
            elif name == "traversability":
                m = self.get_traversability()
            elif name == "time":
                m = self.get_time()
            elif name == "upper_bound":
                m = self.get_upper_bound()
            elif name == "is_upper_bound":
                m = self.get_is_upper_bound()
            elif name == "normal_x":
                m = self.normal_map.copy()[0, 1:-1, 1:-1]
            elif name == "normal_y":
                m = self.normal_map.copy()[1, 1:-1, 1:-1]
            elif name == "normal_z":
                m = self.normal_map.copy()[2, 1:-1, 1:-1]
            elif name in self.semantic_map.layer_names:
                m = self.semantic_map.get_map_with_name(name)
            elif name in self.plugin_manager.layer_names:
                self.plugin_manager.update_with_name(
                    name,
                    self.elevation_map,
                    self.layer_names,
                    semantic_map=self.semantic_map.semantic_map,
                    semantic_params=self.semantic_map.layer_names,
                    rotation=self.base_rotation,
                    elements_to_shift=self.semantic_map.elements_to_shift,
                )
                m = self.plugin_manager.get_map_with_name(name)
                p = self.plugin_manager.get_param_with_name(name)
                xp = self.xp_of_array(m)
                m = self.process_map_for_publish(m, fill_nan=p.fill_nan, add_z=p.is_height_layer, xp=xp)
            else:
                raise KeyError(f"Layer '{name}' is not in the map.")
        # Transform to align elevation_mapping_cupy with grid_map coordinate convention.
        #
        # elevation_mapping_cupy uses Row=Y, Col=X (see kernels/custom_kernels.py:35)
        # grid_map uses Row→-X, Col→-Y (see grid_map_core/src/GridMapMath.cpp:64-67
        #   transformBufferOrderToMapFrame returns {-index[0], -index[1]})
        #
        # Required transformation:
        #   1. Transpose: swap axes so Row=X, Col=Y (matching grid_map's axis assignment)
        #   2. Flip axis 0: so increasing row → decreasing X (matching grid_map's -X)
        #   3. Flip axis 1: so increasing col → decreasing Y (matching grid_map's -Y)
        #
        # This is equivalent to: rot90(m.T, k=2) or flip(flip(m.T, 0), 1)
        #
        # Old 180° rotation (incorrect - missing transpose, caused 90° CCW error in RViz):
        # m = xp.flip(m, 0)
        # m = xp.flip(m, 1)
        m = self._transform_to_grid_map_coordinate_convention(m)
        if use_stream:
            stream = cp.cuda.Stream(non_blocking=False)
        else:
            stream = None
        self.copy_to_cpu(m, data, stream=stream)

    def _transform_to_grid_map_coordinate_convention(self, m):
        """Transform the map to the grid_map coordinate convention.

        elevation_mapping_cupy uses Row=Y, Col=X (see kernels/custom_kernels.py:35)
        grid_map uses Row→-X, Col→-Y (see grid_map_core/src/GridMapMath.cpp:64-67
        transformBufferOrderToMapFrame returns {-index[0], -index[1]})
        Required transformation:
           1. Transpose: swap axes so Row=X, Col=Y (matching grid_map's axis assignment)
           2. Flip axis 0: so increasing row → decreasing X (matching grid_map's -X)
           3. Flip axis 1: so increasing col → decreasing Y (matching grid_map's -Y)

        This is equivalent to: rot90(m.T, k=2) or flip(flip(m.T, 0), 1)

        Args:
            m (cupy._core.core.ndarray):

        Returns:
            cupy._core.core.ndarray:
        """
        m = m.T
        m = xp.flip(m, 0)
        m = xp.flip(m, 1)
        return m

    def _transform_to_elevation_mapping_coordinate_convention(self, m):
        """Transform the map to the grid_map coordinate convention.

        elevation_mapping_cupy uses Row=Y, Col=X (see kernels/custom_kernels.py:35)
        grid_map uses Row→-X, Col→-Y (see grid_map_core/src/GridMapMath.cpp:64-67
        transformBufferOrderToMapFrame returns {-index[0], -index[1]})
        To transform back to a normal array, we need to apply the inverse transformation:
        Flip axis 0: so increasing row → decreasing X (matching grid_map's -X)
        Flip axis 1: so increasing col → decreasing Y (matching grid_map's -Y)
        Transpose: swap axes so Row=X, Col=Y (matching grid_map's axis assignment)
        This is equivalent to: flip(flip(m, 0), 1).T

        Args:
            m (cupy._core.core.ndarray):

        Returns:
            cupy._core.core.ndarray:
        """
        m = xp.flip(m, 0)
        m = xp.flip(m, 1)
        m = m.T
        return m

    def get_normal_maps(self):
        """Get the normal maps.

        Returns:
            maps: the three normal values for each cell
        """
        normal = self.normal_map.copy()
        normal_x = normal[0, 1:-1, 1:-1]
        normal_y = normal[1, 1:-1, 1:-1]
        normal_z = normal[2, 1:-1, 1:-1]
        maps = xp.stack([normal_x, normal_y, normal_z], axis=0)
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        maps = xp.asnumpy(maps)
        return maps

    def get_normal_ref(self, normal_x_data, normal_y_data, normal_z_data):
        """Get the normal maps as reference.

        Args:
            normal_x_data:
            normal_y_data:
            normal_z_data:
        """
        maps = self.get_normal_maps()
        self.stream = cp.cuda.Stream(non_blocking=True)
        normal_x_data[...] = xp.asnumpy(maps[0], stream=self.stream)
        normal_y_data[...] = xp.asnumpy(maps[1], stream=self.stream)
        normal_z_data[...] = xp.asnumpy(maps[2], stream=self.stream)

    def get_layer(self, name):
        """Return the layer with the name input.

        Args:
            name: The layers name.

        Returns:
            return_map: The rqeuested layer.

        """
        if name in self.layer_names:
            idx = self.layer_names.index(name)
            return_map = self.elevation_map[idx]
        elif name in self.semantic_map.layer_names:
            idx = self.semantic_map.layer_names.index(name)
            return_map = self.semantic_map.semantic_map[idx]
        elif name in self.plugin_manager.layer_names:
            self.plugin_manager.update_with_name(
                name,
                self.elevation_map,
                self.layer_names,
                semantic_map=self.semantic_map.semantic_map,
                semantic_params=self.semantic_map.layer_names,
                rotation=self.base_rotation,
                elements_to_shift=self.semantic_map.elements_to_shift,
            )
            return_map = self.plugin_manager.get_map_with_name(name)
        else:
            print("Layer {} is not in the map, returning traversabiltiy!".format(name))
            return
        return return_map

    def get_polygon_traversability(self, polygon, result):
        """Check if input polygons are traversable.

        Args:
            polygon (cupy._core.core.ndarray):
            result (numpy.ndarray):

        Returns:
            Union[None, int]:
        """
        polygon = xp.asarray(polygon)
        area = calculate_area(polygon)
        polygon = polygon.astype(self.data_type)
        pmin = self.center[:2] - self.map_length / 2 + self.resolution
        pmax = self.center[:2] + self.map_length / 2 - self.resolution
        polygon[:, 0] = polygon[:, 0].clip(pmin[0], pmax[0])
        polygon[:, 1] = polygon[:, 1].clip(pmin[1], pmax[1])
        polygon_min = polygon.min(axis=0)
        polygon_max = polygon.max(axis=0)
        polygon_bbox = cp.concatenate([polygon_min, polygon_max]).flatten()
        polygon_n = xp.array(polygon.shape[0], dtype=np.int16)
        clipped_area = calculate_area(polygon)
        self.polygon_mask_kernel(
            polygon,
            self.center[0],
            self.center[1],
            polygon_n,
            polygon_bbox,
            self.mask,
            size=(self.cell_n * self.cell_n),
        )
        tmp_map = self.get_layer(self.param.checker_layer)
        masked, masked_isvalid = get_masked_traversability(self.elevation_map, self.mask, tmp_map)
        if masked_isvalid.sum() > 0:
            t = masked.sum() / masked_isvalid.sum()
        else:
            t = cp.asarray(0.0, dtype=self.data_type)
        is_safe, un_polygon = is_traversable(
            masked,
            self.param.safe_thresh,
            self.param.safe_min_thresh,
            self.param.max_unsafe_n,
        )
        untraversable_polygon_num = 0
        if un_polygon is not None:
            un_polygon = transform_to_map_position(un_polygon, self.center[:2], self.cell_n, self.resolution)
            untraversable_polygon_num = un_polygon.shape[0]
        if clipped_area < 0.001:
            is_safe = False
            print("requested polygon is outside of the map")
        result[...] = np.array([is_safe, t.get(), area.get()])
        self.untraversable_polygon = un_polygon
        return untraversable_polygon_num

    def get_untraversable_polygon(self, untraversable_polygon):
        """Copy the untraversable polygons to input untraversable_polygons.

        Args:
            untraversable_polygon (numpy.ndarray):
        """
        untraversable_polygon[...] = xp.asnumpy(self.untraversable_polygon)

    def initialize_map(self, points, method="cubic"):
        """Initializes the map according to some points and using an approximation according to method.

        Args:
            points (numpy.ndarray):
            method (str): Interpolation method ['linear', 'cubic', 'nearest']
        """
        self.clear()
        with self.map_lock:
            points = cp.asarray(points, dtype=self.data_type)
            indices = transform_to_map_index(points[:, :2], self.center[:2], self.cell_n, self.resolution)
            points[:, :2] = indices.astype(points.dtype)
            points[:, 2] -= self.center[2]
            self.map_initializer(self.elevation_map, points, method)
            if self.param.dilation_size_initialize > 0:
                for i in range(2):
                    self.dilation_filter_kernel_initializer(
                        self.elevation_map[0],
                        self.elevation_map[2],
                        self.elevation_map[0],
                        self.elevation_map[2],
                        size=(self.cell_n * self.cell_n),
                    )
            self.update_upper_bound_with_valid_elevation()
            self.plugin_manager.reset_layers()

    def list_layers(self) -> List[str]:
        ordered: List[str] = []
        for container in (
            self.layer_names,
            self.semantic_map.layer_names,
            getattr(self.plugin_manager, "layer_names", []),
        ):
            for name in container:
                if name and name not in ordered:
                    ordered.append(name)
        return ordered

    def export_layers(self, layer_names: List[str]) -> Dict[str, np.ndarray]:
        exported: Dict[str, np.ndarray] = {}
        buffer = np.zeros((self.cell_n - 2, self.cell_n - 2), dtype=np.float32)
        for name in layer_names:
            if not self.exists_layer(name):
                continue
            self.get_map_with_name_ref(name, buffer)
            exported[name] = buffer.copy()
        return exported

    def apply_masked_replace(
        self,
        layer_data: Dict[str, np.ndarray],
        mask: Optional[np.ndarray],
        geometry: GridGeometry,
    ) -> None:
        if not layer_data:
            raise ValueError("No layer data provided for masked replace.")

        # Transform the layer data from grid_map coordinate convention to the elevation_mapping_cupy coordinate convention
        for name, array in layer_data.items():
            layer_data[name] = self._transform_to_elevation_mapping_coordinate_convention(array)
        if mask is not None:
            mask = self._transform_to_elevation_mapping_coordinate_convention(mask)

        sample_shape: Optional[Tuple[int, int]] = None
        for array in layer_data.values():
            if sample_shape is None:
                sample_shape = array.shape
            elif sample_shape != array.shape:
                raise ValueError("All incoming layers must share the same shape.")

        if sample_shape is None:
            raise ValueError("Unable to infer incoming layer shape.")

        if mask is None:
            mask = np.ones(sample_shape, dtype=np.float32)
        if mask.shape != sample_shape:
            raise ValueError("Mask shape does not match provided layers.")

        self._validate_geometry_against_shape(sample_shape, geometry)
        overlap = self._compute_overlap_indices(sample_shape, geometry)
        if overlap is None:
            raise ValueError("Incoming grid does not overlap with the active map.")

        map_rows = overlap["map"][0]
        map_cols = overlap["map"][1]
        patch_rows = overlap["patch"][0]
        patch_cols = overlap["patch"][1]

        mask_slice = mask[patch_rows, patch_cols]
        valid_mask = np.isfinite(mask_slice)
        if not np.any(valid_mask):
            return

        cp_mask = cp.asarray(valid_mask)
        center_z = float(cp.asnumpy(self.center)[2])

        with self.map_lock:
            for name, array in layer_data.items():
                target = self._resolve_layer_target(name)
                if target is None:
                    raise ValueError(f"Layer '{name}' does not exist in the map.")
                incoming_slice = array[patch_rows, patch_cols]
                if incoming_slice.shape != mask_slice.shape:
                    raise ValueError("Mismatch between mask and incoming slice dimensions.")
                if name in ("elevation", "upper_bound"):
                    incoming_slice = incoming_slice - center_z
                incoming_cp = cp.asarray(incoming_slice, dtype=self.data_type)
                target_region = target[map_rows, map_cols]
                before = target_region[cp_mask].copy()
                target_region[cp_mask] = incoming_cp[cp_mask]
                written = int(cp_mask.sum())
                # Debug diagnostics for field coverage
                min_max = None
                if np.any(valid_mask):
                    vals = incoming_slice[valid_mask]
                    min_max = (float(np.nanmin(vals)), float(np.nanmax(vals)))
                map_extent = self._map_extent_from_mask(map_rows, map_cols, valid_mask) or self._map_extent_from_slices(map_rows, map_cols)
                print(
                    f"[ElevationMap] masked_replace layer '{name}': wrote {written} cells, "
                    f"X∈[{map_extent['x_min']:.2f},{map_extent['x_max']:.2f}], "
                    f"Y∈[{map_extent['y_min']:.2f},{map_extent['y_max']:.2f}], "
                    f"values {min_max if min_max else 'n/a'}",
                    flush=True
                )

        self._invalidate_caches()

    def set_full_map(
        self,
        fused_layers: Dict[str, np.ndarray],
        raw_layers: Dict[str, np.ndarray],
        geometry: GridGeometry,
    ) -> None:
        if not raw_layers:
            raise ValueError("Raw layer data required to restore the map.")

        # Transform the raw layer data from grid_map coordinate convention to the elevation_mapping_cupy coordinate convention
        for name, array in raw_layers.items():
            raw_layers[name] = self._transform_to_elevation_mapping_coordinate_convention(array)

        sample_shape = next(iter(raw_layers.values())).shape
        self._validate_geometry_against_shape(sample_shape, geometry)

        center_np = np.asarray(geometry.center, dtype=np.float32)
        provided_plugin_layers = set()
        total_plugin_layers = len(getattr(self.plugin_manager, "layer_names", []))

        with self.map_lock:
            self.center[:] = cp.asarray(center_np, dtype=self.data_type)
            for name, data in raw_layers.items():
                target = self._resolve_layer_target(name, allow_semantic_creation=False)
                if target is None:
                    continue
                incoming = data
                if name in ("elevation", "upper_bound"):
                    incoming = incoming - center_np[2]
                incoming_cp = cp.asarray(incoming, dtype=self.data_type)
                target[...] = incoming_cp
                if name in getattr(self.plugin_manager, "layer_names", []):
                    provided_plugin_layers.add(name)

            if total_plugin_layers > 0:
                if len(provided_plugin_layers) != total_plugin_layers:
                    self.plugin_manager.reset_layers()

        self._invalidate_caches(reset_plugins=True)

    def _resolve_layer_target(self, name: str, allow_semantic_creation: bool = False):
        if name in BASE_LAYER_TO_INDEX:
            idx = BASE_LAYER_TO_INDEX[name]
            return self.elevation_map[idx, 1:-1, 1:-1]
        if name in NORMAL_LAYER_TO_INDEX:
            idx = NORMAL_LAYER_TO_INDEX[name]
            return self.normal_map[idx, 1:-1, 1:-1]
        if name in getattr(self.plugin_manager, "layer_names", []):
            idx = self.plugin_manager.layer_names.index(name)
            return self.plugin_manager.layers[idx, 1:-1, 1:-1]
        if name in self.semantic_map.layer_names:
            idx = self.semantic_map.layer_names.index(name)
            return self.semantic_map.semantic_map[idx, 1:-1, 1:-1]
        if allow_semantic_creation:
            self.semantic_map.add_layer(name)
            idx = self.semantic_map.layer_names.index(name)
            return self.semantic_map.semantic_map[idx, 1:-1, 1:-1]
        return None

    def _validate_geometry_against_shape(self, shape: Tuple[int, int], geometry: GridGeometry) -> None:
        expected_shape = geometry.shape
        if shape != expected_shape:
            raise ValueError(
                f"Grid shape mismatch: expected {expected_shape}, received {shape}."
            )
        if not math.isclose(float(geometry.resolution), float(self.resolution), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"Resolution mismatch: map uses {self.resolution}, incoming grid uses {geometry.resolution}."
            )

    def _compute_overlap_indices(
        self, incoming_shape: Tuple[int, int], geometry: GridGeometry
    ) -> Optional[Dict[str, Tuple[slice, slice]]]:
        map_length = (self.cell_n - 2) * self.resolution
        center_cpu = np.asarray(cp.asnumpy(self.center))
        map_min_x = center_cpu[0] - map_length / 2.0
        map_max_x = center_cpu[0] + map_length / 2.0
        map_min_y = center_cpu[1] - map_length / 2.0
        map_max_y = center_cpu[1] + map_length / 2.0

        patch_min_x, patch_max_x = geometry.bounds_x
        patch_min_y, patch_max_y = geometry.bounds_y

        overlap_min_x = max(map_min_x, patch_min_x)
        overlap_max_x = min(map_max_x, patch_max_x)
        overlap_min_y = max(map_min_y, patch_min_y)
        overlap_max_y = min(map_max_y, patch_max_y)

        if overlap_max_x <= overlap_min_x or overlap_max_y <= overlap_min_y:
            return None

        map_width = self.cell_n - 2
        patch_rows, patch_cols = incoming_shape

        map_origin_x = map_min_x
        map_origin_y = map_min_y
        patch_origin_x = patch_min_x
        patch_origin_y = patch_min_y

        map_start_x = int(np.clip(np.floor((overlap_min_x - map_origin_x) / self.resolution), 0, map_width))
        map_end_x = int(
            np.clip(np.ceil((overlap_max_x - map_origin_x) / self.resolution), 0, map_width)
        )
        map_start_y = int(np.clip(np.floor((overlap_min_y - map_origin_y) / self.resolution), 0, map_width))
        map_end_y = int(
            np.clip(np.ceil((overlap_max_y - map_origin_y) / self.resolution), 0, map_width)
        )

        patch_start_x = int(
            np.clip(np.floor((overlap_min_x - patch_origin_x) / geometry.resolution), 0, patch_cols)
        )
        patch_end_x = int(
            np.clip(np.ceil((overlap_max_x - patch_origin_x) / geometry.resolution), 0, patch_cols)
        )
        patch_start_y = int(
            np.clip(np.floor((overlap_min_y - patch_origin_y) / geometry.resolution), 0, patch_rows)
        )
        patch_end_y = int(
            np.clip(np.ceil((overlap_max_y - patch_origin_y) / geometry.resolution), 0, patch_rows)
        )

        width = min(map_end_x - map_start_x, patch_end_x - patch_start_x)
        height = min(map_end_y - map_start_y, patch_end_y - patch_start_y)

        if width <= 0 or height <= 0:
            return None

        return {
            "map": (slice(map_start_y, map_start_y + height), slice(map_start_x, map_start_x + width)),
            "patch": (
                slice(patch_start_y, patch_start_y + height),
                slice(patch_start_x, patch_start_x + width),
            ),
        }

    def _map_extent_from_slices(self, rows: slice, cols: slice) -> Dict[str, float]:
        map_length = (self.cell_n - 2) * self.resolution
        center_cpu = np.asarray(cp.asnumpy(self.center))
        map_min_x = center_cpu[0] - map_length / 2.0
        map_min_y = center_cpu[1] - map_length / 2.0
        x_min = map_min_x + (cols.start + 0.5) * self.resolution
        x_max = map_min_x + (cols.stop - 0.5) * self.resolution
        y_min = map_min_y + (rows.start + 0.5) * self.resolution
        y_max = map_min_y + (rows.stop - 0.5) * self.resolution
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def _map_extent_from_mask(self, rows: slice, cols: slice, valid_mask: np.ndarray) -> Optional[Dict[str, float]]:
        """Compute extent based on the actual mask footprint; returns None if mask is empty."""
        if valid_mask is None or not np.any(valid_mask):
            return None
        row_idx, col_idx = np.nonzero(valid_mask)
        row_min = rows.start + int(row_idx.min())
        row_max = rows.start + int(row_idx.max())
        col_min = cols.start + int(col_idx.min())
        col_max = cols.start + int(col_idx.max())

        map_length = (self.cell_n - 2) * self.resolution
        center_cpu = np.asarray(cp.asnumpy(self.center))
        map_min_x = center_cpu[0] - map_length / 2.0
        map_min_y = center_cpu[1] - map_length / 2.0

        x_min = map_min_x + (col_min + 0.5) * self.resolution
        x_max = map_min_x + (col_max + 0.5) * self.resolution
        y_min = map_min_y + (row_min + 0.5) * self.resolution
        y_max = map_min_y + (row_max + 0.5) * self.resolution
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def _invalidate_caches(self, reset_plugins: bool = True):
        self.traversability_buffer[...] = cp.nan
        if reset_plugins:
            self.plugin_manager.reset_layers()


if __name__ == "__main__":
    #  Test script for profiling.
    #  $ python -m cProfile -o profile.stats elevation_mapping.py
    #  $ snakeviz profile.stats
    xp.random.seed(123)
    R = xp.random.rand(3, 3)
    t = xp.random.rand(3)
    print(R, t)
    param = Parameter(
        use_chainer=False,
        weight_file="../config/weights.dat",
        plugin_config_file="../config/plugin_config.yaml",
    )
    param.additional_layers = ["rgb", "grass", "tree", "people"]
    param.fusion_algorithms = ["color", "class_bayesian", "class_bayesian", "class_bayesian"]
    param.update()
    elevation = ElevationMap(param)
    layers = [
        "elevation",
        "variance",
        "traversability",
        "min_filter",
        "smooth",
        "inpaint",
        "rgb",
    ]
    points = xp.random.rand(100000, len(layers))

    channels = ["x", "y", "z"] + param.additional_layers
    print(channels)
    data = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
    for i in range(50):
        elevation.input_pointcloud(points, channels, R, t, 0, 0)
        elevation.update_normal(elevation.elevation_map[0])
        pos = np.array([i * 0.01, i * 0.02, i * 0.01])
        elevation.move_to(pos, R)
        for layer in layers:
            elevation.get_map_with_name_ref(layer, data)
        print(i)
        polygon = cp.array([[0, 0], [2, 0], [0, 2]], dtype=param.data_type)
        result = np.array([0, 0, 0])
        elevation.get_polygon_traversability(polygon, result)
        print(result)
