#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List
from .plugin_manager import PluginBase


class RobotCentricElevation(PluginBase):
    """Generates an elevation map with respect to the robot frame.

    Args:
        cell_n (int):
        resolution (ruamel.yaml.scalarfloat.ScalarFloat):
        threshold (ruamel.yaml.scalarfloat.ScalarFloat):
        use_threshold (bool):
        **kwargs ():
    """

    def __init__(
        self, cell_n: int = 100, resolution: float = 0.05, threshold: float = 0.4, use_threshold: bool = 0, **kwargs
    ):
        super().__init__()
        self.width = cell_n
        self.height = cell_n
        self.min_filtered = cp.zeros((self.width, self.height), dtype=cp.float32)

        self.base_elevation_kernel = cp.ElementwiseKernel(
            in_params="raw U map, raw U mask, raw U R",
            out_params="raw U newmap",
            preamble=string.Template(
                """
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }

                __device__ bool is_inside(int idx) {
                    int idx_x = idx / ${width};
                    int idx_y = idx % ${width};
                    if (idx_x <= 0 || idx_x >= ${width} - 1) {
                        return false;
                    }
                    if (idx_y <= 0 || idx_y >= ${height} - 1) {
                        return false;
                    }
                    return true;
                }
                __device__ float get_map_x(int idx){
                    float idx_x = idx / ${width}* ${resolution};
                    return idx_x;
                }
                __device__ float get_map_y(int idx){
                    float idx_y = idx % ${width}* ${resolution};
                    return idx_y;
                }
                __device__ float transform_p(float x, float y, float z,
                                     float r0, float r1, float r2) {
                    return r0 * x + r1 * y + r2 * z ;
                }
                """
            ).substitute(width=self.width, height=self.height, resolution=resolution),
            operation=string.Template(
                """
                U rz = map[get_map_idx(i, 0)];
                U valid = mask[get_map_idx(i, 0)];
                if (valid > 0.5) {
                    U rx = get_map_x(get_map_idx(i, 0));
                    U ry = get_map_y(get_map_idx(i, 0));
                    U x_b = transform_p(rx, ry, rz, R[0], R[1], R[2]);
                    U y_b = transform_p(rx, ry, rz, R[3], R[4], R[5]);
                    U z_b = transform_p(rx, ry, rz, R[6], R[7], R[8]);
                    if (${use_threshold} && z_b>= ${threshold} ) {
                        newmap[get_map_idx(i, 0)] = 1.0;
                    }
                    else if (${use_threshold} && z_b< ${threshold} ){
                        newmap[get_map_idx(i, 0)] = 0.0;
                    }
                    else{
                        newmap[get_map_idx(i, 0)] = z_b;
                    }
                }
                """
            ).substitute(threshold=threshold, use_threshold=int(use_threshold)),
            name="base_elevation_kernel",
        )

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        rotation,
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_map (elevation_mapping_cupy.semantic_map.SemanticMap):
            rotation (cupy._core.core.ndarray):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        # Process maps here
        # check that transform is a ndarray
        self.min_filtered = elevation_map[0].copy()
        self.base_elevation_kernel(
            elevation_map[0], elevation_map[2], rotation, self.min_filtered, size=(self.width * self.height),
        )
        return self.min_filtered
