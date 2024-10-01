#
# Copyright (c) 2024, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cv2 as cv
import cupy as cp
import numpy as np

from typing import List

from .plugin_manager import PluginBase


class Erosion(PluginBase):
    """
    This class is used for applying erosion to an elevation map or specific layers within it.
    Erosion is a morphological operation that is used to remove small-scale details from a binary image.

    Args:
        kernel_size (int): Size of the erosion kernel. Default is 3, which means a 3x3 square kernel.
        iterations (int): Number of times erosion is applied. Default is 1.
        **kwargs (): Additional keyword arguments.
    """

    def __init__(
        self,
        input_layer_name="traversability",
        kernel_size: int = 3,
        iterations: int = 1,
        reverse: bool = False,
        default_layer_name: str = "traversability",
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.reverse = reverse
        self.default_layer_name = default_layer_name

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """
        Applies erosion to the given elevation map.

        Args:
            elevation_map (cupy._core.core.ndarray): The elevation map to be eroded.
            layer_names (List[str]): Names of the layers in the elevation map.
            plugin_layers (cupy._core.core.ndarray): Layers provided by other plugins.
            plugin_layer_names (List[str]): Names of the layers provided by other plugins.
            *args (): Additional arguments.

        Returns:
            cupy._core.core.ndarray: The eroded elevation map.
        """
        # Convert the elevation map to a format suitable for erosion (if necessary)
        layer_data = self.get_layer_data(
            elevation_map,
            layer_names,
            plugin_layers,
            plugin_layer_names,
            semantic_map,
            semantic_layer_names,
            self.input_layer_name,
        )
        if layer_data is None:
            print(f"No layers are found, using {self.default_layer_name}!")
            layer_data = self.get_layer_data(
                elevation_map,
                layer_names,
                plugin_layers,
                plugin_layer_names,
                semantic_map,
                semantic_layer_names,
                self.default_layer_name,
            )
            if layer_data is None:
                print(f"No layers are found, using traversability!")
                layer_data = self.get_layer_data(
                    elevation_map,
                    layer_names,
                    plugin_layers,
                    plugin_layer_names,
                    semantic_map,
                    semantic_layer_names,
                    "traversability",
                )
        layer_np = cp.asnumpy(layer_data)

        # Define the erosion kernel
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        if self.reverse:
            layer_np = 1 - layer_np
        # Apply erosion
        layer_min = float(layer_np.min())
        layer_max = float(layer_np.max())
        layer_np_normalized = ((layer_np - layer_min) * 255 / (layer_max - layer_min)).astype("uint8")
        eroded_map_np = cv.erode(layer_np_normalized, kernel, iterations=self.iterations)
        eroded_map_np = eroded_map_np.astype(np.float32) * (layer_max - layer_min) / 255 + layer_min
        if self.reverse:
            eroded_map_np = 1 - eroded_map_np

        # Convert back to cupy array and return
        return cp.asarray(eroded_map_np)
