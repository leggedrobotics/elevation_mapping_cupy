#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage

from .plugin_manager import PluginBase


class SmoothFilter(PluginBase):
    """
    SmoothFilter is a class that applies a smoothing filter
    to the elevation map. The filter is applied to the layer specified by the input_layer_name parameter.
    If the specified layer is not found, the filter is applied to the elevation layer.

    Args:
        cell_n (int): The width and height of the elevation map. Default is 100.
        input_layer_name (str): The name of the layer to which the filter should be applied. Default is "elevation".
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, cell_n: int = 100, input_layer_name: str = "elevation", **kwargs):
        super().__init__()
        self.input_layer_name = input_layer_name

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        if self.input_layer_name in layer_names:
            idx = layer_names.index(self.input_layer_name)
            h = elevation_map[idx]
        elif self.input_layer_name in plugin_layer_names:
            idx = plugin_layer_names.index(self.input_layer_name)
            h = plugin_layers[idx]
        else:
            print("layer name {} was not found. Using elevation layer.".format(self.input_layer_name))
            h = elevation_map[0]
        hs1 = ndimage.uniform_filter(h, size=3)
        hs1 = ndimage.uniform_filter(hs1, size=3)
        return hs1
