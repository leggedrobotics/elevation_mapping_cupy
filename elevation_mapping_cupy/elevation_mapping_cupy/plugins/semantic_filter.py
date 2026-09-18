#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List
import re

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticFilter(PluginBase):
    """This is a filter to create a one hot encoded map of the class probabilities.

    Args:
        cell_n (int): width and height of the elevation map.
        classes (list): List of classes for semantic filtering. Default is ["person", "grass"].
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self, cell_n: int = 100, classes: list = ["person", "grass"], **kwargs,
    ):
        super().__init__()
        self.indices = []
        self.classes = classes
        self.color_encoding = self.transform_color()

    def color_map(self, N: int = 256, normalized: bool = False):
        """
        Creates a color map with N different colors.

        Args:
            N (int, optional): The number of colors in the map. Defaults to 256.
            normalized (bool, optional): If True, the colors are normalized to the range [0,1]. Defaults to False.

        Returns:
            np.ndarray: The color map.
        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N + 1, 3), dtype=dtype)
        for i in range(N + 1):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        cmap[1] = np.array([81, 113, 162])
        cmap[2] = np.array([81, 113, 162])
        cmap[3] = np.array([188, 63, 59])
        return cmap[1:]

    def transform_color(self):
        """
        Transforms the color map into a format that can be used for semantic filtering.

        Returns:
            cp.ndarray: The transformed color map.
        """
        color_classes = self.color_map(255)
        r = np.asarray(color_classes[:, 0], dtype=np.uint32)
        g = np.asarray(color_classes[:, 1], dtype=np.uint32)
        b = np.asarray(color_classes[:, 2], dtype=np.uint32)
        rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
        rgb_arr.dtype = np.float32
        return cp.asarray(rgb_arr)

    def get_layer_indices(self, layer_names: List[str]) -> List[int]:
        """ Get the indices of the layers that are to be processed using regular expressions.
        Args:
            layer_names (List[str]): List of layer names.
        Returns:
            List[int]: List of layer indices.
        """
        indices = []
        for i, layer_name in enumerate(layer_names):
            if any(re.match(pattern, layer_name) for pattern in self.classes):
                indices.append(i)
        return indices

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        rotation,
        elements_to_shift,
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_map (elevation_mapping_cupy.semantic_map.SemanticMap):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        # get indices of all layers that contain semantic class information
        data = []
        for m, layer_names in zip(
            [elevation_map, plugin_layers, semantic_map], [layer_names, plugin_layer_names, semantic_layer_names]
        ):
            layer_indices = self.get_layer_indices(layer_names)
            if len(layer_indices) > 0:
                data.append(m[layer_indices])
        if len(data) > 0:
            data = cp.concatenate(data, axis=0)
            class_map = cp.amax(data, axis=0)
            class_map_id = cp.argmax(data, axis=0)
        else:
            class_map = cp.zeros_like(elevation_map[0])
            class_map_id = cp.zeros_like(elevation_map[0], dtype=cp.int32)
        enc = self.color_encoding[class_map_id]
        return enc
