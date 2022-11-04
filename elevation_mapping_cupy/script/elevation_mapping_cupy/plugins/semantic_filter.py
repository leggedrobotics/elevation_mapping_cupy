#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticFilter(PluginBase):
    """This is a filter to create colors

    ...

    Attributes
    ----------
    cell_n: int
        width and height of the elevation map.
    """

    def __init__(
        self,
        cell_n: int = 100,
        classes: list = ["person", "grass"],
        colors: list = [
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 0],
        ],
        **kwargs,
    ):
        super().__init__()
        self.indices = []
        self.classes = classes
        color_classes = np.array(colors)
        self.color_encoding = self.transform_color(color_classes)

    def transform_color(self, color_classes):
        r = np.asarray(color_classes[:, 0], dtype=np.uint32)
        g = np.asarray(color_classes[:, 1], dtype=np.uint32)
        b = np.asarray(color_classes[:, 2], dtype=np.uint32)
        rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
        rgb_arr.dtype = np.float32
        return cp.asarray(rgb_arr)

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map,
        **kwargs,
    ) -> cp.ndarray:
        # get indices of all layers that
        layer_indices = cp.array([], dtype=cp.int32)
        for it, fusion_alg in enumerate(semantic_map.param.fusion_algorithms):
            if fusion_alg == "class_average" and (
                semantic_map.param.additional_layers[it] in self.classes
            ):
                layer_indices = cp.append(layer_indices, it).astype(cp.int32)

        # check which has the highest value
        class_map = cp.argmax(semantic_map.map[layer_indices], axis=0)
        # create color coding
        enc = self.color_encoding[class_map]
        return enc
