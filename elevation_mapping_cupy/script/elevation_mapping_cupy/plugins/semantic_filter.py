#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticFilter(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        classes: list = ["person", "grass"],
        **kwargs,
    ):
        """This is a filter to create a one hot encoded map of the class probabilities.

        Args:
            cell_n (int): width and height of the elevation map.
            classes (ruamel.yaml.comments.CommentedSeq):
            **kwargs ():
        """
        super().__init__()
        self.indices = []
        self.classes = classes
        self.color_encoding = self.transform_color()

    def color_map(self, N=256, normalized=False):
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
        return cmap[1:]

    def transform_color(self):
        color_classes = self.color_map(255)
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
        layer_indices = cp.array([], dtype=cp.int32)
        max_idcs = cp.array([], dtype=cp.int32)
        for it, fusion_alg in enumerate(semantic_map.param.fusion_algorithms):
            if fusion_alg in ["class_bayesian", "class_average"]:
                layer_indices = cp.append(layer_indices, it).astype(cp.int32)
            # we care only for the first max in the display
            if fusion_alg in ["class_max"] and len(max_idcs) < 1:
                max_idcs = cp.append(max_idcs, it).astype(cp.int32)

        # check which has the highest value
        # todo we are using the new_map because of the bayesian
        if len(layer_indices) > 0:
            class_map = cp.amax(semantic_map.semantic_map[layer_indices], axis=0)
            class_map_id = cp.argmax(semantic_map.semantic_map[layer_indices], axis=0)
        else:
            class_map = cp.zeros_like(semantic_map.semantic_map[0])
            class_map_id = cp.zeros_like(semantic_map.semantic_map[0], dtype=cp.int32)

        if "class_max" in semantic_map.param.fusion_algorithms:
            max_map = cp.amax(semantic_map.semantic_map[max_idcs], axis=0)
            max_map_id = semantic_map.elements_to_shift["id_max"][max_idcs]
            map = cp.where(max_map > class_map, max_map_id, class_map_id)
        else:
            map = class_map_id
        # create color coding
        enc = self.color_encoding[map]
        return enc
