#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticTraversability(PluginBase):
    """Extracts traversability and elevations from layers and generates an updated traversability that can be used by checker.

    Args:
        cell_n (int): The width and height of the elevation map.
        layers (list): List of layers for semantic traversability. Default is ["traversability"].
        thresholds (list): List of thresholds for each layer. Default is [0.5].
        type (list): List of types for each layer. Default is ["traversability"].
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        cell_n: int = 100,
        layers: list = ["traversability"],
        thresholds: list = [0.5],
        type: list = ["traversability"],
        **kwargs,
    ):
        super().__init__()
        self.layers = layers
        self.thresholds = cp.asarray(thresholds)
        self.type = type

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
        # get indices of all layers that
        map = cp.zeros(elevation_map[2].shape, np.float32)
        tempo = cp.zeros(elevation_map[2].shape, np.float32)
        for it, name in enumerate(self.layers):
            if name in layer_names:
                idx = layer_names.index(name)
                tempo = elevation_map[idx]
            # elif name in semantic_params.additional_layers:
            #     idx = semantic_params.additional_layers.index(name)
            #     tempo = semantic_map[idx]
            elif name in plugin_layer_names:
                idx = plugin_layer_names.index(name)
                tempo = plugin_layers[idx]
            else:
                print("Layer {} is not in the map, returning traversabiltiy!".format(name))
                return
            if self.type[it] == "traversability":
                tempo = cp.where(tempo <= self.thresholds[it], 1, 0)
                map += tempo
            else:
                tempo = cp.where(tempo >= self.thresholds[it], 1, 0)
                map += tempo
        map = cp.where(map <= 0.9, 0.1, 1)

        return map
