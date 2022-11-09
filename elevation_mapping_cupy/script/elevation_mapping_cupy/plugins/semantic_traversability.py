#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticTraversability(PluginBase):
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
        semantic_map,
        *args,
    ) -> cp.ndarray:
        # get indices of all layers that
        map = cp.zeros(elevation_map[2].shape, np.float32)
        tempo = cp.zeros(elevation_map[2].shape, np.float32)
        for it, name in enumerate(self.layers):
            if name in layer_names:
                idx = layer_names.index(name)
                tempo = elevation_map[idx]
            elif name in semantic_map.param.additional_layers:
                idx = semantic_map.param.additional_layers.index(name)
                tempo = semantic_map.map[idx]
            elif name in plugin_layer_names:
                idx = plugin_layer_names.index(name)
                tempo = plugin_layers[idx]
            else:
                print(
                    "Layer {} is not in the map, returning traversabiltiy!".format(name)
                )
                return
            if self.type[it] == "traversability":
                tempo = cp.where(tempo <= self.thresholds[it], 1, 0)
                map += tempo
            else:
                tempo = cp.where(tempo >= self.thresholds[it], 1, 0)
                map += tempo
        map = cp.where(map <= 0.9, 0.1,1)

        return map
