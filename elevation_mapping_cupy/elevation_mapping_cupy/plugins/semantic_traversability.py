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
        combine (str): How per-layer votes are combined. "any" (upstream behaviour) marks a cell
            traversable if at least one layer votes for it; "all" requires every layer to vote.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        cell_n: int = 100,
        layers: list = ["traversability"],
        thresholds: list = [0.5],
        type: list = ["traversability"],
        combine: str = "any",
        **kwargs,
    ):
        super().__init__()
        self.layers = layers
        self.thresholds = cp.asarray(thresholds)
        self.type = type
        if combine not in ("any", "all"):
            raise ValueError(f"semantic_traversability: combine must be 'any' or 'all', got {combine!r}")
        self.combine = combine

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
            elif semantic_layer_names is not None and name in semantic_layer_names:
                # Semantic layers (e.g. the VLM-relabelled risk/steppable/preferred_area
                # channels) live in SemanticMap.semantic_map, indexed by its own layer_names.
                idx = semantic_layer_names.index(name)
                tempo = semantic_map[idx]
            elif name in plugin_layer_names:
                idx = plugin_layer_names.index(name)
                tempo = plugin_layers[idx]
            else:
                raise KeyError(
                    f"semantic_traversability: configured layer '{name}' is not available. "
                    f"elevation layers={layer_names}, semantic layers={semantic_layer_names}, "
                    f"plugin layers={plugin_layer_names}"
                )
            if self.type[it] == "traversability":
                tempo = cp.where(tempo <= self.thresholds[it], 1, 0)
                map += tempo
            else:
                tempo = cp.where(tempo >= self.thresholds[it], 1, 0)
                map += tempo
        required = len(self.layers) if self.combine == "all" else 1
        map = cp.where(map <= required - 0.1, 0.1, 1)

        return map
