#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase

class InpaintTravFilter(PluginBase):
    """Applies a maximum filter to the input layers and updates the traversability map.
    This can be used to enhance navigation by identifying traversable areas.

    Args:
        input_layer_name (str): The name of the input layer to be processed. Default is "traversability".
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_layer_name="traversability",
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        
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
        trav_idx = layer_names.index(self.input_layer_name)
        trav = elevation_map[trav_idx]
        valid_idx = layer_names.index("is_valid")
        valid = elevation_map[valid_idx]
        new_trav = cp.where(valid > 0.5, trav, 0.0)
        return new_trav
