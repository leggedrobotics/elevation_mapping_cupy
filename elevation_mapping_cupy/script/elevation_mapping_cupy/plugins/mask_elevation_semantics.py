#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage
import numpy as np
import cv2 as cv

from .plugin_manager import PluginBase


class MaskElevationSemantics(PluginBase):
    def __init__(self, semantic_reference_layer_name: str, **kwargs):
        super().__init__()
        self.semantic_reference_layer_name = semantic_reference_layer_name

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
        
        
        ma = cp.zeros_like(elevation_map[0])
        ma[:,:] = np.nan
        
        try:
            idx = semantic_layer_names.index(self.semantic_reference_layer_name)
        except:
            return ma 

        m = ~cp.isnan(semantic_map[idx])
        ma[m] = elevation_map[0][m]
        return ma