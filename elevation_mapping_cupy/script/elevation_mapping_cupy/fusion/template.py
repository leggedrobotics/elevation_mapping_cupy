import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase





class Color(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        print("Initialize fusion kernel")
        self.cell_n = params.cell_n
        self.resolution = params.resolution


    def __call__(self, points_all, R, t, pcl_ids, layer_ids, elevation_map,semantic_map, new_map,*args):
        pass


