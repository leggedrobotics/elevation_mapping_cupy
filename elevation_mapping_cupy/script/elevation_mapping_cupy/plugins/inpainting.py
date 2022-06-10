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


class Inpainting(PluginBase):
    """This is a filter to smoothen

    ...

    Attributes
    ----------
    cell_n: int
        width and height of the elevation map.
    """

    def __init__(self, cell_n: int = 100, method: str = "telea", **kwargs):
        super().__init__()
        if method == "telea":
            self.method = cv.INPAINT_TELEA
        elif method == "ns":  # Navier-Stokes
            self.method = cv.INPAINT_NS
        else:  # default method
            self.method = cv.INPAINT_TELEA

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
    ) -> cp.ndarray:
        mask = cp.asnumpy((elevation_map[2] < 0.5).astype("uint8"))
        if (mask < 1).any():
            h = elevation_map[0]
            h_max = float(h[mask < 1].max())
            h_min = float(h[mask < 1].min())
            h = cp.asnumpy((elevation_map[0] - h_min) * 255 / (h_max - h_min)).astype("uint8")
            dst = np.array(cv.inpaint(h, mask, 1, self.method))
            h_inpainted = dst.astype(np.float32) * (h_max - h_min) / 255 + h_min
            return cp.asarray(h_inpainted).astype(np.float64)
        else:
            return elevation_map[0]
