#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import logging
from typing import List

import cupy as cp
import cv2 as cv
import numpy as np

from .plugin_manager import PluginBase

_LOGGER = logging.getLogger(__name__)


class Inpainting(PluginBase):
    """
    This class is used for inpainting, a process of reconstructing lost or deteriorated parts of images and videos.

    Args:
        cell_n (int): The number of cells. Default is 100.
        method (str): The inpainting method. Options are 'telea' or 'ns' (Navier-Stokes). Default is 'telea'.
        **kwargs (): Additional keyword arguments.
    """

    def __init__(
        self,
        cell_n: int = 100,
        method: str = "telea",
        input_layer_name: str = "elevation",
        max_hole_area: int = 64,
        fill_border_holes: bool = False,
        inpaint_radius: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        if method == "telea":
            self.method = cv.INPAINT_TELEA
        elif method == "ns":  # Navier-Stokes
            self.method = cv.INPAINT_NS
        else:  # default method
            self.method = cv.INPAINT_TELEA
        self.max_hole_area = None if int(max_hole_area) <= 0 else int(max_hole_area)
        self.fill_border_holes = bool(fill_border_holes)
        self.inpaint_radius = float(inpaint_radius)

    def _select_holes_to_fill(self, invalid_mask: np.ndarray):
        """Label the bounded invalid components worth filling.

        Returns the component labels (0 where nothing is filled) and the
        (label, left, top, width, height) box of every selected component.
        """
        height, width = invalid_mask.shape
        if self.max_hole_area is None and self.fill_border_holes:
            return invalid_mask.astype(np.int32), [(1, 0, 0, width, height)]

        label_count, labels, stats, _ = cv.connectedComponentsWithStats(invalid_mask, connectivity=4)
        left, top, component_width, component_height, area = stats.T
        keep = np.ones(label_count, dtype=bool)
        keep[0] = False  # label 0 is the valid background
        if self.max_hole_area is not None:
            keep &= area <= self.max_hole_area
        if not self.fill_border_holes:
            keep &= (left > 0) & (top > 0) & (left + component_width < width) & (top + component_height < height)
        kept = np.flatnonzero(keep)
        fill_labels = np.where(keep[labels], labels, 0).astype(np.int32)
        return fill_labels, [(int(i), *(int(v) for v in stats[i, :4])) for i in kept]

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        valid_layer = elevation_map[2]
        if self.input_layer_name in layer_names:
            elevation = elevation_map[layer_names.index(self.input_layer_name)]
        elif self.input_layer_name in plugin_layer_names:
            elevation = plugin_layers[plugin_layer_names.index(self.input_layer_name)]
        else:
            raise ValueError(f"Inpainting could not find layer '{self.input_layer_name}'")

        finite_elevation = cp.isfinite(elevation)
        valid_mask = cp.logical_and(valid_layer > 0.5, finite_elevation)
        output = cp.full(elevation.shape, cp.nan, dtype=cp.float32)
        output = cp.where(valid_mask, elevation, output)

        if not cp.any(valid_mask):
            return output.astype(cp.float64)

        invalid_mask_np = cp.asnumpy(cp.logical_not(valid_mask).astype(cp.uint8))
        if not invalid_mask_np.any():
            return elevation.astype(cp.float64)

        fill_labels, holes = self._select_holes_to_fill(invalid_mask_np)
        if not holes:
            return output.astype(cp.float64)

        h_valid = elevation[valid_mask]
        h_max = float(cp.asnumpy(h_valid.max()))
        h_min = float(cp.asnumpy(h_valid.min()))
        denom = h_max - h_min
        fill_mask = cp.asarray(fill_labels > 0)

        if denom <= 1e-6:
            _LOGGER.warning(
                "Inpainting detected near-flat terrain (h_min=%.3f, h_max=%.3f); filling only bounded holes.",
                h_min,
                h_max,
            )
            output = cp.where(fill_mask, h_max, output)
            return output.astype(cp.float64)

        # Keep the full invalid mask when running OpenCV so large unknown regions do not
        # contribute placeholder values to nearby hole filling. Only bounded components are
        # copied back into the published layer. cv.inpaint fills every masked pixel, and a
        # survey-sized map is mostly unknown, so inpaint each hole in a window that only
        # adds the inpaint radius around it.
        safe_elevation = cp.where(valid_mask, elevation, h_min)
        scaled = cp.asnumpy(cp.clip((safe_elevation - h_min) * 255.0 / denom, 0.0, 255.0)).astype("uint8")
        dst = np.zeros_like(scaled)
        margin = int(np.ceil(self.inpaint_radius)) + 1
        height, width = scaled.shape
        for label, left, top, hole_width, hole_height in holes:
            y0, y1 = max(top - margin, 0), min(top + hole_height + margin, height)
            x0, x1 = max(left - margin, 0), min(left + hole_width + margin, width)
            window = cv.inpaint(scaled[y0:y1, x0:x1], invalid_mask_np[y0:y1, x0:x1], self.inpaint_radius, self.method)
            own = fill_labels[y0:y1, x0:x1] == label
            dst[y0:y1, x0:x1][own] = window[own]
        h_inpainted = cp.asarray(dst.astype(np.float32) * denom / 255.0 + h_min, dtype=cp.float32)
        output = cp.where(fill_mask, h_inpainted, output)
        return output.astype(cp.float64)
