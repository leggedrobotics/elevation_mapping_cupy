#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def latest_correspondences_to_map_kernel(resolution, width, height, alpha):
    latest_correspondences_to_map_kernel = cp.ElementwiseKernel(
        in_params="raw U sem_map, raw U map_idx, raw U image_mono, raw U uv_correspondence, raw B valid_correspondence, raw U image_height, raw U image_width",
        out_params="raw U new_sem_map",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            int cell_idx = get_map_idx(i, 0);
            if (valid_correspondence[cell_idx]){
                int cell_idx_2 = get_map_idx(i, 1);
                int idx = int(uv_correspondence[cell_idx]) + int(uv_correspondence[cell_idx_2]) * image_width; 
                new_sem_map[get_map_idx(i, map_idx)] = image_mono[idx];
            }else{
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)];
            }

            """
        ).substitute(alpha=alpha),
        name="latest_correspondences_to_map_kernel",
    )
    return latest_correspondences_to_map_kernel


class ImageLatest(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "image_latest"
        self.cell_n = params.cell_n
        self.resolution = params.resolution

        self.latest_correspondences_to_map_kernel = latest_correspondences_to_map_kernel(
            resolution=self.resolution, width=self.cell_n, height=self.cell_n, alpha=0.7,
        )

    def __call__(
        self,
        sem_map_idx,
        image,
        j,
        uv_correspondence,
        valid_correspondence,
        image_height,
        image_width,
        semantic_map,
        new_map,
    ):
        self.latest_correspondences_to_map_kernel(
            semantic_map,
            sem_map_idx,
            image[j],
            uv_correspondence,
            valid_correspondence,
            image_height,
            image_width,
            new_map,
            size=int(self.cell_n * self.cell_n),
        )
        semantic_map[sem_map_idx] = new_map[sem_map_idx]
