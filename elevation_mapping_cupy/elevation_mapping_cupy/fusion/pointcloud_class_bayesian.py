#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def alpha_kernel(
    resolution, width, height,
):
    # input the list of layers, amount of channels can slo be input through kernel
    alpha_kernel = cp.ElementwiseKernel(
        in_params="raw U p, raw W pcl_chan, raw W map_lay, raw W pcl_channels",
        out_params="raw U newmap",
        preamble=string.Template(
            """
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }
            """
        ).substitute(resolution=resolution, width=width, height=height),
        operation=string.Template(
            """
            U id = floorf(i/pcl_channels[1]);
            int layer = i % pcl_channels[1];
            U idx = p[id * pcl_channels[0]];
            U valid = p[id * pcl_channels[0] + 1];
            U inside = p[id * pcl_channels[0] + 2];
            if (valid) {
                if (inside) {
                    U theta_max = 0;
                    W arg_max = 0;
                    U theta = p[id * pcl_channels[0] + pcl_chan[layer]];
                        if (theta >=theta_max){
                            arg_max = map_lay[layer];
                            theta_max = theta;
                        }
                    atomicAdd(&newmap[get_map_idx(idx, arg_max)], theta_max);
                }
            }
            """
        ).substitute(),
        name="alpha_kernel",
    )
    return alpha_kernel


class ClassBayesian(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "pointcloud_class_bayesian"
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.alpha_kernel = alpha_kernel(self.resolution, self.cell_n, self.cell_n,)

    def __call__(self, points_all, R, t, pcl_ids, layer_ids, elevation_map, semantic_map, new_map, *args):
        self.alpha_kernel(
            points_all,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            new_map,
            size=(points_all.shape[0]),
        )
        # calculate new thetas
        sum_alpha = cp.sum(new_map[layer_ids], axis=0)
        # do not divide by zero
        sum_alpha[sum_alpha == 0] = 1
        semantic_map[layer_ids] = new_map[layer_ids] / cp.expand_dims(sum_alpha, axis=0)
