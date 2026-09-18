#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def sum_compact_kernel(
    resolution, width, height,
):
    # input the list of layers, amount of channels can slo be input through kernel
    sum_compact_kernel = cp.ElementwiseKernel(
        in_params="raw U p, raw U R, raw U t, raw W pcl_chan, raw W map_lay, raw W pcl_channels",
        out_params=" raw U newmap",
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
                    U feat = p[id * pcl_channels[0] + pcl_chan[layer]];
                    atomicAdd(&newmap[get_map_idx(idx, layer)], feat);
                }
            }
            """
        ).substitute(),
        name="sum_compact_kernel",
    )
    return sum_compact_kernel


def bayesian_inference_kernel(
    width, height,
):
    bayesian_inference_kernel = cp.ElementwiseKernel(
        in_params=" raw W pcl_chan, raw W map_lay, raw W pcl_channels, raw U new_elmap",
        out_params="raw U newmap, raw U sum_mean, raw U map",
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
            U id = floorf(i/pcl_channels[1]);
            int layer = i % pcl_channels[1];
            U cnt = new_elmap[get_map_idx(id, 2)];
            if (cnt>0){
                    U feat_ml = sum_mean[get_map_idx(id,  layer)]/cnt;
                    U feat_old = map[get_map_idx(id,  map_lay[layer])];
                    U sigma_old = newmap[get_map_idx(id,  map_lay[layer])];
                    U sigma = 1.0;
                    U feat_new = sigma*feat_old /(cnt*sigma_old + sigma) +cnt*sigma_old *feat_ml /(cnt*sigma_old+sigma);
                    U sigma_new = sigma*sigma_old /(cnt*sigma_old +sigma);
                    map[get_map_idx(id,  map_lay[layer])] = feat_new;
                    newmap[get_map_idx(id,  map_lay[layer])] = sigma_new;
            }
            """
        ).substitute(),
        name="bayesian_inference_kernel",
    )
    return bayesian_inference_kernel


class BayesianInference(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize bayesian inference kernel")
        self.name = "pointcloud_bayesian_inference"

        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.fusion_algorithms = params.fusion_algorithms
        self.data_type = params.data_type

        self.sum_mean = cp.ones(
            (self.fusion_algorithms.count("bayesian_inference"), self.cell_n, self.cell_n,), self.data_type,
        )
        # TODO initialize the variance with a value different than 0
        self.sum_compact_kernel = sum_compact_kernel(self.resolution, self.cell_n, self.cell_n,)
        self.bayesian_inference_kernel = bayesian_inference_kernel(self.cell_n, self.cell_n,)

    def __call__(self, points_all, R, t, pcl_ids, layer_ids, elevation_map, semantic_map, new_map, *args):
        self.sum_mean *= 0
        self.sum_compact_kernel(
            points_all,
            R,
            t,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            self.sum_mean,
            size=(points_all.shape[0]),
        )
        self.bayesian_inference_kernel(
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            elevation_map,
            new_map,
            self.sum_mean,
            semantic_map,
            size=(self.cell_n * self.cell_n),
        )
