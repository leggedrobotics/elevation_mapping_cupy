#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def add_color_kernel(
    width, height,
):
    add_color_kernel = cp.ElementwiseKernel(
        in_params="raw T p, raw U R, raw U t, raw W pcl_chan, raw W map_lay, raw W pcl_channels",
        out_params="raw V color_map",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ unsigned int get_r(unsigned int color){
                unsigned int red = 0xFF0000;
                unsigned int reds = (color & red) >> 16;
                return reds;
            }
            __device__ unsigned int get_g(unsigned int color){
                unsigned int green = 0xFF00;
                unsigned int greens = (color & green) >> 8;
                return greens;
            }
            __device__ unsigned int get_b(unsigned int color){
                unsigned int blue = 0xFF;
                unsigned int blues = ( color & blue);
                return blues;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            U id = floorf(i/pcl_channels[1]);
            int layer = i % pcl_channels[1];
            U idx = p[id * pcl_channels[0]];
            U valid = p[id * pcl_channels[0] + 1];
            U inside = p[id * pcl_channels[0] + 2];
            if (valid && inside){
                    unsigned int color = __float_as_uint(p[id * pcl_channels[0] + pcl_chan[layer]]);                    
                    atomicAdd(&color_map[get_map_idx(idx, layer*3)], get_r(color));
                    atomicAdd(&color_map[get_map_idx(idx, layer*3+1)], get_g(color));
                    atomicAdd(&color_map[get_map_idx(idx, layer*3 + 2)], get_b(color));
                    atomicAdd(&color_map[get_map_idx(idx, pcl_channels[1]*3)], 1);
            }
            """
        ).substitute(width=width),
        name="add_color_kernel",
    )
    return add_color_kernel


def color_average_kernel(
    width, height,
):
    color_average_kernel = cp.ElementwiseKernel(
        in_params="raw V color_map, raw W pcl_chan, raw W map_lay, raw W pcl_channels",
        out_params="raw U map",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ unsigned int get_r(unsigned int color){
                unsigned int red = 0xFF0000;
                unsigned int reds = (color & red) >> 16;
                return reds;
            }
            __device__ unsigned int get_g(unsigned int color){
                unsigned int green = 0xFF00;
                unsigned int greens = (color & green) >> 8;
                return greens;
            }
            __device__ unsigned int get_b(unsigned int color){
                unsigned int blue = 0xFF;
                unsigned int blues = (color & blue);
                return blues;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            U id = floorf(i/pcl_channels[1]);
            int layer = i % pcl_channels[1];
            unsigned int cnt = color_map[get_map_idx(id, pcl_channels[1]*3)];
            if (cnt>0){
                    // U prev_color = map[get_map_idx(id, map_lay[layer])];
                    unsigned int r = color_map[get_map_idx(id, layer*3)]/(1*cnt);
                    unsigned int g = color_map[get_map_idx(id, layer*3+1)]/(1*cnt);
                    unsigned int b = color_map[get_map_idx(id, layer*3+2)]/(1*cnt);
                    //if (prev_color>=0){
                    //    unsigned int prev_r = get_r(prev_color);
                    //    unsigned int prev_g = get_g(prev_color);
                    //    unsigned int prev_b = get_b(prev_color);
                    //    unsigned int r = prev_r/2 + color_map[get_map_idx(i, layer*3)]/(2*cnt);
                    //    unsigned int g = prev_g/2 + color_map[get_map_idx(i, layer*3+1)]/(2*cnt);
                    //    unsigned int b = prev_b/2 + color_map[get_map_idx(i, layer*3+2)]/(2*cnt);
                    //}
                    unsigned int rgb = (r<<16) + (g << 8) + b;
                    float rgb_ = __uint_as_float(rgb);
                    map[get_map_idx(id,  map_lay[layer])] = rgb_;
            }
            """
        ).substitute(),
        name="color_average_kernel",
    )
    return color_average_kernel


class Color(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "pointcloud_color"
        self.cell_n = params.cell_n
        self.resolution = params.resolution

        self.add_color_kernel = add_color_kernel(params.cell_n, params.cell_n,)
        self.color_average_kernel = color_average_kernel(self.cell_n, self.cell_n)

    def __call__(self, points_all, R, t, pcl_ids, layer_ids, elevation_map, semantic_map, new_map, *args):
        self.color_map = cp.zeros((1 + 3 * layer_ids.shape[0], self.cell_n, self.cell_n), dtype=cp.uint32,)

        points_all = points_all.astype(cp.float32)
        self.add_color_kernel(
            points_all,
            R,
            t,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            self.color_map,
            size=(points_all.shape[0]),
        )
        self.color_average_kernel(
            self.color_map,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            semantic_map,
            size=(self.cell_n * self.cell_n),
        )
