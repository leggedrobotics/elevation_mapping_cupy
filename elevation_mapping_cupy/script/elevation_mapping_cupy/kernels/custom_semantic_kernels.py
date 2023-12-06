#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string


def sum_kernel(
    resolution, width, height,
):
    """Sums the semantic values of the classes for the exponentiala verage or for the average.

    Args:
        resolution:
        width:
        height:

    Returns:

    """
    # input the list of layers, amount of channels can slo be input through kernel
    sum_kernel = cp.ElementwiseKernel(
        in_params="raw U p, raw U R, raw U t, raw W pcl_chan, raw W map_lay, raw W pcl_channels",
        out_params="raw U map, raw U newmap",
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
                    atomicAdd(&newmap[get_map_idx(idx, map_lay[layer])], feat);
                }
            }
            """
        ).substitute(),
        name="sum_kernel",
    )
    return sum_kernel


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


def sum_max_kernel(
    resolution, width, height,
):
    # input the list of layers, amount of channels can slo be input through kernel
    sum_max_kernel = cp.ElementwiseKernel(
        in_params="raw U p, raw U max_pt, raw T max_id, raw W pcl_chan, raw W map_lay, raw W pcl_channels",
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
            U idx = p[i * pcl_channels[0]];
            U valid = p[i * pcl_channels[0] + 1];
            U inside = p[i * pcl_channels[0] + 2];
            if (valid) {
                if (inside) {
                    // for every max value
                    for ( W it=0;it<pcl_channels[2];it++){
                        U prob = max_pt[i * pcl_channels[2] + it];
                        T id = max_id[i * pcl_channels[2] + it];
                        atomicAdd(&newmap[get_map_idx(idx, id)], prob);
                    }
                }
            }
            """
        ).substitute(),
        name="sum_max_kernel",
    )
    return sum_max_kernel


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


def average_kernel(
    width, height,
):
    average_kernel = cp.ElementwiseKernel(
        in_params="raw V newmap, raw W pcl_chan, raw W map_lay, raw W pcl_channels, raw U new_elmap",
        out_params="raw U map",
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
                U feat = newmap[get_map_idx(id,  map_lay[layer])]/(1*cnt);
                map[get_map_idx(id,  map_lay[layer])] = feat;
            }
            """
        ).substitute(),
        name="average_map_kernel",
    )
    return average_kernel


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


def class_average_kernel(
    width, height, alpha,
):
    class_average_kernel = cp.ElementwiseKernel(
        in_params="raw V newmap, raw W pcl_chan, raw W map_lay, raw W pcl_channels, raw U new_elmap",
        out_params="raw U map",
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
                U prev_val = map[get_map_idx(id,  map_lay[layer])];
                if (prev_val==0){
                    U val = newmap[get_map_idx(id, map_lay[layer])]/(1*cnt);
                    map[get_map_idx(id,  map_lay[layer])] = val;
                }
                else{
                    U val = ${alpha} *prev_val + (1-${alpha}) * newmap[get_map_idx(id, map_lay[layer])]/(cnt);
                    map[get_map_idx(id,  map_lay[layer])] = val;
                }
            }
            """
        ).substitute(alpha=alpha,),
        name="class_average_kernel",
    )
    return class_average_kernel


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
