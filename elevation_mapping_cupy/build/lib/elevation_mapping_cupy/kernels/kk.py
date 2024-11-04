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



def map_utils(
    resolution,
    width,
    height,
    sensor_noise_factor,
    min_valid_distance,
    max_height_range,
    ramped_height_range_a,
    ramped_height_range_b,
    ramped_height_range_c,
):
    util_preamble = string.Template(
        """
        __device__ float16 clamp(float16 x, float16 min_x, float16 max_x) {

            return max(min(x, max_x), min_x);
        }
        __device__ int get_x_idx(float16 x, float16 center) {
            int i = (x - center) / ${resolution} + 0.5 * ${width};
            return i;
        }
        __device__ int get_y_idx(float16 y, float16 center) {
            int i = (y - center) / ${resolution} + 0.5 * ${height};
            return i;
        }
        __device__ bool is_inside(int idx) {
            int idx_x = idx / ${width};
            int idx_y = idx % ${width};
            if (idx_x == 0 || idx_x == ${width} - 1) {
                return false;
            }
            if (idx_y == 0 || idx_y == ${height} - 1) {
                return false;
            }
            return true;
        }
        __device__ int get_idx(float16 x, float16 y, float16 center_x, float16 center_y) {
            int idx_x = clamp(get_x_idx(x, center_x), 0, ${width} - 1);
            int idx_y = clamp(get_y_idx(y, center_y), 0, ${height} - 1);
            return ${width} * idx_x + idx_y;
        }
        __device__ int get_map_idx(int idx, int layer_n) {
            const int layer = ${width} * ${height};
            return layer * layer_n + idx;
        }
        __device__ float transform_p(float16 x, float16 y, float16 z,
                                     float16 r0, float16 r1, float16 r2, float16 t) {
            return r0 * x + r1 * y + r2 * z + t;
        }
        __device__ float z_noise(float16 z){
            return ${sensor_noise_factor} * z * z;
        }

        __device__ float point_sensor_distance(float16 x, float16 y, float16 z,
                                               float16 sx, float16 sy, float16 sz) {
            float d = (x - sx) * (x - sx) + (y - sy) * (y - sy) + (z - sz) * (z - sz);
            return d;
        }

        __device__ bool is_valid(float16 x, float16 y, float16 z,
                               float16 sx, float16 sy, float16 sz) {
            float d = point_sensor_distance(x, y, z, sx, sy, sz);
            float dxy = max(sqrt(x * x + y * y) - ${ramped_height_range_b}, 0.0);
            if (d < ${min_valid_distance} * ${min_valid_distance}) {
                return false;
            }
            else if (z - sz > dxy * ${ramped_height_range_a} + ${ramped_height_range_c} || z - sz > ${max_height_range}) {
                return false;
            }
            else {
                return true;
            }
        }

        __device__ float ray_vector(float16 tx, float16 ty, float16 tz,
                                    float16 px, float16 py, float16 pz,
                                    float16& rx, float16& ry, float16& rz){
            float16 vx = px - tx;
            float16 vy = py - ty;
            float16 vz = pz - tz;
            float16 norm = sqrt(vx * vx + vy * vy + vz * vz);
            if (norm > 0) {
                rx = vx / norm;
                ry = vy / norm;
                rz = vz / norm;
            }
            else {
                rx = 0;
                ry = 0;
                rz = 0;
            }
            return norm;
        }

        __device__ float inner_product(float16 x1, float16 y1, float16 z1,
                                       float16 x2, float16 y2, float16 z2) {

            float product = (x1 * x2 + y1 * y2 + z1 * z2);
            return product;
       }

        """
    ).substitute(
        resolution=resolution,
        width=width,
        height=height,
        sensor_noise_factor=sensor_noise_factor,
        min_valid_distance=min_valid_distance,
        max_height_range=max_height_range,
        ramped_height_range_a=ramped_height_range_a,
        ramped_height_range_b=ramped_height_range_b,
        ramped_height_range_c=ramped_height_range_c,
    )
    return util_preamble


def add_points_kernel(
    resolution,
    width,
    height,
    sensor_noise_factor,
    mahalanobis_thresh,
    outlier_variance,
    wall_num_thresh,
    max_ray_length,
    cleanup_step,
    min_valid_distance,
    max_height_range,
    cleanup_cos_thresh,
    ramped_height_range_a,
    ramped_height_range_b,
    ramped_height_range_c,
    enable_edge_shaped=True,
    enable_visibility_cleanup=True,
):
    add_points_kernel = cp.ElementwiseKernel(
        in_params="raw U center_x, raw U center_y, raw U R, raw U t, raw U norm_map",
        out_params="raw U p, raw U map, raw T newmap",
        preamble=map_utils(
            resolution,
            width,
            height,
            sensor_noise_factor,
            min_valid_distance,
            max_height_range,
            ramped_height_range_a,
            ramped_height_range_b,
            ramped_height_range_c,
        ),
        operation=string.Template(
            """
            U rx = p[i * 3];
            U ry = p[i * 3 + 1];
            U rz = p[i * 3 + 2];
            U x = transform_p(rx, ry, rz, R[0], R[1], R[2], t[0]);
            U y = transform_p(rx, ry, rz, R[3], R[4], R[5], t[1]);
            U z = transform_p(rx, ry, rz, R[6], R[7], R[8], t[2]);
            U v = z_noise(rz);
            int idx = get_idx(x, y, center_x[0], center_y[0]);
            if (is_valid(x, y, z, t[0], t[1], t[2])) {
                if (is_inside(idx)) {
                    U map_h = map[get_map_idx(idx, 0)];
                    U map_v = map[get_map_idx(idx, 1)];
                    U num_points = newmap[get_map_idx(idx, 4)];
                    if (abs(map_h - z) > (map_v * ${mahalanobis_thresh})) {
                        atomicAdd(&map[get_map_idx(idx, 1)], ${outlier_variance});
                    }
                    else {
                        if (${enable_edge_shaped} && (num_points > ${wall_num_thresh}) && (z < map_h - map_v * ${mahalanobis_thresh} / num_points)) {
                          // continue;
                        }
                        else {
                            T new_h = (map_h * v + z * map_v) / (map_v + v);
                            T new_v = (map_v * v) / (map_v + v);
                            atomicAdd(&newmap[get_map_idx(idx, 0)], new_h);
                            atomicAdd(&newmap[get_map_idx(idx, 1)], new_v);
                            atomicAdd(&newmap[get_map_idx(idx, 2)], 1.0);
                            // is Valid
                            map[get_map_idx(idx, 2)] = 1;
                            // Time layer
                            map[get_map_idx(idx, 4)] = 0.0;
                            // Upper bound
                            map[get_map_idx(idx, 5)] = new_h;
                            map[get_map_idx(idx, 6)] = 0.0;
                        }
                        // visibility cleanup
                    }
                }
            }
            if (${enable_visibility_cleanup}) {
                float16 ray_x, ray_y, ray_z;
                float16 ray_length = ray_vector(t[0], t[1], t[2], x, y, z, ray_x, ray_y, ray_z);
                ray_length = min(ray_length, (float16)${max_ray_length});
                int last_nidx = -1;
                for (float16 s=${ray_step}; s < ray_length; s+=${ray_step}) {
                    // iterate through ray
                    U nx = t[0] + ray_x * s;
                    U ny = t[1] + ray_y * s;
                    U nz = t[2] + ray_z * s;
                    int nidx = get_idx(nx, ny, center_x[0], center_y[0]);
                    if (last_nidx == nidx) {continue;}  // Skip if we're still in the same cell
                    else {last_nidx = nidx;}
                    if (!is_inside(nidx)) {continue;}

                    U nmap_h = map[get_map_idx(nidx, 0)];
                    U nmap_v = map[get_map_idx(nidx, 1)];
                    U nmap_valid = map[get_map_idx(nidx, 2)];
                    // traversability
                    U nmap_trav = map[get_map_idx(nidx, 3)];
                    // Time layer
                    U non_updated_t = map[get_map_idx(nidx, 4)];
                    // upper bound
                    U nmap_upper = map[get_map_idx(nidx, 5)];
                    U nmap_is_upper = map[get_map_idx(nidx, 6)];

                    // If point is close or is farther away than ray length, skip.
                    float16 d = (x - nx) * (x - nx) + (y - ny) * (y - ny) + (z - nz) * (z - nz);
                    if (d < 0.1 || !is_valid(x, y, z, t[0], t[1], t[2])) {continue;}

                    // If invalid, do upper bound check, then skip
                    if (nmap_valid < 0.5) {
                      if (nz < nmap_upper || nmap_is_upper < 0.5) {
                        map[get_map_idx(nidx, 5)] = nz;
                        map[get_map_idx(nidx, 6)] = 1.0f;
                      }
                      continue;
                    }
                    // If updated recently, skip
                    if (non_updated_t < 0.5) {continue;}

                    if (nmap_h > nz + 0.01 - min(nmap_v, 1.0) * 0.05) {
                        // If ray and norm is vertical, skip
                        U norm_x = norm_map[get_map_idx(nidx, 0)];
                        U norm_y = norm_map[get_map_idx(nidx, 1)];
                        U norm_z = norm_map[get_map_idx(nidx, 2)];
                        float product = inner_product(ray_x, ray_y, ray_z, norm_x, norm_y, norm_z);
                        if (fabs(product) < ${cleanup_cos_thresh}) {continue;}
                        U num_points = newmap[get_map_idx(nidx, 3)];
                        if (num_points > ${wall_num_thresh} && non_updated_t < 1.0) {continue;}

                        // Finally, this cell is penetrated by the ray.
                        atomicAdd(&map[get_map_idx(nidx, 2)], -${cleanup_step}/(ray_length / ${max_ray_length}));
                        atomicAdd(&map[get_map_idx(nidx, 1)], ${outlier_variance});
                        // Do upper bound check.
                        if (nz < nmap_upper || nmap_is_upper < 0.5) {
                            map[get_map_idx(nidx, 5)] = nz;
                            map[get_map_idx(nidx, 6)] = 1.0f;
                        }
                    }
                }
            }
            p[i * 3]= idx;
            p[i * 3 + 1] = is_valid(x, y, z, t[0], t[1], t[2]);
            p[i * 3 + 2] = is_inside(idx);
            """
        ).substitute(
            mahalanobis_thresh=mahalanobis_thresh,
            outlier_variance=outlier_variance,
            wall_num_thresh=wall_num_thresh,
            ray_step=resolution / 2 ** 0.5,
            max_ray_length=max_ray_length,
            cleanup_step=cleanup_step,
            cleanup_cos_thresh=cleanup_cos_thresh,
            enable_edge_shaped=int(enable_edge_shaped),
            enable_visibility_cleanup=int(enable_visibility_cleanup),
        ),
        name="add_points_kernel",
    )
    return add_points_kernel


def error_counting_kernel(
    resolution,
    width,
    height,
    sensor_noise_factor,
    mahalanobis_thresh,
    outlier_variance,
    traversability_inlier,
    min_valid_distance,
    max_height_range,
    ramped_height_range_a,
    ramped_height_range_b,
    ramped_height_range_c,
):
    error_counting_kernel = cp.ElementwiseKernel(
        in_params="raw U map, raw U p, raw U center_x, raw U center_y, raw U R, raw U t",
        out_params="raw U newmap, raw T error, raw T error_cnt",
        preamble=map_utils(
            resolution,
            width,
            height,
            sensor_noise_factor,
            min_valid_distance,
            max_height_range,
            ramped_height_range_a,
            ramped_height_range_b,
            ramped_height_range_c,
        ),
        operation=string.Template(
            """
            U rx = p[i * 3];
            U ry = p[i * 3 + 1];
            U rz = p[i * 3 + 2];
            U x = transform_p(rx, ry, rz, R[0], R[1], R[2], t[0]);
            U y = transform_p(rx, ry, rz, R[3], R[4], R[5], t[1]);
            U z = transform_p(rx, ry, rz, R[6], R[7], R[8], t[2]);
            U v = z_noise(rz);
            // if (!is_valid(z, t[2])) {return;}
            if (!is_valid(x, y, z, t[0], t[1], t[2])) {return;}
            // if ((x - t[0]) * (x - t[0]) + (y - t[1]) * (y - t[1]) + (z - t[2]) * (z - t[2]) < 0.5) {return;}
            int idx = get_idx(x, y, center_x[0], center_y[0]);
            if (!is_inside(idx)) {
                return;
            }
            U map_h = map[get_map_idx(idx, 0)];
            U map_v = map[get_map_idx(idx, 1)];
            U map_valid = map[get_map_idx(idx, 2)];
            U map_t = map[get_map_idx(idx, 3)];
            if (map_valid > 0.5 && (abs(map_h - z) < (map_v * ${mahalanobis_thresh}))
                && map_v < ${outlier_variance} / 2.0
                && map_t > ${traversability_inlier}) {
                T e = z - map_h;
                atomicAdd(&error[0], e);
                atomicAdd(&error_cnt[0], 1);
                atomicAdd(&newmap[get_map_idx(idx, 3)], 1.0);
            }
            atomicAdd(&newmap[get_map_idx(idx, 4)], 1.0);
            """
        ).substitute(
            mahalanobis_thresh=mahalanobis_thresh,
            outlier_variance=outlier_variance,
            traversability_inlier=traversability_inlier,
        ),
        name="error_counting_kernel",
    )
    return error_counting_kernel


def average_map_kernel(width, height, max_variance, initial_variance):
    average_map_kernel = cp.ElementwiseKernel(
        in_params="raw U newmap",
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
            U h = map[get_map_idx(i, 0)];
            U v = map[get_map_idx(i, 1)];
            U valid = map[get_map_idx(i, 2)];
            U new_h = newmap[get_map_idx(i, 0)];
            U new_v = newmap[get_map_idx(i, 1)];
            U new_cnt = newmap[get_map_idx(i, 2)];
            if (new_cnt > 0) {
                if (new_v / new_cnt > ${max_variance}) {
                    map[get_map_idx(i, 0)] = 0;
                    map[get_map_idx(i, 1)] = ${initial_variance};
                    map[get_map_idx(i, 2)] = 0;
                }
                else {
                    map[get_map_idx(i, 0)] = new_h / new_cnt;
                    map[get_map_idx(i, 1)] = new_v / new_cnt;
                    map[get_map_idx(i, 2)] = 1;
                }
            }
            if (valid < 0.5) {
                map[get_map_idx(i, 0)] = 0;
                map[get_map_idx(i, 1)] = ${initial_variance};
                map[get_map_idx(i, 2)] = 0;
            }
            """
        ).substitute(max_variance=max_variance, initial_variance=initial_variance),
        name="average_map_kernel",
    )
    return average_map_kernel


def dilation_filter_kernel(width, height, dilation_size):
    dilation_filter_kernel = cp.ElementwiseKernel(
        in_params="raw U map, raw U mask",
        out_params="raw U newmap, raw U newmask",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }

            __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n) {
                const int layer = ${width} * ${height};
                const int relative_idx = idx + ${width} * dy + dx;
                return layer * layer_n + relative_idx;
            }
            __device__ bool is_inside(int idx) {
                int idx_x = idx / ${width};
                int idx_y = idx % ${width};
                if (idx_x <= 0 || idx_x >= ${width} - 1) {
                    return false;
                }
                if (idx_y <= 0 || idx_y >= ${height} - 1) {
                    return false;
                }
                return true;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            U h = map[get_map_idx(i, 0)];
            U valid = mask[get_map_idx(i, 0)];
            newmap[get_map_idx(i, 0)] = h;
            if (valid < 0.5) {
                U distance = 100;
                U near_value = 0;
                for (int dy = -${dilation_size}; dy <= ${dilation_size}; dy++) {
                    for (int dx = -${dilation_size}; dx <= ${dilation_size}; dx++) {
                        int idx = get_relative_map_idx(i, dx, dy, 0);
                        if (!is_inside(idx)) {continue;}
                        U valid = mask[idx];
                        if(valid > 0.5 && dx + dy < distance) {
                            distance = dx + dy;
                            near_value = map[idx];
                        }
                    }
                }
                if(distance < 100) {
                    newmap[get_map_idx(i, 0)] = near_value;
                    newmask[get_map_idx(i, 0)] = 1.0;
                }
            }
            """
        ).substitute(dilation_size=dilation_size),
        name="dilation_filter_kernel",
    )
    return dilation_filter_kernel


def normal_filter_kernel(width, height, resolution):
    normal_filter_kernel = cp.ElementwiseKernel(
        in_params="raw U map, raw U mask",
        out_params="raw U newmap",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }

            __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n) {
                const int layer = ${width} * ${height};
                const int relative_idx = idx + ${width} * dy + dx;
                return layer * layer_n + relative_idx;
            }
            __device__ bool is_inside(int idx) {
                int idx_x = idx / ${width};
                int idx_y = idx % ${width};
                if (idx_x <= 0 || idx_x >= ${width} - 1) {
                    return false;
                }
                if (idx_y <= 0 || idx_y >= ${height} - 1) {
                    return false;
                }
                return true;
            }
            __device__ float resolution() {
                return ${resolution};
            }
            """
        ).substitute(width=width, height=height, resolution=resolution),
        operation=string.Template(
            """
            U h = map[get_map_idx(i, 0)];
            U valid = mask[get_map_idx(i, 0)];
            if (valid > 0.5) {
                int idx_x = get_relative_map_idx(i, 1, 0, 0);
                int idx_y = get_relative_map_idx(i, 0, 1, 0);
                if (!is_inside(idx_x) || !is_inside(idx_y)) { return; }
                float dzdx = (map[idx_x] - h);
                float dzdy = (map[idx_y] - h);
                float nx = -dzdy / resolution();
                float ny = -dzdx / resolution();
                float nz = 1;
                float norm = sqrt((nx * nx) + (ny * ny) + 1);
                newmap[get_map_idx(i, 0)] = nx / norm;
                newmap[get_map_idx(i, 1)] = ny / norm;
                newmap[get_map_idx(i, 2)] = nz / norm;
            }
            """
        ).substitute(),
        name="normal_filter_kernel",
    )
    return normal_filter_kernel


def polygon_mask_kernel(width, height, resolution):
    polygon_mask_kernel = cp.ElementwiseKernel(
        in_params="raw U polygon, raw U center_x, raw U center_y, raw int16 polygon_n, raw U polygon_bbox",
        out_params="raw U mask",
        preamble=string.Template(
            """
            __device__ struct Point
            {
                int x;
                int y;
            };
            // Given three colinear points p, q, r, the function checks if
            // point q lies on line segment 'pr'
            __device__ bool onSegment(Point p, Point q, Point r)
            {
                if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
                        q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
                    return true;
                return false;
            }
            // To find orientation of ordered triplet (p, q, r).
            // The function returns following values
            // 0 --> p, q and r are colinear
            // 1 --> Clockwise
            // 2 --> Counterclockwise
            __device__ int orientation(Point p, Point q, Point r)
            {
                int val = (q.y - p.y) * (r.x - q.x) -
                          (q.x - p.x) * (r.y - q.y);
                if (val == 0) return 0;  // colinear
                return (val > 0)? 1: 2; // clock or counterclock wise
            }
            // The function that returns true if line segment 'p1q1'
            // and 'p2q2' intersect.
            __device__ bool doIntersect(Point p1, Point q1, Point p2, Point q2)
            {
                // Find the four orientations needed for general and
                // special cases
                int o1 = orientation(p1, q1, p2);
                int o2 = orientation(p1, q1, q2);
                int o3 = orientation(p2, q2, p1);
                int o4 = orientation(p2, q2, q1);
                // General case
                if (o1 != o2 && o3 != o4)
                    return true;
                // Special Cases
                // p1, q1 and p2 are colinear and p2 lies on segment p1q1
                if (o1 == 0 && onSegment(p1, p2, q1)) return true;
                // p1, q1 and p2 are colinear and q2 lies on segment p1q1
                if (o2 == 0 && onSegment(p1, q2, q1)) return true;
                // p2, q2 and p1 are colinear and p1 lies on segment p2q2
                if (o3 == 0 && onSegment(p2, p1, q2)) return true;
                 // p2, q2 and q1 are colinear and q1 lies on segment p2q2
                if (o4 == 0 && onSegment(p2, q1, q2)) return true;
                return false; // Doesn't fall in any of the above cases
            }
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }

            __device__ int get_idx_x(int idx){
                int idx_x = idx / ${width};
                return idx_x;
            }

            __device__ int get_idx_y(int idx){
                int idx_y = idx % ${width};
                return idx_y;
            }

            __device__ float16 clamp(float16 x, float16 min_x, float16 max_x) {

                return max(min(x, max_x), min_x);
            }
            __device__ float16 round(float16 x) {
                return (int)x + (int)(2 * (x - (int)x));
            }
            __device__ int get_x_idx(float16 x, float16 center) {
                const float resolution = ${resolution};
                const float width = ${width};
                int i = (x - center) / resolution + 0.5 * width;
                return i;
            }
            __device__ int get_y_idx(float16 y, float16 center) {
                const float resolution = ${resolution};
                const float height = ${height};
                int i = (y - center) / resolution + 0.5 * height;
                return i;
            }
            __device__ int get_idx(float16 x, float16 y, float16 center_x, float16 center_y) {
                int idx_x = clamp(get_x_idx(x, center_x), 0, ${width} - 1);
                int idx_y = clamp(get_y_idx(y, center_y), 0, ${height} - 1);
                return ${width} * idx_x + idx_y;
            }

            """
        ).substitute(width=width, height=height, resolution=resolution),
        operation=string.Template(
            """
            // Point p = {get_idx_x(i, center_x[0]), get_idx_y(i, center_y[0])};
            Point p = {get_idx_x(i), get_idx_y(i)};
            Point extreme = {100000, p.y};
            int bbox_min_idx = get_idx(polygon_bbox[0], polygon_bbox[1], center_x[0], center_y[0]);
            int bbox_max_idx = get_idx(polygon_bbox[2], polygon_bbox[3], center_x[0], center_y[0]);
            Point bmin = {get_idx_x(bbox_min_idx), get_idx_y(bbox_min_idx)};
            Point bmax = {get_idx_x(bbox_max_idx), get_idx_y(bbox_max_idx)};
            if (p.x < bmin.x || p.x > bmax.x || p.y < bmin.y || p.y > bmax.y){
                mask[i] = 0;
                return;
            }
            else {
                int intersect_cnt = 0;
                for (int j = 0; j < polygon_n[0]; j++) {
                    Point p1, p2;
                    int i1 = get_idx(polygon[j * 2 + 0], polygon[j * 2 + 1], center_x[0], center_y[0]);
                    p1.x = get_idx_x(i1);
                    p1.y = get_idx_y(i1);
                    int j2 = (j + 1) % polygon_n[0];
                    int i2 = get_idx(polygon[j2 * 2 + 0], polygon[j2 * 2 + 1], center_x[0], center_y[0]);
                    p2.x = get_idx_x(i2);
                    p2.y = get_idx_y(i2);
                    if (doIntersect(p1, p2, p, extreme))
                    {
                        // If the point 'p' is colinear with line segment 'i-next',
                        // then check if it lies on segment. If it lies, return true,
                        // otherwise false
                        if (orientation(p1, p, p2) == 0) {
                            if (onSegment(p1, p, p2)){
                                mask[i] = 1;
                                return;
                            }
                        }
                        else if(((p1.y <= p.y) && (p2.y > p.y)) || ((p1.y > p.y) && (p2.y <= p.y))){
                            intersect_cnt++;
                        }
                    }
                }
                if (intersect_cnt % 2 == 0) { mask[i] = 0; }
                else { mask[i] = 1; }
            }
            """
        ).substitute(a=1),
        name="polygon_mask_kernel",
    )
    return polygon_mask_kernel



def image_to_map_correspondence_kernel(resolution, width, height, tolerance_z_collision):
    """
    This function calculates the correspondence between the image and the map.
    It takes in the resolution, width, height, and tolerance_z_collision as parameters.
    The function returns a kernel that can be used to perform the correspondence calculation.
    """
    _image_to_map_correspondence_kernel = cp.ElementwiseKernel(
        in_params="raw U map, raw U x1, raw U y1, raw U z1, raw U P, raw U K, raw U D, raw U image_height, raw U image_width, raw U center",
        out_params="raw U uv_correspondence, raw B valid_correspondence",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ bool is_inside_map(int x, int y) {
                return (x >= 0 && y >= 0 && x<${width} && y<${height});
            }
            __device__ float get_l2_distance(int x0, int y0, int x1, int y1) {
                float dx = x0-x1;
                float dy = y0-y1;
                return sqrt( dx*dx + dy*dy);
            }
            """
        ).substitute(width=width, height=height, resolution=resolution),
        operation=string.Template(
            """
            int cell_idx = get_map_idx(i, 0);
            
            // return if gridcell has no valid height
            if (map[get_map_idx(i, 2)] != 1){
                return;
            }
            
            // get current cell position
            int y0 = i % ${width};
            int x0 = i / ${width};
            
            // gridcell 3D point in worldframe TODO reverse x and y
            float p1 = (x0-(${width}/2)) * ${resolution} + center[0];
            float p2 = (y0-(${height}/2)) * ${resolution} + center[1];
            float p3 = map[cell_idx] +  center[2];
            
            // reproject 3D point into image plane
            float u = p1 * P[0]  + p2 * P[1] + p3 * P[2] + P[3];      
            float v = p1 * P[4]  + p2 * P[5] + p3 * P[6] + P[7];
            float d = p1 * P[8]  + p2 * P[9] + p3 * P[10] + P[11];
            
            // filter point behind image plane
            if (d <= 0) {
                return;
            }
            u = u/d;
            v = v/d;

            // Check if D is all zeros
            bool is_D_zero = (D[0] == 0 && D[1] == 0 && D[2] == 0 && D[3] == 0 && D[4] == 0);

            // Apply undistortion using distortion matrix D if not all zeros
            if (!is_D_zero) {
                float k1 = D[0];
                float k2 = D[1];
                float p1 = D[2];
                float p2 = D[3];
                float k3 = D[4];
                float fx = K[0];
                float fy = K[4];
                float cx = K[2];
                float cy = K[5];
                float x = (u - cx) / fx;
                float y = (v - cy) / fy;
                float r2 = x * x + y * y;
                float radial_distortion = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
                float u_corrected = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
                float v_corrected = y * radial_distortion + 2 * p2 * x * y + p1 * (r2 + 2 * y * y);
                u = fx * u_corrected + cx;
                v = fy * v_corrected + cy;
            }
            
            // filter point next to image plane
            if ((u < 0) || (v < 0) || (u >= image_width) || (v >= image_height)){
                return;
            }
            
            int y0_c = y0;
            int x0_c = x0;
            float total_dis = get_l2_distance(x0_c, y0_c, x1,y1);
            float z0 = map[cell_idx];
            float delta_z = z1-z0;
            
            
            // bresenham algorithm to iterate over cells in line between camera center and current gridmap cell
            // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
            int dx = abs(x1-x0);
            int sx = x0 < x1 ? 1 : -1;
            int dy = -abs(y1 - y0);
            int sy = y0 < y1 ? 1 : -1;
            int error = dx + dy;

            bool is_valid = true;
            
            // iterate over all cells along line
            while (1){
                // assumption we do not need to check the height for camera center cell
                if (x0 == x1 && y0 == y1){
                    break;
                }
                
                // check if height is invalid
                if (is_inside_map(x0,y0)){
                    int idx = y0 + (x0 * ${width});
                    if (map[get_map_idx(idx, 2)]){
                        float dis = get_l2_distance(x0_c, y0_c, x0, y0);
                        float rayheight = z0 + ( dis / total_dis * delta_z);
                        if ( map[idx] - ${tolerance_z_collision} > rayheight){
                            is_valid = false;
                            break;
                        }
                    }
                }

                
                // computation of next gridcell index in line
                int e2 = 2 * error;
                if (e2 >= dy){
                    if(x0 == x1){
                        break;
                    }
                    error = error + dy;
                    x0 = x0 + sx;
                }
                if (e2 <= dx){
                    if (y0 == y1){
                        break;
                    }
                    error = error + dx;
                    y0 = y0 + sy;        
                }
            }
            
            // mark the correspondence
            uv_correspondence[get_map_idx(i, 0)] = u;
            uv_correspondence[get_map_idx(i, 1)] = v;
            valid_correspondence[get_map_idx(i, 0)] = is_valid;
            """
        ).substitute(height=height, width=width, resolution=resolution, tolerance_z_collision=tolerance_z_collision),
        name="image_to_map_correspondence_kernel",
    )
    return _image_to_map_correspondence_kernel


def average_correspondences_to_map_kernel(width, height):
    """
    This function calculates the average correspondences to the map.
    It takes in the width and height as parameters.
    The function returns a kernel that can be used to perform the correspondence calculation.
    """
    _average_correspondences_to_map_kernel = cp.ElementwiseKernel(
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
        ).substitute(),
        name="average_correspondences_to_map_kernel",
    )
    return _average_correspondences_to_map_kernel


def exponential_correspondences_to_map_kernel(width, height, alpha):
    """
    This function calculates the exponential correspondences to the map.
    It takes in the width, height, and alpha as parameters.
    The function returns a kernel that can be used to perform the correspondence calculation.
    """
    _exponential_correspondences_to_map_kernel = cp.ElementwiseKernel(
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
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)] * (1-${alpha}) +  ${alpha} * image_mono[idx];
            }else{
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)];
            }
            
            """
        ).substitute(alpha=alpha),
        name="exponential_correspondences_to_map_kernel",
    )
    return _exponential_correspondences_to_map_kernel


def color_correspondences_to_map_kernel(width, height):
    """
    This function calculates the color correspondences to the map.
    It takes in the width and height as parameters.
    The function returns a kernel that can be used to perform the correspondence calculation.
    """
    _color_correspondences_to_map_kernel = cp.ElementwiseKernel(
        in_params="raw U sem_map, raw U map_idx, raw U image_rgb, raw U uv_correspondence, raw B valid_correspondence, raw U image_height, raw U image_width",
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
                
                int idx_red = int(uv_correspondence[cell_idx]) + int(uv_correspondence[cell_idx_2]) * image_width; 
                int idx_green = image_width * image_height + idx_red;
                int idx_blue = image_width * image_height * 2 + idx_red;
                
                unsigned int r = image_rgb[idx_red];
                unsigned int g = image_rgb[idx_green];
                unsigned int b = image_rgb[idx_blue];
                
                unsigned int rgb = (r<<16) + (g << 8) + b;
                float rgb_ = __uint_as_float(rgb);
                new_sem_map[get_map_idx(i, map_idx)] = rgb_;
            }else{
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)];
            }
            """
        ).substitute(),
        name="color_correspondences_to_map_kernel",
    )
    return _color_correspondences_to_map_kernel
