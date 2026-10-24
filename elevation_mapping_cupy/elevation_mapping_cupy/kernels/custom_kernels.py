#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import string

import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt


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
        __device__ int clamp_index(int x, int min_x, int max_x) {
            return max(min(x, max_x), min_x);
        }
        __device__ int get_x_idx(float x, float center) {
            float fi = (x - center) / ${resolution} + 0.5 * (${width} - 1);
            int i = (int)floorf(fi + 0.5f);
            return i;
        }
        __device__ int get_y_idx(float y, float center) {
            float fi = (y - center) / ${resolution} + 0.5 * (${height} - 1);
            int i = (int)floorf(fi + 0.5f);
            return i;
        }
        __device__ bool is_inside(int idx) {
            // Fixed: Row-Major (Row=Y, Col=X)
            // Row index (Y)
            int idx_y = idx / ${width};
            // Column index (X)
            int idx_x = idx % ${width};
            // Check Col bounds (Width)
            if (idx_x == 0 || idx_x == ${width} - 1) {
                return false;
            }
            // Check Row bounds (Height)
            if (idx_y == 0 || idx_y == ${height} - 1) {
                return false;
            }
            return true;
        }
        __device__ int get_idx(float x, float y, float center_x, float center_y) {
            int idx_x = clamp_index(get_x_idx(x, center_x), 0, ${width} - 1);
            int idx_y = clamp_index(get_y_idx(y, center_y), 0, ${height} - 1);
            // Fixed: Row-Major (Row=Y, Col=X)
            return ${width} * idx_y + idx_x;
        }
        __device__ int get_map_idx(int idx, int layer_n) {
            const int layer = ${width} * ${height};
            return layer * layer_n + idx;
        }
        __device__ float transform_p(float x, float y, float z,
                                     float r0, float r1, float r2, float t) {
            return r0 * x + r1 * y + r2 * z + t;
        }
        __device__ float point_noise(float x, float y, float z){
            // Noise model based on squared range in the sensor frame.
            // This avoids v=0 for flat ground points (z=0) and works for both
            // depth-camera optical frames (where z is range) and generic frames.
            return ${sensor_noise_factor} * (x * x + y * y + z * z);
        }

        __device__ float point_sensor_distance(float x, float y, float z,
                                               float sx, float sy, float sz) {
            float d = (x - sx) * (x - sx) + (y - sy) * (y - sy) + (z - sz) * (z - sz);
            return d;
        }

        __device__ bool is_valid(float x, float y, float z,
                                 float sx, float sy, float sz) {
            if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
                return false;
            }
            float d = point_sensor_distance(x, y, z, sx, sy, sz);
            float dxy = fmaxf(
                sqrtf((x - sx) * (x - sx) + (y - sy) * (y - sy)) - ${ramped_height_range_b},
                0.0f
            );
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

        __device__ float ray_vector(float tx, float ty, float tz,
                                    float px, float py, float pz,
                                    float& rx, float& ry, float& rz){
            float vx = px - tx;
            float vy = py - ty;
            float vz = pz - tz;
            float norm = sqrtf(vx * vx + vy * vy + vz * vz);
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

        __device__ float inner_product(float x1, float y1, float z1,
                                       float x2, float y2, float z2) {

            float product = (x1 * x2 + y1 * y2 + z1 * z2);
            return product;
       }

        __device__ float atomic_min_float(float* address, float value) {
            int* address_as_int = (int*)address;
            int old = *address_as_int;
            while (value < __int_as_float(old)) {
                int assumed = old;
                old = atomicCAS(address_as_int, assumed, __float_as_int(value));
                if (old == assumed) { break; }
            }
            return __int_as_float(old);
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
        in_params="raw U center_x, raw U center_y, raw U R, raw U t, raw U map, raw U norm_map",
        out_params="raw U p, raw T newmap, raw T visibility",
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
            U v = point_noise(rx, ry, rz);
            int idx = get_idx(x, y, center_x[0], center_y[0]);
            bool valid = is_valid(x, y, z, t[0], t[1], t[2]);
            p[i * 3] = idx;
            p[i * 3 + 1] = valid;
            p[i * 3 + 2] = is_inside(idx);
            if (!valid) { return; }

            if (is_inside(idx)) {
                    U map_h = map[get_map_idx(idx, 0)];
                    U map_v = map[get_map_idx(idx, 1)];
                    U num_points = newmap[get_map_idx(idx, 4)];
                    if (abs(map_h - z) > (sqrt(map_v + v) * ${mahalanobis_thresh})) {
                        atomicAdd(&newmap[get_map_idx(idx, 5)], 1.0);
                    }
                    else {
                        if (${enable_edge_shaped} && (num_points > ${wall_num_thresh}) && (z < map_h - sqrt(map_v + v) * ${mahalanobis_thresh} / num_points)) {
                          // continue;
                        }
                        else {
                            T new_h = (map_h * v + z * map_v) / (map_v + v);
                            T new_v = (map_v * v) / (map_v + v);
                            atomicAdd(&newmap[get_map_idx(idx, 0)], new_h);
                            atomicAdd(&newmap[get_map_idx(idx, 1)], new_v);
                            atomicAdd(&newmap[get_map_idx(idx, 2)], 1.0);
                        }
                        // visibility cleanup
                    }
            }
            if (${enable_visibility_cleanup}) {
                float ray_x, ray_y, ray_z;
                float ray_length = ray_vector(t[0], t[1], t[2], x, y, z, ray_x, ray_y, ray_z);
                ray_length = fminf(ray_length, (float)${max_ray_length});
                float end_x = t[0] + ray_x * ray_length;
                float end_y = t[1] + ray_y * ray_length;
                float end_z = t[2] + ray_z * ray_length;
                float start_grid_x = (
                    (t[0] - center_x[0]) / ${resolution} + 0.5f * (${width} - 1) + 0.5f
                );
                float start_grid_y = (
                    (t[1] - center_y[0]) / ${resolution} + 0.5f * (${height} - 1) + 0.5f
                );
                float end_grid_x = (
                    (end_x - center_x[0]) / ${resolution} + 0.5f * (${width} - 1) + 0.5f
                );
                float end_grid_y = (
                    (end_y - center_y[0]) / ${resolution} + 0.5f * (${height} - 1) + 0.5f
                );
                float grid_dx = end_grid_x - start_grid_x;
                float grid_dy = end_grid_y - start_grid_y;
                int cell_x = (int)floorf(start_grid_x);
                int cell_y = (int)floorf(start_grid_y);
                int end_cell_x = (int)floorf(end_grid_x);
                int end_cell_y = (int)floorf(end_grid_y);
                int step_x = (grid_dx > 0.0f) - (grid_dx < 0.0f);
                int step_y = (grid_dy > 0.0f) - (grid_dy < 0.0f);
                float delta_x = step_x == 0 ? 1.0e30f : fabsf(1.0f / grid_dx);
                float delta_y = step_y == 0 ? 1.0e30f : fabsf(1.0f / grid_dy);
                float boundary_x = cell_x + (step_x > 0 ? 1.0f : 0.0f);
                float boundary_y = cell_y + (step_y > 0 ? 1.0f : 0.0f);
                float next_x = step_x == 0 ? 1.0e30f : (boundary_x - start_grid_x) / grid_dx;
                float next_y = step_y == 0 ? 1.0e30f : (boundary_y - start_grid_y) / grid_dy;

                for (int step = 0; step < ${width} + ${height}; ++step) {
                    float entry;
                    if (next_x < next_y) {
                        entry = next_x;
                        next_x += delta_x;
                        cell_x += step_x;
                    }
                    else if (next_y < next_x) {
                        entry = next_y;
                        next_y += delta_y;
                        cell_y += step_y;
                    }
                    else {
                        entry = next_x;
                        next_x += delta_x;
                        next_y += delta_y;
                        cell_x += step_x;
                        cell_y += step_y;
                    }
                    if (entry >= 1.0f || (cell_x == end_cell_x && cell_y == end_cell_y)) {break;}
                    if (
                        cell_x <= 0 || cell_x >= ${width} - 1
                        || cell_y <= 0 || cell_y >= ${height} - 1
                    ) {continue;}

                    int nidx = cell_y * ${width} + cell_x;
                    float exit = fminf(fminf(next_x, next_y), 1.0f);
                    float sample = 0.5f * (entry + exit);
                    U nz = t[2] + (end_z - t[2]) * sample;

                    U nmap_h = map[get_map_idx(nidx, 0)];
                    U nmap_v = map[get_map_idx(nidx, 1)];
                    U nmap_valid = map[get_map_idx(nidx, 2)];
                    // Time layer
                    U non_updated_t = map[get_map_idx(nidx, 4)];
                    // If invalid, do upper bound check, then skip
                    if (nmap_valid < 0.5) {
                      atomic_min_float(&visibility[get_map_idx(nidx, 2)], nz);
                      continue;
                    }
                    // If updated recently, skip
                    if (non_updated_t < 0.5) {continue;}

                    if (nmap_h > nz + 0.01 - fminf(nmap_v, 1.0f) * 0.05) {
                        // If ray and norm is vertical, skip
                        U norm_x = norm_map[get_map_idx(nidx, 0)];
                        U norm_y = norm_map[get_map_idx(nidx, 1)];
                        U norm_z = norm_map[get_map_idx(nidx, 2)];
                        float product = inner_product(ray_x, ray_y, ray_z, norm_x, norm_y, norm_z);
                        if (fabs(product) < ${cleanup_cos_thresh}) {continue;}
                        U num_points = newmap[get_map_idx(nidx, 3)];
                        if (num_points > ${wall_num_thresh} && non_updated_t < 1.0) {continue;}

                        // Finally, this cell is penetrated by the ray.
                        atomicAdd(
                            &visibility[get_map_idx(nidx, 0)],
                            ${cleanup_step}/(ray_length / ${max_ray_length})
                        );
                        atomicAdd(&visibility[get_map_idx(nidx, 1)], 1.0);
                        atomic_min_float(&visibility[get_map_idx(nidx, 2)], nz);
                    }
                }
            }
            """
        ).substitute(
            mahalanobis_thresh=mahalanobis_thresh,
            outlier_variance=outlier_variance,
            wall_num_thresh=wall_num_thresh,
            resolution=resolution,
            width=width,
            height=height,
            max_ray_length=max_ray_length,
            cleanup_step=cleanup_step,
            cleanup_cos_thresh=cleanup_cos_thresh,
            enable_edge_shaped=int(enable_edge_shaped),
            enable_visibility_cleanup=int(enable_visibility_cleanup),
        ),
        name="add_points_kernel",
    )
    return add_points_kernel


def finalize_map_kernel(width, height, max_variance, initial_variance, outlier_variance):
    """Finalize immutable endpoint and visibility proposals once per map cell."""
    return cp.ElementwiseKernel(
        in_params="raw U previous, raw U newmap, raw U visibility",
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
            for (int layer = 0; layer < 7; ++layer) {
                map[get_map_idx(i, layer)] = previous[get_map_idx(i, layer)];
            }

            U new_count = newmap[get_map_idx(i, 2)];
            U outlier_count = newmap[get_map_idx(i, 5)];
            if (new_count > 0) {
                U new_height = newmap[get_map_idx(i, 0)] / new_count;
                U new_variance = newmap[get_map_idx(i, 1)] / new_count;
                map[get_map_idx(i, 0)] = new_height;
                map[get_map_idx(i, 1)] = new_variance;
                map[get_map_idx(i, 2)] = 1.0;
                map[get_map_idx(i, 4)] = 0.0;
                map[get_map_idx(i, 5)] = new_height;
                map[get_map_idx(i, 6)] = 0.0;
                if (new_variance > ${max_variance}) {
                    map[get_map_idx(i, 0)] = 0.0;
                    map[get_map_idx(i, 1)] = ${initial_variance};
                    map[get_map_idx(i, 2)] = 0.0;
                }
            }
            else {
                map[get_map_idx(i, 1)] += outlier_count * ${outlier_variance};
                U cleanup = visibility[get_map_idx(i, 0)];
                if (cleanup > 0.0) {
                    map[get_map_idx(i, 2)] -= cleanup;
                    map[get_map_idx(i, 1)] += (
                        visibility[get_map_idx(i, 1)] * ${outlier_variance}
                    );
                }

                U proposed_upper_bound = visibility[get_map_idx(i, 2)];
                U previous_upper_bound = previous[get_map_idx(i, 5)];
                U previous_has_upper_bound = previous[get_map_idx(i, 6)];
                if (
                    isfinite(proposed_upper_bound)
                    && (
                        previous_has_upper_bound < 0.5
                        || proposed_upper_bound < previous_upper_bound
                    )
                ) {
                    map[get_map_idx(i, 5)] = proposed_upper_bound;
                    map[get_map_idx(i, 6)] = 1.0;
                }
            }

            if (map[get_map_idx(i, 2)] < 0.5) {
                map[get_map_idx(i, 0)] = 0.0;
                map[get_map_idx(i, 1)] = ${initial_variance};
                map[get_map_idx(i, 2)] = 0.0;
            }
            """
        ).substitute(
            max_variance=max_variance,
            initial_variance=initial_variance,
            outlier_variance=outlier_variance,
        ),
        name=f"finalize_map_{width}_{height}",
    )


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
            U v = point_noise(rx, ry, rz);
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
            if (map_valid > 0.5 && (abs(map_h - z) < (sqrt(map_v + v) * ${mahalanobis_thresh}))
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
                    valid = 0;
                }
                else {
                    map[get_map_idx(i, 0)] = new_h / new_cnt;
                    map[get_map_idx(i, 1)] = new_v / new_cnt;
                    map[get_map_idx(i, 2)] = 1;
                    valid = 1;
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
    """Return an alias-safe nearest-valid-cell dilation operation.

    Small neighborhoods use a direct kernel. Large neighborhoods use CuPy's
    Euclidean distance transform so runtime does not grow with radius squared.
    """
    if dilation_size < 0:
        raise ValueError("dilation_size must be non-negative")

    direct_kernel = cp.ElementwiseKernel(
        in_params="raw U map, raw U mask",
        out_params="raw U newmap, raw U newmask",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }

            __device__ bool is_inside(int row, int col) {
                return row > 0 && row < ${height} - 1 && col > 0 && col < ${width} - 1;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            U h = map[get_map_idx(i, 0)];
            U valid = mask[get_map_idx(i, 0)];
            newmap[get_map_idx(i, 0)] = h;
            newmask[get_map_idx(i, 0)] = valid;
            if (valid < 0.5) {
                int row = i / ${width};
                int col = i % ${width};
                int distance_squared = ${dilation_size} * ${dilation_size} + 1;
                U near_value = 0;
                for (int dy = -${dilation_size}; dy <= ${dilation_size}; dy++) {
                    for (int dx = -${dilation_size}; dx <= ${dilation_size}; dx++) {
                        int candidate_distance = dx * dx + dy * dy;
                        if (candidate_distance >= distance_squared) {continue;}
                        int candidate_row = row + dy;
                        int candidate_col = col + dx;
                        if (!is_inside(candidate_row, candidate_col)) {continue;}
                        int idx = candidate_row * ${width} + candidate_col;
                        U candidate_valid = mask[idx];
                        if(candidate_valid > 0.5) {
                            distance_squared = candidate_distance;
                            near_value = map[idx];
                        }
                    }
                }
                if(distance_squared <= ${dilation_size} * ${dilation_size}) {
                    newmap[get_map_idx(i, 0)] = near_value;
                    newmask[get_map_idx(i, 0)] = 1.0;
                }
            }
            """
        ).substitute(
            dilation_size=dilation_size,
            width=width,
            height=height,
        ),
        name=f"dilation_filter_{width}_{height}_{dilation_size}",
    )

    def run(map_in, mask_in, map_out, mask_out, size=None):
        if size is not None and size != width * height:
            raise ValueError(f"Expected size {width * height}, got {size}")

        if dilation_size <= 8:
            source_map = map_in.copy() if map_in.data.ptr == map_out.data.ptr else map_in
            source_mask = mask_in.copy() if mask_in.data.ptr == mask_out.data.ptr else mask_in
            direct_kernel(source_map, source_mask, map_out, mask_out, size=width * height)
            return

        valid = mask_in > 0.5
        seeds = valid.copy()
        seeds[[0, -1], :] = False
        seeds[:, [0, -1]] = False
        distances, indices = distance_transform_edt(
            ~seeds,
            return_indices=True,
            float64_distances=False,
        )
        rows = cp.clip(indices[0], 0, height - 1)
        cols = cp.clip(indices[1], 0, width - 1)
        nearest = map_in[rows, cols]
        replace = ~valid & (distances <= dilation_size)
        map_out[...] = cp.where(replace, nearest, map_in)
        mask_out[...] = valid | replace

    return run


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
                // Fixed: Row-Major (Row=Y, Col=X)
                // Row index (Y)
                int idx_y = idx / ${width};
                // Column index (X)
                int idx_x = idx % ${width};
                // Check Col bounds (Width)
                if (idx_x <= 0 || idx_x >= ${width} - 1) {
                    return false;
                }
                // Check Row bounds (Height)
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
                // Fixed: Normal = (-dH/dx, -dH/dy, 1)
                float nx = -dzdx / resolution();
                float ny = -dzdy / resolution();
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
                // Fixed: Column index (X)
                int idx_x = idx % ${width};
                return idx_x;
            }

            __device__ int get_idx_y(int idx){
                // Fixed: Row index (Y)
                int idx_y = idx / ${width};
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
                // Fixed: Row-Major (Row=Y, Col=X)
                return ${width} * idx_y + idx_x;
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


if __name__ == "__main__":
    for i in range(10):
        import random

        a = cp.zeros((100, 100))
        n = random.randint(3, 5)

        # polygon = cp.array([[-1, -1], [3, 4], [2, 4], [1, 3]], dtype=float)
        polygon = cp.array(
            [[(random.random() - 0.5) * 10, (random.random() - 0.5) * 10] for i in range(n)], dtype=float
        )
        print(polygon)
        polygon_min = polygon.min(axis=0)
        polygon_max = polygon.max(axis=0)
        polygon_bbox = cp.concatenate([polygon_min, polygon_max]).flatten()
        polygon_n = polygon.shape[0]
        print(polygon_bbox)
        # polygon_bbox = cp.array([-5, -5, 5, 5], dtype=float)
        polygon_mask = polygon_mask_kernel(100, 100, 0.1)
        import time

        start = time.time()
        polygon_mask(polygon, 0.0, 0.0, polygon_n, polygon_bbox, a, size=(100 * 100))
        print(time.time() - start)
        import pylab as plt

        print(a)
        plt.imshow(cp.asnumpy(a))
        plt.show()
