#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string


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
        in_params="raw U p, raw U center_x, raw U center_y, raw U R, raw U t, raw U norm_map",
        out_params="raw U map, raw T newmap",
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
            if (is_valid(x, y, z, t[0], t[1], t[2])) {
                int idx = get_idx(x, y, center_x[0], center_y[0]);
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
            """
        ).substitute(
            mahalanobis_thresh=mahalanobis_thresh,
            outlier_variance=outlier_variance,
            wall_num_thresh=wall_num_thresh,
            ray_step=resolution / 2**0.5,
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
