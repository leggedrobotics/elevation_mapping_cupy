#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string


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
