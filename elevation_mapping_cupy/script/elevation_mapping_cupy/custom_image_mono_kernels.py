import cupy as cp
import string


def image_to_map_correspondence_kernel(resolution, width, height, tolerance_z_collision):
    image_to_map_correspondence_kernel = cp.ElementwiseKernel(
        in_params="raw U map, raw U x1, raw U y1, raw U z1, raw U P, raw U image_height, raw U image_width, raw U center",
        out_params="raw U uv_correspondence, raw B valid_correspondence",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ bool is_inside_map(int x, int y) {
                return (x >= 0 && y >= 0 && x<${width} && x<${height});
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
            
            // filter point nexto image plane
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
            valid_correspondence[get_map_idx(i, 0)] = 1;
            """
        ).substitute(height=height, width=width, resolution=resolution, tolerance_z_collision=tolerance_z_collision),
        name="image_to_map_correspondence_kernel",
    )
    return image_to_map_correspondence_kernel


def average_correspondences_to_map_kernel(resolution, width, height):
    average_correspondences_to_map_kernel = cp.ElementwiseKernel(
        in_params="raw U sem_map, raw U image_mono, raw U uv_correspondence, raw B valid_correspondence, raw U image_height, raw U image_width",
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
            int cell_idx_2 = get_map_idx(i, 1);
            if (valid_correspondence[cell_idx]){
                int idx = int(uv_correspondence[cell_idx]) + int(uv_correspondence[cell_idx_2]) * image_width; 
                new_sem_map[cell_idx] = image_mono[idx];
            }
            """
        ).substitute(),
        name="average_correspondences_to_map_kernel",
    )
    return average_correspondences_to_map_kernel
