import cupy as cp
import string


def image_to_map_corrospondence_kernel(
    resolution,
    width,
    height
):
    image_to_map_corrospondence_kernel = cp.ElementwiseKernel(
        in_params="raw U center_x, raw U center_y, raw U R, raw U t, raw U norm_map",
        out_params="raw U p, raw U map, raw T newmap",
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
            """
        ).substitute(
        ),
        name="image_to_map_corrospondence_kernel",
    )
    return image_to_map_corrospondence_kernel


def test_kernel(
    resolution,
    width,
    height, 
    tolerance_z_collision
):
    test_kernel = cp.ElementwiseKernel(
        in_params="raw U map, raw U x1, raw U y1, raw U z1",
        out_params="raw U newmap, raw U debug1, raw U debug2",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ int get_x_idx(float16 x, float16 center) {
                int i = (x - center) / ${resolution} + 0.5 * ${width};
                return i;
            }
            __device__ int get_y_idx(float16 y, float16 center) {
                int i = (y - center) / ${resolution} + 0.5 * ${height};
                return i;
            }
            """
        ).substitute(width=width, height=height, resolution=resolution),
        operation=string.Template(
            """
            // for each cell use Bersenham to calculate projection
            unsigned int y0 = i % ${width};
            unsigned int x0 = i / ${width};
            
            unsigned int y0_c = y0;
            unsigned int x0_c = x0;
            
            
            float z0 = map[get_map_idx(i, 0)];
            float t1 = x1-x0_c;
            float t2 = y1 - y0_c;
            float total_dis = sqrt( t1*t1 + t2*t2);
            debug2[get_map_idx(i, 0)] = total_dis;
            
            if (total_dis == 0){
                return;
            }
            float delta_z = z1-z0;
            
            
            // following plotLine implementation wikipedia https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
            
            unsigned int dx = abs(x1-x0);
            unsigned int sx = x0 < x1 ? 1 : -1;
            unsigned int dy = -abs(y1 - y0);
            unsigned int sy = y0 < y1 ? 1 : -1;
            unsigned int error = dx + dy;
            
            unsigned int k = 0;
            while (1){
                // TODO do here the calculation
                k++;
                debug1[get_map_idx(i, 0)] = k;
                
                unsigned int idx = y0 + (x0 * ${width});
                
                float t1 = x0 - x0_c;
                float t2 = y0 - y0_c;
            
                float dis = sqrt( t1*t1 + t2*t2);
                float rayheight = z0 + dis/total_dis * (delta_z);
                
                if ( map[ get_map_idx(idx, 0)] > rayheight-${tolerance_z_collision}){
                    newmap[get_map_idx(i, 0)] = 0;
                }
                
                if (x0 == x1 && y0 == y1){
                    return;
                }
                unsigned int e2 = 2 * error;
                if (e2 >= dy){
                    if(x0 == x1){
                        return;
                    }
                    error = error + dy;
                    x0 = x0 + sx;
                }
                if (e2 <= dx){
                    if (y0 == y1){
                        return;
                    }
                    error = error + dx;
                    y0 = y0 + sy;
                }
            }
            """
        ).substitute(
            height=height, width=width, resolution=resolution, tolerance_z_collision=tolerance_z_collision
        ),
        name="test_kernel",
    )
    return test_kernel




if __name__ == "__main__":
    import numpy as np
    kernel = test_kernel(
        resolution = 0.1,
        width = 100,
        height = 100,
        tolerance_z_collision = 0.0
    )
    arr1 = cp.asarray( np.ones( (100,100), dtype=np.float32), dtype=np.float32)
    arr1[:2,:2] = 10
    
    
    arr_out = cp.asarray( np.ones( (100,100), dtype=np.float32), dtype=np.float32)
    
    debug1 = cp.asarray( np.ones( (100,100), dtype=np.float32), dtype=np.float32)
    debug2 = cp.asarray( np.ones( (100,100), dtype=np.float32), dtype=np.float32)
    camera_x_idx = cp.uint32( 0 )
    camera_y_idx = cp.uint32( 10 ) 
    camera_z_meter = cp.float32( 2.0 ) 
    kernel(arr1, camera_x_idx, camera_y_idx, camera_z_meter, arr_out, debug1, debug2, size=int(arr1.shape[0] * arr1.shape[1]) )
    
    print(arr1)
    print(arr_out)
    
    print("K")
    print(debug1)
    
    print("distance")
    print(debug2)