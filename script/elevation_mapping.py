import numpy as np
import scipy as nsp
import scipy.ndimage

import chainer
import chainer.links as L
import chainer.functions as F
import yaml
import string
import time

use_cupy = False
xp = np
sp = nsp


def load_backend(enable_cupy):
    if enable_cupy:
        global use_cupy, xp, sp, cp
        import cupy as cp
        import cupyx.scipy as csp
        import cupyx.scipy.ndimage
        use_cupy = True
        xp = cp
        sp = csp
	pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
	cp.cuda.set_allocator(pool.malloc)
    else:
        xp = np
        sp = nsp


class Parameter(object):
    def __init__(self):
        self.use_cupy = True
        self.resolution = 0.02
        self.gather_mode = 'mean'

        self.map_length = 10.0
        self.sensor_noise_factor = 0.05
        self.mahalanobis_thresh = 2.0
        self.outlier_variance = 0.01
        self.time_variance = 0.01

        self.max_variance = 1.0

        self.initial_variance = 10.0
        self.w1 = np.zeros((4, 1, 3, 3))
        self.w2 = np.zeros((4, 1, 3, 3))
        self.w3 = np.zeros((4, 1, 3, 3))
        self.w_out = np.zeros((1, 12, 1, 1))

    def load_weights(self, filename):
        with open(filename) as file:
            weights = yaml.load(file)
            self.w1 = np.array(weights['w1'])
            self.w2 = np.array(weights['w2'])
            self.w3 = np.array(weights['w3'])
            self.w_out = np.array(weights['w_out'])

    def set_use_cupy(self, use_cupy):
        self.use_cupy = use_cupy

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_gather_mode(self, gather_mode):
        self.gather_mode = gather_mode

    def set_map_length(self, map_length):
        self.map_length = map_length

    def set_sensor_noise_factor(self, sensor_noise_factor):
        self.sensor_noise_factor = sensor_noise_factor

    def set_mahalanobis_thresh(self, mahalanobis_thresh):
        self.mahalanobis_thresh = mahalanobis_thresh

    def set_outlier_variance(self, outlier_variance):
        self.outlier_variance = outlier_variance

    def set_time_variance(self, time_variance):
        self.time_variance = time_variance

    def set_max_variance(self, max_variance):
        self.max_variance = max_variance

    def set_initial_variance(self, initial_variance):
        self.initial_variance = initial_variance


class TraversabilityFilter(chainer.Chain):
    def __init__(self, w1, w2, w3, w_out):
        super(TraversabilityFilter, self).__init__()
        self.conv1 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=1,
                                     nobias=True, initialW=w1)
        self.conv2 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=2,
                                     nobias=True, initialW=w2)
        self.conv3 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=3,
                                     nobias=True, initialW=w3)
        self.conv_out = L.Convolution2D(12, 1, ksize=1,
                                        nobias=True, initialW=w_out)

        if use_cupy:
            self.conv1.to_gpu()
            self.conv2.to_gpu()
            self.conv3.to_gpu()
            self.conv_out.to_gpu()
        chainer.config.train = False
        chainer.config.enable_backprop = False

    def __call__(self, elevation_map):
        elevation = elevation_map[0]
        out1 = self.conv1(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))
        out2 = self.conv2(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))
        out3 = self.conv3(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))

        out1 = out1[:, :, 2:-2, 2:-2]
        out2 = out2[:, :, 1:-1, 1:-1]
        out = F.concat((out1, out2, out3), axis=1)
        return self.conv_out(F.absolute(out)).array


class ElevationMap(object):
    def __init__(self, param):
        param.use_cupy = True
        load_backend(param.use_cupy)
        self.resolution = param.resolution
        self.center = xp.array([0, 0], dtype=float)
        self.center_cpu = np.array([0, 0], dtype=float)
        self.map_length = param.map_length
        # +2 is a border for outside map
        self.cell_n = int(self.map_length / self.resolution) + 2

        # 'mean' or 'max'
        self.gather_mode = 'mean'
        # self.gather_mode = param.gather_mode
        if self.gather_mode == 'max':
            self.atomic_func = 'atomicMax'
        else:
            self.atomic_func = 'atomicAdd'

        self.sensor_noise_factor = param.sensor_noise_factor
        self.mahalanobis_thresh = param.mahalanobis_thresh
        self.outlier_variance = param.outlier_variance
        self.time_variance = param.time_variance

        self.max_variance = param.max_variance

        # layers: elevation, variance, is_valid, traversability
        self.elevation_map = xp.zeros((4, self.cell_n, self.cell_n))
        # Initial variance
        self.initial_variance = param.initial_variance
        self.elevation_map[1] += self.initial_variance

        self.points_cpu = None
        self.points_gpu = None

        if use_cupy:
            self.compile_add_points_kernel()

        self.traversability_filter = TraversabilityFilter(param.w1,
                                                          param.w2,
                                                          param.w3,
                                                          param.w_out)
        if use_cupy:
            self.traversability_filter.to_gpu()

    def get_resolution(self):
        return self.resolution

    def get_length(self):
        return self.map_length

    def get_position(self, position):
        if use_cupy:
            position[0][:] = xp.asnumpy(self.center)
            # position[0][:] = self.center_cpu
            # position[0][:] = self.center_cpu
            # return xp.asnumpy(self.center)
        else:
            position[0][:] = self.center
        # return self.center

    def move(self, delta_position):
        delta_position = xp.asarray(delta_position)
        delta_pixel = xp.round(delta_position / self.resolution)
        delta_position = delta_pixel * self.resolution
        self.center += xp.asarray(delta_position)
        self.shift_map(delta_pixel)

    def move_to(self, position):
        position = xp.asarray(position)
        delta = position - self.center
        delta_pixel = xp.around(delta / self.resolution)
        delta = delta_pixel * self.resolution
        self.center += delta
        self.center_cpu += xp.asnumpy(delta)
        self.shift_map(-delta_pixel)

    def shift_map(self, delta_pixel):
        shift_value = delta_pixel
        # shift_value = xp.round(delta_pixel)
        shift_fn = sp.ndimage.interpolation.shift
        # elevation
        self.elevation_map[0] = shift_fn(self.elevation_map[0], shift_value,
                                         cval=0.0)
        # variance
        self.elevation_map[1] = shift_fn(self.elevation_map[1], shift_value,
                                         cval=self.initial_variance)
        # is valid (1 is valid 0 is not valid)
        self.elevation_map[2] = shift_fn(self.elevation_map[2], shift_value,
                                         cval=0)

    def add_noise(self, points):
        z = points[:, -1]
        n = self.sensor_noise_factor * z * z
        n = xp.expand_dims(n, axis=1)
        points = xp.hstack([points, n])
        return points

    def transform_points(self, points, R, t):
        p = points[:, 0:3]
        transformed = xp.einsum('ij, kj->ki', R, p) + t
        new_points = points.copy()
        new_points[:, 0:3] = transformed
        return new_points

    def get_cell_index(self, points):
        index = (points[:, 0:2] - self.center) / self.resolution
        index = xp.around(index).astype(int)
        index += self.cell_n / 2
        index = xp.clip(index, 0, self.cell_n - 1)
        return index

    def get_indices(self, index):
        # get unique indices for averaging new values
        flatten_index = index[:, 0] * self.cell_n + index[:, 1]
        unique = xp.unique(flatten_index,
                           return_inverse=True,
                           return_counts=True)
        flatten_unique_index, unique_inverse, unique_count = unique
        unique_index_size = (len(flatten_unique_index), 2)
        unique_index = xp.zeros(unique_index_size, dtype=int)
        unique_index[:, 0] = flatten_unique_index // self.cell_n
        unique_index[:, 1] = flatten_unique_index % self.cell_n
        return index, unique_index, unique_inverse, unique_count

    def large_variance_rejection(self):

        outliers = self.elevation_map[1] > self.max_variance
        self.elevation_map[0] = xp.where(outliers, 0,
                                         self.elevation_map[0])
        self.elevation_map[1] = xp.where(outliers, self.initial_variance,
                                         self.elevation_map[1])
        self.elevation_map[2] = xp.where(outliers, 0,
                                         self.elevation_map[2])

    def add_variance_to_outliers(self, outlier_index):
        if len(outlier_index) > 0:
            outlier_unique = self.get_indices(outlier_index)
            _, outlier_unique, inverse, _ = outlier_unique
            variance_addition = xp.bincount(inverse, xp.ones_like(inverse))
            variance_addition *= self.outlier_variance
            index_x, index_y = outlier_unique[:, 0], outlier_unique[:, 1]
            self.elevation_map[1][index_x, index_y] += variance_addition

    def outlier_rejection(self, index, map_h, map_v, point_h, point_v):
        outliers = xp.abs(map_h - point_h) > (map_v * self.mahalanobis_thresh)
        outlier_index = index[outliers]
        self.add_variance_to_outliers(outlier_index)
        index = index[~outliers]
        map_h = map_h[~outliers]
        map_v = map_v[~outliers]
        point_h = point_h[~outliers]
        point_v = point_v[~outliers]

    def gather_into_unique_cell(self, index, new_h, new_v, mode='mean'):
        unique_tuple = self.get_indices(index)
        index, unique_index, unique_inverse, unique_count = unique_tuple
        if mode == 'mean':
            new_unique_h = xp.bincount(unique_inverse, new_h) / unique_count
            new_unique_v = xp.bincount(unique_inverse, new_v)
        elif mode == 'max':
            new_unique_h = self.get_max_unique_values(unique_inverse, new_h)
            new_unique_v = xp.bincount(unique_inverse, new_v) / unique_count
        else:
            print('ERROR[gather_into_unique_cell]: use mean or max')
            print('Using mean...')
            new_unique_h = xp.bincount(unique_inverse, new_h) / unique_count
            new_unique_v = xp.bincount(unique_inverse, new_v)
        index_x, index_y = unique_index[:, 0], unique_index[:, 1]
        return (index_x, index_y), new_unique_h, new_unique_v

    def get_max_unique_values(self, unique, values):
        order = xp.lexsort(xp.stack([values, unique]))
        unique = unique[order]
        values = values[order]
        index = xp.empty(len(unique), 'bool')
        index[-1] = True
        index[:-1] = unique[1:] != unique[:-1]
        new_values = values[index]
        return new_values

    def to_pinned_memory(self, array):
        # print(array)
        # print(array.dtype)
        # print(array.shape)
        if use_cupy:
            mem = xp.cuda.alloc_pinned_memory(array.nbytes)
            src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
            # src[...] = array
            src = array
            return src
        else:
            return array

    def load_points(self, points):
        if use_cupy:
            if self.points_cpu is None:
                # points = np.concatenate([points for i in range(5)], axis=0)
                self.points_cpu = self.to_pinned_memory(points)
                self.points_gpu = xp.ndarray(self.points_cpu.shape,
                                             self.points_cpu.dtype)
                self.stream = xp.cuda.Stream(non_blocking=True)

                self.points_gpu.set(self.points_cpu, stream=self.stream)
            else:
                self.points_cpu[...] = points
                self.points_gpu.set(self.points_cpu, stream=self.stream)
        else:
            self.points_cpu = points

    def update_map_xp(self, points):
        self.update_variance()
        index = self.get_cell_index(points)
        map_h = self.elevation_map[0][index[:, 0], index[:, 1]]
        map_v = self.elevation_map[1][index[:, 0], index[:, 1]]
        point_h = points[:, 2]
        point_v = points[:, 3]
        # outlier rejection
        self.outlier_rejection(index, map_h, map_v, point_h, point_v)

        # calculate new value
        new_h = ((map_h * point_v + point_h * map_v) / (map_v + point_v))
        new_v = (map_v * point_v) / (map_v + point_v)

        # get value for each cell (choose max or mean)
        idx, h, v = self.gather_into_unique_cell(index, new_h,
                                                 new_v, mode=self.gather_mode)
        self.elevation_map[0][idx] = h
        self.elevation_map[1][idx] = v
        self.elevation_map[2][idx] = 1

        self.large_variance_rejection()

        # calculate traversability
        traversability = self.traversability_filter(self.elevation_map)
        self.elevation_map[3][3:-3, 3:-3] = traversability.reshape((traversability.shape[2], traversability.shape[3]))

    def compile_add_points_kernel(self):
        self.new_map = cp.zeros((3, self.cell_n, self.cell_n))
        self.add_points_kernel = cp.ElementwiseKernel(
                in_params='raw U p, U center_x, U center_y, raw U R, raw U t',
                out_params='raw U map, raw T newmap',
                preamble=\
                string.Template('''
                __device__ float16 clamp(float16 x, float16 min_x, float16 max_x) {
                    return max(min(x, max_x), min_x);
                }
                __device__ float16 round(float16 x) {
                    return (int)x + (int)(2 * (x - (int)x));
                }
                __device__ int get_xy_idx(float16 x, float16 center) {
                    const float resolution = ${resolution};
                    int i = round((x - center) / resolution);
                    return i;
                }
                __device__ int get_idx(float16 x, float16 y, float16 center_x, float16 center_y) {
                    int idx_x = clamp(get_xy_idx(x, center_x) + ${width} / 2, 0, ${width} - 1);
                    int idx_y = clamp(get_xy_idx(y, center_y) + ${height} / 2, 0, ${height} - 1);
                    return ${width} * idx_x + idx_y;
                }
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }
                __device__ int floatToOrderedInt( float floatVal )
                {
                    int intVal = __float_as_int( floatVal );
                    return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
                }
                __device__ float orderedIntToFloat( int intVal )
                {
                    return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );
                }
                __device__ float transform_p(float16 x, float16 y, float16 z,
                                             float16 r0, float16 r1, float16 r2, float16 t) {
                    return r0 * x + r1 * y + r2 * z + t;
                }
                __device__ float z_noise(float16 z){
                    return ${sensor_noise_factor} * z * z;
                }

                ''').substitute(resolution=self.resolution, width=self.cell_n, height=self.cell_n,
                                sensor_noise_factor=self.sensor_noise_factor),
                operation=\
                string.Template(
                '''
                U rx = p[i * 3];
                U ry = p[i * 3 + 1];
                U rz = p[i * 3 + 2];
                U x = transform_p(rx, ry, rz, R[0], R[1], R[2], t[0]);
                U y = transform_p(rx, ry, rz, R[3], R[4], R[5], t[1]);
                U z = transform_p(rx, ry, rz, R[6], R[7], R[8], t[2]);
                U v = z_noise(rz);
                int idx = get_idx(x, y, center_x, center_y);
                U map_h = map[get_map_idx(idx, 0)];
                U map_v = map[get_map_idx(idx, 1)];
                if (abs(map_h - z) > (map_v * ${mahalanobis_thresh})) {
                    atomicAdd(&map[get_map_idx(idx, 1)], ${outlier_variance});
                }
                else {
                    T new_h = (map_h * v + z * map_v) / (map_v + v);
                    T new_v = (map_v * v) / (map_v + v);
                    ${atomic_func}(&newmap[get_map_idx(idx, 0)], new_h);
                    atomicAdd(&newmap[get_map_idx(idx, 1)], new_v);
                    ${atomic_func}(&newmap[get_map_idx(idx, 2)], 1.0);
                    map[get_map_idx(idx, 2)] = 1;
                }
                ''').substitute(mahalanobis_thresh=self.mahalanobis_thresh,
                                outlier_variance=self.outlier_variance,
                                atomic_func=self.atomic_func),
                name='add_points_kernel')
        self.average_map_kernel = cp.ElementwiseKernel(
                in_params='raw U newmap',
                out_params='raw U map',
                preamble=\
                string.Template('''
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }
                ''').substitute(width=self.cell_n, height=self.cell_n),
                operation=\
                '''
                U h = map[get_map_idx(i, 0)];
                U v = map[get_map_idx(i, 1)];
                U new_h = newmap[get_map_idx(i, 0)];
                U new_v = newmap[get_map_idx(i, 1)];
                U new_cnt = newmap[get_map_idx(i, 2)];
                if (new_cnt > 0) {
                    map[get_map_idx(i, 0)] = new_h / new_cnt;
                    map[get_map_idx(i, 1)] = new_v;
                    map[get_map_idx(i, 2)] = 1;
                }
                ''',
                name='average_map_kernel')

    def update_map_kernel(self, points, R, t):
        self.new_map *= 0.0
        self.add_points_kernel(points, self.center[0], self.center[1], R, t,
                          self.elevation_map, self.new_map,
                          size=(points.shape[0]))
        self.average_map_kernel(self.new_map, self.elevation_map, size=(self.cell_n * self.cell_n))

        # calculate traversability
        traversability = self.traversability_filter(self.elevation_map)
        self.elevation_map[3][3:-3, 3:-3] = traversability.reshape((traversability.shape[2], traversability.shape[3]))

    def update_variance(self):
        self.elevation_map[1] += self.time_variance * self.elevation_map[2]

    def load(self, points, R, t):
        self.load_points(points)
        if use_cupy:
            self.raw_points = self.points_gpu[~xp.isnan(self.points_gpu).any(axis=1)]
        else:
            self.raw_points = self.points_cpu[~xp.isnan(self.points_cpu).any(axis=1)]
        self.R = xp.asarray(R)
        self.t = xp.asarray(t)

    def calculate(self):
        self.update_map_kernel(self.raw_points, self.R, self.t)

    def input(self, raw_points, R, t):
        raw_points = xp.asarray(raw_points)
        raw_points = raw_points[~xp.isnan(raw_points).any(axis=1)]
        # print(raw_points)
        # print(R)
        # print(t)
        # print(use_cupy)
        if use_cupy:
            self.update_map_kernel(raw_points, xp.asarray(R), xp.asarray(t))
        else:
            points = self.add_noise(raw_points)
            points = self.transform_points(points, xp.asarray(R), xp.asarray(t))
            self.update_map_xp(points)
        # print('finish processing')

    def get_maps(self):
        elevation = xp.where(self.elevation_map[2] > 0.5,
                             self.elevation_map[0].copy(), xp.nan)
        variance = self.elevation_map[1].copy()
        traversability = self.elevation_map[3].copy()
        elevation = elevation[1:-1, 1:-1]
        variance = variance[1:-1, 1:-1]
        traversability = traversability[1:-1, 1:-1]

        maps = xp.stack([elevation, variance, traversability], axis=0)
        # maps = xp.transpose(maps, axes=(0, 2, 1))
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        if use_cupy:
            maps = xp.asnumpy(maps)
        return maps

    def get_maps_ref(self, elevation_data, variance_data, traversability_data):
        elevation = xp.where(self.elevation_map[2] > 0.5,
                             self.elevation_map[0].copy(), xp.nan)
        variance = self.elevation_map[1].copy()
        traversability = self.elevation_map[3].copy()
        elevation = elevation[1:-1, 1:-1]
        variance = variance[1:-1, 1:-1]
        traversability = traversability[1:-1, 1:-1]

        maps = xp.stack([elevation, variance, traversability], axis=0)
        # maps = xp.transpose(maps, axes=(0, 2, 1))
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        if use_cupy:
            elevation_data[...] = xp.asnumpy(maps[0])
            stream = cp.cuda.Stream(non_blocking=True)
            # elevation_data[...] = xp.asnumpy(maps[0], stream=stream)
            variance_data[...] = xp.asnumpy(maps[1], stream=stream)
            traversability_data[...] = xp.asnumpy(maps[2], stream=stream)
        else:
            elevation_data[...] = maps[0]
            variance_data[...] = maps[1]
            traversability_data[...] = maps[2]


if __name__ == '__main__':
    load_backend(True)
    xp.random.seed(123)
    points = xp.random.rand(100000, 3)
    R = xp.random.rand(3, 3)
    t = xp.random.rand(3)
    print(R, t)
    param = Parameter()
    param.load_weights('../config/weights.yaml')
    elevation = ElevationMap(param)
    for i in range(200):
        elevation.input(points, R, t)
        print(i)
