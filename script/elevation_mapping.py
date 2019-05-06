import numpy as np
import scipy as nsp
import scipy.ndimage

import matplotlib.pylab as plt

import chainer
import chainer.links as L
import chainer.functions as F

# import open3d as od

import time

use_cupy = True
# use_cupy = False
if use_cupy:
    import cupy as cp
    import cupyx.scipy as csp
    import cupyx.scipy.ndimage
    xp = cp
    sp = csp
else:
    xp = np
    sp = nsp


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


    def __call__(self, elevation_map):
        elevation = elevation_map[0]
        padded = F.pad(elevation, 1, 'reflect')
        out1 = self.conv1(padded.reshape(-1, 1, padded.shape[0], padded.shape[1]))
        padded = F.pad(elevation, 2, 'reflect')
        out2 = self.conv2(padded.reshape(-1, 1, padded.shape[0], padded.shape[1]))
        padded = F.pad(elevation, 3, 'reflect')
        out3 = self.conv3(padded.reshape(-1, 1, padded.shape[0], padded.shape[1]))
        out = F.concat((out1, out2, out3), axis=1)
        return F.absolute(self.conv_out(out)).array


class ElevationMap(object):
    def __init__(self):
        self.resolution = 0.02
        self.center = xp.array([0, 0], dtype=float)
        self.map_length = 8 
        # +2 is a border for outside map
        self.cell_n = int(self.map_length / self.resolution) + 2

        # 'mean' or 'max'
        # self.gather_mode = 'mean'
        self.gather_mode = 'max'

        self.noise_factor = 0.05
        self.mahalanobis_thresh = 2.0
        self.outlier_variance = 0.01
        self.time_variance = 0.01

        self.max_variance = 1.0

        # layers: elevation, variance, is_valid, traversability
        self.elevation_map = xp.zeros((4, self.cell_n, self.cell_n))
        # Initial variance
        self.initial_variance = 10.0
        self.elevation_map[1] += self.initial_variance

        w1 = np.array([[[[-1.6650e-01, -1.6682e-01, -1.6663e-01],
                        [-7.5112e-06, -4.0844e-05,  3.7997e-05],
                        [ 1.6670e-01,  1.6667e-01,  1.6660e-01]]],
                      [[[-1.6653e-01,  3.0289e-05,  1.6651e-01],
                        [-1.6677e-01, -5.4666e-05,  1.6669e-01],
                        [-1.6665e-01,  1.2168e-04,  1.6665e-01]]],
                      [[[-1.3175e-04,  1.2513e-01,  2.4983e-01],
                        [-1.2505e-01,  3.6682e-06,  1.2503e-01],
                        [-2.4988e-01, -1.2494e-01,  9.7242e-06]]],
                      [[[ 2.4968e-01,  1.2499e-01, -1.5196e-05],
                        [ 1.2511e-01, -2.3487e-04, -1.2501e-01],
                        [ 2.1485e-04, -1.2496e-01, -2.4978e-01]]]])
        w2 = np.array([[[[-1.6604e-01, -1.6719e-01, -1.6669e-01],
                        [ 4.1097e-04, -7.9650e-05,  9.1703e-05],
                        [ 1.6636e-01,  1.6711e-01,  1.6602e-01]]],
                      [[[-1.6644e-01,  1.6302e-04,  1.6659e-01],
                        [-1.6714e-01,  3.3449e-05,  1.6717e-01],
                        [-1.6642e-01,  1.3836e-04,  1.6591e-01]]],
                      [[[-2.8285e-04,  1.2527e-01,  2.4939e-01],
                        [-1.2541e-01,  1.6673e-05,  1.2501e-01],
                        [-2.4916e-01, -1.2515e-01,  3.0937e-04]]],
                      [[[ 2.4936e-01,  1.2532e-01,  5.7826e-05],
                        [ 1.2510e-01, -7.0027e-05, -1.2552e-01],
                        [ 1.5795e-04, -1.2541e-01, -2.4900e-01]]]])
        w3 = np.array([[[[-1.6498e-01, -1.6708e-01, -1.6560e-01],
                        [ 9.8741e-04,  6.1481e-04, -2.3462e-03],
                        [ 1.6511e-01,  1.6745e-01,  1.6584e-01]]],
                      [[[-1.6636e-01,  5.5259e-04,  1.6628e-01],
                        [-1.6795e-01,  8.4194e-04,  1.6589e-01],
                        [-1.6569e-01,  3.2529e-04,  1.6612e-01]]],
                      [[[ 3.5657e-04,  1.2612e-01,  2.4948e-01],
                        [-1.2560e-01,  3.0604e-04,  1.2374e-01],
                        [-2.4845e-01, -1.2534e-01, -6.1795e-04]]],
                      [[[ 2.4942e-01,  1.2524e-01, -1.5642e-04],
                        [ 1.2534e-01, -3.7394e-04, -1.2408e-01],
                        [-3.9571e-04, -1.2554e-01, -2.4946e-01]]]])
        w_out = np.array([[[[1.2696]],
                           [[1.4777]],
                           [[1.7409]],
                           [[1.7916]],
                           [[0.9110]],
                           [[0.8669]],
                           [[1.3450]],
                           [[1.3525]],
                           [[0.2810]],
                           [[0.0376]],
                           [[0.5175]],
                           [[0.5921]]]])
        self.traversability_filter = TraversabilityFilter(w1, w2, w3, w_out)
        if use_cupy:
            self.traversability_filter.to_gpu()

        self.time_sum = 0.0
        self.time_cnt = 0.0


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
        n = self.noise_factor * z * z
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
        outlier_map_index = (outlier_index[:, 0], outlier_index[:, 1])
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

    def update_map(self, points):
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
        idx, h, v = self.gather_into_unique_cell(index, new_h, new_v, mode=self.gather_mode)
        self.elevation_map[0][idx] = h
        self.elevation_map[1][idx] = v
        self.elevation_map[2][idx] = 1

        self.large_variance_rejection()

        # calculate traversability
        traversability = self.traversability_filter(self.elevation_map)
        self.elevation_map[3] = traversability.reshape((self.cell_n, self.cell_n))

    def update_variance(self):
        self.elevation_map[1] += self.time_variance * self.elevation_map[2]

    def input(self, raw_points, R, t):
        points = self.add_noise(xp.asarray(raw_points))
        points = self.transform_points(points, xp.asarray(R), xp.asarray(t))
        self.update_map(points)

    def get_maps(self):
        elevation = xp.where(self.elevation_map[2] > 0.5,
                             self.elevation_map[0].copy(), xp.nan)
        variance = self.elevation_map[1].copy()
        traversability = self.elevation_map[3].copy()
        elevation = elevation[1:-1, 1:-1]
        variance = variance[1:-1, 1:-1]
        traversability = traversability[1:-1, 1:-1]

        maps = xp.stack([elevation, variance, traversability], axis=0)
        if use_cupy:
            maps = xp.asnumpy(maps)
        maps = xp.transpose(maps, axes=(0, 2, 1))
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        return maps

    def show(self):
        if use_cupy:
            plt.imshow(xp.asnumpy(self.elevation_map[0]))
            plt.show()
            plt.imshow(xp.asnumpy(self.elevation_map[1]))
            plt.show()
            plt.imshow(xp.asnumpy(self.elevation_map[2]))
            plt.show()
        else:
            plt.imshow(self.elevation_map[0])
            plt.show()
            plt.imshow(self.elevation_map[1])
            plt.show()
            plt.imshow(self.elevation_map[2])
            plt.show()
