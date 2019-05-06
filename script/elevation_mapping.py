import numpy as np
import scipy as nsp
import scipy.ndimage

import matplotlib.pylab as plt

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


class ElevationMap(object):
    def __init__(self):
        self.resolution = 0.05
        self.center = xp.array([0, 0], dtype=float)
        self.map_length =10 
        # +2 is a border for outside map
        self.cell_n = int(self.map_length / self.resolution) + 2

        self.noise_factor = 0.05
        self.mahalanobis_thresh = 2.0
        self.outlier_variance = 0.10
        self.time_variance = 0.01

        self.max_variance = 1.0

        # layers: elevation, variance, is_valid
        self.elevation_map = xp.zeros((3, self.cell_n, self.cell_n))
        # Initial variance
        self.initial_variance = 10.0
        self.elevation_map[1] += self.initial_variance

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

    def get_indices(self, points, index):
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

    def outlier_rejection(self):

        outliers = self.elevation_map[1] > self.max_variance
        self.elevation_map[0] = xp.where(outliers, 0,
                                         self.elevation_map[0].copy())
        self.elevation_map[1] = xp.where(outliers, self.initial_variance,
                                         self.elevation_map[1].copy())
        self.elevation_map[2] = xp.where(outliers, 0,
                                         self.elevation_map[2].copy())
        # outliers = variance > self.max_variance
        # print(outliers)
        # outlier_index = [outliers[:, 0], outliers[:, 1]]
        # print(outlier_index)
        # self.elevation_map[2][outlier_index] = 0

    def update_map(self, points):
        self.update_variance()
        index = self.get_cell_index(points)
        map_h = self.elevation_map[0][index[:, 0], index[:, 1]]
        map_v = self.elevation_map[1][index[:, 0], index[:, 1]]
        point_h = points[:, 2]
        point_v = points[:, 3]
        # outlier rejection
        outliers = xp.abs(map_h - point_h) > (map_v * self.mahalanobis_thresh)
        outlier_index = index[outliers]
        outlier_map_index = (outlier_index[:, 0], outlier_index[:, 1])
        self.elevation_map[1][outlier_map_index] += self.outlier_variance
        index = index[~outliers]
        map_h = map_h[~outliers]
        map_v = map_v[~outliers]
        point_h = point_h[~outliers]
        point_v = point_v[~outliers]
        unique_tuple = self.get_indices(points, index)
        index, unique_index, unique_inverse, unique_count = unique_tuple
        new_h = ((map_h * point_v + point_h * map_v) / (map_v + point_v))
        new_v = (map_v * point_v) / (map_v + point_v)

        new_unique_h = xp.bincount(unique_inverse, new_h) / unique_count
        new_unique_v = xp.bincount(unique_inverse, new_v) / unique_count
        index_x, index_y = unique_index[:, 0], unique_index[:, 1]
        self.elevation_map[0][index_x, index_y] = new_unique_h
        self.elevation_map[1][index_x, index_y] = new_unique_v
        self.elevation_map[2][index_x, index_y] = 1
        self.outlier_rejection()

    def update_variance(self):
        self.elevation_map[1] += self.time_variance * self.elevation_map[2]

    def input(self, raw_points, R, t):
        start = time.time()
        points = self.add_noise(xp.asarray(raw_points))
        # print('add noise', time.time() - start)
        points = self.transform_points(points, xp.asarray(R), xp.asarray(t))
        # print('transform', time.time() - start)
        self.update_map(points)
        # print('update map', time.time() - start)

    def get_maps(self):
        elevation = xp.where(self.elevation_map[2] > 0.5,
                             self.elevation_map[0].copy(), xp.nan)
        variance = self.elevation_map[1].copy()
        elevation = elevation[1:-2, 1:-2]
        variance = variance[1:-2, 1:-2]
        maps = xp.stack([elevation, variance], axis=0)
        if use_cupy:
            maps = xp.asnumpy(maps)
        # return elevation, variance
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


# def load_points():
#     print("Load a ply point cloud, print it, and render it")
#     pcd = od.read_point_cloud("./fragment.ply")
#     print(pcd)
#     downpcd = od.voxel_down_sample(pcd, voxel_size=0.010)
#
#     return downpcd
#     # return pcd
#
#
# # @xp.fuse()
# def main():
#     pcd = load_points()
#     # od.draw_geometries([pcd])
#     # print(points)
#     elevation_map = ElevationMap()
#
#     theta = -0.25 * xp.pi
#     # theta = 0
#     R = xp.array([[xp.cos(theta), -xp.sin(theta), 0],
#                  [xp.sin(theta), xp.cos(theta), 0],
#                  [0, 0, 1]], dtype=xp.float)
#     t = xp.array([0.1, 0.2, -0.1])
#     start = time.time()
#     for i in range(1000):
#         points = xp.asarray(pcd.points)
#         elevation_map.input(points, R, t)
#     elapsed = time.time() - start
#     print(elapsed)
#     elevation_map.show()
#
#
# if __name__ == '__main__':
#     main()
