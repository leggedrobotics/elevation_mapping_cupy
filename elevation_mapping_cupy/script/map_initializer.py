#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from scipy.interpolate import griddata
import numpy as np
import cupy as cp


class MapInitializer(object):
    def __init__(self, initial_variance, new_variance, xp=np, method="points"):
        self.methods = ["points"]
        assert method in self.methods, "method should be chosen from {}".format(self.methods)
        self.method = method
        self.xp = xp
        self.initial_variance = initial_variance
        self.new_variance = new_variance

    def __call__(self, *args, **kwargs):
        if self.method == "points":
            self.points_initializer(*args, **kwargs)
        else:
            return

    def points_initializer(self, elevation_map, points, method="linear"):
        """
        Initialize the map using interpolation between given poitns
        Args:
        elevation_map: elevation_map data.
        points: points used to interpolate.
        method: method for interpolation. (nearest, linear, cubic)
        """

        # points from existing map.
        points_idx = self.xp.where(elevation_map[2] > 0.5)
        values = elevation_map[0, points_idx[0], points_idx[1]]

        # Add external points for interpolation.
        points_idx = self.xp.stack(points_idx).T
        points_idx = self.xp.vstack([points_idx, points[:, :2]])
        values = self.xp.hstack([values, points[:, 2]])

        assert points_idx.shape[0] > 3, "Initialization points must be more than 3."

        # Interpolation using griddata function.
        w = elevation_map.shape[1]
        h = elevation_map.shape[2]
        grid_x, grid_y = np.mgrid[0:w, 0:h]
        if self.xp == cp:
            points_idx = cp.asnumpy(points_idx)
            values = cp.asnumpy(values)
        interpolated = griddata(points_idx, values, (grid_x, grid_y), method=method)
        if self.xp == cp:
            interpolated = cp.asarray(interpolated)

        # Update elevation map.
        elevation_map[0] = self.xp.nan_to_num(interpolated)
        elevation_map[1] = self.xp.where(
            self.xp.invert(self.xp.isnan(interpolated)), self.new_variance, self.initial_variance
        )
        elevation_map[2] = self.xp.where(self.xp.invert(self.xp.isnan(interpolated)), 1.0, 0.0)
        return


if __name__ == "__main__":
    initializer = MapInitializer(100, 10, method="points", xp=cp)
    m = np.zeros((4, 10, 10))
    m[0, 0:5, 2:5] = 0.3
    m[2, 0:5, 2:5] = 1.0
    np.set_printoptions(threshold=100)
    print(m[0])
    print(m[1])
    print(m[2])
    points = cp.array([[0, 0, 0.2], [8, 0, 0.2], [6, 9, 0.2]])
    # [3, 3, 0.2]])
    m = cp.asarray(m)
    initializer(m, points, method="cubic")
    print(m[0])
    print(m[1])
    print(m[2])
