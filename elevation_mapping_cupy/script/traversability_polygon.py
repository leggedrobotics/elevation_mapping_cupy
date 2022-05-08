#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np
import cupy as cp
from shapely.geometry import Polygon, MultiPoint


def get_masked_traversability(map_array, mask):
    traversability = map_array[3][1:-1, 1:-1]
    is_valid = map_array[2][1:-1, 1:-1]
    mask = mask[1:-1, 1:-1]

    # invalid place is 0 traversability value
    untraversability = cp.where(is_valid > 0.5, 1 - traversability, 0)
    masked = untraversability * mask
    masked_isvalid = is_valid * mask
    return masked, masked_isvalid


def is_traversable(masked_untraversability, thresh, min_thresh, max_over_n):
    untraversable_thresh = 1 - thresh
    max_thresh = 1 - min_thresh
    over_thresh = cp.where(masked_untraversability > untraversable_thresh, 1, 0)
    polygon = calculate_untraversable_polygon(over_thresh)
    max_untraversability = masked_untraversability.max()
    if over_thresh.sum() > max_over_n:
        is_safe = False
    elif max_untraversability > max_thresh:
        is_safe = False
    else:
        is_safe = True
    return is_safe, polygon


def calculate_area(polygon):
    area = 0
    for i in range(len(polygon)):
        p1 = polygon[i - 1]
        p2 = polygon[i]
        area += (p1[0] * p2[1] - p1[1] * p2[0]) / 2.0
    return abs(area)


def calculate_untraversable_polygon(over_thresh):
    x, y = cp.where(over_thresh > 0.5)
    points = cp.stack([x, y]).T
    convex_hull = MultiPoint(points).convex_hull
    if convex_hull.is_empty or convex_hull.geom_type == "Point" or convex_hull.geom_type == "LineString":
        return None
    else:
        return cp.array(convex_hull.exterior.coords)


def transform_to_map_position(polygon, center, cell_n, resolution):
    polygon = center.reshape(1, 2) + (polygon - cell_n / 2.0) * resolution
    return polygon


def transform_to_map_index(points, center, cell_n, resolution):
    indices = ((points - center.reshape(1, 2)) / resolution + cell_n / 2).astype(cp.int)
    return indices


if __name__ == "__main__":
    polygon = [[0, 0], [2, 0], [0, 2]]
    print(calculate_area(polygon))

    under_thresh = cp.zeros((20, 20))
    # under_thresh[10:12, 8:10] = 1.0
    under_thresh[14:18, 8:10] = 1.0
    under_thresh[1:8, 2:9] = 1.0
    print(under_thresh)
    polygon = calculate_untraversable_polygon(under_thresh)
    print(polygon)
    transform_to_map_position(polygon, cp.array([0.5, 1.0]), 6.0, 0.05)
