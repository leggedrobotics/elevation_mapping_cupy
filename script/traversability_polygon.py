import numpy as np
import cupy as cp


def get_masked_traversability(map_array, mask):
    traversability = map_array[1:-1, 1:-1]
    mask = mask[1:-1, 1:-1]
    masked = traversability * mask
    return masked


def is_traversable(masked_traversability, thresh, max_over_n):
    traversability_map = cp.where(masked_traversability == 0, cp.nan, masked_traversability)
    over_thresh = cp.where(masked_traversability > thresh, 1, 0)
    # print(traversability_map)
    # print('over_thresh', over_thresh.sum())
    max_traversability = masked_traversability.max()
    # print('max_traversability ', max_traversability)
    # print('mean', masked_traversability.mean())
    if over_thresh.sum() > max_over_n:
        return False
    else:
        return True


def calculate_area(polygon):
    area = 0
    for i in range(len(polygon)):
        p1 = polygon[i - 1]
        p2 = polygon[i]
        area += (p1[0] * p2[1] - p1[1] * p2[0]) / 2.
    return abs(area)


if __name__ == '__main__':
    polygon = [[0, 0], [2, 0], [0, 2]]
    print(calculate_area(polygon))
