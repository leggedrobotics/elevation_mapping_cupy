def get_weighted_sum(map_array, mask):
    traversability = map_array[1:-1, 1:-1]
    mask = mask[1:-1, 1:-1]
    masked = traversability * mask
    return masked.sum()
