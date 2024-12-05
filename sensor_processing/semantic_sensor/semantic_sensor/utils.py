import numpy as np
import cupy as cp

from detectron2.utils.logger import setup_logger

setup_logger()


def encode_max(maxim, index):
    maxim, index = cp.asarray(maxim, dtype=cp.float32), cp.asarray(index, dtype=cp.uint32)
    # fuse them
    maxim = maxim.astype(cp.float16)
    maxim = maxim.view(cp.uint16)
    maxim = maxim.astype(cp.uint32)
    index = index.astype(cp.uint32)
    mer = cp.array(cp.left_shift(index, 16) | maxim, dtype=cp.uint32)
    mer = mer.view(cp.float32)
    return mer


def decode_max(mer):
    mer = mer.astype(cp.float32)
    mer = mer.view(dtype=cp.uint32)
    ma = cp.bitwise_and(mer, 0xFFFF, dtype=np.uint16)
    ma = ma.view(np.float16)
    ma = ma.astype(np.float32)
    ind = cp.right_shift(mer, 16)
    return ma, ind
