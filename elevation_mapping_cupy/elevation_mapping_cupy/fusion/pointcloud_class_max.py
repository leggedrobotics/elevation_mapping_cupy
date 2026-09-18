#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def sum_max_kernel(
    resolution, width, height,
):
    # input the list of layers, amount of channels can slo be input through kernel
    sum_max_kernel = cp.ElementwiseKernel(
        in_params="raw U p, raw U max_pt, raw T max_id, raw W pcl_chan, raw W map_lay, raw W pcl_channels",
        out_params=" raw U newmap",
        preamble=string.Template(
            """
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }
            """
        ).substitute(resolution=resolution, width=width, height=height),
        operation=string.Template(
            """
            U idx = p[i * pcl_channels[0]];
            U valid = p[i * pcl_channels[0] + 1];
            U inside = p[i * pcl_channels[0] + 2];
            if (valid) {
                if (inside) {
                    // for every max value
                    for ( W it=0;it<pcl_channels[2];it++){
                        U prob = max_pt[i * pcl_channels[2] + it];
                        T id = max_id[i * pcl_channels[2] + it];
                        atomicAdd(&newmap[get_map_idx(idx, id)], prob);
                    }
                }
            }
            """
        ).substitute(),
        name="sum_max_kernel",
    )
    return sum_max_kernel


class ClassMax(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "pointcloud_class_max"
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.fusion_algorithms = params.fusion_algorithms

        self.sum_max_kernel = sum_max_kernel(self.resolution, self.cell_n, self.cell_n,)
        layer_cnt = self.fusion_algorithms.count("class_max")

        self.unique_id = cp.array([0])

    def decode_max(self, mer):
        """Decode the float32 value into two 16 bit value containing the class probability and the class id.

        Args:
            mer:

        Returns:
            cp.array: probability
            cp.array: class id
        """
        mer = mer.astype(cp.float32)
        mer = mer.view(dtype=cp.uint32)
        ma = cp.bitwise_and(mer, 0xFFFF, dtype=np.uint16)
        ma = ma.view(np.float16)
        ma = ma.astype(np.float32)
        ind = cp.right_shift(mer, 16)
        return ma, ind

    def __call__(self, points_all, R, t, pcl_ids, layer_ids, elevation_map, semantic_map, new_map, elements_to_shift):
        max_pt, pt_id = self.decode_max(points_all[:, pcl_ids])
        # find unique ids in new measurement and in existing map
        unique_idm = cp.unique(pt_id)
        unique_ida = cp.unique(self.unique_id[elements_to_shift["id_max"]])
        # get all unique ids, where index is the position in the prob_sum and the value in the NN class
        self.unique_id = cp.unique(cp.concatenate((unique_idm, unique_ida)))
        # contains the sum of the new measurement probabilities
        self.prob_sum = cp.zeros((len(self.unique_id), self.cell_n, self.cell_n), dtype=np.float32,)
        # transform the index matrix of the classes to the index matrix of the prob_sum
        pt_id_zero = pt_id.copy()
        for it, val in enumerate(self.unique_id):
            pt_id_zero[pt_id_zero == val] = it

        # sum all measurements probabilities
        self.sum_max_kernel(
            points_all,
            max_pt,
            pt_id_zero,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0], pt_id.shape[1]], dtype=cp.int32,),
            self.prob_sum,
            size=(points_all.shape[0]),
        )
        # add the previous alpha
        for i, lay in enumerate(layer_ids):
            c = cp.mgrid[0 : new_map.shape[1], 0 : new_map.shape[2]]
            # self.prob_sum[self.elements_to_shift["id_max"][i], c[0], c[1]] += self.new_map[lay]
            # TODO add residual of prev alpha to the prob_sum
            # res = 1- self.new_map[lay]
            # res /= (len(self.unique_id)-1)

        # find the alpha we want to keep
        for i, lay in enumerate(layer_ids):
            new_map[lay] = cp.amax(self.prob_sum, axis=0)
            elements_to_shift["id_max"][lay] = self.unique_id[cp.argmax(self.prob_sum, axis=0)]
            self.prob_sum[cp.argmax(self.prob_sum, axis=0)] = 0
        # update map calculate new thetas
        sum_alpha = cp.sum(new_map[layer_ids], axis=0)
        # do not divide by zero
        sum_alpha[sum_alpha == 0] = 1
        semantic_map[layer_ids] = new_map[layer_ids] / cp.expand_dims(sum_alpha, axis=0)
