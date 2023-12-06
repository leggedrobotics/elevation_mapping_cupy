import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def sum_kernel(
    resolution, width, height,
):
    """Sums the semantic values of the classes for the exponentiala verage or for the average.

    Args:
        resolution:
        width:
        height:

    Returns:

    """
    # input the list of layers, amount of channels can slo be input through kernel
    sum_kernel = cp.ElementwiseKernel(
        in_params="raw U p, raw U R, raw U t, raw W pcl_chan, raw W map_lay, raw W pcl_channels",
        out_params="raw U map, raw U newmap",
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
            U id = floorf(i/pcl_channels[1]);
            int layer = i % pcl_channels[1];
            U idx = p[id * pcl_channels[0]];
            U valid = p[id * pcl_channels[0] + 1];
            U inside = p[id * pcl_channels[0] + 2];
            if (valid) {
                if (inside) {
                    U feat = p[id * pcl_channels[0] + pcl_chan[layer]];
                    atomicAdd(&newmap[get_map_idx(idx, map_lay[layer])], feat);
                }
            }
            """
        ).substitute(),
        name="sum_kernel",
    )
    return sum_kernel


def average_kernel(
    width, height,
):
    average_kernel = cp.ElementwiseKernel(
        in_params="raw V newmap, raw W pcl_chan, raw W map_lay, raw W pcl_channels, raw U new_elmap",
        out_params="raw U map",
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
            U id = floorf(i/pcl_channels[1]);
            int layer = i % pcl_channels[1];
            U cnt = new_elmap[get_map_idx(id, 2)];
            if (cnt>0){
                U feat = newmap[get_map_idx(id,  map_lay[layer])]/(1*cnt);
                map[get_map_idx(id,  map_lay[layer])] = feat;
            }
            """
        ).substitute(),
        name="average_map_kernel",
    )
    return average_kernel


class Average(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "pointcloud_average"
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.sum_kernel = sum_kernel(self.resolution, self.cell_n, self.cell_n,)
        self.average_kernel = average_kernel(self.cell_n, self.cell_n,)

    def __call__(self, points_all, R, t, pcl_ids, layer_ids, elevation_map, semantic_map, new_map, *args):
        self.sum_kernel(
            points_all,
            R,
            t,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            semantic_map,
            new_map,
            size=(points_all.shape[0] * pcl_ids.shape[0]),
        )
        self.average_kernel(
            new_map,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            elevation_map,
            semantic_map,
            size=(self.cell_n * self.cell_n * pcl_ids.shape[0]),
        )
