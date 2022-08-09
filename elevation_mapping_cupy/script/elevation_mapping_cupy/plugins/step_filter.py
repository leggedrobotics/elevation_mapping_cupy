import cupy as cp
import string
from typing import List

from .plugin_manager import PluginBase


class StepFilter(PluginBase):
    def __init__(self, cell_n: int = 100, radius: int = 1, include_negative_steps: bool = False, **kwargs):
        super().__init__()

        self.width = cell_n
        self.height = cell_n

        include_negative_steps_string = "true" if include_negative_steps else "false"

        self.step_filtered = cp.zeros((self.width, self.height))
        self.step_filter_kernel = cp.ElementwiseKernel(
            in_params="raw U map",
            out_params="raw U resultmap",
            preamble=string.Template(
                """
                __device__ int get_map_idx(int idx, int layer_n)
                {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }

                __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n)
                {
                    const int layer = ${width} * ${height};
                    const int relative_idx = idx + ${width} * dy + dx;
                    return layer * layer_n + relative_idx;
                }

                __device__ bool is_inside(int idx)
                {
                    int idx_x = idx / ${width};
                    int idx_y = idx % ${width};
                    if (idx_x <= 0 || idx_x >= ${width} - 1)
                    {
                        return false;
                    }
                    if (idx_y <= 0 || idx_y >= ${height} - 1)
                    {
                        return false;
                    }
                    return true;
                }
                """
            ).substitute(width=self.width, height=self.height),
            operation=string.Template(
                """
                U center_value = map[get_map_idx(i, 0)];

                U max_distance = 0.0;
                for (int dy = -${radius}; dy <= ${radius}; ++dy)
                {
                  for (int dx = -${radius}; dx <= ${radius}; ++dx)
                  {
                    int idx = get_relative_map_idx(i, dx, dy, 0);
                    int valid_idx = get_relative_map_idx(i, dx, dy, 2);
                    U is_valid = map[valid_idx];
                    if (!is_inside(idx) || is_valid < 0.5)
                    {
                      continue;
                    }

                    U distance = center_value - map[idx];

                    if (${include_negative_steps} && distance < 0.0)
                    {
                      distance = -distance;
                    }

                    if (distance > max_distance)
                    {
                      max_distance = distance;
                    }
                  }
                }

                resultmap[get_map_idx(i, 0)] = max_distance;
                """
            ).substitute(radius=radius, include_negative_steps=include_negative_steps_string),
            name="step_filter_kernel",
        )

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
    ) -> cp.ndarray:

        elevation_idx = layer_names.index("elevation")

        self.step_filtered *= 0.0

        self.step_filter_kernel(
            elevation_map,
            self.step_filtered,
            size=(self.width * self.height),
        )

        return self.step_filtered
