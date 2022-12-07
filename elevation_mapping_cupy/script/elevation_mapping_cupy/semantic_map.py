from elevation_mapping_cupy.parameter import Parameter
import cupy as cp
import numpy as np
from typing import List
from elevation_mapping_cupy.kernels import sum_kernel, add_color_kernel, color_average_kernel
from elevation_mapping_cupy.kernels import average_kernel, class_average_kernel, alpha_kernel
from elevation_mapping_cupy.kernels import average_correspondences_to_map_kernel, color_correspondences_to_map_kernel

xp = cp


class SemanticMap:
    def __init__(self, param: Parameter):
        """

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """

        self.param = param

        # TODO: maps channel to fusion strategy (maybe rename)
        self.layer_specs = {}
        self.layer_names = []
        self.unique_fusion = []

        for k, config in self.param.subscriber_cfg.items():
            for f, c in zip(config["fusion"], config["channels"]):
                if c not in self.layer_names:
                    self.layer_names.append(c)
                    self.layer_specs[c] = f
                else:
                    assert self.layer_specs[c] == f, "Error: Single layer has multiple fusion algorithms!"

                if f not in self.unique_fusion:
                    self.unique_fusion.append(f)

        self.amount_layer_names = len(self.layer_names)

        self.semantic_map = xp.zeros(
            (self.amount_layer_names, self.param.cell_n, self.param.cell_n),
            dtype=param.data_type,
        )
        self.new_map = xp.zeros(
            (self.amount_layer_names, self.param.cell_n, self.param.cell_n),
            param.data_type,
        )

    def clear(self):
        self.semantic_map *= 0.0

    def compile_kernels(self) -> None:
        """
        Returns:
            None:
        """
        # TODO: maybe this could be improved by creating functions for each single
        # create a base class containing compile kernel as well as update and this then can be created for various
        # then for each method this two functions canbe overloaded. and called through a list of all the algporithms
        # that need to be called
        # but I am not sure as we would need to pass a lot of arguments... check plugin
        if "average" in self.unique_fusion:
            print("Initialize fusion kernel")
            self.sum_kernel = sum_kernel(
                self.param.resolution,
                self.param.cell_n,
                self.param.cell_n,
            )
            self.average_kernel = average_kernel(
                self.param.cell_n,
                self.param.cell_n,
            )

        if "color" in self.unique_fusion:
            print("Initialize color kernel")

            self.add_color_kernel = add_color_kernel(
                self.param.cell_n,
                self.param.cell_n,
            )
            self.color_average_kernel = color_average_kernel(self.param.cell_n, self.param.cell_n)
        if "class_average" in self.unique_fusion:
            print("Initialize class average kernel")
            self.sum_kernel = sum_kernel(
                self.param.resolution,
                self.param.cell_n,
                self.param.cell_n,
            )
            self.class_average_kernel = class_average_kernel(
                self.param.cell_n,
                self.param.cell_n,
            )
        if "class_bayesian" in self.unique_fusion:
            print("Initialize class bayesian kernel")
            self.alpha_kernel = alpha_kernel(
                self.param.resolution,
                self.param.cell_n,
                self.param.cell_n,
            )

        if "image_exponential" in self.unique_fusion:
            self.average_correspondences_to_map_kernel = average_correspondences_to_map_kernel(
                resolution=self.param.resolution, width=self.param.cell_n, height=self.param.cell_n
            )

        if "image_color" in self.unique_fusion:
            self.color_correspondences_to_map_kernel = color_correspondences_to_map_kernel(
                resolution=self.param.resolution, width=self.param.cell_n, height=self.param.cell_n
            )

    def get_fusion_of_pcl(self, channels: List[str]) -> List[str]:
        """Get all fusion algorithms that need to be applied to a specific pointcloud

        Args:
            channels (List[str]):
        """
        fusion_list = []
        for channel in channels:
            x = self.layer_specs[channel]
            if x not in fusion_list:
                fusion_list.append(x)
        return fusion_list

    def get_indices_fusion(self, pcl_channels: List[str], fusion_alg: str):
        """Computes the indices that are used for the additional kernel update of the popintcloud and the semantic map.

        Args:
            pcl_channels (List[str]):
            fusion_alg (str):

        Returns:
            Union[Tuple[List[int], List[int]], Tuple[cupy._core.core.ndarray, cupy._core.core.ndarray]]:


        """
        # this contains exactly the fusion alg type for each channel of the pcl
        pcl_val_list = [self.layer_specs[x] for x in pcl_channels]
        # this contains the indeces of the pointcloud where we have to perform a certain fusion
        pcl_indices = cp.array(
            [idp + 3 for idp, x in enumerate(pcl_val_list) if x == fusion_alg],
            dtype=np.int32,
        )
        # create a list of indeces of the layers that will be updated by the pointcloud with specific fusion alg
        layer_indices = cp.array([], dtype=np.int32)
        for it, (key, val) in enumerate(self.layer_specs.items()):
            if key in pcl_channels and val == fusion_alg:
                layer_indices = cp.append(layer_indices, it).astype(np.int32)
        return pcl_indices, layer_indices

    def update_layers_pointcloud(self, points_all, channels, R, t, elevation_map):
        additional_fusion = self.get_fusion_of_pcl(channels)
        self.new_map *= 0.0
        if "average" in additional_fusion:
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "average")
            self.sum_kernel(
                points_all,
                R,
                t,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=np.int32),
                self.semantic_map,
                self.new_map,
                size=(points_all.shape[0]),
            )
            self.average_kernel(
                self.new_map,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=np.int32),
                elevation_map,
                self.semantic_map,
                size=(self.param.cell_n * self.param.cell_n),
            )
        if "class_average" in additional_fusion:
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "class_average")
            self.sum_kernel(
                points_all,
                R,
                t,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=np.int32),
                self.semantic_map,
                self.new_map,
                size=(points_all.shape[0]),
            )
            self.class_average_kernel(
                self.new_map,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=np.int32),
                elevation_map,
                self.semantic_map,
                size=(self.param.cell_n * self.param.cell_n),
            )

        if "class_bayesian" in additional_fusion:
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "class_bayesian")
            # alpha sum get points as input and calculate for each point to what cell it belongs and then
            # adds to the right channel a one
            self.alpha_kernel(
                points_all,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=np.int32),
                self.new_map,
                size=(points_all.shape[0]),
            )
            # calculate new thetas
            sum_alpha = cp.sum(self.new_map[layer_ids], axis=0)
            self.semantic_map[layer_ids] = self.new_map[layer_ids] / cp.expand_dims(sum_alpha, axis=0)
            # assert  cp.unique(cp.sum(self.semantic_map[layer_ids], axis=0)) equal to zero or to nan

        if "color" in additional_fusion:
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "color")
            self.color_map = cp.zeros(
                (1 + 3 * layer_ids.shape[0], self.param.cell_n, self.param.cell_n),
                dtype=np.uint32,
            )

            points_all = points_all.astype(cp.float32)
            self.add_color_kernel(
                points_all,
                R,
                t,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=np.int32),
                self.color_map,
                size=(points_all.shape[0]),
            )
            self.color_average_kernel(
                self.color_map,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=np.int32),
                self.semantic_map,
                size=(self.param.cell_n * self.param.cell_n),
            )

    def update_layers_image(
        self,
        sub_key: str,
        image: cp._core.core.ndarray,
        uv_correspondence: cp._core.core.ndarray,
        valid_correspondence: cp._core.core.ndarray,
        image_height: cp._core.core.ndarray,
        image_width: cp._core.core.ndarray,
    ):

        self.new_map *= 0
        config = self.param.subscriber_cfg[sub_key]

        for j, (fusion, channel) in enumerate(zip(config["fusion"], config["channels"])):
            sem_map_idx = self.get_index(channel)

            if fusion == "image_exponential":
                self.average_correspondences_to_map_kernel(
                    self.semantic_map,
                    cp.uint64(sem_map_idx),
                    image[j],
                    uv_correspondence,
                    valid_correspondence,
                    image_height,
                    image_width,
                    self.new_map,
                    size=int(self.param.cell_n * self.param.cell_n),
                )
                self.semantic_map[sem_map_idx] = self.new_map[sem_map_idx]

            elif fusion == "image_color":
                self.color_correspondences_to_map_kernel(
                    self.semantic_map,
                    cp.uint64(sem_map_idx),
                    image,
                    uv_correspondence,
                    valid_correspondence,
                    image_height,
                    image_width,
                    self.new_map,
                    size=int(self.param.cell_n * self.param.cell_n),
                )
                self.semantic_map[sem_map_idx] = self.new_map[sem_map_idx]

            else:
                raise ValueError("Fusion for image is unkown.")

    def get_map_with_name(self, name):
        if self.layer_specs[name] == "color":
            m = self.get_rgb(name)
            return m
        else:
            m = self.get_semantic(name)
            return m

    def get_rgb(self, name):
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        c = c.astype(np.float32)
        # c = xp.uint32(c.get())
        # c.dtype = np.float32
        return c

    def get_semantic(self, name):
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        return c

    def process_map_for_publish(self, input_map):
        m = input_map.copy()
        return m[1:-1, 1:-1]

    def get_index(self, name):
        return self.layer_names.index(name)
