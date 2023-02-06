from elevation_mapping_cupy.parameter import Parameter
import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.kernels import (
    average_kernel,
    class_average_kernel,
    alpha_kernel,
    bayesian_inference_kernel,
    add_color_kernel,
    color_average_kernel,
    sum_compact_kernel,
    sum_max_kernel,
    sum_kernel,
)
from elevation_mapping_cupy.kernels import (
    average_correspondences_to_map_kernel,
    exponential_correspondences_to_map_kernel,
    color_correspondences_to_map_kernel,
)

xp = cp


class SemanticMap:
    def __init__(self, param: Parameter):
        """

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """

        self.param = param

        self.layer_specs = {}
        self.layer_names = []
        self.unique_fusion = []
        self.elements_to_shift = {}

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
        # which layers should be reset to zero at each update, per default everyone,
        # if a layer should not be reset, it is defined in compile_kernels function
        self.delete_new_layers = cp.ones(self.new_map.shape[0], cp.bool8)

    def clear(self):
        """Clear the semantic map.

        """
        self.semantic_map *= 0.0

    def compile_kernels(self) -> None:
        """
        Returns:
            None:
        """
        # TODO: maybe this could be improved by creating functions for each single fusion algorithm
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
        if "bayesian_inference" in self.unique_fusion:
            print("Initialize bayesian inference kernel")
            self.sum_mean = xp.ones(
                (
                    self.param.fusion_algorithms.count("bayesian_inference"),
                    self.param.cell_n,
                    self.param.cell_n,
                ),
                self.param.data_type,
            )
            # TODO initialize the variance with a value different than 0
            self.sum_compact_kernel = sum_compact_kernel(
                self.param.resolution,
                self.param.cell_n,
                self.param.cell_n,
            )
            self.bayesian_inference_kernel = bayesian_inference_kernel(
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
                self.param.average_weight,
            )
        if "class_bayesian" in self.unique_fusion:
            print("Initialize class bayesian kernel")
            pcl_ids = self.get_layer_indices("class_bayesian")
            self.delete_new_layers[pcl_ids] = 0
            self.alpha_kernel = alpha_kernel(
                self.param.resolution,
                self.param.cell_n,
                self.param.cell_n,
            )
        if "class_max" in self.unique_fusion:
            print("Initialize class max kernel")
            pcl_ids = self.get_layer_indices("class_max")
            self.delete_new_layers[pcl_ids] = 0
            self.sum_max_kernel = sum_max_kernel(
                self.param.resolution,
                self.param.cell_n,
                self.param.cell_n,
            )
            layer_cnt = self.param.fusion_algorithms.count("class_max")
            id_max = cp.zeros(
                (layer_cnt, self.param.cell_n, self.param.cell_n),
                dtype=cp.uint32,
            )
            self.elements_to_shift["id_max"] = id_max
            self.unique_id = cp.array([0])

        if "image_exponential" in self.unique_fusion:
            self.exponential_correspondences_to_map_kernel = exponential_correspondences_to_map_kernel(
                resolution=self.param.resolution,
                width=self.param.cell_n,
                height=self.param.cell_n,
                alpha=0.7
            )

        if "image_color" in self.unique_fusion:
            self.color_correspondences_to_map_kernel = color_correspondences_to_map_kernel(
                resolution=self.param.resolution,
                width=self.param.cell_n,
                height=self.param.cell_n,
            )

    def pad_value(self, x, shift_value, idx=None, value=0.0):
        """Create a padding of the map along x,y-axis according to amount that has shifted.

        Args:
            x (cupy._core.core.ndarray):
            shift_value (cupy._core.core.ndarray):
            idx (Union[None, int, None, None]):
            value (float):
        """
        if idx is None:
            if shift_value[0] > 0:
                x[:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:, :, shift_value[1] :] = value
        else:
            if shift_value[0] > 0:
                x[idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[idx, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[idx, :, shift_value[1] :] = value

    def shift_map_xy(self, shift_value):
        """Shift the map along x,y-axis according to shift values.

        Args:
            shift_value:
        """
        self.semantic_map = cp.roll(self.semantic_map, shift_value, axis=(1, 2))
        self.pad_value(self.semantic_map, shift_value, value=0.0)
        self.new_map = cp.roll(self.new_map, shift_value, axis=(1, 2))
        self.pad_value(self.new_map, shift_value, value=0.0)
        for el in self.elements_to_shift.values():
            el = cp.roll(el, shift_value, axis=(1, 2))
            self.pad_value(el, shift_value, value=0.0)

    def get_fusion_of_pcl(self, channels: List[str]) -> List[str]:
        """Get all fusion algorithms that need to be applied to a specific pointcloud.

        Args:
            channels (List[str]):
        """
        fusion_list = []
        for channel in channels:
            x = self.layer_specs[channel]
            if x not in fusion_list:
                fusion_list.append(x)
        return fusion_list

    def get_layer_indices(self, fusion_alg):
        """Get the indices of the layers that are used for a specific fusion algorithm.

        Args:
            fusion_alg(str): fusion algorithm name

        Returns:
            cp.array: indices of the layers
        """
        layer_indices = cp.array([], dtype=cp.int32)
        for it, (key, val) in enumerate(self.layer_specs.items()):
            if key in val == fusion_alg:
                layer_indices = cp.append(layer_indices, it).astype(cp.int32)
        return layer_indices

    def get_indices_fusion(self, pcl_channels: List[str], fusion_alg: str):
        """Computes the indices of the channels of the pointcloud and the layers of the semantic map of type fusion_alg.

        Args:
            pcl_channels (List[str]): list of all channel names
            fusion_alg (str): fusion algorithm type we want to use for channel selection

        Returns:
            Union[Tuple[List[int], List[int]], Tuple[cupy._core.core.ndarray, cupy._core.core.ndarray]]:


        """
        # this contains exactly the fusion alg type for each channel of the pcl
        pcl_val_list = [self.layer_specs[x] for x in pcl_channels]
        # this contains the indices of the point cloud where we have to perform a certain fusion
        pcl_indices = cp.array(
            [idp + 3 for idp, x in enumerate(pcl_val_list) if x == fusion_alg],
            dtype=cp.int32,
        )
        # create a list of indices of the layers that will be updated by the point cloud with specific fusion alg
        layer_indices = cp.array([], dtype=cp.int32)
        for it, (key, val) in enumerate(self.layer_specs.items()):
            if key in pcl_channels and val == fusion_alg:
                layer_indices = cp.append(layer_indices, it).astype(cp.int32)
        return pcl_indices, layer_indices

    def update_layers_pointcloud(self, points_all, channels, R, t, elevation_map):
        """Update the semantic map with the pointcloud.

        Args:
            points_all: semantic point cloud
            channels: list of channel names
            R: rotation matrix
            t: translation vector
            elevation_map: elevation map object
        """
        additional_fusion = self.get_fusion_of_pcl(channels)
        self.new_map[self.delete_new_layers] = 0.0
        if "average" in additional_fusion:
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "average")
            self.sum_kernel(
                points_all,
                R,
                t,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
                self.semantic_map,
                self.new_map,
                size=(points_all.shape[0]*pcl_ids.shape[0]),
            )
            self.average_kernel(
                self.new_map,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
                elevation_map,
                self.semantic_map,
                size=(self.param.cell_n * self.param.cell_n*pcl_ids.shape[0]),
            )
            np.save("features.npy",self.semantic_map[layer_ids])
        if "bayesian_inference" in additional_fusion:
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "bayesian_inference")
            self.sum_mean *= 0
            self.sum_compact_kernel(
                points_all,
                R,
                t,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
                self.sum_mean,
                size=(points_all.shape[0]),
            )
            self.bayesian_inference_kernel(
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
                elevation_map,
                self.new_map,
                self.sum_mean,
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
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
                self.semantic_map,
                self.new_map,
                size=(points_all.shape[0]*pcl_ids.shape[0]),
            )
            self.class_average_kernel(
                self.new_map,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
                elevation_map,
                self.semantic_map,
                size=(self.param.cell_n * self.param.cell_n*pcl_ids.shape[0]),
            )

        if "class_bayesian" in additional_fusion:
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "class_bayesian")
            # alpha sum get points as input and calculate for each point to what cell it belongs and then
            # adds to the right channel a one
            self.alpha_kernel(
                points_all,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
                self.new_map,
                size=(points_all.shape[0]),
            )
            # calculate new thetas
            sum_alpha = cp.sum(self.new_map[layer_ids], axis=0)
            # do not divide by zero
            sum_alpha[sum_alpha == 0] = 1
            self.semantic_map[layer_ids] = self.new_map[layer_ids] / cp.expand_dims(sum_alpha, axis=0)

        if "class_max" in additional_fusion:
            # get indices that are of type class_max in pointclopud and in layers
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "class_max")
            # decode float32 into to float16
            max_pt, pt_id = self.decode_max(points_all[:, pcl_ids])
            # find unique ids in new measurement and in existing map
            unique_idm = cp.unique(pt_id)
            unique_ida = cp.unique(self.unique_id[self.elements_to_shift["id_max"]])
            # get all unique ids, where index is the position in the prob_sum and the value in the NN class
            self.unique_id = cp.unique(cp.concatenate((unique_idm, unique_ida)))
            # contains the sum of the new measurement probabilities
            self.prob_sum = cp.zeros(
                (len(self.unique_id), self.param.cell_n, self.param.cell_n),
                dtype=np.float32,
            )
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
                cp.array(
                    [points_all.shape[1], pcl_ids.shape[0], pt_id.shape[1]],
                    dtype=cp.int32,
                ),
                self.prob_sum,
                size=(points_all.shape[0]),
            )
            # add the previous alpha
            for i, lay in enumerate(layer_ids):
                c = cp.mgrid[0 : self.new_map.shape[1], 0 : self.new_map.shape[2]]
                # self.prob_sum[self.elements_to_shift["id_max"][i], c[0], c[1]] += self.new_map[lay]
                # TODO add residual of prev alpha to the prob_sum
                # res = 1- self.new_map[lay]
                # res /= (len(self.unique_id)-1)

            # find the alpha we want to keep
            for i, lay in enumerate(layer_ids):
                self.new_map[lay] = cp.amax(self.prob_sum, axis=0)
                self.elements_to_shift["id_max"][lay] = self.unique_id[cp.argmax(self.prob_sum, axis=0)]
                self.prob_sum[cp.argmax(self.prob_sum, axis=0)] = 0
            # update map calculate new thetas
            sum_alpha = cp.sum(self.new_map[layer_ids], axis=0)
            # do not divide by zero
            sum_alpha[sum_alpha == 0] = 1
            self.semantic_map[layer_ids] = self.new_map[layer_ids] / cp.expand_dims(sum_alpha, axis=0)

        if "color" in additional_fusion:
            pcl_ids, layer_ids = self.get_indices_fusion(channels, "color")
            self.color_map = cp.zeros(
                (1 + 3 * layer_ids.shape[0], self.param.cell_n, self.param.cell_n),
                dtype=cp.uint32,
            )

            points_all = points_all.astype(cp.float32)
            self.add_color_kernel(
                points_all,
                R,
                t,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
                self.color_map,
                size=(points_all.shape[0]),
            )
            self.color_average_kernel(
                self.color_map,
                pcl_ids,
                layer_ids,
                cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
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
        """Update the semantic map with the new image.

        Args:
            sub_key:
            image:
            uv_correspondence:
            valid_correspondence:
            image_height:
            image_width:
        """
        self.new_map *= 0
        config = self.param.subscriber_cfg[sub_key]

        for j, (fusion, channel) in enumerate(zip(config["fusion"], config["channels"])):
            sem_map_idx = self.get_index(channel)

            if fusion == "image_exponential":
                self.exponential_correspondences_to_map_kernel(
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
                raise ValueError("Fusion for image is unknown.")

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

    def get_map_with_name(self, name):
        """Return the map with the given name.

        Args:
            name: layer name

        Returns:
            cp.array: map
        """
        if self.layer_specs[name] == "color":
            m = self.get_rgb(name)
            return m
        else:
            m = self.get_semantic(name)
            return m

    def get_rgb(self, name):
        """Return the rgb map with the given name.

        Args:
            name:

        Returns:
            cp.array: rgb map
        """
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        c = c.astype(np.float32)
        return c

    def get_semantic(self, name):
        """Return the semantic map layer with the given name.

        Args:
            name(str): layer name

        Returns:
            cp.array: semantic map layer
        """
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        return c

    def process_map_for_publish(self, input_map):
        """Remove padding.

        Args:
            input_map(cp.array): map layer

        Returns:
            cp.array: map layer without padding
        """
        m = input_map.copy()
        return m[1:-1, 1:-1]

    def get_index(self, name):
        """Return the index of the layer with the given name.

        Args:
            name(str):

        Returns:
            int: index
        """
        return self.layer_names.index(name)
