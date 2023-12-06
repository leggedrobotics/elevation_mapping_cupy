#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List
import re

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase
from sklearn.decomposition import PCA


class FeaturesPca(PluginBase):
    """This is a filter to create a pca layer of the semantic features in the map.

    Args:
        cell_n (int): width and height of the elevation map.
        classes (ruamel.yaml.comments.CommentedSeq):
        **kwargs ():
    """

    def __init__(
        self, cell_n: int = 100, process_layer_names: List[str] = [], **kwargs,
    ):
        super().__init__()
        self.process_layer_names = process_layer_names

    def get_layer_indices(self, layer_names: List[str]) -> List[int]:
        """ Get the indices of the layers that are to be processed using regular expressions.
        Args:
            layer_names (List[str]): List of layer names.
        Returns:
            List[int]: List of layer indices.
        """
        indices = []
        for i, layer_name in enumerate(layer_names):
            if any(re.match(pattern, layer_name) for pattern in self.process_layer_names):
                indices.append(i)
        return indices

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_map (elevation_mapping_cupy.semantic_map.SemanticMap):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        # get indices of all layers that contain semantic features information
        data = []
        for m, layer_names in zip(
            [elevation_map, plugin_layers, semantic_map], [layer_names, plugin_layer_names, semantic_layer_names]
        ):
            layer_indices = self.get_layer_indices(layer_names)
            if len(layer_indices) > 0:
                n_c = m[layer_indices].shape[1]
                data_i = cp.reshape(m[layer_indices], (len(layer_indices), -1)).T.get()
                data_i = np.clip(data_i, -1, 1)
                data.append(data_i)
        if len(data) > 0:
            data = np.concatenate(data, axis=1)
            # check which has the highest value
            n_components = 3
            pca = PCA(n_components=n_components).fit(data)
            pca_descriptors = pca.transform(data)
            img_pca = pca_descriptors.reshape(n_c, n_c, n_components)
            comp = img_pca  # [:, :, -3:]
            comp_min = comp.min(axis=(0, 1))
            comp_max = comp.max(axis=(0, 1))
            if (comp_max - comp_min).any() != 0:
                comp_img = np.divide((comp - comp_min), (comp_max - comp_min))
            pca_map = (comp_img * 255).astype(np.uint8)
            r = np.asarray(pca_map[:, :, 0], dtype=np.uint32)
            g = np.asarray(pca_map[:, :, 1], dtype=np.uint32)
            b = np.asarray(pca_map[:, :, 2], dtype=np.uint32)
            rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
            rgb_arr.dtype = np.float32
            return cp.asarray(rgb_arr)
        else:
            return cp.zeros_like(elevation_map[0])
