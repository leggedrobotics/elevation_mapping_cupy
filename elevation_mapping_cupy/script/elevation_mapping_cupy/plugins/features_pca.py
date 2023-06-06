#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase
from sklearn.decomposition import PCA


class FeaturesPca(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        **kwargs,
    ):
        """This is a filter to create a pca layer of the semantic features in the map.

        Args:
            cell_n (int): width and height of the elevation map.
            classes (ruamel.yaml.comments.CommentedSeq):
            **kwargs ():
        """
        super().__init__()
        self.indices = []

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map,
        semantic_params,
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
        layer_indices = cp.array([], dtype=cp.int32)
        for it, fusion_alg in enumerate(semantic_params.fusion_algorithms):
            if fusion_alg in ["average", "bayesian_inference", "image_exponential"]:
                layer_indices = cp.append(layer_indices, it).astype(cp.int32)

        n_c = semantic_map[layer_indices].shape[1]
        comp_img = np.zeros((n_c, n_c, 3), dtype=np.float32)
        # check which has the highest value
        if len(layer_indices) > 0:
            data = cp.reshape(semantic_map[layer_indices], (len(layer_indices), -1)).T.get()
            # data = np.clip(data, -1, 1)
            n_components = 3
            pca = PCA(n_components=n_components).fit(data)
            pca_descriptors = pca.transform(data)
            img_pca = pca_descriptors.reshape(n_c, n_c, n_components)
            comp = img_pca  # [:, :, -3:]
            var = comp.std(axis=(0, 1))
            comp_min = comp.min(axis=(0, 1))
            comp_max = comp.max(axis=(0, 1))
            if (comp_max - comp_min).any() != 0:
                comp_img = np.divide((comp - comp_min), (comp_max - comp_min))
        map = (comp_img * 255).astype(np.uint8)
        r = np.asarray(map[:, :, 0], dtype=np.uint32)
        g = np.asarray(map[:, :, 1], dtype=np.uint32)
        b = np.asarray(map[:, :, 2], dtype=np.uint32)
        rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
        rgb_arr.dtype = np.float32
        return cp.asarray(rgb_arr)
