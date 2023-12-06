#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from abc import ABC, abstractmethod
import cupy as cp
from typing import List, Dict, Any
from dataclasses import dataclass
import importlib
import inspect


# @dataclass
# class FusionParams:
#     name: str


class FusionBase(ABC):
    """
    This is a base class of Fusion
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """

        Args:
            fusion_params : FusionParams
            The parameter of callback
        """
        self.name = None

    @abstractmethod
    def __call__(self, points_all, R, t, pcl_ids, layer_ids, elevation_map, semantic_map, new_map):
        pass


class FusionManager(object):
    def __init__(self, params):
        self.fusion_plugins: Dict[str, FusionBase] = {}
        self.params = params
        self.plugins = []

    def register_plugin(self, plugin):
        """
        Register a new fusion plugin
        """
        try:
            m = importlib.import_module("." + plugin, package="elevation_mapping_cupy.fusion")  # -> 'module'
        except:
            raise ValueError("Plugin {} does not exist.".format(plugin))
        for name, obj in inspect.getmembers(m):

            if inspect.isclass(obj) and issubclass(obj, FusionBase) and name != "FusionBase":
                self.plugins.append(obj(self.params))

    def get_plugin_idx(self, name: str, data_type: str):
        """
        Get a registered fusion plugin
        """
        name = data_type + "_" + name
        for idx, plugin in enumerate(self.plugins):
            if plugin.name == name:
                return idx
        print("[WARNING] Plugin {} is not in the list: {}".format(name, self.plugins))
        return None

    def execute_plugin(
        self, name: str, points_all, R, t, pcl_ids, layer_ids, elevation_map, semantic_map, new_map, elements_to_shift
    ):
        """
        Execute a registered fusion plugin
        """
        idx = self.get_plugin_idx(name, "pointcloud")
        if idx is not None:
            self.plugins[idx](
                points_all, R, t, pcl_ids, layer_ids, elevation_map, semantic_map, new_map, elements_to_shift
            )
        # else:
        #     raise ValueError("Plugin {} is not registered".format(name))

    def execute_image_plugin(
        self,
        name: str,
        sem_map_idx,
        image,
        j,
        uv_correspondence,
        valid_correspondence,
        image_height,
        image_width,
        semantic_map,
        new_map,
    ):
        """
        Execute a registered fusion plugin
        """
        idx = self.get_plugin_idx(name, "image")
        if idx is not None:
            self.plugins[idx](
                sem_map_idx,
                image,
                j,
                uv_correspondence,
                valid_correspondence,
                image_height,
                image_width,
                semantic_map,
                new_map,
            )
        # else:
        #     raise ValueError("Plugin {} is not registered".format(name))
