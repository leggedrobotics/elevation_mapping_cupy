#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from abc import ABC
import cupy as cp
from typing import List, Dict, Optional
import importlib
import inspect
from dataclasses import dataclass
from ruamel.yaml import YAML
from inspect import signature


@dataclass
class PluginParams:
    name: str
    layer_name: str
    fill_nan: bool = False  # fill nan to invalid region
    is_height_layer: bool = False  # if this is a height layer


class PluginBase(ABC):
    """
    This is a base class of Plugins
    """

    def __init__(self, *args, **kwargs):
        """

        Args:
            plugin_params : PluginParams
            The parameter of callback
        """

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
        **kwargs,
    ) -> cp.ndarray:
        """This gets the elevation map data and plugin layers as a cupy array.


        Args:
            elevation_map ():
            layer_names ():
            plugin_layers ():
            plugin_layer_names ():
            semantic_map ():
            semantic_layer_names ():

        Run your processing here and return the result.
        layer of elevation_map  0: elevation
                                1: variance
                                2: is_valid
                                3: traversability
                                4: time
                                5: upper_bound
                                6: is_upper_bound
        You can also access to the other plugins' layer with plugin_layers and plugin_layer_names

        """
        pass

    def get_layer_data(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        name: str,
    ) -> Optional[cp.ndarray]:
        """
        Retrieve a copy of the layer data from the elevation, plugin, or semantic maps based on the layer name.

        Args:
            elevation_map (cp.ndarray): The elevation map containing various layers.
            layer_names (List[str]): A list of names for each layer in the elevation map.
            plugin_layers (cp.ndarray): The plugin layers containing additional data.
            plugin_layer_names (List[str]): A list of names for each layer in the plugin layers.
            semantic_map (cp.ndarray): The semantic map containing various layers.
            semantic_layer_names (List[str]): A list of names for each layer in the semantic map.
            name (str): The name of the layer to retrieve.

        Returns:
            Optional[cp.ndarray]: A copy of the requested layer as a cupy ndarray if found, otherwise None.
        """
        if name in layer_names:
            idx = layer_names.index(name)
            layer = elevation_map[idx].copy()
        elif name in plugin_layer_names:
            idx = plugin_layer_names.index(name)
            layer = plugin_layers[idx].copy()
        elif name in semantic_layer_names:
            idx = semantic_layer_names.index(name)
            layer = semantic_map[idx].copy()
        else:
            print(f"Could not find layer {name}!")
            layer = None
        return layer


class PluginManager(object):
    """
    This manages the plugins.
    """

    def __init__(self, cell_n: int):
        self.cell_n = cell_n

    def init(self, plugin_params: List[PluginParams], extra_params: List[Dict]):
        self.plugin_params = plugin_params

        self.plugins = []
        for param, extra_param in zip(plugin_params, extra_params):
            m = importlib.import_module("." + param.name, package="elevation_mapping_cupy.plugins")  # -> 'module'
            for name, obj in inspect.getmembers(m):
                if inspect.isclass(obj) and issubclass(obj, PluginBase) and name != "PluginBase":
                    # Add cell_n to params
                    extra_param["cell_n"] = self.cell_n
                    self.plugins.append(obj(**extra_param))
        self.layers = cp.zeros((len(self.plugins), self.cell_n, self.cell_n), dtype=cp.float32)
        self.layer_names = self.get_layer_names()
        self.plugin_names = self.get_plugin_names()

    def load_plugin_settings(self, file_path: str):
        print("Start loading plugins...")
        cfg = YAML().load(open(file_path, "r"))
        plugin_params = []
        extra_params = []
        for k, v in cfg.items():
            if v["enable"]:
                plugin_params.append(
                    PluginParams(
                        name=k if not "type" in v else v["type"],
                        layer_name=v["layer_name"],
                        fill_nan=v["fill_nan"],
                        is_height_layer=v["is_height_layer"],
                    )
                )
                extra_params.append(v["extra_params"])
        self.init(plugin_params, extra_params)
        print("Loaded plugins are ", *self.plugin_names)

    def get_layer_names(self):
        names = []
        for obj in self.plugin_params:
            names.append(obj.layer_name)
        return names

    def get_plugin_names(self):
        names = []
        for obj in self.plugin_params:
            names.append(obj.name)
        return names

    def get_plugin_index_with_name(self, name: str) -> int:
        try:
            idx = self.plugin_names.index(name)
            return idx
        except Exception as e:
            print("Error with plugin {}: {}".format(name, e))
            return None

    def get_layer_index_with_name(self, name: str) -> int:
        try:
            idx = self.layer_names.index(name)
            return idx
        except Exception as e:
            print("Error with layer {}: {}".format(name, e))
            return None

    def update_with_name(
        self,
        name: str,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        semantic_map=None,
        semantic_params=None,
        rotation=None,
        elements_to_shift={},
    ):
        idx = self.get_layer_index_with_name(name)
        if idx is not None and idx < len(self.plugins):
            n_param = len(signature(self.plugins[idx]).parameters)
            if n_param == 5:
                self.layers[idx] = self.plugins[idx](elevation_map, layer_names, self.layers, self.layer_names)
            elif n_param == 7:
                self.layers[idx] = self.plugins[idx](
                    elevation_map,
                    layer_names,
                    self.layers,
                    self.layer_names,
                    semantic_map,
                    semantic_params,
                )
            elif n_param == 8:
                self.layers[idx] = self.plugins[idx](
                    elevation_map,
                    layer_names,
                    self.layers,
                    self.layer_names,
                    semantic_map,
                    semantic_params,
                    rotation,
                )
            else:
                self.layers[idx] = self.plugins[idx](
                    elevation_map,
                    layer_names,
                    self.layers,
                    self.layer_names,
                    semantic_map,
                    semantic_params,
                    rotation,
                    elements_to_shift,
                )

    def get_map_with_name(self, name: str) -> cp.ndarray:
        idx = self.get_layer_index_with_name(name)
        if idx is not None:
            return self.layers[idx]

    def get_param_with_name(self, name: str) -> PluginParams:
        idx = self.get_layer_index_with_name(name)
        if idx is not None:
            return self.plugin_params[idx]


if __name__ == "__main__":
    plugins = [
        PluginParams(name="min_filter", layer_name="min_filter"),
        PluginParams(name="smooth_filter", layer_name="smooth"),
    ]
    extra_params = [
        {"dilation_size": 5, "iteration_n": 5},
        {"input_layer_name": "elevation2"},
    ]
    manager = PluginManager(200)
    manager.load_plugin_settings("../config/plugin_config.yaml")
    print(manager.layer_names)
    print(manager.plugin_names)
    elevation_map = cp.zeros((7, 200, 200)).astype(cp.float32)
    layer_names = [
        "elevation",
        "variance",
        "is_valid",
        "traversability",
        "time",
        "upper_bound",
        "is_upper_bound",
    ]
    elevation_map[0] = cp.random.randn(200, 200)
    elevation_map[2] = cp.abs(cp.random.randn(200, 200))
    print("map", elevation_map[0])
    print("layer map ", manager.layers)
    manager.update_with_name("min_filter", elevation_map, layer_names)
    manager.update_with_name("smooth_filter", elevation_map, layer_names)
    # manager.update_with_name("sem_fil", elevation_map, layer_names, semantic_map=semantic_map)
    print(manager.get_map_with_name("smooth"))
