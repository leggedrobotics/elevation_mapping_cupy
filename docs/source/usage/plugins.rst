.. _plugins:

Plugins
******************************************************************

You can add your own plugin to process the elevation map and publish as a layer in GridMap message.

This page is structured in two parts:

* `Create a plugin`_

* `Existing plugins`_


Create a plugin
==================================================================
You can create your own plugin to process the elevation map and publish as a layer in GridMap message.

Let's look at the example.

First, create your plugin file in `elevation_mapping_cupy/script/plugins/` and save as `example.py`.

.. code-block:: python

  import cupy as cp
  from typing import List
  from .plugin_manager import PluginBase


  class NameOfYourPlugin(PluginBase):
      def __init__(self, add_value:float=1.0, **kwargs):
          super().__init__()
          self.add_value = float(add_value)

      def __call__(self,
          elevation_map: cp.ndarray,
          layer_names: List[str],
          plugin_layers: cp.ndarray,
          plugin_layer_names: List[str],
          semantic_layers: cp.ndarray,
          semantic_layer_names: List[str],
          *args
      )->cp.ndarray:
          # Process maps here
          # You can also use the other plugin's data through plugin_layers.
          new_elevation = elevation_map[0] + self.add_value
          return new_elevation

Then, add your plugin setting to `config/plugin_config.yaml`

.. code-block:: yaml

  example:                                      # Name of your filter
    type: "example"                             # Specify the name of your plugin (the name of your file name).
    enable: True                                # weather to load this plugin
    fill_nan: True                              # Fill nans to invalid cells of elevation layer.
    is_height_layer: True                       # If this is a height layer (such as elevation) or not (such as traversability)
    layer_name: "example_layer"                 # The layer name.
    extra_params:                               # This params are passed to the plugin class on initialization.
      add_value: 2.0                            # Example param

  example_large:                                # You can apply same filter with different name.
    type: "example"                             # Specify the name of your plugin (the name of your file name).
    enable: True                                # weather to load this plugin
    fill_nan: True                              # Fill nans to invalid cells of elevation layer.
    is_height_layer: True                       # If this is a height layer (such as elevation) or not (such as traversability)
    layer_name: "example_layer_large"           # The layer name.
    extra_params:                               # This params are passed to the plugin class on initialization.
      add_value: 100.0                          # Example param with larger value.


Finally, add your layer name to publishers in your config. You can create a new topic or add to existing topics.

.. code-block:: yaml

    plugin_example: # Topic name
      layers: [ 'elevation', 'example_layer', 'example_layer_large' ]
      basic_layers: [ 'example_layer' ]
      fps: 1.0        # The plugin is called with this fps.





Existing plugins
==================================================================

This section lists the plugins that are already installed and available for use.




1. Min filter
-------------------------------------------------------------------
.. automodule:: elevation_mapping_cupy.plugins.min_filter
    :members:

2. Inpainting
-------------------------------------------------------------------

.. automodule:: elevation_mapping_cupy.plugins.inpainting
    :members:

3. Smooth Filter
-------------------------------------------------------------------
.. automodule:: elevation_mapping_cupy.plugins.smooth_filter
    :members:

4. Robot centric elevation
-------------------------------------------------------------------
.. automodule:: elevation_mapping_cupy.plugins.robot_centric_elevation
    :members:

5. Semantic Filter
-------------------------------------------------------------------
.. automodule:: elevation_mapping_cupy.plugins.semantic_filter
    :members:

6. Semantic traversability
-------------------------------------------------------------------
.. automodule:: elevation_mapping_cupy.plugins.semantic_traversability
    :members:

7. Features PCA
-------------------------------------------------------------------
.. automodule:: elevation_mapping_cupy.plugins.features_pca
    :members: