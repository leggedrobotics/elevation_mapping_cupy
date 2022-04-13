# Elevation Mapping cupy
[![Build Status](https://ci.leggedrobotics.com/buildStatus/icon?job=bitbucket_leggedrobotics/elevation_mapping_cupy/master)](https://ci.leggedrobotics.com/job/<repo_host>_leggedrobotics/job/elevation_mapping_cupy/job/master/)

## Overview
This is a ros package of elevation mapping on GPU.  
Code are written in python and uses cupy for GPU calculation.  
![screenshot](doc/main_repo.png)

## Installation

### CUDA & cuDNN
The tested versions are CUDA10.2.89, cuDNN8.0.0.180

[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)  
[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux).


#### On Jetson
`CUDA` and `cuDNN` can be installed via apt. It comes with nvidia-jetpack. The tested version is jetpack 4.5 with L4T 32.5.0.

### Python dependencies
The versions are the tested version on Jetson Xavier.
Newer versions might also work.
- [numpy==1.19.5](https://www.numpy.org/)
- [scipy==1.5.4](https://www.scipy.org/)
- [cupy==9.1.0](https://cupy.chainer.org/)
- [shapely==1.7.1](https://github.com/Toblerity/Shapely)

#### Cupy
cupy can be installed with specific CUDA versions. (On jetson, only "from source" i.e. via pip3 could work)  
On jetson, pip3 builds the packages from source so it would take time.
> For CUDA 10.2
> pip install cupy-cuda102
> 
> For CUDA 11.0
> pip install cupy-cuda110
> 
> For CUDA 11.1
> pip install cupy-cuda111
> 
> For CUDA 11.2
> pip install cupy-cuda112
> 
> For CUDA 11.3
> pip install cupy-cuda113
> 
> For CUDA 11.4
> pip install cupy-cuda114
> 
> For CUDA 11.5
> pip install cupy-cuda115
> 
> For CUDA 11.6
> pip install cupy-cuda116
> 
> For AMD ROCm 4.0
> pip install cupy-rocm-4-0
> 
> For AMD ROCm 4.2
> pip install cupy-rocm-4-2
> 
> For AMD ROCm 4.3
> pip install cupy-rocm-4-3
> 
> For AMD ROCm 5.0
> pip install cupy-rocm-5-0
>
> (Install CuPy from source)
> % pip install cupy


#### Traversability filter
You can choose either pytorch, or chainer to run the CNN based traversability filter.

- [torch](https://pytorch.org/)
- [chainer](https://chainer.org/)

Pytorch uses ~2GB more GPU memory than Chainer, but runs a bit faster.  
Use parameter `use_chainer` to select which backend to use.

On jetson, you need the version for its CPU arch:
```bash
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install Cython
pip3 install numpy==1.19.5 torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

To install others, run the following.
```bash
pip3 install -r requirements.txt
```

Also, on jetson you need fortran (should already be installed).
```bash
sudo apt install gfortran
```
### ROS package dependencies

- [pybind11_catkin](https://github.com/ipab-slmc/pybind11_catkin)
- [grid_map_msgs](https://github.com/ANYbotics/grid_map)

```bash
sudo apt install ros-noetic-pybind11-catkin
sudo apt install ros-noetic-grid-map-msgs
```

#### On Jetson

- [pybind11_catkin](https://github.com/ipab-slmc/pybind11_catkin)
- [grid_map_msgs](https://github.com/ANYbotics/grid_map)

```bash
sudo apt install ros-melodic-pybind11-catkin
```

For interaction with ROS Noetic the recomended way is to compile grid map from source via the open source package or ANYmal Research.

If the Jetson is set up with Jetpack 4.5 with ROS Melodic the following package is additionally required:

```bash
git clone git@github.com:ros/filters.git -b noetic-devel
```

## Usage
### Build
```bash
catkin build elevation_mapping_cupy
```

### Run
```bash
roslaunch elevation_mapping_cupy elevation_mapping_cupy.launch
```
### Subscribed Topics

* topics specified in **`pointcloud_topics`** in **`parameters.yaml`** ([sensor_msgs/PointCloud2])

    The distance measurements.

* **`/tf`** ([tf/tfMessage])

    The transformation tree.


### Published Topics
The topics are published as set in the rosparam.  
You can specify which layers to publish in which fps.

Example setting in `config/parameters.yaml`.

* **`elevation_map_raw`** ([grid_map_msg/GridMap])

    The entire elevation map.


* **`elevation_map_recordable`** ([grid_map_msg/GridMap])

    The entire elevation map with slower update rate for visualization and logging.

* **`elevation_map_filter`** ([grid_map_msg/GridMap])

    The filtered maps using plugins.



# Plugins
You can create your own plugin to process the elevation map and publish as a layer in GridMap message.

Let's look at the example.

First, create your plugin file in `elevation_mapping_cupy/script/plugins/` and save as `example.py`.
```python
import cupy as cp
from typing import List
from .plugin_manager import PluginBase


class NameOfYourPlugin(PluginBase):
    def __init__(self, add_value:float=1.0, **kwargs):
        super().__init__()
        self.add_value = float(add_value)

    def __call__(self, elevation_map: cp.ndarray, layer_names: List[str],
            plugin_layers: cp.ndarray, plugin_layer_names: List[str])->cp.ndarray:
        # Process maps here
        # You can also use the other plugin's data through plugin_layers.
        new_elevation = elevation_map[0] + self.add_value
        return new_elevation
```

Then, add your plugin setting to `config/plugin_config.yaml`
```yaml
example:                                      # Use the same name as your file name.
  enable: True                                # weather to laod this plugin
  fill_nan: True                              # Fill nans to invalid cells of elevation layer.
  is_height_layer: True                       # If this is a height layer (such as elevation) or not (such as traversability)
  layer_name: "example_layer"                 # The layer name.
  extra_params:                               # This params are passed to the plugin class on initialization.
    add_value: 2.0                            # Example param
```

Finally, add your layer name to publishers in `config/parameters.yaml`.
You can create a new topic or add to existing topics.
```yaml
  plugin_example:   # Topic name
    layers: ['elevation', 'example_layer']
    basic_layers: ['example_layer']
    fps: 1.0        # The plugin is called with this fps.
```
