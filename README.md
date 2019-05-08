# Elevation Mapping cupy
## Overview
This is a ros package of elevation mapping on GPU.  
Code are written in python and numpy, cupy backend can be selected.

![screenshot](doc/sim.png)

## Installation
### CUDA & cuDNN
First, install CUDA and cuDNN.
Instructions are here [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation),
[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux).

The tested versions are CUDA10.0, cuDNN7.

### python dependencies
- [numpy](https://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [cupy](https://cupy.chainer.org/)
- [chainer](https://chainer.org/)

```bash
pip install numpy, scipy, cupy, chainer
```

cupy can be installed with specific CUDA versions.
> (For CUDA 9.0)
> % pip install cupy-cuda90
> 
> (For CUDA 9.1)
> % pip install cupy-cuda91
> 
> (For CUDA 9.2)
> % pip install cupy-cuda92
> 
> (For CUDA 10.0)
> % pip install cupy-cuda100
> 
> (Install CuPy from source)
> % pip install cupy

### ROS package dependencies
- [ros_numpy](https://github.com/eric-wieser/ros_numpy)
- [grid_map_msgs](https://github.com/ANYbotics/grid_map)

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

* **`/points`** ([sensor_msgs/PointCloud2])

    The distance measurements.

* **`/pose`** ([geometry_msgs/PoseWithCovarianceStamped])

    The robot pose and covariance.

* **`/tf`** ([tf/tfMessage])

    The transformation tree.


### Published Topics

* **`elevation_map`** ([grid_map_msg/GridMap])

    The entire elevation map.

### Processing Time
The processing time of

- pointcloud transform
- elevation map update
- traversability calculation

is measured in P52 laptop which has `Intel� Core� i7-8850H CPU` and `Quadro P3200 `.  
101760 = 424 x 240 is the realsense's number of points.

![graph](doc/processing_time.png)
