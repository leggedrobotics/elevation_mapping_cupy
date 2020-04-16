# loco_perception #

## Overview
This is a C++ ROS package for extracting convex polygons from elevation maps created by elevation_mapping.  

![screenshot](convex_plane_decomposition_ros/data/entire_map_decomposed.png)

## Installation

### Dependencies

#### OpenCV
Make sure you have openCV installed.  
You can execute the following command to install it.
```bash
sudo apt-get install libopencv-dev
```

#### Eigen
```bash
sudo apt install libeigen3-dev
```

#### GLOG
```bash
sudo apt-get install libgoogle-glog-dev
```

#### CGAL
CGAL5 is required. You can download it from [here](https://github.com/CGAL/cgal/releases/tag/releases%2FCGAL-5.0.2)  
and follow the instructions [here](https://doc.cgal.org/latest/Manual/installation.html#installation_idealworld).  
Make sure you have the third-party libaries installed on you machine:
```bash
sudo apt-get install libgmp-dev
sudo apt-get install libgmp3-dev
sudo apt-get install libmpfr-dev
sudo apt-get install libboost-all-dev
```


### ROS package dependencies

#### JSK-visualization
For rviz-visualization the jsk-library is used.
```bash
sudo apt-get install ros-melodic-jsk-visualization
```

#### Grid Map
Grid map has to be located in you workspace. You can clone it using:
```bash
git clone https://github.com/ANYbotics/grid_map.git
```

## Usage
### Build
```bash
catkin build convex_plane_extraction
```
### Run demo
```bash
roslaunch convex_plane_extraction_ros convex_plane_extraction_demo.launch
```

### Parameters
You can select input map topics, pipeline parameters etc. in the respective yaml file in ```bash
convex_plane_decomposition_ros/config/
```
