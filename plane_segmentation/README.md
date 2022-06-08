# Plane Segmentation

## Overview
This is a C++ ROS package for extracting convex polygons from elevation maps.
In a first step, the terrain is segmented into planes, and their non-convex contour is extracted.
In a second step, a local convex approximation can be obtained.

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

#### CGAL
CGAL5 is required. It will be automatically downloaded and installed into the catkin workspace by the cgal5_catkin package.
Make sure you have the third-party libaries installed on you machine:
```bash
sudo apt-get install libgmp-dev
sudo apt-get install libmpfr-dev
sudo apt-get install libboost-all-dev
```

#### PCL
PCL is required, but the ANYbotics distributed version does not contain visualization components. 
With pcl_visualization_catkin, the missing components are provided into your catkin workspace (for pcl 1.10). 
Additionally vtk7 is required, DO NOT install this on the ANYmal onboard PCs, only on OPC and simulation PCs.
```bash
sudo apt-get install libvtk7-dev
```

### ROS package dependencies

#### JSK-visualization
For rviz-visualization the jsk-library is used.
```bash
sudo apt-get install ros-noetic-jsk-visualization
```

#### Grid Map
Grid map is available through ANYmal-research, apt install, or you can add it to your workspace. You can clone it using:
```bash
git clone https://github.com/ANYbotics/grid_map.git
```

## Usage
### Build
```bash
catkin build convex_plane_decomposition_ros
```

### Run as node
```bash
roslaunch convex_plane_decomposition_ros convex_plane_decomposition.launch
```

### Run demo
```bash
roslaunch convex_plane_decomposition_ros demo.launch
```

### Convex approximation demo
A simple 2D demo the convex inner approximation is available:
```bash
rosrun convex_plane_decomposition_ros convex_plane_decomposition_ros_TestShapeGrowing
```

### Parameters
You can select input map topics, pipeline parameters etc. in the respective yaml file in
```bash
convex_plane_decomposition_ros/config/
```
Some other parameters are set through the launch files.
