# Plane Segmentation

## Overview
This is a C++ ROS package for extracting convex polygons from elevation maps.
In a first step, the terrain is segmented into planes, and their non-convex contour is extracted.
In a second step, a local convex approximation can be obtained.

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
