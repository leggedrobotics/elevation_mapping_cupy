//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
// Grid Map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "elevation_mapping_cupy/elevation_mapping_ros.hpp"

using namespace elevation_mapping_cupy;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "elevation_mapping");
  ros::NodeHandle nh("~");

  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  ElevationMappingNode mapNode(nh);
  py::gil_scoped_release release;

  // Spin
  ros::AsyncSpinner spinner(1); // Use n threads
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
