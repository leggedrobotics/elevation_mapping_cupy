//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

// Pybind
#include <pybind11/embed.h>  // everything needed for embedding

// ROS
#include <ros/ros.h>

#include "elevation_mapping_cupy/elevation_mapping_ros.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "elevation_mapping");
  ros::NodeHandle nh("~");

  py::scoped_interpreter guard{};  // start the interpreter and keep it alive
  elevation_mapping_cupy::ElevationMappingNode mapNode(nh);
  py::gil_scoped_release release;

  // Spin
  ros::AsyncSpinner spinner(1);  // Use n threads
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
