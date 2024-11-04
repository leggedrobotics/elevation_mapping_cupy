//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

// Pybind
#include <pybind11/embed.h>  // everything needed for embedding

// ROS2
#include <rclcpp/rclcpp.hpp>
#include "elevation_mapping_cupy/elevation_mapping_ros.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  
  py::scoped_interpreter guard{};  // start the interpreter and keep it alive
  auto mapNode = std::make_shared<elevation_mapping_cupy::ElevationMappingNode>();  // Correct instantiation  
  py::gil_scoped_release release;
  

  // TODO: Create a multi-threaded executor
  rclcpp::spin(mapNode);
  rclcpp::shutdown();
  return 0;
}

