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
  auto node = std::make_shared<rclcpp::Node>("elevation_mapping");

  py::scoped_interpreter guard{};  // start the interpreter and keep it alive
  // elevation_mapping_cupy::ElevationMappingNode mapNode(node);
  py::gil_scoped_release release;

  // Spin
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}