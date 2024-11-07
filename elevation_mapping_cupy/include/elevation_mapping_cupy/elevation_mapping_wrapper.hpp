//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#pragma once

// STL
#include <iostream>

// Eigen
#include <Eigen/Dense>

// Pybind
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

// ROS2
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/empty.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

// Grid Map
#include <grid_map_msgs/srv/get_grid_map.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

// PCL
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace py = pybind11;

namespace elevation_mapping_cupy {

std::vector<std::string> extract_unique_names(const std::map<std::string, rclcpp::Parameter>& subscriber_params);

class ElevationMappingWrapper {
 public:
  using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ColMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

  ElevationMappingWrapper(); 

  void initialize(const std::shared_ptr<rclcpp::Node>& node);

  void input(const RowMatrixXd& points, const std::vector<std::string>& channels, const RowMatrixXd& R, const Eigen::VectorXd& t,
             const double positionNoise, const double orientationNoise);
  void input_image(const std::vector<ColMatrixXf>& multichannel_image, const std::vector<std::string>& channels, const RowMatrixXd& R,
                   const Eigen::VectorXd& t, const RowMatrixXd& cameraMatrix, const Eigen::VectorXd& D, const std::string distortion_model, int height, int width);
  void move_to(const Eigen::VectorXd& p, const RowMatrixXd& R);
  void clear();
  void update_variance();
  void update_time();
  bool exists_layer(const std::string& layerName);
  void get_layer_data(const std::string& layerName, RowMatrixXf& map);
  void get_grid_map(grid_map::GridMap& gridMap, const std::vector<std::string>& layerNames);
  void get_polygon_traversability(std::vector<Eigen::Vector2d>& polygon, Eigen::Vector3d& result,
                                  std::vector<Eigen::Vector2d>& untraversable_polygon);
  double get_additive_mean_error();
  void initializeWithPoints(std::vector<Eigen::Vector3d>& points, std::string method);
  void addNormalColorLayer(grid_map::GridMap& map);
  
 private:  
  
  std::shared_ptr<rclcpp::Node> node_;
  void setParameters();  
 
  py::object map_;
  py::object param_;
  double resolution_;
  double map_length_;
  int map_n_;
  bool enable_normal_;
  bool enable_normal_color_;
};

}  // namespace elevation_mapping_cupy
