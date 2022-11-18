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

// ROS
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <tf/transform_listener.h>

// Grid Map
#include <grid_map_msgs/GetGridMap.h>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>

// PCL
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace py = pybind11;

namespace elevation_mapping_cupy {

class ElevationMappingWrapper {
 public:
  using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  ElevationMappingWrapper();

  void initialize(ros::NodeHandle& nh);

  void input(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, const RowMatrixXd& R, const Eigen::VectorXd& t,
             const double positionNoise, const double orientationNoise);
  void move_to(const Eigen::VectorXd& p);
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
  void pointCloudToMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, RowMatrixXd& points);
  void addNormalColorLayer(grid_map::GridMap& map);

 private:
  void setParameters(ros::NodeHandle& nh);
  py::object map_;
  py::object param_;
  double resolution_;
  double map_length_;
  int map_n_;
  bool enable_normal_;
  bool enable_normal_color_;
};

}  // namespace elevation_mapping_cupy
