#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"
#include <pybind11/embed.h> 
#include <pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/common/projection_matrix.h>
#include <tf_conversions/tf_eigen.h>
#include <ros/package.h>

namespace elevation_mapping_cupy{
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

ElevationMappingWrapper::ElevationMappingWrapper() {
}

void ElevationMappingWrapper::initialize(ros::NodeHandle& nh) {
  // Add the elevation_mapping_cupy path to sys.path
  auto sys = py::module::import("sys");
  auto path = sys.attr("path");
  std::string module_path = ros::package::getPath("elevation_mapping_cupy");
  module_path = module_path + "/script";
  path.attr("insert")(0, module_path);

  auto elevation_mapping = py::module::import("elevation_mapping");
  auto parameter = py::module::import("parameter");
  param_ = parameter.attr("Parameter")();
  setParameters(nh);
  map_ = elevation_mapping.attr("ElevationMap")(param_);
}

void ElevationMappingWrapper::setParameters(ros::NodeHandle& nh) {
  bool enable_edge_sharpen, enable_drift_compensation;
  float resolution, map_length, sensor_noise_factor, mahalanobis_thresh, outlier_variance;
  float time_variance, initial_variance, traversability_inlier;
  int dilation_size, wall_num_thresh, min_height_drift_cnt;
  std::string gather_mode, weight_file;
  nh.param<bool>("enable_edge_sharpen", enable_edge_sharpen, true);
  param_.attr("set_enable_edge_sharpen")(enable_edge_sharpen);

  nh.param<bool>("enable_drift_compensation", enable_drift_compensation, true);
  param_.attr("set_enable_drift_compensation")(enable_drift_compensation);

  nh.param<float>("resolution", resolution, 0.02);
  param_.attr("set_resolution")(resolution);

  nh.param<float>("map_length", map_length, 5.0);
  param_.attr("set_map_length")(map_length);

  nh.param<float>("sensor_noise_factor", sensor_noise_factor, 0.05);
  param_.attr("set_sensor_noise_factor")(sensor_noise_factor);

  nh.param<float>("mahalanobis_thresh", mahalanobis_thresh, 2.0);
  param_.attr("set_mahalanobis_thresh")(mahalanobis_thresh);

  nh.param<float>("outlier_variance", outlier_variance, 0.01);
  param_.attr("set_outlier_variance")(outlier_variance);

  nh.param<float>("time_variance", time_variance, 0.01);
  param_.attr("set_time_variance")(time_variance);

  nh.param<float>("initial_variance", initial_variance, 10.0);
  param_.attr("set_initial_variance")(initial_variance);

  nh.param<float>("traversability_inlier", traversability_inlier, 0.1);
  param_.attr("set_traversability_inlier")(traversability_inlier);

  nh.param<int>("dilation_size", dilation_size, 2);
  param_.attr("set_dilation_size")(dilation_size);

  nh.param<int>("wall_num_thresh", wall_num_thresh, 100);
  param_.attr("set_wall_num_thresh")(wall_num_thresh);

  nh.param<int>("min_height_drift_cnt", min_height_drift_cnt, 100);
  param_.attr("set_min_height_drift_cnt")(min_height_drift_cnt);

  nh.param<std::string>("gather_mode", gather_mode, "mean");
  param_.attr("set_gather_mode")(gather_mode);

  nh.param<std::string>("weight_file", weight_file, "config/weights.yaml");
  std::string path = ros::package::getPath("elevation_mapping_cupy");

  weight_file = path + "/" + weight_file;
  param_.attr("load_weights")(weight_file);

  resolution_ = resolution;
  map_length_ = map_length;
  map_n_ = (int)(map_length_ / resolution_);
}


void ElevationMappingWrapper::input(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, const RowMatrixXd& R, const Eigen::VectorXd& t) {
  
  RowMatrixXd points;
  pointCloudToMatrix(pointCloud, points);
  map_.attr("input")(static_cast<Eigen::Ref<const RowMatrixXd>>(points),
                     static_cast<Eigen::Ref<const RowMatrixXd>>(R),
                     static_cast<Eigen::Ref<const Eigen::VectorXd>>(t));
}


void ElevationMappingWrapper::move_to(const Eigen::VectorXd& p) {
  map_.attr("move_to")(static_cast<Eigen::Ref<const Eigen::VectorXd>>(p));
}


void ElevationMappingWrapper::clear() {
  map_.attr("clear")();
}


void ElevationMappingWrapper::get_maps(std::vector<Eigen::MatrixXd>& maps) {

  RowMatrixXd elevation(map_n_, map_n_);
  RowMatrixXd variance(map_n_, map_n_);
  RowMatrixXd traversability(map_n_, map_n_);
  map_.attr("get_maps_ref")(static_cast<Eigen::Ref<RowMatrixXd>>(elevation),
                            static_cast<Eigen::Ref<RowMatrixXd>>(variance),
                            static_cast<Eigen::Ref<RowMatrixXd>>(traversability));
  maps.clear();
  maps.push_back(elevation);
  maps.push_back(variance);
  maps.push_back(traversability);
  return;
}


void ElevationMappingWrapper::get_grid_map(grid_map::GridMap& gridMap) {
  RowMatrixXd pos(1, 2);
  map_.attr("get_position")(static_cast<Eigen::Ref<RowMatrixXd>>(pos));

  grid_map::Position position(pos(0, 0), pos(0, 1));
  grid_map::Length length(map_length_, map_length_);
  gridMap.setGeometry(length, resolution_, position);
  std::vector<Eigen::MatrixXd> maps;
  get_maps(maps);
  std::vector<std::string> layerNames = {"elevation", "variance", "traversability"};
  for(int i = 0; i < maps.size() ; ++i) {
    gridMap.add(layerNames[i], maps[i].cast<float>());
  }
}


double ElevationMappingWrapper::get_polygon_traversability(std::vector<Eigen::Vector2d> &polygon) {
  RowMatrixXd polygon_m(polygon.size(), 2);
  if (polygon.size() < 3)
    return 0;
  int i = 0;
  for (auto& p: polygon) {
    polygon_m(i, 0) = p.x();
    polygon_m(i, 1) = p.y();
    i++;
  }
  double traversability = map_.attr("get_polygon_traversability")(
      static_cast<Eigen::Ref<const RowMatrixXd>>(polygon_m)).cast<double>();
  return traversability;
}


void ElevationMappingWrapper::pointCloudToMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud,
                                                 RowMatrixXd& points)
{
  points = RowMatrixXd(pointCloud->size(), 3);
  for (unsigned int i = 0; i < pointCloud->size(); ++i) {
    const auto& point = pointCloud->points[i];
    points(i, 0) = static_cast<double>(point.x);
    points(i, 1) = static_cast<double>(point.y);
    points(i, 2) = static_cast<double>(point.z);
  }
  return;
}


}
