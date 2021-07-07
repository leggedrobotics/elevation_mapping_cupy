#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"
#include <pybind11_catkin/pybind11/embed.h>
#include <pybind11_catkin/pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/common/projection_matrix.h>
#include <tf_conversions/tf_eigen.h>
#include <ros/package.h>
#include <grid_map_core/grid_map_core.hpp>

namespace elevation_mapping_cupy{
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

ElevationMappingWrapper::ElevationMappingWrapper() {
}

void ElevationMappingWrapper::initialize(ros::NodeHandle& nh) {
  // Add the elevation_mapping_cupy path to sys.path
  auto threading = py::module::import("threading");
  py::gil_scoped_acquire acquire;

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
  bool enable_edge_sharpen, enable_drift_compensation, enable_visibility_cleanup, enable_overlap_clearance;
  float resolution, map_length, sensor_noise_factor, mahalanobis_thresh, outlier_variance, drift_compensation_variance_inlier, max_drift, drift_compensation_alpha;
  float time_variance, initial_variance, traversability_inlier, position_noise_thresh, cleanup_cos_thresh, max_variance, time_interval,
        orientation_noise_thresh, max_ray_length, cleanup_step, min_valid_distance, max_height_range, safe_thresh, safe_min_thresh, overlap_clear_range_xy, overlap_clear_range_z;
  int dilation_size, dilation_size_initialize, wall_num_thresh, min_height_drift_cnt, max_unsafe_n, min_filter_size, min_filter_iteration;
  std::string gather_mode, weight_file;
  py::gil_scoped_acquire acquire;
  nh.param<bool>("enable_edge_sharpen", enable_edge_sharpen, true);
  param_.attr("set_enable_edge_sharpen")(enable_edge_sharpen);

  nh.param<bool>("enable_drift_compensation", enable_drift_compensation, true);
  param_.attr("set_enable_drift_compensation")(enable_drift_compensation);

  nh.param<bool>("enable_visibility_cleanup", enable_visibility_cleanup, true);
  param_.attr("set_enable_visibility_cleanup")(enable_visibility_cleanup);

  nh.param<bool>("enable_normal", enable_normal_, false);
  nh.param<bool>("enable_normal_color", enable_normal_color_, false);

  nh.param<bool>("enable_overlap_clearance", enable_overlap_clearance, true);
  param_.attr("set_enable_overlap_clearance")(enable_overlap_clearance);

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

  nh.param<float>("max_variance", max_variance, 10.0);
  param_.attr("set_max_variance")(max_variance);

  nh.param<float>("drift_compensation_variance_inlier", drift_compensation_variance_inlier, 0.1);
  param_.attr("set_drift_compensation_variance_inlier")(drift_compensation_variance_inlier);

  nh.param<float>("max_drift", max_drift, 0.1);
  param_.attr("set_max_drift")(max_drift);

  nh.param<float>("drift_compensation_alpha", drift_compensation_alpha, 1.0);
  param_.attr("set_drift_compensation_alpha")(drift_compensation_alpha);

  nh.param<float>("time_variance", time_variance, 0.01);
  param_.attr("set_time_variance")(time_variance);

  nh.param<float>("time_interval", time_interval, 0.1);
  param_.attr("set_time_interval")(time_interval);

  nh.param<float>("initial_variance", initial_variance, 10.0);
  param_.attr("set_initial_variance")(initial_variance);

  nh.param<float>("traversability_inlier", traversability_inlier, 0.1);
  param_.attr("set_traversability_inlier")(traversability_inlier);

  nh.param<float>("position_noise_thresh", position_noise_thresh, 0.1);
  param_.attr("set_position_noise_thresh")(position_noise_thresh);

  nh.param<float>("orientation_noise_thresh", orientation_noise_thresh, 0.1);
  param_.attr("set_orientation_noise_thresh")(orientation_noise_thresh);

  nh.param<float>("max_ray_length", max_ray_length, 2.0);
  param_.attr("set_max_ray_length")(max_ray_length);

  nh.param<float>("cleanup_step", cleanup_step, 0.01);
  param_.attr("set_cleanup_step")(cleanup_step);

  nh.param<float>("cleanup_cos_thresh", cleanup_cos_thresh, 0.5);
  param_.attr("set_cleanup_cos_thresh")(cleanup_cos_thresh);

  nh.param<float>("min_valid_distance", min_valid_distance, 0.5);
  param_.attr("set_min_valid_distance")(min_valid_distance);

  nh.param<float>("max_height_range", max_height_range, 1.0);
  param_.attr("set_max_height_range")(max_height_range);

  nh.param<float>("safe_min_thresh", safe_min_thresh, 0.5);
  param_.attr("set_safe_min_thresh")(safe_min_thresh);

  nh.param<float>("safe_thresh", safe_thresh, 0.5);
  param_.attr("set_safe_thresh")(safe_thresh);

  nh.param<float>("overlap_clear_range_xy", overlap_clear_range_xy, 4.0);
  param_.attr("set_overlap_clear_range_xy")(overlap_clear_range_xy);

  nh.param<float>("overlap_clear_range_z", overlap_clear_range_z, 2.0);
  param_.attr("set_overlap_clear_range_z")(overlap_clear_range_z);

  nh.param<int>("dilation_size", dilation_size, 2);
  param_.attr("set_dilation_size")(dilation_size);

  nh.param<int>("dilation_size_initialize", dilation_size_initialize, 10);
  param_.attr("set_dilation_size_initialize")(dilation_size_initialize);

  nh.param<int>("wall_num_thresh", wall_num_thresh, 100);
  param_.attr("set_wall_num_thresh")(wall_num_thresh);

  nh.param<int>("min_height_drift_cnt", min_height_drift_cnt, 100);
  param_.attr("set_min_height_drift_cnt")(min_height_drift_cnt);

  nh.param<int>("max_unsafe_n", max_unsafe_n, 20);
  param_.attr("set_max_unsafe_n")(max_unsafe_n);

  nh.param<int>("min_filter_size", min_filter_size, 5);
  param_.attr("set_min_filter_size")(min_filter_size);

  nh.param<int>("min_filter_iteration", min_filter_iteration, 3);
  param_.attr("set_min_filter_iteration")(min_filter_iteration);

  nh.param<std::string>("gather_mode", gather_mode, "mean");
  param_.attr("set_gather_mode")(gather_mode);

  nh.param<std::string>("weight_file", weight_file, "config/weights.dat");
  std::string path = ros::package::getPath("elevation_mapping_cupy");

  weight_file = path + "/" + weight_file;
  param_.attr("load_weights")(weight_file);

  resolution_ = resolution;
  map_length_ = map_length;
  map_n_ = (int)(map_length_ / resolution_);
}


void ElevationMappingWrapper::input(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, const RowMatrixXd& R, const Eigen::VectorXd& t, const double positionNoise, const double orientationNoise) {
  py::gil_scoped_acquire acquire;
  RowMatrixXd points;
  pointCloudToMatrix(pointCloud, points);
  map_.attr("input")(static_cast<Eigen::Ref<const RowMatrixXd>>(points),
                     static_cast<Eigen::Ref<const RowMatrixXd>>(R),
                     static_cast<Eigen::Ref<const Eigen::VectorXd>>(t),
                     positionNoise, orientationNoise);
}


void ElevationMappingWrapper::move_to(const Eigen::VectorXd& p) {
  py::gil_scoped_acquire acquire;
  map_.attr("move_to")(static_cast<Eigen::Ref<const Eigen::VectorXd>>(p));
}


void ElevationMappingWrapper::clear() {
  py::gil_scoped_acquire acquire;
  map_.attr("clear")();
}

double ElevationMappingWrapper::get_additive_mean_error() {
  py::gil_scoped_acquire acquire;
  double additive_error = map_.attr("get_additive_mean_error")().cast<double>();
  return additive_error;
}

void ElevationMappingWrapper::get_maps(std::vector<Eigen::MatrixXd>& maps, const std::vector<int>& selection) {
  RowMatrixXd elevation(map_n_, map_n_);
  RowMatrixXd variance(map_n_, map_n_);
  RowMatrixXd traversability(map_n_, map_n_);
  RowMatrixXd min_filtered(map_n_, map_n_);
  RowMatrixXd time_layer(map_n_, map_n_);
  RowMatrixXd normal_x(map_n_, map_n_);
  RowMatrixXd normal_y(map_n_, map_n_);
  RowMatrixXd normal_z(map_n_, map_n_);

  // selection
  RowMatrixXd selection_matrix(1, selection.size());
  for (int i=0; i<selection.size(); i++)
    selection_matrix(0, i) = selection[i];

  py::gil_scoped_acquire acquire;
  map_.attr("get_maps_ref")(static_cast<Eigen::Ref<RowMatrixXd>>(selection_matrix),
                            static_cast<Eigen::Ref<RowMatrixXd>>(elevation),
                            static_cast<Eigen::Ref<RowMatrixXd>>(variance),
                            static_cast<Eigen::Ref<RowMatrixXd>>(traversability),
                            static_cast<Eigen::Ref<RowMatrixXd>>(min_filtered),
                            static_cast<Eigen::Ref<RowMatrixXd>>(time_layer),
                            static_cast<Eigen::Ref<RowMatrixXd>>(normal_x),
                            static_cast<Eigen::Ref<RowMatrixXd>>(normal_y),
                            static_cast<Eigen::Ref<RowMatrixXd>>(normal_z),
                            enable_normal_
                           );
  maps.clear();
  for (const int idx: selection) {
    if (idx == 0)
      maps.push_back(elevation);
    if (idx == 1)
      maps.push_back(variance);
    if (idx == 2)
      maps.push_back(traversability);
    if (idx == 3)
      maps.push_back(min_filtered);
    if (idx == 4)
      maps.push_back(time_layer);
  }
  if (enable_normal_) {
    maps.push_back(normal_x);
    maps.push_back(normal_y);
    maps.push_back(normal_z);
  }
  return;
}


void ElevationMappingWrapper::get_grid_map(grid_map::GridMap& gridMap, const std::vector<std::string>& layerNames) {
  std::vector<std::string> basicLayerNames;
  std::vector<int> selection;
  for (const auto& layerName: layerNames) {
    if (layerName == "elevation") {
      selection.push_back(0);
      basicLayerNames.push_back("elevation");
    }
    if (layerName == "variance")
      selection.push_back(1);
    if (layerName == "traversability") {
      selection.push_back(2);
      basicLayerNames.push_back("traversability");
    }
    if (layerName == "min_filtered")
      selection.push_back(3);
    if (layerName == "time_since_update")
      selection.push_back(4);
  }
  // if (enable_normal_) {
  //   layerNames.push_back("normal_x");
  //   layerNames.push_back("normal_y");
  //   layerNames.push_back("normal_z");
  // }

  RowMatrixXd pos(1, 3);
  py::gil_scoped_acquire acquire;
  map_.attr("get_position")(static_cast<Eigen::Ref<RowMatrixXd>>(pos));
  grid_map::Position position(pos(0, 0), pos(0, 1));
  grid_map::Length length(map_length_, map_length_);
  gridMap.setGeometry(length, resolution_, position);
  std::vector<Eigen::MatrixXd> maps;
  get_maps(maps, selection);
  // gridMap.add("elevation", maps[0].cast<float>());
  // gridMap.add("traversability", maps[2].cast<float>());
  // std::vector<std::string> layerNames = {"elevation", "traversability"};
  for(int i = 0; i < maps.size() ; ++i) {
    gridMap.add(layerNames[i], maps[i].cast<float>());
  }
  // gridMap.setBasicLayers({"elevation", "traversability"});
  gridMap.setBasicLayers(basicLayerNames);
  if (enable_normal_ && enable_normal_color_) {
    addNormalColorLayer(gridMap);
  }
  // Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(map_n_, map_n_);
  // gridMap.add("horizontal_variance_x", zero.cast<float>());
  // gridMap.add("horizontal_variance_y", zero.cast<float>());
  // gridMap.add("horizontal_variance_xy", zero.cast<float>());
  // gridMap.add("time", zero.cast<float>());
}


void ElevationMappingWrapper::get_polygon_traversability(std::vector<Eigen::Vector2d> &polygon, Eigen::Vector3d& result,
                                                           std::vector<Eigen::Vector2d> &untraversable_polygon) {
  RowMatrixXd polygon_m(polygon.size(), 2);
  if (polygon.size() < 3)
    return;
  int i = 0;
  for (auto& p: polygon) {
    polygon_m(i, 0) = p.x();
    polygon_m(i, 1) = p.y();
    i++;
  }
  py::gil_scoped_acquire acquire;
  const int untraversable_polygon_num = map_.attr("get_polygon_traversability")(
      static_cast<Eigen::Ref<const RowMatrixXd>>(polygon_m),
      static_cast<Eigen::Ref<Eigen::VectorXd>>(result)).cast<int>();

  untraversable_polygon.clear();
  if (untraversable_polygon_num > 0) {
    RowMatrixXd untraversable_polygon_m(untraversable_polygon_num, 2);
    map_.attr("get_untraversable_polygon")(static_cast<Eigen::Ref<RowMatrixXd>>(untraversable_polygon_m));
    for (int i = 0; i < untraversable_polygon_num; i++) {
      Eigen::Vector2d p;
      p.x() = untraversable_polygon_m(i, 0);
      p.y() = untraversable_polygon_m(i, 1);
      untraversable_polygon.push_back(p);
    }
  }

  return;
}

void ElevationMappingWrapper::initializeWithPoints(std::vector<Eigen::Vector3d> &points, std::string method) {
  RowMatrixXd points_m(points.size(), 3);
  int i = 0;
  for (auto& p: points) {
    points_m(i, 0) = p.x();
    points_m(i, 1) = p.y();
    points_m(i, 2) = p.z();
    i++;
  }
  py::gil_scoped_acquire acquire;
  map_.attr("initialize_map")(static_cast<Eigen::Ref<const RowMatrixXd>>(points_m), method);
  return;
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

void ElevationMappingWrapper::addNormalColorLayer(grid_map::GridMap& map)
{
  const auto& normalX = map["normal_x"];
  const auto& normalY = map["normal_y"];
  const auto& normalZ = map["normal_z"];

  map.add("color");
  auto& color = map["color"];

  // X: -1 to +1 : Red: 0 to 255
  // Y: -1 to +1 : Green: 0 to 255
  // Z:  0 to  1 : Blue: 128 to 255

  // For each cell in map.
  for (size_t i = 0; i < color.size(); ++i) {
    const Eigen::Vector3f colorVector((normalX(i) + 1.0) / 2.0,
                                      (normalY(i) + 1.0) / 2.0,
                                      (normalZ(i)));
    Eigen::Vector3i intColorVector = (colorVector * 255.0).cast<int>();
    grid_map::colorVectorToValue(intColorVector, color(i));
  }
  return;
}

void ElevationMappingWrapper::update_variance() {
  py::gil_scoped_acquire acquire;
  map_.attr("update_variance")();
  return;
}

void ElevationMappingWrapper::update_time() {
  py::gil_scoped_acquire acquire;
  map_.attr("update_time")();
  return;
}

}
