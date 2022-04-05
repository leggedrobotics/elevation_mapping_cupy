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
  std::string weight_file;
  py::list paramNames = param_.attr("get_names")();
  py::list paramTypes = param_.attr("get_types")();
  py::gil_scoped_acquire acquire;
  for (int i = 0; i < paramNames.size(); i++) {
    std::string type = py::cast<std::string>(paramTypes[i]);
    std::string name = py::cast<std::string>(paramNames[i]);
    if (type == "float") {
      float param;
      if (nh.getParam(name, param))
        param_.attr("set_value")(name, param);
    }
    else if (type == "str") {
      std::string param;
      if (nh.getParam(name, param))
        param_.attr("set_value")(name, param);
    }
    else if (type == "bool") {
      bool param;
      if (nh.getParam(name, param))
        param_.attr("set_value")(name, param);
    }
    else if (type == "int") {
      int param;
      if (nh.getParam(name, param))
        param_.attr("set_value")(name, param);
    }
  }

  nh.param<std::string>("weight_file", weight_file, "config/weights.dat");
  std::string path = ros::package::getPath("elevation_mapping_cupy");

  weight_file = path + "/" + weight_file;
  param_.attr("load_weights")(weight_file);

  resolution_ = py::cast<float>(param_.attr("get_value")("resolution"));
  map_length_ = py::cast<float>(param_.attr("get_value")("map_length"));
  map_n_ = static_cast<int>(round(map_length_ / resolution_));
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
  RowMatrixXd upper_bound(map_n_, map_n_);
  RowMatrixXd is_upper_bound(map_n_, map_n_);
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
                            static_cast<Eigen::Ref<RowMatrixXd>>(upper_bound),
                            static_cast<Eigen::Ref<RowMatrixXd>>(is_upper_bound),
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
    if (idx == 5)
      maps.push_back(upper_bound);
    if (idx == 6)
      maps.push_back(is_upper_bound);
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
    if (layerName == "upper_bound")
      selection.push_back(5);
    if (layerName == "is_upper_bound")
      selection.push_back(6);
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
