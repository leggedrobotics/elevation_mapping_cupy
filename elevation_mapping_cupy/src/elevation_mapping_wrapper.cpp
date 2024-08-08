//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"

// Pybind
#include <pybind11/eigen.h>

// PCL
#include <pcl/common/projection_matrix.h>

// ROS
#include <ros/package.h>

#include <utility>

namespace elevation_mapping_cupy {

ElevationMappingWrapper::ElevationMappingWrapper() {}

void ElevationMappingWrapper::initialize(ros::NodeHandle& nh) {
  // Add the elevation_mapping_cupy path to sys.path
  auto threading = py::module::import("threading");
  py::gil_scoped_acquire acquire;

  auto sys = py::module::import("sys");
  auto path = sys.attr("path");
  std::string module_path = ros::package::getPath("elevation_mapping_cupy");
  module_path = module_path + "/script";
  path.attr("insert")(0, module_path);

  auto elevation_mapping = py::module::import("elevation_mapping_cupy.elevation_mapping");
  auto parameter = py::module::import("elevation_mapping_cupy.parameter");
  param_ = parameter.attr("Parameter")();
  setParameters(nh);
  map_ = elevation_mapping.attr("ElevationMap")(param_);
}

/**
 *  Load ros parameters into Parameter class.
 *  Search for the same name within the name space.
 */
void ElevationMappingWrapper::setParameters(ros::NodeHandle& nh) {
  // Get all parameters names and types.
  py::list paramNames = param_.attr("get_names")();
  py::list paramTypes = param_.attr("get_types")();
  py::gil_scoped_acquire acquire;

  // Try to find the parameter in the ros parameter server.
  // If there was a parameter, set it to the Parameter variable.
  for (int i = 0; i < paramNames.size(); i++) {
    std::string type = py::cast<std::string>(paramTypes[i]);
    std::string name = py::cast<std::string>(paramNames[i]);
    if (type == "float") {
      float param;
      if (nh.getParam(name, param)) {
        param_.attr("set_value")(name, param);
      }
    } else if (type == "str") {
      std::string param;
      if (nh.getParam(name, param)) {
        param_.attr("set_value")(name, param);
      }
    } else if (type == "bool") {
      bool param;
      if (nh.getParam(name, param)) {
        param_.attr("set_value")(name, param);
      }
    } else if (type == "int") {
      int param;
      if (nh.getParam(name, param)) {
        param_.attr("set_value")(name, param);
      }
    }
  }

  XmlRpc::XmlRpcValue subscribers;
  nh.getParam("subscribers", subscribers);

  py::dict sub_dict;
  for (auto& subscriber : subscribers) {
    const char* const name = subscriber.first.c_str();
    const auto& subscriber_params = subscriber.second;
    if (!sub_dict.contains(name)) {
      sub_dict[name] = py::dict();
    }
    for (auto iterat : subscriber_params) {
      const char* const key = iterat.first.c_str();
      const auto val = iterat.second;
      std::vector<std::string> arr;
      switch (val.getType()) {
        case XmlRpc::XmlRpcValue::TypeString:
          sub_dict[name][key] = static_cast<std::string>(val);
          break;
        case XmlRpc::XmlRpcValue::TypeInt:
          sub_dict[name][key] = static_cast<int>(val);
          break;
        case XmlRpc::XmlRpcValue::TypeDouble:
          sub_dict[name][key] = static_cast<double>(val);
          break;
        case XmlRpc::XmlRpcValue::TypeBoolean:
          sub_dict[name][key] = static_cast<bool>(val);
          break;
        case XmlRpc::XmlRpcValue::TypeArray:
          for (int32_t i = 0; i < val.size(); ++i) {
            auto elem = static_cast<std::string>(val[i]);
            arr.push_back(elem);
          }
          sub_dict[name][key] = arr;
          arr.clear();
          break;
        case XmlRpc::XmlRpcValue::TypeStruct:
          break;
        default:
          sub_dict[name][key] = py::cast(val);
          break;
      }
    }
  }
  param_.attr("subscriber_cfg") = sub_dict;

  // point cloud channel fusion
  if (!nh.hasParam("pointcloud_channel_fusions")) {
    ROS_WARN("No pointcloud_channel_fusions parameter found. Using default values.");
  }
  else {
    XmlRpc::XmlRpcValue pointcloud_channel_fusion;
    nh.getParam("pointcloud_channel_fusions", pointcloud_channel_fusion);

    py::dict pointcloud_channel_fusion_dict;
    for (auto& channel_fusion : pointcloud_channel_fusion) {
      const char* const name = channel_fusion.first.c_str();
      std::string fusion = static_cast<std::string>(channel_fusion.second);
      if (!pointcloud_channel_fusion_dict.contains(name)) {
        pointcloud_channel_fusion_dict[name] = fusion;
      }
    }
    ROS_INFO_STREAM("pointcloud_channel_fusion_dict: " << pointcloud_channel_fusion_dict);
    param_.attr("pointcloud_channel_fusions") = pointcloud_channel_fusion_dict;
  }

  // image channel fusion
  if (!nh.hasParam("image_channel_fusions")) {
    ROS_WARN("No image_channel_fusions parameter found. Using default values.");
  }
  else {
    XmlRpc::XmlRpcValue image_channel_fusion;
    nh.getParam("image_channel_fusions", image_channel_fusion);

    py::dict image_channel_fusion_dict;
    for (auto& channel_fusion : image_channel_fusion) {
      const char* const name = channel_fusion.first.c_str();
      std::string fusion = static_cast<std::string>(channel_fusion.second);
      if (!image_channel_fusion_dict.contains(name)) {
        image_channel_fusion_dict[name] = fusion;
      }
    }
    ROS_INFO_STREAM("image_channel_fusion_dict: " << image_channel_fusion_dict);
    param_.attr("image_channel_fusions") = image_channel_fusion_dict;
  }

  param_.attr("update")();
  resolution_ = py::cast<float>(param_.attr("get_value")("resolution"));
  map_length_ = py::cast<float>(param_.attr("get_value")("true_map_length"));
  map_n_ = py::cast<int>(param_.attr("get_value")("true_cell_n"));

  nh.param<bool>("enable_normal", enable_normal_, false);
  nh.param<bool>("enable_normal_color", enable_normal_color_, false);
}

void ElevationMappingWrapper::input(const RowMatrixXd& points, const std::vector<std::string>& channels, const RowMatrixXd& R,
                                    const Eigen::VectorXd& t, const double positionNoise, const double orientationNoise) {
  py::gil_scoped_acquire acquire;
  map_.attr("input_pointcloud")(Eigen::Ref<const RowMatrixXd>(points), channels, Eigen::Ref<const RowMatrixXd>(R),
                     Eigen::Ref<const Eigen::VectorXd>(t), positionNoise, orientationNoise);
}

void ElevationMappingWrapper::input_image(const std::vector<ColMatrixXf>& multichannel_image, const std::vector<std::string>& channels, const RowMatrixXd& R,
                                          const Eigen::VectorXd& t, const RowMatrixXd& cameraMatrix, const Eigen::VectorXd& D, const std::string distortion_model, int height, int width) {
  py::gil_scoped_acquire acquire;
  map_.attr("input_image")(multichannel_image, channels, Eigen::Ref<const RowMatrixXd>(R), Eigen::Ref<const Eigen::VectorXd>(t),
                           Eigen::Ref<const RowMatrixXd>(cameraMatrix), Eigen::Ref<const Eigen::VectorXd>(D), distortion_model, height, width);
}

void ElevationMappingWrapper::move_to(const Eigen::VectorXd& p, const RowMatrixXd& R) {
  py::gil_scoped_acquire acquire;
  map_.attr("move_to")(Eigen::Ref<const Eigen::VectorXd>(p), Eigen::Ref<const RowMatrixXd>(R));
}

void ElevationMappingWrapper::clear() {
  py::gil_scoped_acquire acquire;
  map_.attr("clear")();
}

double ElevationMappingWrapper::get_additive_mean_error() {
  py::gil_scoped_acquire acquire;
  return map_.attr("get_additive_mean_error")().cast<double>();
}

bool ElevationMappingWrapper::exists_layer(const std::string& layerName) {
  py::gil_scoped_acquire acquire;
  return py::cast<bool>(map_.attr("exists_layer")(layerName));
}

void ElevationMappingWrapper::get_layer_data(const std::string& layerName, RowMatrixXf& map) {
  py::gil_scoped_acquire acquire;
  map = RowMatrixXf(map_n_, map_n_);
  map_.attr("get_map_with_name_ref")(layerName, Eigen::Ref<RowMatrixXf>(map));
}

void ElevationMappingWrapper::get_grid_map(grid_map::GridMap& gridMap, const std::vector<std::string>& requestLayerNames) {
  std::vector<std::string> basicLayerNames;
  std::vector<std::string> layerNames = requestLayerNames;
  std::vector<int> selection;
  for (const auto& layerName : layerNames) {
    if (layerName == "elevation") {
      basicLayerNames.push_back("elevation");
    }
  }

  RowMatrixXd pos(1, 3);
  py::gil_scoped_acquire acquire;
  map_.attr("get_position")(Eigen::Ref<RowMatrixXd>(pos));
  grid_map::Position position(pos(0, 0), pos(0, 1));
  grid_map::Length length(map_length_, map_length_);
  gridMap.setGeometry(length, resolution_, position);
  std::vector<Eigen::MatrixXf> maps;

  for (const auto& layerName : layerNames) {
    bool exists = map_.attr("exists_layer")(layerName).cast<bool>();
    if (exists) {
      RowMatrixXf map(map_n_, map_n_);
      map_.attr("get_map_with_name_ref")(layerName, Eigen::Ref<RowMatrixXf>(map));
      gridMap.add(layerName, map);
    }
  }
  if (enable_normal_color_) {
    RowMatrixXf normal_x(map_n_, map_n_);
    RowMatrixXf normal_y(map_n_, map_n_);
    RowMatrixXf normal_z(map_n_, map_n_);
    map_.attr("get_normal_ref")(Eigen::Ref<RowMatrixXf>(normal_x), Eigen::Ref<RowMatrixXf>(normal_y), Eigen::Ref<RowMatrixXf>(normal_z));
    gridMap.add("normal_x", normal_x);
    gridMap.add("normal_y", normal_y);
    gridMap.add("normal_z", normal_z);
  }
  gridMap.setBasicLayers(basicLayerNames);
  if (enable_normal_color_) {
    addNormalColorLayer(gridMap);
  }
}

void ElevationMappingWrapper::get_polygon_traversability(std::vector<Eigen::Vector2d>& polygon, Eigen::Vector3d& result,
                                                         std::vector<Eigen::Vector2d>& untraversable_polygon) {
  if (polygon.size() < 3) {
    return;
  }
  RowMatrixXf polygon_m(polygon.size(), 2);
  int i = 0;
  for (auto& p : polygon) {
    polygon_m(i, 0) = p.x();
    polygon_m(i, 1) = p.y();
    i++;
  }
  py::gil_scoped_acquire acquire;
  const int untraversable_polygon_num =
      map_.attr("get_polygon_traversability")(Eigen::Ref<const RowMatrixXf>(polygon_m), Eigen::Ref<Eigen::VectorXd>(result)).cast<int>();

  untraversable_polygon.clear();
  if (untraversable_polygon_num > 0) {
    RowMatrixXf untraversable_polygon_m(untraversable_polygon_num, 2);
    map_.attr("get_untraversable_polygon")(Eigen::Ref<RowMatrixXf>(untraversable_polygon_m));
    for (int j = 0; j < untraversable_polygon_num; j++) {
      Eigen::Vector2d p;
      p.x() = untraversable_polygon_m(j, 0);
      p.y() = untraversable_polygon_m(j, 1);
      untraversable_polygon.push_back(p);
    }
  }
}

void ElevationMappingWrapper::initializeWithPoints(std::vector<Eigen::Vector3d>& points, std::string method) {
  RowMatrixXd points_m(points.size(), 3);
  int i = 0;
  for (auto& p : points) {
    points_m(i, 0) = p.x();
    points_m(i, 1) = p.y();
    points_m(i, 2) = p.z();
    i++;
  }
  py::gil_scoped_acquire acquire;
  map_.attr("initialize_map")(Eigen::Ref<const RowMatrixXd>(points_m), method);
}

void ElevationMappingWrapper::addNormalColorLayer(grid_map::GridMap& map) {
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
    const Eigen::Vector3f colorVector((normalX(i) + 1.0) / 2.0, (normalY(i) + 1.0) / 2.0, (normalZ(i)));
    Eigen::Vector3i intColorVector = (colorVector * 255.0).cast<int>();
    grid_map::colorVectorToValue(intColorVector, color(i));
  }
}

void ElevationMappingWrapper::update_variance() {
  py::gil_scoped_acquire acquire;
  map_.attr("update_variance")();
}

void ElevationMappingWrapper::update_time() {
  py::gil_scoped_acquire acquire;
  map_.attr("update_time")();
}

}  // namespace elevation_mapping_cupy
