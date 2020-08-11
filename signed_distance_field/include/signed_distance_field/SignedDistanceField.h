//
// Created by rgrandia on 10.07.20.
//

#pragma once

#include <vector>

#include <Eigen/Dense>

#include <grid_map_core/TypeDefs.hpp>
#include <grid_map_core/GridMap.hpp>

#include <pcl/point_types.h>
#include <pcl/conversions.h>

namespace signed_distance_field {

class SignedDistanceField {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SignedDistanceField(const grid_map::GridMap& gridMap, const std::string& elevationLayer, double minHeight, double maxHeight);

  double atPosition(const Eigen::Vector3d& position) const;
  Eigen::Vector3d derivativeAtPosition(const Eigen::Vector3d& position) const;

  pcl::PointCloud<pcl::PointXYZI> asPointCloud() const;

 private:
  void computeSignedDistance(const grid_map::Matrix& elevation);
  // Must be called in ascending layers numbers [0, gridSize_[2]-1]
  void emplacebackLayerData(const grid_map::Matrix& signedDistance, const grid_map::Matrix& dx, const grid_map::Matrix& dy, const grid_map::Matrix& dz);
  size_t nearestNode(const Eigen::Vector3d& position) const;
  Eigen::Vector3d nodePosition(size_t rowX, size_t colY, size_t layerZ) const;
  size_t linearIndex(size_t rowX, size_t colY, size_t layerZ) const ;

  using node_data_t = std::array<float, 4>;
  static double distance(const node_data_t& nodeData) noexcept { return nodeData[0]; }
  static Eigen::Vector3d derivative(const node_data_t& nodeData) noexcept { return {nodeData[1], nodeData[2], nodeData[3]}; }

  float resolution_;
  std::array<size_t, 3> gridsize_;
  Eigen::Vector3d gridOrigin_;
  std::vector<node_data_t> data_;
};

}  // namespace signed_distance_field
