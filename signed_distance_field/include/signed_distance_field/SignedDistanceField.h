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

#include "Gridmap3dLookup.h"

namespace signed_distance_field {

class SignedDistanceField {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SignedDistanceField(const grid_map::GridMap& gridMap, const std::string& elevationLayer, double minHeight, double maxHeight);

  double atPosition(const Eigen::Vector3d& position) const;

  Eigen::Vector3d derivativeAtPosition(const Eigen::Vector3d& position) const;

  std::pair<double, Eigen::Vector3d> distanceAndDerivativeAt(const Eigen::Vector3d& position) const;

  pcl::PointCloud<pcl::PointXYZI> asPointCloud() const;

 private:
  void computeSignedDistance(const grid_map::Matrix& elevation);
  void emplacebackLayerData(const grid_map::Matrix& signedDistance, const grid_map::Matrix& dx, const grid_map::Matrix& dy, const grid_map::Matrix& dz);

  using node_data_t = std::array<float, 4>;
  static double distance(const node_data_t& nodeData) noexcept { return nodeData[0]; }
  static Eigen::Vector3d derivative(const node_data_t& nodeData) noexcept { return {nodeData[1], nodeData[2], nodeData[3]}; }

  Gridmap3dLookup gridmap3DLookup_;
  std::vector<node_data_t> data_;
};

}  // namespace signed_distance_field
