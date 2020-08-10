//
// Created by rgrandia on 10.07.20.
//

#pragma once

#include <vector>

#include <Eigen/Dense>

namespace signed_distance_field {

class SignedDistanceField {
 public:
  double atPosition(const Eigen::Vector3d& position) const;
  Eigen::Vector3d derivativeAtPosition(const Eigen::Vector3d& position) const;

 private:
  size_t nearestNode(const Eigen::Vector3d& position) const;
  Eigen::Vector3d nodePosition(const Eigen::Vector3d& position) const;

  using node_data_t = std::array<float, 4>;
  static double distance(const node_data_t& nodeData) noexcept { return nodeData[0]; }
  static Eigen::Vector3d derivative(const node_data_t& nodeData) noexcept { return {nodeData[1], nodeData[2], nodeData[3]}; }

  double resolution_;
  std::array<size_t, 3> gridsize_;
  std::vector<node_data_t> data_;
};

}  // namespace signed_distance_field
