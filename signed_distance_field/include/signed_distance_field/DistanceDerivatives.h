//
// Created by rgrandia on 10.08.20.
//

#pragma once

#include <Eigen/Dense>

namespace signed_distance_field {

inline Eigen::MatrixXf columnwiseCentralDifference(const Eigen::MatrixXf& data, float resolution) {
  assert(data.cols() >= 2);  // Minimum size to take finite differences.

  const int m = data.cols();
  Eigen::MatrixXf centralDifference(data.rows(), data.cols());

  // First column
  centralDifference.col(0) = (data.col(1) - data.col(0)) / resolution;

  // All the middle columns
  float doubleResolution = 2.0F * resolution;
  for (int i = 1; i + 1 < m; ++i) {
    centralDifference.col(i) = (data.col(i + 1) - data.col(i - 1)) / doubleResolution;
  }

  // Last column
  centralDifference.col(m - 1) = (data.col(m - 1) - data.col(m - 2)) / resolution;

  return centralDifference;
}

inline Eigen::MatrixXf rowwiseCentralDifference(const Eigen::MatrixXf& data, float resolution) {
  return columnwiseCentralDifference(data.transpose(), resolution).transpose();
}

inline Eigen::MatrixXf layerFiniteDifference(const Eigen::MatrixXf& data_k, const Eigen::MatrixXf& data_kp1, float resolution) {
  return (data_kp1 - data_k) / resolution;
}

inline Eigen::MatrixXf layerCentralDifference(const Eigen::MatrixXf& data_km1, const Eigen::MatrixXf& data_kp1, float resolution) {
  float doubleResolution = 2.0F * resolution;
  return (data_kp1 - data_km1) / doubleResolution;
}

}  // namespace signed_distance_field
