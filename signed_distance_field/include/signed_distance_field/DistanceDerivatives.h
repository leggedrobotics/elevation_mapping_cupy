//
// Created by rgrandia on 10.08.20.
//

#pragma once

#include <Eigen/Dense>

namespace signed_distance_field {

inline void columnwiseCentralDifference(const Eigen::MatrixXf& data, Eigen::MatrixXf& centralDifference, float resolution) {
  assert(data.cols() >= 2);  // Minimum size to take finite differences.

  const int m = data.cols();

  // First column
  centralDifference.col(0) = (data.col(1) - data.col(0)) / resolution;

  // All the middle columns
  float doubleResolution = 2.0F * resolution;
  for (int i = 1; i + 1 < m; ++i) {
    centralDifference.col(i) = (data.col(i + 1) - data.col(i - 1)) / doubleResolution;
  }

  // Last column
  centralDifference.col(m - 1) = (data.col(m - 1) - data.col(m - 2)) / resolution;
}

inline void layerFiniteDifference(const Eigen::MatrixXf& data_k, const Eigen::MatrixXf& data_kp1, Eigen::MatrixXf& result,
                                             float resolution) {
  result = (1.0F / resolution) * (data_kp1 - data_k);
}

inline void layerCentralDifference(const Eigen::MatrixXf& data_km1, const Eigen::MatrixXf& data_kp1, Eigen::MatrixXf& result, float resolution) {
  float doubleResolution = 2.0F * resolution;
  result = (1.0F / doubleResolution) * (data_kp1 - data_km1);
}

}  // namespace signed_distance_field
