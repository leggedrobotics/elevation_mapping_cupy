/*
 * naiveSignedDistance.h
 *
 *  Created on: Aug 10, 2020
 *      Author: Ruben Grandia
 *   Institute: ETH Zurich
 */

#pragma once

namespace grid_map {
namespace signed_distance_field {

inline Eigen::Matrix<bool, -1, -1> occupancyAtHeight(const Matrix& elevationMap, float height) {
  Eigen::Matrix<bool, -1, -1> occupany = elevationMap.unaryExpr([=](float val) { return val > height; });
  return occupany;
}

inline bool isEqualSdf(const Matrix& sdf0, const Matrix& sdf1, float tol) {
  Matrix error = (sdf0 - sdf1).array().abs();
  float maxDifference = error.maxCoeff();
  return maxDifference < tol;
}

// N^2 naive implementation, for testing purposes
inline Matrix naiveSignedDistanceAtHeight(const Matrix& elevationMap, float height, float resolution) {
  Matrix signedDistance(elevationMap.rows(), elevationMap.cols());

  // For each point
  for (int row = 0; row < elevationMap.rows(); ++row) {
    for (int col = 0; col < elevationMap.cols(); ++col) {
      if (elevationMap(row, col) >= height) {  // point is below surface
        signedDistance(row, col) = -INF;
        // find closest open space over all other points
        for (int i = 0; i < elevationMap.rows(); ++i) {
          for (int j = 0; j < elevationMap.cols(); ++j) {
            // Compute distance to free cube at location (i, j)
            float dx = resolution * pixelBorderDistance(i, row);
            float dy = resolution * pixelBorderDistance(j, col);
            float dz = std::max(elevationMap(i, j) - height, 0.0F);
            float currentSignedDistance = -std::sqrt(dx * dx + dy * dy + dz * dz);
            signedDistance(row, col) = std::max(signedDistance(row, col), currentSignedDistance);
          }
        }
      } else {  // point is above surface
        signedDistance(row, col) = INF;
        // find closest object over all other points
        for (int i = 0; i < elevationMap.rows(); ++i) {
          for (int j = 0; j < elevationMap.cols(); ++j) {
            // Compute distance to occupied cube at location (i, j)
            float dx = resolution * pixelBorderDistance(i, row);
            float dy = resolution * pixelBorderDistance(j, col);
            float dz = std::max(height - elevationMap(i, j), 0.0F);
            float currentSignedDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
            signedDistance(row, col) = std::min(signedDistance(row, col), currentSignedDistance);
          }
        }
      }
    }
  }

  return signedDistance;
}

inline Matrix naiveSignedDistanceFromOccupancy(const Eigen::Matrix<bool, -1, -1>& occupancyGrid, float resolution) {
  Matrix signedDistance(occupancyGrid.rows(), occupancyGrid.cols());

  // For each point
  for (int row = 0; row < occupancyGrid.rows(); ++row) {
    for (int col = 0; col < occupancyGrid.cols(); ++col) {
      if (occupancyGrid(row, col)) {  // This point is an obstable
        signedDistance(row, col) = -INF;
        // find closest open space over all other points
        for (int i = 0; i < occupancyGrid.rows(); ++i) {
          for (int j = 0; j < occupancyGrid.cols(); ++j) {
            if (!occupancyGrid(i, j)) {
              float dx = resolution * pixelBorderDistance(i, row);
              float dy = resolution * pixelBorderDistance(j, col);
              float currentSignedDistance = -std::sqrt(dx * dx + dy * dy);
              signedDistance(row, col) = std::max(signedDistance(row, col), currentSignedDistance);
            }
          }
        }
      } else {  // This point is in free space
        signedDistance(row, col) = INF;
        // find closest object over all other points
        for (int i = 0; i < occupancyGrid.rows(); ++i) {
          for (int j = 0; j < occupancyGrid.cols(); ++j) {
            if (occupancyGrid(i, j)) {
              float dx = resolution * pixelBorderDistance(i, row);
              float dy = resolution * pixelBorderDistance(j, col);
              float currentSignedDistance = std::sqrt(dx * dx + dy * dy);
              signedDistance(row, col) = std::min(signedDistance(row, col), currentSignedDistance);
            }
          }
        }
      }
    }
  }

  return signedDistance;
}

}  // namespace signed_distance_field
}  // namespace grid_map