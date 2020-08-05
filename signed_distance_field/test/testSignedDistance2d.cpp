//
// Created by rgrandia on 10.07.20.
//

#include <gtest/gtest.h>

#include "signed_distance_field/SignedDistance2d.h"

namespace signed_distance_field {

// N^2 naive implementation, for testing purposes
grid_map::Matrix naiveSignedDistance(const grid_map::Matrix& elevationMap, float height, float resolution) {
  grid_map::Matrix signedDistance(elevationMap.rows(), elevationMap.cols());

  // For each point
  for (int row = 0; row < elevationMap.rows(); ++row) {
    for (int col = 0; col < elevationMap.cols(); ++col) {
      if (elevationMap(row, col) >= height) {
        signedDistance(row, col) = -INF;
        // find closest open space over all other points
        for (int i = 0; i < elevationMap.rows(); ++i) {
          for (int j = 0; j < elevationMap.cols(); ++j) {
            if (elevationMap(i, j) < height) {
              float dx = resolution * (i - row);
              float dy = resolution * (j - col);
              float currentSignedDistance = -std::sqrt(dx * dx + dy * dy);
              signedDistance(row, col) = std::max(signedDistance(row, col), currentSignedDistance);
            }
          }
        }
      } else {
        signedDistance(row, col) = INF;
        // find closest object over all other points
        for (int i = 0; i < elevationMap.rows(); ++i) {
          for (int j = 0; j < elevationMap.cols(); ++j) {
            if (elevationMap(i, j) >= height) {
              float dx = resolution * (i - row);
              float dy = resolution * (j - col);
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

TEST(testSignedDistance2d, sparsityInfo_noObstacles) {
  const int n = 3;
  const int m = 4;
  const grid_map::Matrix map = grid_map::Matrix::Ones(n, m);

  const auto sparsityInfo = signed_distance_field::collectSparsityInfo(map, 2.0);

  ASSERT_TRUE(sparsityInfo.hasFreeSpace);
  ASSERT_FALSE(sparsityInfo.hasObstacles);
  ASSERT_EQ(sparsityInfo.numColsHomogeneous, m);
  ASSERT_EQ(sparsityInfo.numRowsHomogeneous, n);
}

TEST(testSignedDistance2d, sparsityInfo_allObstacles) {
  const int n = 3;
  const int m = 4;
  const grid_map::Matrix map = grid_map::Matrix::Ones(n, m);

  const auto sparsityInfo = signed_distance_field::collectSparsityInfo(map, 0.0);

  ASSERT_FALSE(sparsityInfo.hasFreeSpace);
  ASSERT_TRUE(sparsityInfo.hasObstacles);
  ASSERT_EQ(sparsityInfo.numColsHomogeneous, m);
  ASSERT_EQ(sparsityInfo.numRowsHomogeneous, n);
}

TEST(testSignedDistance2d, sparsityInfo_mixed) {
  const int n = 2;
  const int m = 3;
  grid_map::Matrix map(n, m);
  map << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0;

  const auto sparsityInfo = signed_distance_field::collectSparsityInfo(map, 0.5);

  ASSERT_TRUE(sparsityInfo.hasFreeSpace);
  ASSERT_TRUE(sparsityInfo.hasObstacles);
  ASSERT_TRUE(sparsityInfo.colsIsHomogenous[0]);
  ASSERT_TRUE(sparsityInfo.rowsIsHomogenous[1]);
  ASSERT_EQ(sparsityInfo.numColsHomogeneous, 1);
  ASSERT_EQ(sparsityInfo.numRowsHomogeneous, 1);
}

TEST(testSignedDistance2d, signedDistance2d_noObstacles) {
  const int n = 3;
  const int m = 4;
  const float resolution = 0.1;
  const grid_map::Matrix map = grid_map::Matrix::Ones(n, m);

  const auto signedDistance = signed_distance_field::computeSignedDistanceAtHeight(map, 2.0, resolution);

  ASSERT_TRUE((signedDistance.array() == signed_distance_field::INF).all());
}

TEST(testSignedDistance2d, signedDistance2d_allObstacles) {
  const int n = 3;
  const int m = 4;
  const float resolution = 0.1;
  const grid_map::Matrix map = grid_map::Matrix::Ones(n, m);

  const auto signedDistance = signed_distance_field::computeSignedDistanceAtHeight(map, 0.0, resolution);

  ASSERT_TRUE((signedDistance.array() == 0.0).all());
}

TEST(testSignedDistance2d, signedDistance2d_mixed) {
  const int n = 2;
  const int m = 3;
  const float resolution = 0.1;
  grid_map::Matrix map(n, m);
  map << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0;

  const auto naiveSignedDistance = signed_distance_field::naiveSignedDistance(map, 0.5, resolution);
  const auto signedDistance = signed_distance_field::computeSignedDistanceAtHeight(map, 0.5, resolution);
  ASSERT_TRUE(signedDistance.isApprox(naiveSignedDistance));
}

TEST(testSignedDistance2d, signedDistance2d_oneObstacle) {
  const int n = 20;
  const int m = 30;
  const float resolution = 0.1;
  grid_map::Matrix map =  grid_map::Matrix::Zero(n, m);
  map(n/2, m/2) = 1.0;

  const auto naiveSignedDistance = signed_distance_field::naiveSignedDistance(map, 0.5, resolution);
  const auto signedDistance = signed_distance_field::computeSignedDistanceAtHeight(map, 0.5, resolution);
  ASSERT_TRUE(signedDistance.isApprox(naiveSignedDistance));
}

TEST(testSignedDistance2d, signedDistance2d_oneFreeSpace) {
  const int n = 20;
  const int m = 30;
  const float resolution = 0.1;
  grid_map::Matrix map =  grid_map::Matrix::Ones(n, m);
  map(n/2, m/2) = 0.0;

  const auto naiveSignedDistance = signed_distance_field::naiveSignedDistance(map, 0.5, resolution);
  const auto signedDistance = signed_distance_field::computeSignedDistanceAtHeight(map, 0.5, resolution);
  ASSERT_TRUE(signedDistance.isApprox(naiveSignedDistance));
}

TEST(testSignedDistance2d, signedDistance2d_random) {
  const int n = 20;
  const int m = 30;
  const float resolution = 0.1;
  grid_map::Matrix map =  grid_map::Matrix::Random(n, m); // random [-1.0, 1.0]

  const auto naiveSignedDistance = signed_distance_field::naiveSignedDistance(map, 0.0, resolution);
  const auto signedDistance = signed_distance_field::computeSignedDistanceAtHeight(map, 0.0, resolution);
  ASSERT_TRUE(signedDistance.isApprox(naiveSignedDistance));
}