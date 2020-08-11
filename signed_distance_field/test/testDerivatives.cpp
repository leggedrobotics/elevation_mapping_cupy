//
// Created by rgrandia on 10.08.20.
//

#include <gtest/gtest.h>

#include "signed_distance_field/DistanceDerivatives.h"


TEST(testDerivatives, columnwise) {
  Eigen::MatrixXf data(2, 3);
  data << 1.0, 2.0, 4.0,
      2.0, 4.0, 6.0;

  float resolution = 0.1;
  float doubleResolution = 2.0F * resolution;

  Eigen::MatrixXf manualDifference(2, 3);
  manualDifference << 1.0 /resolution, 3.0 /doubleResolution, 2.0 /resolution,
      2.0 /resolution, 4.0 /doubleResolution, 2.0 /resolution;

  const auto computedDifference = signed_distance_field::columnwiseCentralDifference(data, resolution);

  ASSERT_TRUE(manualDifference.isApprox(computedDifference));
}

TEST(testDerivatives, rowwise) {
  Eigen::MatrixXf data(3, 2);
  data << 1.0, 2.0,
        4.0,  2.0,
        4.0, 6.0;

  float resolution = 0.1;
  float doubleResolution = 2.0F * resolution;

  Eigen::MatrixXf manualDifference(3, 2);
  manualDifference << 3.0 /resolution, 0.0 /resolution,
      3.0 /doubleResolution, 4.0 /doubleResolution,
      0.0 /resolution, 4.0 /resolution;

  const auto computedDifference = signed_distance_field::rowwiseCentralDifference(data, resolution);

  ASSERT_TRUE(manualDifference.isApprox(computedDifference));
}