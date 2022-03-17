//
// Created by rgrandia on 10.08.20.
//

#include <gtest/gtest.h>

#include "signed_distance_field/DistanceDerivatives.h"

using namespace grid_map;
using namespace signed_distance_field;

TEST(testDerivatives, columnwise) {
  Matrix data(2, 3);
  data << 1.0, 2.0, 4.0,
      2.0, 4.0, 6.0;

  float resolution = 0.1;
  float doubleResolution = 2.0F * resolution;

  Matrix manualDifference(2, 3);
  manualDifference << 1.0 /resolution, 3.0 /doubleResolution, 2.0 /resolution,
      2.0 /resolution, 4.0 /doubleResolution, 2.0 /resolution;

  Matrix computedDifference = Matrix::Zero(data.rows(), data.cols());
  columnwiseCentralDifference(data, computedDifference, resolution);

  ASSERT_TRUE(manualDifference.isApprox(computedDifference));
}
