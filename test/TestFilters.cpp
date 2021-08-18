/**
 * @authors     Fabian Jenelten
 * @affiliation RSL
 * @brief       Tests for grid map filters.
 */

#include <gtest/gtest.h>

#include <grid_map_filters_rsl/inpainting.hpp>

using namespace grid_map;

TEST(TestInpainting, initialization) {  // NOLINT
  // Grid map with constant gradient.
  GridMap map;
  map.setGeometry(Length(1.0, 2.0), 0.1, Position(0.0, 0.0));
  map.add("elevation", 1.0);
  const Eigen::MatrixXf H0 = map.get("elevation");
  Eigen::MatrixXf& H_in = map.get("elevation");

  // Set nan patches.
  H_in.topLeftCorner<3, 3>(3, 3).setConstant(NAN);
  H_in.middleRows<2>(5).setConstant(NAN);

  // Fill in nan values.
  inpainting::nonlinearInterpolation(map, "elevation", "filled_nonlin", 0.1);
  inpainting::minValues(map, "elevation", "filled_min");
  inpainting::biLinearInterpolation(map, "elevation", "filled_bilinear", 3);

  // Check if open-cv in-painting was successful.
  constexpr double threshold = 1.0e-9;
  const Eigen::MatrixXf& H_out_ref = map.get("filled_nonlin");
  EXPECT_FALSE(std::isnan(H_out_ref.norm()));
  EXPECT_TRUE((H_out_ref - H0).norm() < threshold);

  // Compare against open-cv in-painting.
  EXPECT_TRUE((H_out_ref - map.get("filled_min")).norm() < threshold);
  EXPECT_TRUE((H_out_ref - map.get("filled_bilinear")).norm() < threshold);
}

TEST(TestInpainting, minValuesAroundContour) {  // NOLINT
  // Grid map with constant gradient.
  GridMap map;
  map.setGeometry(Length(3.0, 3.0), 1.0, Position(0.0, 0.0));
  map.add("input", 1.0);
  const Eigen::MatrixXf H0 = map.get("input");
  Eigen::MatrixXf& H_in = map.get("input");

  // Set nan patches.
  // clang-format off
  H_in << NAN,  0.0, -1.0,
          1.0,  NAN,  NAN,
         -2.0,  0.0,  NAN;
  Eigen::MatrixXf H_expected(3, 3);
  H_expected <<  0.0,  0.0, -1.0,
                 1.0, -1.0, -1.0,
                -2.0,  0.0, -1.0;
  // clang-format on

  inpainting::minValues(map, "input", "filled_min");
  constexpr double threshold = 1.0e-9;
  EXPECT_TRUE((H_expected - map.get("filled_min")).norm() < threshold);
}

TEST(TestInpainting, minValuesOnlyNaN) {  // NOLINT
  // Grid map with only NaN, check that we don't end in infinite loop
  GridMap map;
  map.setGeometry(Length(3.0, 3.0), 1.0, Position(0.0, 0.0));
  map.add("input_nan", NAN);    // layer with only NaN
  map.add("input_nonan", 1.0);  // layer with constant value

  inpainting::minValues(map, "input_nan", "filled_min_nan");
  inpainting::minValues(map, "input_nonan", "filled_min_nonan");

  EXPECT_TRUE(map.get("filled_min_nan").hasNaN());
  EXPECT_DOUBLE_EQ(map.get("filled_min_nonan").minCoeff(), 1.0);
}