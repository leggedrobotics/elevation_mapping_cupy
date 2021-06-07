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
  inpainting::biLinearInterpolation(map, "elevation", "filled_bilinear");

  // Check if open-cv in-painting was successful.
  constexpr double threshold = 1.0e-9;
  const Eigen::MatrixXf& H_out_ref = map.get("filled_nonlin");
  EXPECT_FALSE(std::isnan(H_out_ref.norm()));
  EXPECT_TRUE((H_out_ref - H0).norm() < threshold);

  // Compare against open-cv in-painting.
  EXPECT_TRUE((H_out_ref - map.get("filled_min")).norm() < threshold);
  EXPECT_TRUE((H_out_ref - map.get("filled_bilinear")).norm() < threshold);
}