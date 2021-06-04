/**
 * @file        GridMapDerivative.cpp
 * @authors     Fabian Jenelten
 * @date        20.05, 2021
 * @affiliation ETH RSL
 * @brief       Computes first and second order derivatives for grid map. Intended to be used for online applications (i.e, derivatives
 *              are not precomputed but computed at the time and location where they are needed).
 */

// grid map filters rsl.
#include <grid_map_filters_rsl/GridMapDerivative.hpp>

// robot utils.
#include <robot_utils/math/math.hpp>

namespace tamols_mapping {
constexpr std::array<double, GridMapDerivative::kernelSize_> GridMapDerivative::kernelD1_;
constexpr std::array<double, GridMapDerivative::kernelSize_> GridMapDerivative::kernelD2_;
constexpr double GridMapDerivative::res_;
constexpr int GridMapDerivative::kernelSize_;

GridMapDerivative::GridMapDerivative() {}

void GridMapDerivative::estimateGradient(const grid_map::GridMap& gridMap, Gradient& gradient, const grid_map::Index& index,
                                         const grid_map::Matrix& H) const {
  // Init.
  gradient.setZero();

  constexpr int maxId = (kernelSize_ - 1) / 2;
  for (auto dim = 0U; dim < 2U; ++dim) {
    auto tempIndex = index;
    for (auto id = -maxId; id <= maxId; ++id) {  // x or y
      tempIndex(dim) = index(dim) + id;
      mapIndexToGrid(gridMap, tempIndex);
      gradient(dim) += kernelD1_[maxId + id] * H(tempIndex.x(), tempIndex.y());
    }
  }
}

void GridMapDerivative::estimateGradientAndCurvature(const grid_map::GridMap& gridMap, Gradient& gradient, Curvature& curvature,
                                                     const grid_map::Index& index, const grid_map::Matrix& H) const {
  // Init.
  gradient.setZero();
  curvature.setZero();

  // Gradient in Y for different x (used for computing the cross hessian).
  constexpr int maxId = (kernelSize_ - 1) / 2;
  std::array<double, kernelSize_> gradientYArray{0.0};
  for (auto idY = -maxId; idY <= maxId; ++idY) {    // y
    for (auto idX = -maxId; idX <= maxId; ++idX) {  // x
      grid_map::Index tempIndex = index + grid_map::Index(idX, idY);
      mapIndexToGrid(gridMap, tempIndex);
      gradientYArray[maxId + idX] += kernelD1_[maxId + idY] * H(tempIndex.x(), tempIndex.y());
    }
  }

  for (auto dim = 0U; dim < 2U; ++dim) {
    auto tempIndex = index;
    for (auto id = -maxId; id <= maxId; ++id) {  // x or y
      tempIndex(dim) = index(dim) + id;
      mapIndexToGrid(gridMap, tempIndex);
      const auto arrayId = maxId + id;

      const double height = H(tempIndex.x(), tempIndex.y());
      gradient(dim) += kernelD1_[arrayId] * height;
      curvature(dim, dim) += kernelD2_[arrayId] * height;

      if (dim == 0U) {
        curvature(0U, 1U) += kernelD1_[arrayId] * gradientYArray[arrayId];
      }
    }
  }

  // Curvature is symmetric.
  curvature(1U, 0U) = curvature(0U, 1U);
}

void GridMapDerivative::mapIndexToGrid(const grid_map::GridMap& gridMap, grid_map::Index& index) const {
  for (auto dim = 0U; dim < 2U; ++dim) {
    robot_utils::boundToRange(&index(dim), 0, gridMap.getSize()(dim) - 1);
  }
}

}  // namespace tamols_mapping
