/**
 * @file        GridMapDerivative.cpp
 * @authors     Fabian Jenelten
 * @date        20.05, 2021
 * @affiliation ETH RSL
 * @brief       Computes first and second order derivatives for a grid map. Intended to be used for online applications (i.e, derivatives
 *              are not precomputed but computed at the time and location where they are needed).
 */

// grid map filters rsl.
#include <grid_map_filters_rsl/GridMapDerivative.hpp>

namespace grid_map {
namespace derivative {
constexpr std::array<double, GridMapDerivative::kernelSize_> GridMapDerivative::kernelD1_;
constexpr std::array<double, GridMapDerivative::kernelSize_> GridMapDerivative::kernelD2_;
constexpr int GridMapDerivative::kernelSize_;

GridMapDerivative::GridMapDerivative() : oneDivRes_(0.0), oneDivResSquared_(0.0) {}

bool GridMapDerivative::initialize(double res) {
  oneDivRes_ = 1.0 / res;
  oneDivResSquared_ = oneDivRes_ * oneDivRes_;
  return res > 0.0;
}

void GridMapDerivative::estimateGradient(const grid_map::GridMap& gridMap, Gradient& gradient, const grid_map::Index& index,
                                         const grid_map::Matrix& H) const {
  // Init.
  gradient.setZero();

  constexpr int maxId = (kernelSize_ - 1) / 2;
  for (auto dim = 0U; dim < 2U; ++dim) {
    auto tempIndex = index;
    const auto centerId = getKernelCenter(gridMap, index, dim, maxId);
    for (auto id = centerId - maxId; id <= centerId + maxId; ++id) {  // x or y
      tempIndex(dim) = index(dim) + id;
      gradient(dim) += kernelD1_[maxId + id] * H(tempIndex.x(), tempIndex.y());
    }
  }

  // Normalize.
  gradient *= oneDivRes_;
}

void GridMapDerivative::estimateGradientAndCurvature(const grid_map::GridMap& gridMap, Gradient& gradient, Curvature& curvature,
                                                     const grid_map::Index& index, const grid_map::Matrix& H) const {
  // Init.
  gradient.setZero();
  curvature.setZero();

  // Gradient in Y for different x (used for computing the cross hessian).
  constexpr int maxId = (kernelSize_ - 1) / 2;
  std::array<double, kernelSize_> gradientYArray{0.0};
  const auto centerIdY = getKernelCenter(gridMap, index, 1, maxId);
  for (auto idY = centerIdY - maxId; idY <= centerIdY + maxId; ++idY) {  // y
    const auto centerIdX = getKernelCenter(gridMap, index, 0, maxId);
    for (auto idX = centerIdX - maxId; idX <= centerIdX + maxId; ++idX) {  // x
      grid_map::Index tempIndex = index + grid_map::Index(idX, idY);
      gradientYArray[maxId + idX] += kernelD1_[maxId + idY] * H(tempIndex.x(), tempIndex.y());
    }
  }

  for (auto dim = 0U; dim < 2U; ++dim) {
    auto tempIndex = index;
    const auto centerId = getKernelCenter(gridMap, index, dim, maxId);
    for (auto id = centerId - maxId; id <= centerId + maxId; ++id) {  // x or y
      tempIndex(dim) = index(dim) + id;
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

  // Normalize.
  gradient *= oneDivRes_;
  curvature *= oneDivResSquared_;
}

int GridMapDerivative::getKernelCenter(const grid_map::GridMap& gridMap, const grid_map::Index& centerIndex, unsigned int dim,
                                       int maxKernelId) const {
  constexpr int minId = 0;
  const int maxId = gridMap.getSize()(dim) - 1;
  return -std::min(centerIndex(dim) - maxKernelId, minId) - std::max(centerIndex(dim) + maxKernelId - maxId, 0);
}
}  // namespace derivative
}  // namespace grid_map
