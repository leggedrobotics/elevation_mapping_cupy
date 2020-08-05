//
// Created by rgrandia on 10.07.20.
//

#include "signed_distance_field/SignedDistance2d.h"

namespace signed_distance_field {

namespace internal {
template <typename T>
T square(T x) {
  return x * x;
}

/**
 * 1D Euclidean Distance Transform based on: http://cs.brown.edu/people/pfelzens/dt/
 * Adapted to work on Eigen objects directly
 * Optimized computation of s
 */
void squaredDistanceTransform_1d_inplace(Eigen::Ref<Eigen::VectorXf> squareDistance1d, Eigen::Ref<Eigen::VectorXf> d,
                                         Eigen::Ref<Eigen::VectorXf> z, std::vector<int>& v) {
  const int n = squareDistance1d.size();
  const auto& f = squareDistance1d;
  assert(d.size() == n);
  assert(z.size() == n + 1);
  assert(v.size() == n);

  // Initialize
  int k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;

  // Compute bounds
  for (int q = 1; q < n; q++) {
    auto factor1 = static_cast<float>(q - v[k]);
    auto factor2 = static_cast<float>(q + v[k]);
    float s = 0.5F * ((f[q] - f[v[k]]) / factor1 + factor2);
    while (s <= z[k]) {
      k--;
      factor1 = static_cast<float>(q - v[k]);
      factor2 = static_cast<float>(q + v[k]);
      s = 0.5F * ((f[q] - f[v[k]]) / factor1 + factor2);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }

  // Collect results
  k = 0;
  for (int q = 0; q < n; q++) {
    while (z[k + 1] < static_cast<float>(q)) {
      k++;
    }
    d[q] = square(q - v[k]) + f[v[k]];
  }

  // Write distance result back in place
  squareDistance1d = d;
}

void squaredDistanceTransform_2d_columnwiseInplace(grid_map::Matrix& squareDistance, const std::vector<bool>* skipLine = nullptr) {
  const size_t n = squareDistance.rows();
  const size_t m = squareDistance.cols();
  Eigen::VectorXf workvector(2 * n + 1);
  std::vector<int> intWorkvector(n);

  for (size_t i = 0; i < m; i++) {
    if (skipLine == nullptr || !(*skipLine)[i]) {
      squaredDistanceTransform_1d_inplace(squareDistance.col(i), workvector.segment(0, n), workvector.segment(n, n + 1), intWorkvector);
    }
  }
}

// SquareDistance must be initialized with 0.0 for elements and INF for non-elements
void computePixelDistance2d(grid_map::Matrix& squareDistance, const SparsityInfo& sparsityInfo) {
  // Start with the dimension that maximizes the number of points being skips
  if (sparsityInfo.numColsHomogeneous * squareDistance.rows() > sparsityInfo.numRowsHomogeneous * squareDistance.cols()) {
    // Process columns
    squaredDistanceTransform_2d_columnwiseInplace(squareDistance, &sparsityInfo.colsIsHomogenous);

    // Process rows
    squareDistance.transposeInPlace();
    squaredDistanceTransform_2d_columnwiseInplace(squareDistance);
    squareDistance.transposeInPlace();
  } else {
    // Process rows
    squareDistance.transposeInPlace();
    squaredDistanceTransform_2d_columnwiseInplace(squareDistance, &sparsityInfo.rowsIsHomogenous);
    squareDistance.transposeInPlace();

    // Process columns
    squaredDistanceTransform_2d_columnwiseInplace(squareDistance);
  }

  // Convert square distance to absolute distance
  squareDistance = squareDistance.cwiseSqrt();
}
}  // namespace internal

grid_map::Matrix computeSignedDistanceAtHeight(const grid_map::Matrix& elevationMap, float height, float resolution) {
  const auto sparsityInfo = signed_distance_field::collectSparsityInfo(elevationMap, height);

  if (sparsityInfo.hasObstacles) {
    if (sparsityInfo.hasFreeSpace) {
      // Both obstacles and free space -> compute planar signed distance
      // Compute pixel distance to obstacles
      grid_map::Matrix sdfObstacle = elevationMap.unaryExpr([=](float val) { return (val >= height) ? 0.0F : INF; });
      internal::computePixelDistance2d(sdfObstacle, sparsityInfo);

      // Compute pixel distance to obstacle free space
      grid_map::Matrix sdfObstacleFree = elevationMap.unaryExpr([=](float val) { return (val < height) ? 0.0F : INF; });
      internal::computePixelDistance2d(sdfObstacleFree, sparsityInfo);

      grid_map::Matrix sdf2d = resolution * (sdfObstacle - sdfObstacleFree);
      return sdf2d;
    } else {
      // Only obstacles -> distance is zero everywhere
      return grid_map::Matrix::Zero(elevationMap.rows(), elevationMap.cols());
    }
  } else {
    // No obstacles -> planar distance is infinite
    return grid_map::Matrix::Constant(elevationMap.rows(), elevationMap.cols(), INF);
  }
}

SparsityInfo collectSparsityInfo(const grid_map::Matrix& elevationMap, float height) {
  const size_t n = elevationMap.rows();
  const size_t m = elevationMap.cols();

  SparsityInfo sparsityInfo;
  // Assume false and set true if detected otherwise
  sparsityInfo.hasObstacles = false;
  sparsityInfo.hasFreeSpace = false;

  // Assume true and set false if detected otherwise
  sparsityInfo.rowsIsHomogenous.resize(elevationMap.rows(), true);
  sparsityInfo.colsIsHomogenous.resize(elevationMap.cols(), true);

  Eigen::Matrix<bool, 1, -1> firstColIsObstacle = elevationMap.col(0).array() >= height;

  // Loop according to Eigen column major storage order.
  for (size_t j = 0; j < m; ++j) {
    bool firstRowIsObstacle = elevationMap(0, j) >= height;
    bool thisColIsHomogeneous = true;
    for (size_t i = 0; i < n; ++i) {
      float currentValue = elevationMap(i, j);
      if (currentValue >= height) {  // current location contains obstacle
        sparsityInfo.hasObstacles = true;
        if (!firstRowIsObstacle) {
          thisColIsHomogeneous = false;
        }
        if (!firstColIsObstacle(i)) {
          sparsityInfo.rowsIsHomogenous[i] = false;
        }
      } else {  // current location is free
        sparsityInfo.hasFreeSpace = true;
        if (firstRowIsObstacle) {
          thisColIsHomogeneous = false;
        }
        if (firstColIsObstacle(i)) {
          sparsityInfo.rowsIsHomogenous[i] = false;
        }
      }
    }
    sparsityInfo.colsIsHomogenous[j] = thisColIsHomogeneous;
  }

  sparsityInfo.numRowsHomogeneous = std::count(sparsityInfo.rowsIsHomogenous.begin(), sparsityInfo.rowsIsHomogenous.end(), true);
  sparsityInfo.numColsHomogeneous = std::count(sparsityInfo.colsIsHomogenous.begin(), sparsityInfo.colsIsHomogenous.end(), true);

  return sparsityInfo;
}

}  // namespace signed_distance_field
