//
// Created by rgrandia on 10.07.20.
//

#include "signed_distance_field/SignedDistance2d.h"

#include "signed_distance_field/PixelBorderDistance.h"

namespace signed_distance_field {

namespace internal {
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
    float s = equidistancePoint(q, f[q], v[k], f[v[k]]);
    while (s <= z[k]) {
      k--;
      s = equidistancePoint(q, f[q], v[k], f[v[k]]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }

  // Collect results
  k = 0;
  for (int q = 0; q < n; q++) {
    auto qFloat = static_cast<float>(q);
    while (z[k + 1] < qFloat) {
      k++;
    }
    d[q] = squarePixelBorderDistance(qFloat, v[k], f[v[k]]);
  }

  // Write distance result back in place
  squareDistance1d = d;
}

void squaredDistanceTransform_2d_columnwiseInplace(grid_map::Matrix& squareDistance) {
  const size_t n = squareDistance.rows();
  const size_t m = squareDistance.cols();
  Eigen::VectorXf workvector(2 * n + 1);
  std::vector<int> intWorkvector(n);

  for (size_t i = 0; i < m; i++) {
    squaredDistanceTransform_1d_inplace(squareDistance.col(i), workvector.segment(0, n), workvector.segment(n, n + 1), intWorkvector);
  }
}

// SquareDistance must be initialized with 0.0 for elements and INF for non-elements
void computePixelDistance2d(grid_map::Matrix& squareDistance) {
  // Process columns
  squaredDistanceTransform_2d_columnwiseInplace(squareDistance);

  // Process rows
  squareDistance.transposeInPlace();
  squaredDistanceTransform_2d_columnwiseInplace(squareDistance);
  squareDistance.transposeInPlace();

  // Convert square distance to absolute distance
  squareDistance = squareDistance.cwiseSqrt();
}

// Initialize with square distance in height direction in pixel units if above the surface
grid_map::Matrix initializeObstacleDistance(const grid_map::Matrix& elevationMap, float height, float resolution) {
  return elevationMap.unaryExpr([=](float elevation) {
    if (height > elevation) {
      const auto diff = (height - elevation)  / resolution;
      return diff * diff;
    } else {
      return 0.0F;
    }
  });
}

// Initialize with square distance in height direction in pixel units if below the surface
grid_map::Matrix initializeObstacleFreeDistance(const grid_map::Matrix& elevationMap, float height, float resolution) {
  return elevationMap.unaryExpr([=](float elevation) {
    if (height < elevation) {
      const auto diff = (height - elevation)  / resolution;
      return diff * diff;
    } else {
      return 0.0F;
    }
  });
}

grid_map::Matrix pixelDistanceToFreeSpace(const grid_map::Matrix& elevationMap, float height, float resolution) {
  grid_map::Matrix sdfObstacleFree = internal::initializeObstacleFreeDistance(elevationMap, height, resolution);
  internal::computePixelDistance2d(sdfObstacleFree);
  return sdfObstacleFree;
}

grid_map::Matrix pixelDistanceToObstacle(const grid_map::Matrix& elevationMap, float height, float resolution) {
  grid_map::Matrix sdfObstacle = internal::initializeObstacleDistance(elevationMap, height, resolution);
  internal::computePixelDistance2d(sdfObstacle);
  return sdfObstacle;
}

}  // namespace internal

grid_map::Matrix signedDistanceAtHeight(const grid_map::Matrix& elevationMap, float height, float resolution) {
  auto obstacleCount = (elevationMap.array() > height).count();
  bool allPixelsAreObstacles = obstacleCount == elevationMap.size();
  bool allPixelsAreFreeSpace = obstacleCount == 0;

  if (allPixelsAreObstacles) {
    return -resolution * internal::pixelDistanceToFreeSpace(elevationMap, height, resolution);
  } else if (allPixelsAreFreeSpace) {
    return resolution * internal::pixelDistanceToObstacle(elevationMap, height, resolution);
  } else {  // This layer contains a mix of obstacles and free space
    return resolution * (internal::pixelDistanceToObstacle(elevationMap, height, resolution) -
                         internal::pixelDistanceToFreeSpace(elevationMap, height, resolution));
  }
}

grid_map::Matrix signedDistanceFromOccupancy(const Eigen::Matrix<bool, -1, -1>& occupancyGrid, float resolution) {
  auto obstacleCount = occupancyGrid.count();
  bool hasObstacles = obstacleCount > 0;
  if (hasObstacles){
    bool hasFreeSpace = obstacleCount < occupancyGrid.size();
    if (hasFreeSpace){
      // Compute pixel distance to obstacles
      grid_map::Matrix sdfObstacle = occupancyGrid.unaryExpr([=](bool val) { return (val) ? 0.0F : INF; });
      internal::computePixelDistance2d(sdfObstacle);

      // Compute pixel distance to obstacle free space
      grid_map::Matrix sdfObstacleFree = occupancyGrid.unaryExpr([=](bool val) { return (val) ? INF : 0.0F; });
      internal::computePixelDistance2d(sdfObstacleFree);

      grid_map::Matrix sdf2d = resolution * (sdfObstacle - sdfObstacleFree);
      return sdf2d;
    } else {
      // Only obstacles -> distance is minus infinity everywhere
      return grid_map::Matrix::Constant(occupancyGrid.rows(), occupancyGrid.cols(), -INF);
    }
  } else {
    // No obstacles -> planar distance is infinite
    return grid_map::Matrix::Constant(occupancyGrid.rows(), occupancyGrid.cols(), INF);
  }
}

}  // namespace signed_distance_field
