//
// Created by rgrandia on 10.07.20.
//

#include "signed_distance_field/SignedDistance2d.h"

#include "signed_distance_field/PixelBorderDistance.h"

namespace signed_distance_field {

namespace internal {
struct DistanceLowerBound {
  float v;      // origin of bounding function
  float f;      // functional offset at the origin
  float z_lhs;  // lhs of interval where this bound holds
  float z_rhs;  // rhs of interval where this lower bound holds
};

std::vector<DistanceLowerBound>::iterator fillLowerBounds(const Eigen::Ref<Eigen::VectorXf>& squareDistance1d,
                                                          std::vector<DistanceLowerBound>& lowerBounds) {
  const auto n = squareDistance1d.size();
  const auto nFloat = static_cast<float>(n);

  // Initialize
  auto rhsBoundIt = lowerBounds.begin();
  *rhsBoundIt = DistanceLowerBound{0.0F, squareDistance1d[0], -INF, INF};
  auto firstBoundIt = lowerBounds.begin();

  // Compute bounds to the right of minimum
  float qFloat = 1.0F;
  for (Eigen::Index q = 1; q < n; ++q) {
    // Storing this by value gives better performance (removed indirection?)
    const float fq = squareDistance1d[q];

    float s = equidistancePoint(qFloat, fq, rhsBoundIt->v, rhsBoundIt->f);
    if (s < nFloat) {  // Can ignore the lower bound derived from this point if it is only active outsize of [0, n]
      // Search backwards in bounds until s is within [z_lhs, z_rhs]
      while (s < rhsBoundIt->z_lhs) {
        --rhsBoundIt;
        s = equidistancePoint(qFloat, fq, rhsBoundIt->v, rhsBoundIt->f);
      }
      if (s >= 0.0F) {          // Intersection is within [0, n]. Adjust current lowerbound and insert the new one after
        rhsBoundIt->z_rhs = s;  // Update the bound that comes before
        ++rhsBoundIt;           // insert new bound after.
        *rhsBoundIt = DistanceLowerBound{qFloat, fq, s, INF};  // Valid from s till infinity
      } else {  // Intersection is outside [0, n]. This means that the new bound dominates all previous bounds
        *rhsBoundIt = DistanceLowerBound{qFloat, fq, -INF, INF};
        firstBoundIt = rhsBoundIt;  // No need to keep other bounds, so update the first bound iterator to this one.
      }
    }

    // Increment to follow loop counter as float
    qFloat += 1.0F;
  }

  return firstBoundIt;
}

void extractDistances(Eigen::Ref<Eigen::VectorXf> squareDistance1d, const std::vector<DistanceLowerBound>& lowerBounds,
                      std::vector<DistanceLowerBound>::const_iterator lowerBoundIt) {
  const auto n = squareDistance1d.size();

  // Store active bound by value to remove indirection
  auto lastz = lowerBoundIt->z_rhs;

  float qFloat = 0.0F;
  for (Eigen::Index q = 0; q < n; ++q) {
    // Find the new active lower bound if q no longer belongs to current interval
    if (qFloat > lastz) {
      do {
        ++lowerBoundIt;
      } while (lowerBoundIt->z_rhs < qFloat);
      lastz = lowerBoundIt->z_rhs;
    }

    squareDistance1d[q] = squarePixelBorderDistance(qFloat, lowerBoundIt->v, lowerBoundIt->f);
    qFloat += 1.0F;
  }
}

/**
 * 1D Euclidean Distance Transform based on: http://cs.brown.edu/people/pfelzens/dt/
 * Adapted to work on Eigen objects directly
 * Optimized computation of s
 */
inline void squaredDistanceTransform_1d_inplace(Eigen::Ref<Eigen::VectorXf> squareDistance1d,
                                                std::vector<DistanceLowerBound>& lowerBounds) {
  assert(lowerBounds.size() == squareDistance1d.size());

  // If all distances are zero, then result remains all zeros
  if ((squareDistance1d.array() > 0.0F).any()) {
    auto startIt = fillLowerBounds(squareDistance1d, lowerBounds);
    extractDistances(squareDistance1d, lowerBounds, startIt);
  }
}

void squaredDistanceTransform_2d_columnwiseInplace(grid_map::Matrix& squareDistance) {
  const size_t n = squareDistance.rows();
  const size_t m = squareDistance.cols();
  std::vector<DistanceLowerBound> lowerBounds(n);

  for (size_t i = 0; i < m; i++) {
    squaredDistanceTransform_1d_inplace(squareDistance.col(i), lowerBounds);
  }
}

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
      const auto diff = (height - elevation) / resolution;
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
      const auto diff = (height - elevation) / resolution;
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

grid_map::Matrix signedDistanceAtHeight(const grid_map::Matrix& elevationMap, float height, float resolution, float minHeight,
                                        float maxHeight) {
  const bool allPixelsAreObstacles = height < minHeight;
  const bool allPixelsAreFreeSpace = height > maxHeight;

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
  if (hasObstacles) {
    bool hasFreeSpace = obstacleCount < occupancyGrid.size();
    if (hasFreeSpace) {
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
