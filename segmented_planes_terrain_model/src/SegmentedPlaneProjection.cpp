//
// Created by rgrandia on 12.10.21.
//

#include "segmented_planes_terrain_model/SegmentedPlaneProjection.h"

#include <convex_plane_decomposition/GeometryUtils.h>

namespace switched_model {

// alias to avoid long namespace
namespace cpd = convex_plane_decomposition;

// TODO (rgrandia) : deadzone a parameter
const double zHeightDistanceDeadzone = 0.1;  // [m] This difference in z height is not counted

double distanceCost(const vector3_t& query, const vector3_t& terrainPoint) {
  const double dx = query.x() - terrainPoint.x();
  const double dy = query.y() - terrainPoint.y();
  const double dz = std::max(0.0, std::abs(query.z() - terrainPoint.z()) - zHeightDistanceDeadzone);
  return dx * dx + dy * dy + dz * dz;
}

double distanceCostLowerbound(double distanceSquared) {
  // cost = dx*dx + dy*dy + max(0.0, (|dz| - z0)).^2   with z0 >= 0
  // Need a lower bound for this cost derived from square distance and shift
  //
  // dz*dz - z0*z0   < max(0.0, (|dz| - z0)).^2
  // if |dz| > z0 ==>
  //    dz*dz - 2*|dz|*z0 + z0*z0 = (|dz| - z0).^2
  //    dz*dz - 2*z0*z0 + z0*z0   < (|dz| - z0).^2
  //    dz*dz - z0*z0   < (|dz| - z0).^2
  //
  // if |dz| < z0 ==>
  //    dz*dz - z0*z0  < 0.0  (true)
  return distanceSquared - zHeightDistanceDeadzone * zHeightDistanceDeadzone;
}

double singleSidedSquaredDistance(double value, double min, double max) {
  //   -    |    0     |    +
  //       min        max
  // Returns 0.0 if between min and max. Returns the distance to one boundary otherwise.
  if (value < min) {
    double diff = min - value;
    return diff * diff;
  } else if (value < max) {
    return 0.0;
  } else {
    double diff = max - value;
    return diff * diff;
  }
}

double squaredDistanceToBoundingBox(const vector3_t& positionInWorld, const cpd::PlanarRegion& planarRegion) {
  const auto& positionInTerrainFrame = positionInTerrainFrameFromPositionInWorld(positionInWorld, planarRegion.planeParameters);
  double dxdx = singleSidedSquaredDistance(positionInTerrainFrame.x(), planarRegion.bbox2d.xmin(), planarRegion.bbox2d.xmax());
  double dydy = singleSidedSquaredDistance(positionInTerrainFrame.y(), planarRegion.bbox2d.ymin(), planarRegion.bbox2d.ymax());
  double dzdz = positionInTerrainFrame.z() * positionInTerrainFrame.z();
  return dxdx + dydy + dzdz;
}

const cpd::CgalPolygonWithHoles2d* findInsetContainingThePoint(const cpd::CgalPoint2d& point,
                                                               const std::vector<cpd::CgalPolygonWithHoles2d>& insets) {
  for (const auto& inset : insets) {
    if (cpd::isInside(point, inset.outer_boundary())) {
      return &inset;
    }
  }
  return nullptr;
}

std::pair<double, cpd::CgalPoint2d> squaredDistanceToBoundary(const vector3_t& positionInWorld, const cpd::PlanarRegion& planarRegion) {
  const auto& positionInTerrainFrame = positionInTerrainFrameFromPositionInWorld(positionInWorld, planarRegion.planeParameters);
  const double dzdz = positionInTerrainFrame.z() * positionInTerrainFrame.z();
  const cpd::CgalPoint2d queryProjectedToPlane{positionInTerrainFrame.x(), positionInTerrainFrame.y()};

  // First search if the projected point is inside any of the insets.
  const auto* const insetPtrContainingPoint = findInsetContainingThePoint(queryProjectedToPlane, planarRegion.boundaryWithInset.insets);

  // Compute the projection
  cpd::CgalPoint2d projectedPoint;
  if (insetPtrContainingPoint == nullptr) {
    // Not inside any of the insets. Need to look for the closest one. The projection will be to the boundary
    double minDistSquared = std::numeric_limits<double>::max();
    for (const auto& inset : planarRegion.boundaryWithInset.insets) {
      const auto closestPoint = cpd::projectToClosestPoint(queryProjectedToPlane, inset.outer_boundary());
      double distSquare = cpd::squaredDistance(closestPoint, queryProjectedToPlane);
      if (distSquare < minDistSquared) {
        projectedPoint = closestPoint;
        minDistSquared = distSquare;
      }
    }
  } else {
    // Query point is inside and does not need projection...
    projectedPoint = queryProjectedToPlane;

    // ... unless it is inside a hole
    for (const auto& hole : insetPtrContainingPoint->holes()) {
      if (cpd::isInside(queryProjectedToPlane, hole)) {
        projectedPoint = cpd::projectToClosestPoint(queryProjectedToPlane, hole);
        break;  // No need to search other holes. Holes are not overlapping
      }
    }
  }

  return {dzdz + cpd::squaredDistance(projectedPoint, queryProjectedToPlane), projectedPoint};
}

std::pair<const cpd::PlanarRegion*, cpd::CgalPoint2d> getPlanarRegionAtPositionInWorld(
    const vector3_t& positionInWorld, const std::vector<cpd::PlanarRegion>& planarRegions) {
  // Compute distance to bounding boxes
  std::vector<std::pair<const cpd::PlanarRegion*, double>> regionsAndBboxSquareDistances;
  regionsAndBboxSquareDistances.reserve(planarRegions.size());
  for (const auto& planarRegion : planarRegions) {
    double squareDistance = squaredDistanceToBoundingBox(positionInWorld, planarRegion);
    regionsAndBboxSquareDistances.emplace_back(&planarRegion, distanceCostLowerbound(squareDistance));
  }

  // Sort regions close to far
  std::sort(regionsAndBboxSquareDistances.begin(), regionsAndBboxSquareDistances.end(),
            [](const std::pair<const cpd::PlanarRegion*, double>& lhs, const std::pair<const cpd::PlanarRegion*, double>& rhs) {
              return lhs.second < rhs.second;
            });

  // Look for closest planar region. Use bbox as lower bound to stop searching.
  double minDistSquared = std::numeric_limits<double>::max();
  std::pair<const cpd::PlanarRegion*, cpd::CgalPoint2d> closestRegionAndProjection;
  for (const auto& regionAndBboxSquareDistance : regionsAndBboxSquareDistances) {
    if (regionAndBboxSquareDistance.second > minDistSquared) {
      break;  // regions are sorted. Can exit on the first lower bound being larger than running minimum
    }

    const auto distanceSqrAndProjection = squaredDistanceToBoundary(positionInWorld, *regionAndBboxSquareDistance.first);
    const auto& projectionInWorldFrame =
        positionInWorldFrameFromPositionInTerrain({distanceSqrAndProjection.second.x(), distanceSqrAndProjection.second.y(), 0.0},
                                                  regionAndBboxSquareDistance.first->planeParameters);
    const double distCost = distanceCost(positionInWorld, projectionInWorldFrame);
    if (distCost < minDistSquared) {
      minDistSquared = distCost;
      closestRegionAndProjection = {regionAndBboxSquareDistance.first, distanceSqrAndProjection.second};
    }
  }

  return closestRegionAndProjection;
}

}  // namespace switched_model