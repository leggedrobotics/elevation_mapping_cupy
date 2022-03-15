//
// Created by rgrandia on 12.10.21.
//

#include "segmented_planes_terrain_model/SegmentedPlaneProjection.h"

#include <convex_plane_decomposition/GeometryUtils.h>

namespace switched_model {

// alias to avoid long namespace
namespace cpd = convex_plane_decomposition;

double distanceCost(const vector3_t& query, const vector3_t& terrainPoint) {
  const double dx = query.x() - terrainPoint.x();
  const double dy = query.y() - terrainPoint.y();
  const double dz = query.z() - terrainPoint.z();
  return dx * dx + dy * dy + dz * dz;
}

double distanceCostLowerbound(double distanceSquared) {
  return distanceSquared;
}

double intervalSquareDistance(double value, double min, double max) {
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

double squaredDistanceToBoundingBox(const cpd::CgalPoint2d& point, const cpd::CgalBbox2d& boundingBox) {
  double dxdx = intervalSquareDistance(point.x(), boundingBox.xmin(), boundingBox.xmax());
  double dydy = intervalSquareDistance(point.y(), boundingBox.ymin(), boundingBox.ymax());
  return dxdx + dydy;
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

cpd::CgalPoint2d projectToPlanarRegion(const cpd::CgalPoint2d& queryProjectedToPlane, const cpd::PlanarRegion& planarRegion) {
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

  return projectedPoint;
}

std::vector<RegionSortingInfo> sortWithBoundingBoxes(const vector3_t& positionInWorld,
                                                     const std::vector<convex_plane_decomposition::PlanarRegion>& planarRegions) {
  // Compute distance to bounding boxes
  std::vector<RegionSortingInfo> regionsAndBboxSquareDistances;
  regionsAndBboxSquareDistances.reserve(planarRegions.size());
  for (const auto& planarRegion : planarRegions) {
    const auto& positionInTerrainFrame = planarRegion.transformPlaneToWorld.inverse() * positionInWorld;
    const double dzdz = positionInTerrainFrame.z() * positionInTerrainFrame.z();

    RegionSortingInfo regionSortingInfo;
    regionSortingInfo.regionPtr = &planarRegion;
    regionSortingInfo.positionInTerrainFrame = {positionInTerrainFrame.x(), positionInTerrainFrame.y()};
    regionSortingInfo.boundingBoxSquareDistance =
        squaredDistanceToBoundingBox(regionSortingInfo.positionInTerrainFrame, planarRegion.bbox2d) + dzdz;

    regionsAndBboxSquareDistances.push_back(regionSortingInfo);
  }

  // Sort regions close to far
  std::sort(regionsAndBboxSquareDistances.begin(), regionsAndBboxSquareDistances.end(),
            [](const RegionSortingInfo& lhs, const RegionSortingInfo& rhs) {
              return lhs.boundingBoxSquareDistance < rhs.boundingBoxSquareDistance;
            });

  return regionsAndBboxSquareDistances;
}

std::pair<const cpd::PlanarRegion*, cpd::CgalPoint2d> getPlanarRegionAtPositionInWorld(
    const vector3_t& positionInWorld, const std::vector<cpd::PlanarRegion>& planarRegions,
    std::function<scalar_t(const vector3_t&)> penaltyFunction) {
  const auto sortedRegions = sortWithBoundingBoxes(positionInWorld, planarRegions);

  // Look for closest planar region.
  scalar_t minCost = std::numeric_limits<scalar_t>::max();
  std::pair<const cpd::PlanarRegion*, cpd::CgalPoint2d> closestRegionAndProjection;
  for (const auto& regionInfo : sortedRegions) {
    // Skip based on lower bound
    if (distanceCostLowerbound(regionInfo.boundingBoxSquareDistance) > minCost) {
      continue;
    }

    // Shorthand
    const auto* regionPtr = regionInfo.regionPtr;

    // Project onto planar region
    const auto projectedPointInTerrainFrame = projectToPlanarRegion(regionInfo.positionInTerrainFrame, *regionPtr);

    // Express projected point in World frame
    const auto projectionInWorldFrame =
        convex_plane_decomposition::positionInWorldFrameFromPosition2dInPlane(projectedPointInTerrainFrame, regionPtr->transformPlaneToWorld);

    const scalar_t cost = distanceCost(positionInWorld, projectionInWorldFrame) + penaltyFunction(projectionInWorldFrame);
    if (cost < minCost) {
      minCost = cost;
      closestRegionAndProjection = {regionPtr, projectedPointInTerrainFrame};
    }
  }

  return closestRegionAndProjection;
}

}  // namespace switched_model