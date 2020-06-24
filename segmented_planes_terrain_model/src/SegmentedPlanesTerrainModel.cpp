//
// Created by rgrandia on 23.06.20.
//

#include "segmented_planes_terrain_model/SegmentedPlanesTerrainModel.h"

#include <algorithm>

#include <convex_plane_decomposition/ConvexRegionGrowing.h>
#include <convex_plane_decomposition/GeometryUtils.h>

namespace switched_model {

SegmentedPlanesTerrainModel::SegmentedPlanesTerrainModel(convex_plane_decomposition::PlanarTerrain planarTerrain)
    : planarTerrain_(std::move(planarTerrain)) {}

TerrainPlane SegmentedPlanesTerrainModel::getLocalTerrainAtPositionInWorld(const vector3_t& positionInWorld) const {
  const auto regionAndSeedPoint = getPlanarRegionAtPositionInWorld(positionInWorld, planarTerrain_);
  const auto& region = *regionAndSeedPoint.first;
  const auto& seedpoint = regionAndSeedPoint.second;
  const auto& seedpointInWorldFrame = positionInWorldFrameFromPositionInTerrain({seedpoint.x(), seedpoint.y(), 0.0}, region.planeParameters);
  return TerrainPlane{seedpointInWorldFrame, region.planeParameters.orientationWorldToTerrain};
}

ConvexTerrain SegmentedPlanesTerrainModel::getConvexTerrainAtPositionInWorld(const vector3_t& positionInWorld) const {
    const auto regionAndSeedPoint = getPlanarRegionAtPositionInWorld(positionInWorld, planarTerrain_);
    const auto& region = *regionAndSeedPoint.first;
    const auto& seedpoint = regionAndSeedPoint.second;
    const auto& seedpointInWorldFrame = positionInWorldFrameFromPositionInTerrain({seedpoint.x(), seedpoint.y(), 0.0}, region.planeParameters);

    // Convert boundary and seedpoint to terrain frame
    const int numberOfVertices = 16;  // Multiple of 4 is nice for symmetry.
    const double growthFactor = 1.05;
    const auto convexRegion = convex_plane_decomposition::growConvexPolygonInsideShape(region.boundaryWithInset.boundary, seedpoint,
    numberOfVertices, growthFactor);

    // Return convex region with origin at the seedpoint
    ConvexTerrain convexTerrain;
    convexTerrain.plane = {seedpointInWorldFrame, region.planeParameters.orientationWorldToTerrain}; // Origin is at the seedpoint
    convexTerrain.boundary.reserve(convexRegion.size());
    for (const auto& point : convexRegion) {
      convexTerrain.boundary.emplace_back(point.x() - seedpoint.x(), point.y() - seedpoint.y()); // Shift points to new origin
    }
    return convexTerrain;
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

double squaredDistanceToBoundingBox(const vector3_t& positionInWorld, const convex_plane_decomposition::PlanarRegion& planarRegion) {
  const auto& positionInTerrainFrame = positionInTerrainFrameFromPositionInWorld(positionInWorld, planarRegion.planeParameters);
  double dxdx = singleSidedSquaredDistance(positionInTerrainFrame.x(), planarRegion.bbox2d.xmin(), planarRegion.bbox2d.xmax());
  double dydy = singleSidedSquaredDistance(positionInTerrainFrame.y(), planarRegion.bbox2d.ymin(), planarRegion.bbox2d.ymax());
  double dzdz = positionInTerrainFrame.z()* positionInTerrainFrame.z();
  return dxdx + dydy + dzdz;
}

const convex_plane_decomposition::CgalPolygonWithHoles2d* findInsetContainingThePoint(const convex_plane_decomposition::CgalPoint2d& point, const std::vector<convex_plane_decomposition::CgalPolygonWithHoles2d>& insets){
  for (const auto& inset : insets) {
    if (convex_plane_decomposition::isInside(point, inset.outer_boundary())) {
      return &inset;
    }
  }
  return nullptr;
}

std::pair<double, convex_plane_decomposition::CgalPoint2d> squaredDistanceToBoundary(const vector3_t& positionInWorld,  const convex_plane_decomposition::PlanarRegion& planarRegion) {
  const auto& positionInTerrainFrame = positionInTerrainFrameFromPositionInWorld(positionInWorld, planarRegion.planeParameters);
  const double dzdz = positionInTerrainFrame.z()* positionInTerrainFrame.z();
  const convex_plane_decomposition::CgalPoint2d queryProjectedToPlane {positionInTerrainFrame.x(), positionInTerrainFrame.y()};

  // First search if the projected point is inside any of the insets.
  const auto insetPtrContainingPoint = findInsetContainingThePoint(queryProjectedToPlane, planarRegion.boundaryWithInset.insets);

  // Compute the projection
  convex_plane_decomposition::CgalPoint2d projectedPoint;
  if (insetPtrContainingPoint == nullptr) {
    // Not inside any of the insets. Need to look for the closest one. The projection will be to the boundary
    double minDistSquared = std::numeric_limits<double>::max();
    for (const auto& inset : planarRegion.boundaryWithInset.insets) {
      const auto closestPoint = convex_plane_decomposition::projectToClosestPoint(queryProjectedToPlane, inset.outer_boundary());
      double distSquare = convex_plane_decomposition::squaredDistance(closestPoint, queryProjectedToPlane);
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
      if (convex_plane_decomposition::isInside(queryProjectedToPlane, hole)) {
        projectedPoint = convex_plane_decomposition::projectToClosestPoint(queryProjectedToPlane, hole);
        break; // No need to search other holes. Holes are not overlapping
      }
    }
  }

  return {dzdz + convex_plane_decomposition::squaredDistance(projectedPoint, queryProjectedToPlane), projectedPoint};
}

std::pair<const convex_plane_decomposition::PlanarRegion*, convex_plane_decomposition::CgalPoint2d>
getPlanarRegionAtPositionInWorld(const vector3_t& positionInWorld,  const convex_plane_decomposition::PlanarTerrain& planarTerrain) {
  // Compute distance to bounding boxes
  std::vector<std::pair<const convex_plane_decomposition::PlanarRegion*, double>> regionsAndBboxSquareDistances;
  regionsAndBboxSquareDistances.reserve(planarTerrain.size());
  for (const auto& planarRegion : planarTerrain) {
    regionsAndBboxSquareDistances.emplace_back(&planarRegion, squaredDistanceToBoundingBox(positionInWorld, planarRegion));
  }

  // Sort regions close to far
  std::sort(regionsAndBboxSquareDistances.begin(), regionsAndBboxSquareDistances.end(),
            [](const std::pair<const convex_plane_decomposition::PlanarRegion*, double>& lhs,
               const std::pair<const convex_plane_decomposition::PlanarRegion*, double>& rhs) {
                return lhs.second < rhs.second;
            });

  // Look for closest planar region. Use bbox as lower bound to stop searching.
  double minDistSquared = std::numeric_limits<double>::max();
  std::pair<const convex_plane_decomposition::PlanarRegion*, convex_plane_decomposition::CgalPoint2d> closestRegionAndProjection;
  for(const auto& regionAndBboxSquareDistance : regionsAndBboxSquareDistances) {
    if (regionAndBboxSquareDistance.second > minDistSquared) {
      break; // regions are sorted. Can exit on the first lower bound being larger than running minimum
    }

    const auto distanceSqrAndProjection = squaredDistanceToBoundary(positionInWorld, *regionAndBboxSquareDistance.first);
    if (distanceSqrAndProjection.first < minDistSquared) {
      minDistSquared = distanceSqrAndProjection.first;
      closestRegionAndProjection = {regionAndBboxSquareDistance.first, distanceSqrAndProjection.second};
    }
  }

  return closestRegionAndProjection;
}

}  // namespace switched_model
