//
// Created by rgrandia on 23.06.20.
//

#include "segmented_planes_terrain_model/SegmentedPlanesTerrainModel.h"

#include <algorithm>

#include <convex_plane_decomposition/ConvexRegionGrowing.h>
#include <signed_distance_field/GridmapSignedDistanceField.h>
#include <grid_map_filters_rsl/lookup.hpp>

#include "segmented_planes_terrain_model/SegmentedPlaneProjection.h"

namespace switched_model {

namespace {
const std::string elevationLayerName = "elevation";
}  // namespace

SegmentedPlanesTerrainModel::SegmentedPlanesTerrainModel(convex_plane_decomposition::PlanarTerrain planarTerrain)
    : planarTerrain_(std::move(planarTerrain)),
      signedDistanceField_(nullptr),
      elevationData_(&planarTerrain_.gridMap.get(elevationLayerName)) {}

TerrainPlane SegmentedPlanesTerrainModel::getLocalTerrainAtPositionInWorldAlongGravity(const vector3_t& positionInWorld) const {
  const auto regionAndSeedPoint = getPlanarRegionAtPositionInWorld(positionInWorld, planarTerrain_.planarRegions);
  const auto& region = *regionAndSeedPoint.first;
  const auto& seedpoint = regionAndSeedPoint.second;
  const auto& seedpointInWorldFrame = positionInWorldFrameFromPosition2dInTerrain(seedpoint, region.planeParameters);
  return TerrainPlane{seedpointInWorldFrame, region.planeParameters.orientationWorldToTerrain};
}

ConvexTerrain SegmentedPlanesTerrainModel::getConvexTerrainAtPositionInWorld(const vector3_t& positionInWorld) const {
  const auto regionAndSeedPoint = getPlanarRegionAtPositionInWorld(positionInWorld, planarTerrain_.planarRegions);
  const auto& region = *regionAndSeedPoint.first;
  const auto& seedpoint = regionAndSeedPoint.second;
  const auto& seedpointInWorldFrame = positionInWorldFrameFromPosition2dInTerrain(seedpoint, region.planeParameters);

  // Convert boundary and seedpoint to terrain frame
  const int numberOfVertices = 16;  // Multiple of 4 is nice for symmetry.
  const double growthFactor = 1.05;
  const auto convexRegion = convex_plane_decomposition::growConvexPolygonInsideShape(region.boundaryWithInset.boundary, seedpoint,
                                                                                     numberOfVertices, growthFactor);

  // Return convex region with origin at the seedpoint
  ConvexTerrain convexTerrain;
  convexTerrain.plane = {seedpointInWorldFrame, region.planeParameters.orientationWorldToTerrain};  // Origin is at the seedpoint
  convexTerrain.boundary.reserve(convexRegion.size());
  for (const auto& point : convexRegion) {
    convexTerrain.boundary.emplace_back(point.x() - seedpoint.x(), point.y() - seedpoint.y());  // Shift points to new origin
  }
  return convexTerrain;
}

void SegmentedPlanesTerrainModel::createSignedDistanceBetween(const Eigen::Vector3d& minCoordinates,
                                                              const Eigen::Vector3d& maxCoordinates) {
  // Compute coordinates of submap
  const auto minXY = planarTerrain_.gridMap.getClosestPositionInMap({minCoordinates.x(), minCoordinates.y()});
  const auto maxXY = planarTerrain_.gridMap.getClosestPositionInMap({maxCoordinates.x(), maxCoordinates.y()});
  const auto centerXY = 0.5 * (minXY + maxXY);
  const auto lengths = maxXY - minXY;

  bool success = true;
  grid_map::GridMap subMap = planarTerrain_.gridMap.getSubmap(centerXY, lengths, success);
  if (success) {
    signedDistanceField_ = std::make_unique<signed_distance_field::GridmapSignedDistanceField>(subMap, elevationLayerName,
                                                                                               minCoordinates.z(), maxCoordinates.z());
  } else {
    std::cerr << "[SegmentedPlanesTerrainModel] Failed to get subMap" << std::endl;
  }
}

vector3_t SegmentedPlanesTerrainModel::getHighestObstacleAlongLine(const vector3_t& position1InWorld,
                                                                   const vector3_t& position2InWorld) const {
  const auto result = grid_map::lookup::maxValueBetweenLocations(
      {position1InWorld.x(), position1InWorld.y()}, {position2InWorld.x(), position2InWorld.y()}, planarTerrain_.gridMap, *elevationData_);
  if (result.isValid) {
    return {result.position.x(), result.position.y(), result.value};
  } else {
    // return highest query point if the map didn't work.
    if (position1InWorld.z() > position2InWorld.z()) {
      return position1InWorld;
    } else {
      return position2InWorld;
    }
  }
}

}  // namespace switched_model
