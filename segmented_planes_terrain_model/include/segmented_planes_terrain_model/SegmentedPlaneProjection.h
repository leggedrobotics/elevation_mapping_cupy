//
// Created by rgrandia on 12.10.21.
//

#pragma once

#include <ocs2_switched_model_interface/core/SwitchedModel.h>
#include <ocs2_switched_model_interface/terrain/TerrainPlane.h>

#include <convex_plane_decomposition/PlanarRegion.h>
#include <convex_plane_decomposition/PolygonTypes.h>

namespace switched_model {

double distanceCost(const vector3_t& query, const vector3_t& terrainPoint);

double distanceCostLowerbound(double distanceSquared);

double intervalSquareDistance(double value, double min, double max);

double squaredDistanceToBoundingBox(const convex_plane_decomposition::CgalPoint2d& point,
                                    const convex_plane_decomposition::CgalBbox2d& boundingBox);

/** Converts a 3D position in world frame to a 2D position in the terrain frame. */
inline convex_plane_decomposition::CgalPoint2d position2dInTerrainFrameFromPositionInWorld(const vector3_t& positionWorld,
                                                                                           const TerrainPlane& terrainPlane) {
  const auto& positionInTerrainFrame = positionInTerrainFrameFromPositionInWorld(positionWorld, terrainPlane);
  return {positionInTerrainFrame.x(), positionInTerrainFrame.y()};
}

/** Converts a 2D position in terrain frame to a 3D position in the world frame. */
inline vector3_t positionInWorldFrameFromPosition2dInTerrain(const convex_plane_decomposition::CgalPoint2d& positionInTerrain,
                                                             const TerrainPlane& terrainPlane) {
  return positionInWorldFrameFromPositionInTerrain({positionInTerrain.x(), positionInTerrain.y(), 0.0}, terrainPlane);
}

const convex_plane_decomposition::CgalPolygonWithHoles2d* findInsetContainingThePoint(
    const convex_plane_decomposition::CgalPoint2d& point, const std::vector<convex_plane_decomposition::CgalPolygonWithHoles2d>& insets);

convex_plane_decomposition::CgalPoint2d projectToPlanarRegion(const convex_plane_decomposition::CgalPoint2d& queryProjectedToPlane,
                                                              const convex_plane_decomposition::PlanarRegion& planarRegion);

struct RegionSortingInfo {
  const convex_plane_decomposition::PlanarRegion* regionPtr;
  convex_plane_decomposition::CgalPoint2d positionInTerrainFrame;
  double boundingBoxSquareDistance;
};
std::vector<RegionSortingInfo> sortWithBoundingBoxes(const vector3_t& positionInWorld,
                                                     const std::vector<convex_plane_decomposition::PlanarRegion>& planarRegions);

std::pair<const convex_plane_decomposition::PlanarRegion*, convex_plane_decomposition::CgalPoint2d> getPlanarRegionAtPositionInWorld(
    const vector3_t& positionInWorld, const std::vector<convex_plane_decomposition::PlanarRegion>& planarRegions,
    std::function<scalar_t(const vector3_t&)> penaltyFunction);

}  // namespace switched_model