//
// Created by rgrandia on 12.10.21.
//

#pragma once

#include <ocs2_switched_model_interface/core/SwitchedModel.h>

#include <convex_plane_decomposition/PlanarRegion.h>
#include <convex_plane_decomposition/PolygonTypes.h>

namespace switched_model {

double distanceCost(const vector3_t& query, const vector3_t& terrainPoint);

double distanceCostLowerbound(double distanceSquared);

double singleSidedSquaredDistance(double value, double min, double max);

double squaredDistanceToBoundingBox(const vector3_t& positionInWorld, const convex_plane_decomposition::PlanarRegion& planarRegion);

const convex_plane_decomposition::CgalPolygonWithHoles2d* findInsetContainingThePoint(
    const convex_plane_decomposition::CgalPoint2d& point, const std::vector<convex_plane_decomposition::CgalPolygonWithHoles2d>& insets);

std::pair<double, convex_plane_decomposition::CgalPoint2d> squaredDistanceToBoundary(
    const vector3_t& positionInWorld, const convex_plane_decomposition::PlanarRegion& planarRegion);

std::pair<const convex_plane_decomposition::PlanarRegion*, convex_plane_decomposition::CgalPoint2d> getPlanarRegionAtPositionInWorld(
    const vector3_t& positionInWorld, const std::vector<convex_plane_decomposition::PlanarRegion>& planarRegions);

}  // namespace switched_model