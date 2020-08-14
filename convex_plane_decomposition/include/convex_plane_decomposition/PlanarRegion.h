//
// Created by rgrandia on 10.06.20.
//

#pragma once

#include <Eigen/Core>

#include <grid_map_core/GridMap.hpp>

#include <ocs2_switched_model_interface/terrain/TerrainPlane.h>

#include "PolygonTypes.h"

namespace convex_plane_decomposition {

using switched_model::TerrainPlane;

struct BoundaryWithInset {
  /// Boundary of the planar region.
  CgalPolygonWithHoles2d boundary;

  /// Encodes an inward offset to the boundary.
  std::vector<CgalPolygonWithHoles2d> insets;
};

struct PlanarRegion {
  /// All 2d points are in the terrain frame
  BoundaryWithInset boundaryWithInset;

  /// 2D bounding box in terrain frame containing all the boundary points
  CgalBbox2d bbox2d;

  /// 3D parameters of the plane
  TerrainPlane planeParameters;
};

struct PlanarTerrain {
  std::vector<PlanarRegion> planarRegions;
  grid_map::GridMap gridMap;
};

}  // namespace convex_plane_decomposition
