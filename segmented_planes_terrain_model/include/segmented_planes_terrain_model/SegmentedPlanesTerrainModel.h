//
// Created by rgrandia on 23.06.20.
//

#pragma once

#include <convex_plane_decomposition/PlanarRegion.h>
#include "ocs2_switched_model_interface/terrain/TerrainModel.h"

namespace switched_model {

class SegmentedPlanesTerrainModel : switched_model::TerrainModel {
 public:
  SegmentedPlanesTerrainModel(convex_plane_decomposition::PlanarTerrain planarTerrain);

  TerrainPlane getLocalTerrainAtPositionInWorld(const vector3_t& positionInWorld) const override;

  ConvexTerrain getConvexTerrainAtPositionInWorld(const vector3_t& positionInWorld) const override;

  const convex_plane_decomposition::PlanarTerrain& PlanarTerrain() const { return planarTerrain_; }

 private:
  convex_plane_decomposition::PlanarTerrain planarTerrain_;
};

std::pair<const convex_plane_decomposition::PlanarRegion*, convex_plane_decomposition::CgalPoint2d> getPlanarRegionAtPositionInWorld(
    const vector3_t& positionInWorld,  const convex_plane_decomposition::PlanarTerrain& planarTerrain);

}  // namespace switched_model
