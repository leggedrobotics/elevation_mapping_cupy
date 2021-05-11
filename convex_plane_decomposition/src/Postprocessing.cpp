#include "convex_plane_decomposition/Postprocessing.h"

namespace convex_plane_decomposition {

Postprocessing::Postprocessing(const PostprocessingParameters& parameters) : parameters_(parameters) {}

void Postprocessing::postprocess(PlanarTerrain& planarTerrain, const std::string& layer) const {
  addHeightOffset(planarTerrain, layer);
}

void Postprocessing::addHeightOffset(PlanarTerrain& planarTerrain, const std::string& layer) const {
  if (parameters_.extracted_planes_height_offset != 0.0) {
    // Lift planar regions
    for (auto& planarRegion : planarTerrain.planarRegions) {
      planarRegion.planeParameters.positionInWorld.z() += parameters_.extracted_planes_height_offset;
    }

    // lift elevation layer
    planarTerrain.gridMap.get(layer).array() += parameters_.extracted_planes_height_offset;
  }
}

}  // namespace convex_plane_decomposition
