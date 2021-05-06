#include "convex_plane_decomposition/Postprocessing.h"

namespace convex_plane_decomposition {

Postprocessing::Postprocessing(const PostprocessingParameters& parameters) : parameters_(parameters) {}

void Postprocessing::postprocess(PlanarTerrain& planarTerrain) const {
  addHeightOffset(planarTerrain);
}

void Postprocessing::addHeightOffset(PlanarTerrain& planarTerrain) const {
  if (parameters_.extracted_planes_height_offset != 0.0) {
    for (auto& planarRegion : planarTerrain.planarRegions) {
      planarRegion.planeParameters.positionInWorld.z() += parameters_.extracted_planes_height_offset;
    }
  }
}

}  // namespace convex_plane_decomposition
