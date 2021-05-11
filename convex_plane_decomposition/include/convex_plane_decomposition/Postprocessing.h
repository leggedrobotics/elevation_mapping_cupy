#pragma once

#include "convex_plane_decomposition/PlanarRegion.h"

namespace convex_plane_decomposition {

struct PostprocessingParameters {
  /// Added offset in z direction to compensate for the location of the foot frame w.r.t. the elevation map
  double extracted_planes_height_offset = 0.0;
};

class Postprocessing {
 public:
  Postprocessing(const PostprocessingParameters& parameters);

  void postprocess(PlanarTerrain& planarTerrain, const std::string& layer) const;

 private:
  void addHeightOffset(PlanarTerrain& planarTerrain, const std::string& layer) const;

  PostprocessingParameters parameters_;
};

}  // namespace convex_plane_decomposition
