#pragma once

#include "convex_plane_decomposition/PlanarRegion.h"

namespace convex_plane_decomposition {

struct PostprocessingParameters {
  /// Added offset in z direction to compensate for the location of the foot frame w.r.t. the elevation map
  double extracted_planes_height_offset = 0.0;

  /// Added offset in z direction added in cells that are not traversable
  double nonplanar_height_offset = 0.0;

  /// Size of the kernel creating the boundary offset. In number of pixels.
  int nonplanar_horizontal_offset = 2;
};

class Postprocessing {
 public:
  Postprocessing(const PostprocessingParameters& parameters);

  /**
   * @param planarTerrain
   * @param elevationLayer : name of the elevation layer
   * @param planeSegmentationLayer : name of the planarity layer with planar = 1.0, non-planar = 0.0
   */
  void postprocess(PlanarTerrain& planarTerrain, const std::string& elevationLayer, const std::string& planeSegmentationLayer) const;

 private:
  void dilationInNonplanarRegions(Eigen::MatrixXf& elevationData, const Eigen::MatrixXf& planarityMask) const;
  void addHeightOffset(Eigen::MatrixXf& elevationData, const Eigen::MatrixXf& planarityMask) const;
  void addHeightOffset(std::vector<PlanarRegion>& planarRegions) const;

  PostprocessingParameters parameters_;
};

}  // namespace convex_plane_decomposition
