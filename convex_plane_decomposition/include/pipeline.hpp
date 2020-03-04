#ifndef CONVEX_PLANE_EXTRACTION__PIPELINE_HPP_
#define CONVEX_PLANE_EXTRACTION__PIPELINE_HPP_

#include "convex_decomposer.hpp"
#include "plane_factory.hpp"
#include "sliding_window_plane_extractor.hpp"

namespace convex_plane_extraction {

struct PipelineParameters{
  sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters sliding_window_plane_extractor_parameters =
      sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters();
  PlaneFactoryParameters plane_factory_parameters_ = PlaneFactoryParameters();
};

struct GridMapParameters{
  grid_map::GridMap& map;
  double resolution;
  std::string& layer_height;
  std::string& normal_layer_prefix;
};


class Pipeline {
 public:

  Pipeline(const PipelineParameters& pipeline_parameters, const GridMapParameters& grid_map_parameters);

  Polygon3dVectorContainer getConvexPolygons() const;

  Polygon3dVectorContainer getPlaneContours() const;

 private:

//  void exportConvexPolygons(const std::string& export_path) const;

  PipelineParameters pipeline_parameters_;
  GridMapParameters grid_map_parameters_;

  sliding_window_plane_extractor::SlidingWindowPlaneExtractor sliding_window_plane_extractor_;
  PlaneFactory plane_factory_;

};
} // namespace_convex_plane_extraction
#endif //CONVEX_PLANE_EXTRACTION__PIPELINE_HPP_
