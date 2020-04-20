#ifndef CONVEX_PLANE_EXTRACTION__PIPELINE_HPP_
#define CONVEX_PLANE_EXTRACTION__PIPELINE_HPP_

#include "convex_decomposer.hpp"
#include "plane_factory.hpp"
#include "sliding_window_plane_extractor.hpp"

namespace convex_plane_decomposition {

struct PipelineParameters{
  sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters sliding_window_plane_extractor_parameters =
      sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters();
  PlaneFactoryParameters plane_factory_parameters = PlaneFactoryParameters();
};

struct GridMapParameters{
  GridMapParameters(grid_map::GridMap& input_map, const std::string& height_layer)
  : map(input_map),
    resolution(input_map.getResolution()),
    layer_height(height_layer){}

  grid_map::GridMap& map;
  double resolution;
  std::string layer_height;
};

class Pipeline {
 public:
  Pipeline(const PipelineParameters& pipeline_parameters, const GridMapParameters& grid_map_parameters);

  Polygon3dVectorContainer getConvexPolygons() const;

  Polygon3dVectorContainer getPlaneContours() const;

  const cv::Mat& getSegmentationImage() const { return sliding_window_plane_extractor_.getLabeledImage(); };

 private:
  // Parameters
  GridMapParameters grid_map_parameters_;
  PipelineParameters pipeline_parameters_;

  PlaneFactory plane_factory_;
  sliding_window_plane_extractor::SlidingWindowPlaneExtractor sliding_window_plane_extractor_;
};
} // namespace_convex_plane_decomposition
#endif //CONVEX_PLANE_EXTRACTION__PIPELINE_HPP_
