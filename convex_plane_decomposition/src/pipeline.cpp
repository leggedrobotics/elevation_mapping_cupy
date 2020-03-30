#include "pipeline.hpp"

using namespace convex_plane_extraction;

Pipeline::Pipeline(const PipelineParameters& pipeline_parameters, const GridMapParameters& grid_map_parameters)
    : pipeline_parameters_(pipeline_parameters),
      grid_map_parameters_(grid_map_parameters),
      sliding_window_plane_extractor_(grid_map_parameters_.map,
          grid_map_parameters_.resolution, grid_map_parameters_.layer_height,
          pipeline_parameters_.sliding_window_plane_extractor_parameters, pipeline_parameters_.sliding_window_plane_extractor_parameters.ransac_parameters),
      plane_factory_(grid_map_parameters_.map, pipeline_parameters_.plane_factory_parameters){
  VLOG(1) << "Starting plane extraction...";
  sliding_window_plane_extractor_.runExtraction();
  VLOG(1) << "done";
  plane_factory_.createPlanesFromLabeledImageAndPlaneParameters(sliding_window_plane_extractor_.getLabeledImage(),
                                                                sliding_window_plane_extractor_.getNumberOfExtractedPlanes(),
                                                                sliding_window_plane_extractor_.getLabelPlaneParameterMap());
  plane_factory_.decomposePlanesInConvexPolygons();
}

Polygon3dVectorContainer Pipeline::getConvexPolygons() const{
  return plane_factory_.getConvexPolygonsInWorldFrame();
}

Polygon3dVectorContainer Pipeline::getPlaneContours() const{
  return plane_factory_.getPlaneContoursInWorldFrame();
}
