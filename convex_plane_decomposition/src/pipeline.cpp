
#include "pipeline.hpp"

using namespace convex_plane_extraction;

Pipeline::Pipeline(const PipelineParameters& pipeline_parameters, const GridMapParameters& grid_map_parameters)
    : pipeline_parameters_(pipeline_parameters),
      grid_map_parameters_(grid_map_parameters),
      sliding_window_plane_extractor_(sliding_window_plane_extractor::SlidingWindowPlaneExtractor(grid_map_parameters_.map,
          grid_map_parameters_.resolution, grid_map_parameters_.layer_height, grid_map_parameters_.normal_layer_prefix,
          pipeline_parameters_.sliding_window_plane_extractor_parameters)),
      plane_factory_(PlaneFactory(grid_map_parameters_.map, pipeline_parameters_.plane_factory_parameters_)){
  sliding_window_plane_extractor_.runExtraction();
  plane_factory_.createPlanesFromLabeledImageAndPlaneParameters(sliding_window_plane_extractor_.getLabeledImage(),
      sliding_window_plane_extractor_.getNumberOfExtractedPlanes(), sliding_window_plane_extractor_.getLabelPlaneParameterMap());
  plane_factory_.decomposePlanesInConvexPolygons();
}

Polygon3dVectorContainer Pipeline::getConvexPolygons() const{
  return plane_factory_.getConvexPolygonsInWorldFrame();
}

Polygon3dVectorContainer Pipeline::getPlaneContours() const{
  return plane_factory_.getPlaneContoursInWorldFrame();
}


//void Pipeline::exportConvexPolygons(const std::string& export_path) const {
//  std::ofstream output_file;
//  output_file.open(export_path + "convex_polygons.txt", std::ofstream::app);
//  std::chrono::milliseconds time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
//      std::chrono::system_clock::now().time_since_epoch());
//  for (const auto& plane : planes_){
//    for (const auto& polygon : plane.getConvexPolygons()){
//      output_file << time_stamp.count() << ", ";
//      for (const auto& vertex : polygon){
//        output_file << vertex.x() << ", ";
//      }
//      for (auto vertex_it = polygon.vertices_begin(); vertex_it != polygon.vertices_end(); ++vertex_it){
//        output_file << vertex_it->y();
//        if (vertex_it != std::prev(polygon.vertices_end())){
//          output_file << ", ";
//        } else {
//          output_file << "\n";
//        }
//      }
//    }
//  }
//  output_file.close();
//  VLOG(1) << "Exported polygons to " << export_path << "convex_polygons.txt !";
//}