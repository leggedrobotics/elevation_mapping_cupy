
#include "pipeline.hpp"

using namespace convex_plane_extraction;

void Pipeline::runPipeline() {
  sliding_window_plane_extractor::SlidingWindowPlaneExtractor sliding_window_plane_extractor(grid_map_parameters_.map,
      grid_map_parameters_.resolution, grid_map_parameters_.layer_height, grid_map_parameters_.normal_layer_prefix,
      pipeline_parameters_.sliding_window_plane_extractor_parameters);
  sliding_window_plane_extractor.runExtraction();
  PlaneFactory plane_factory(grid_map_parameters_.map, pipeline_parameters_.plane_factory_parameters_);
  plane_factory.createPlanesFromLabeledImageAndPlaneParameters(sliding_window_plane_extractor.getLabeledImage(),
      sliding_window_plane_extractor.getNumberOfExtractedPlanes(), sliding_window_plane_extractor.getLabelPlaneParameterMap());

}

void Pipeline::exportConvexPolygons(const std::string& export_path) const {
  std::ofstream output_file;
  output_file.open(export_path + "convex_polygons.txt", std::ofstream::app);
  std::chrono::milliseconds time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
  for (const auto& plane : planes_){
    for (const auto& polygon : plane.getConvexPolygons()){
      output_file << time_stamp.count() << ", ";
      for (const auto& vertex : polygon){
        output_file << vertex.x() << ", ";
      }
      for (auto vertex_it = polygon.vertices_begin(); vertex_it != polygon.vertices_end(); ++vertex_it){
        output_file << vertex_it->y();
        if (vertex_it != std::prev(polygon.vertices_end())){
          output_file << ", ";
        } else {
          output_file << "\n";
        }
      }
    }
  }
  output_file.close();
  VLOG(1) << "Exported polygons to " << export_path << "convex_polygons.txt !";
}