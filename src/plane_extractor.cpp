//
// Created by andrej on 12/6/19.
//

#include "plane_extractor.hpp"

namespace convex_plane_extraction {
using namespace grid_map;

  PlaneExtractor::PlaneExtractor(grid_map::GridMap &map, double resolution, const std::string &normals_layer_prefix,
                               const std::string &height_layer)
                               : map_(map),
                                 resolution_(resolution),
                                 normal_layer_prefix_(normals_layer_prefix),
                                 height_layer_(height_layer),
                                 ransac_plane_extractor_(ransac_plane_extractor::RansacPlaneExtractor(map, resolution, normals_layer_prefix, height_layer)){
      map_size_ = map.getSize();
      ROS_INFO("Plane extractor initialization successful!");
  }

  PlaneExtractor::~PlaneExtractor(){}

  grid_map::GridMap& PlaneExtractor::getMap(){
    return map_;
  }

  void PlaneExtractor::preprocessMapGround(const float& ground_threshold){
    auto& data_from = map_[height_layer_];
    for (GridMapIterator iterator(map_); !iterator.isPastEnd(); ++iterator) {
      const size_t i = iterator.getLinearIndex();
      if (data_from(i) < ground_threshold){
        data_from(i) = 0;
      }
    }
  }

  void PlaneExtractor::setRansacParameters(const ransac_plane_extractor::RansacParameters& parameters){
    ransac_plane_extractor_.setParameters(parameters);
    ROS_INFO("RANSAC parameters set!");
  }

  void PlaneExtractor::runRansacPlaneExtractor(){
    ROS_INFO("Starting RANSAC plane extraction...");
    ransac_plane_extractor_.runDetection();
    ROS_INFO("... done.");
  }

  void PlaneExtractor::augmentMapWithRansacPlanes(){
    ransac_plane_extractor_.ransacPlaneVisualization();
  }

}