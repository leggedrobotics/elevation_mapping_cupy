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
                                 ransac_plane_extractor_(ransac_plane_extractor::RansacPlaneExtractor(map, resolution, normals_layer_prefix, height_layer)),
                                 sliding_window_plane_extractor_(sliding_window_plane_extractor::SlidingWindowPlaneExtractor(map,resolution, normals_layer_prefix, height_layer)){
      map_size_ = map.getSize();
      ROS_INFO("Plane extractor initialization successful!");
  }

  PlaneExtractor::~PlaneExtractor(){}

  grid_map::GridMap& PlaneExtractor::getMap(){
    return map_;
  }

  void PlaneExtractor::setRansacParameters(const ransac_plane_extractor::RansacPlaneExtractorParameters& parameters){
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

  void PlaneExtractor::setSlidingWindowParameters(const sliding_window_plane_extractor::SlidingWindowParameters & parameters) {
    sliding_window_plane_extractor_.setParameters(parameters);
    ROS_INFO("Sliding window parameters set!");
  }

  void PlaneExtractor::runSlidingWindowPlaneExtractor(){
    ROS_INFO("Starting sliding window plane extraction...");
    sliding_window_plane_extractor_.runSlidingWindowDetector();
    ROS_INFO("... done.");
  }

void PlaneExtractor::augmentMapWithSlidingWindowPlanes(){
  sliding_window_plane_extractor_.slidingWindowPlaneVisualization();
}

}