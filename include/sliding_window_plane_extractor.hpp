#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_SLIDING_WINDOW_PLANE_EXTRACTOR_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_SLIDING_WINDOW_PLANE_EXTRACTOR_HPP_

#include <math.h>
#include <numeric>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>

#include "grid_map_ros/grid_map_ros.hpp"


namespace sliding_window_plane_extractor {
  struct SlidingWindowParameters{
    int kernel_size;
    double plane_error_threshold;
  };
  class SlidingWindowPlaneExtractor{
   public:

    SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution, const std::string& layer_height,
        SlidingWindowParameters& parameters);

    SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution, const std::string& layer_height);

    virtual ~SlidingWindowPlaneExtractor();

    void setParameters(const SlidingWindowParameters& parameters);

    void runDetection();

    void slidingWindowPlaneVisualization();

   private:

    grid_map::GridMap& map_;
    std::string elevation_layer_;
    double resolution_;

    int kernel_size_;
    double plane_error_threshold_;
    cv::Mat labeled_image_;

  };
}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_SLIDING_WINDOW_PLANE_EXTRACTOR_HPP_
