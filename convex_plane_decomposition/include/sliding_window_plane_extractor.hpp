#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_SLIDING_WINDOW_PLANE_EXTRACTOR_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_SLIDING_WINDOW_PLANE_EXTRACTOR_HPP_

#include <chrono>
#include <math.h>
#include <numeric>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>

#include "grid_map_ros/grid_map_ros.hpp"
#include "grid_map_core/GridMapMath.hpp"

#include "plane.hpp"
#include "polygon.hpp"
#include "ros_visualizations.hpp"


namespace sliding_window_plane_extractor {
  struct SlidingWindowParameters{
    int kernel_size;
    double plane_error_threshold;
  };
  class SlidingWindowPlaneExtractor{
   public:

    SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution, const std::string& layer_height,
        const std::string& normal_layer_prefix, SlidingWindowParameters& parameters);

    SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution, const std::string& normal_layer_prefix,
        const std::string& layer_height);

    virtual ~SlidingWindowPlaneExtractor();

    void setParameters(const SlidingWindowParameters& parameters);

    void runDetection();

    void slidingWindowPlaneVisualization();

    void generatePlanes();

    void computeMapTransformation();

    void computePlaneFrameFromLabeledImage(const cv::Mat& binary_image, convex_plane_extraction::Plane* plane);

    void visualizeConvexDecomposition(jsk_recognition_msgs::PolygonArray* ros_polygon_array);

    void visualizePlaneContours(jsk_recognition_msgs::PolygonArray* outer_polygons, jsk_recognition_msgs::PolygonArray* hole_poylgons) const;

    void exportConvexPolygons(const std::string& path) const;

   private:

    grid_map::GridMap& map_;
    std::string elevation_layer_;
    std::string normal_layer_prefix_;
    double resolution_;

    Eigen::Matrix2d transformation_xy_to_world_frame_;
    Eigen::Vector2d map_offset_;

    int kernel_size_;
    double plane_error_threshold_;
    cv::Mat labeled_image_;
    int number_of_extracted_planes_;

    convex_plane_extraction::PlaneListContainer planes_;

  };
}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_SLIDING_WINDOW_PLANE_EXTRACTOR_HPP_
