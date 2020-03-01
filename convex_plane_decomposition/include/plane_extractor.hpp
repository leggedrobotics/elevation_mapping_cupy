//
// Created by andrej on 12/6/19.
//

#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_EXTRACTOR_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_EXTRACTOR_HPP_

#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"

#include "ransac_plane_extractor.hpp"
#include "sliding_window_plane_extractor.hpp"
#include "types.hpp"

namespace convex_plane_extraction {
  using namespace grid_map;

  enum PlaneExtractorType : int {
    kRansacExtractor = 0,
    kSlidingWindowExtractor = 1
  };

  class PlaneExtractor {
   public:

    /*!
   * Constructor.
   */
    PlaneExtractor(grid_map::GridMap &map, double resolution, const std::string &normals_layer_prefix,
      const std::string &height_layer);

    /*!
   * Destructor.
   */
    virtual ~PlaneExtractor();

    grid_map::GridMap& getMap();

    // Ransac plane extractor related functions.
    void setRansacParameters(const ransac_plane_extractor::RansacPlaneExtractorParameters& parameters);

    void runRansacPlaneExtractor();

    void augmentMapWithRansacPlanes();

    // Sliding window plane extractor related functions.
    void setSlidingWindowParameters(const sliding_window_plane_extractor::SlidingWindowParameters& parameters);

    void runSlidingWindowPlaneExtractor();

    void augmentMapWithSlidingWindowPlanes();

    void generatePlanes(){
      sliding_window_plane_extractor_.generatePlanes();
    }

    void visualizeConvexDecomposition(jsk_recognition_msgs::PolygonArray* ros_polygon_array){
      CHECK_NOTNULL(ros_polygon_array);
      sliding_window_plane_extractor_.visualizeConvexDecomposition(ros_polygon_array);
      sliding_window_plane_extractor_.exportConvexPolygons("/home/andrej/Desktop/");
    }

    void visualizePlaneContours(jsk_recognition_msgs::PolygonArray* ros_polygon_outer_contours, jsk_recognition_msgs::PolygonArray* ros_polygon_hole_contours){
      sliding_window_plane_extractor_.visualizePlaneContours(ros_polygon_outer_contours, ros_polygon_hole_contours);
    }

   private:

    // Grid map related
    grid_map::GridMap& map_;
    double resolution_;
    std::string normal_layer_prefix_;
    std::string height_layer_;
    Vector2i map_size_;

    ransac_plane_extractor::RansacPlaneExtractor ransac_plane_extractor_;
    sliding_window_plane_extractor::SlidingWindowPlaneExtractor sliding_window_plane_extractor_;
  };

}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_EXTRACTOR_HPP_
