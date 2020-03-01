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
#include "ransac_plane_extractor.hpp"
#include "ros_visualizations.hpp"


namespace sliding_window_plane_extractor {

  struct SlidingWindowPlaneExtractorParameters{
    int kernel_size = 3;
    double plane_patch_error_threshold = 0.004;
    double surface_normal_angle_threshold;
    bool include_curvature_detection;
    bool include_ransac_refinement;
    double global_plane_fit_error_threshold = 0.01;
  };

  class SlidingWindowPlaneExtractor{
   public:

    SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution, const std::string& layer_height,
        const std::string& normal_layer_prefix, const SlidingWindowPlaneExtractorParameters& parameters = SlidingWindowPlaneExtractorParameters(),
        const ransac_plane_extractor::RansacPlaneExtractorParameters& ransac_parameters = ransac_plane_extractor::RansacPlaneExtractorParameters());

    void setParameters(const SlidingWindowPlaneExtractorParameters& parameters);

    void runDetection();

    void runSurfaceNormalCurvatureDetection();

    void slidingWindowPlaneVisualization();

    void generatePlanes();

    void computeMapTransformation();

    void computePlaneFrameFromLabeledImage(const cv::Mat& binary_image, convex_plane_extraction::Plane* plane);

    void extractPlaneParametersFromLabeledImage();

    void computePlaneParametersForLabel(int label);

    double computeAverageErrorToPlane(const Eigen::Vector3d& normal_vector, const Eigen::Vector3d& support_vector,
                                      const std::vector<ransac_plane_extractor::PointWithNormal>& points_with_normal) const;
    const auto& runRansacRefinement(std::vector<ransac_plane_extractor::PointWithNormal>& points_with_normal) const;

    void runSegmentation()

    void visualizeConvexDecomposition(jsk_recognition_msgs::PolygonArray* ros_polygon_array);

    void visualizePlaneContours(jsk_recognition_msgs::PolygonArray* outer_polygons, jsk_recognition_msgs::PolygonArray* hole_poylgons) const;

    void exportConvexPolygons(const std::string& path) const;

   private:

    grid_map::GridMap& map_;
    std::string elevation_layer_;
    std::string normal_layer_prefix_;
    double resolution_;

    SlidingWindowPlaneExtractorParameters parameters_;
    ransac_plane_extractor::RansacPlaneExtractorParameters ransac_parameters_;
    cv::Mat binary_image_patch_;
    cv::Mat binary_image_angle_;
    cv::Mat labeled_image_;
    int number_of_extracted_planes_;
    std::map<int, convex_plane_extraction::PlaneParameters> plane_parameters_;

  };
}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_SLIDING_WINDOW_PLANE_EXTRACTOR_HPP_
