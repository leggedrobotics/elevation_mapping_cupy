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


namespace sliding_window_plane_extractor {

  struct SlidingWindowPlaneExtractorParameters{
    int kernel_size = 3;
    double plane_patch_error_threshold = 0.004;
    double surface_normal_angle_threshold_degrees = 1.0;
    bool include_curvature_detection = false;
    bool include_ransac_refinement = false;
    double global_plane_fit_error_threshold = 0.01;

    ransac_plane_extractor::RansacPlaneExtractorParameters ransac_parameters = ransac_plane_extractor::RansacPlaneExtractorParameters();
  };

  class SlidingWindowPlaneExtractor{
   public:

    SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution, const std::string& layer_height, const SlidingWindowPlaneExtractorParameters& parameters = SlidingWindowPlaneExtractorParameters(),
        const ransac_plane_extractor::RansacPlaneExtractorParameters& ransac_parameters = ransac_plane_extractor::RansacPlaneExtractorParameters());

    void setParameters(const SlidingWindowPlaneExtractorParameters& parameters);

    void runExtraction();

    const cv::Mat& getLabeledImage() const{
      return labeled_image_;
    }

    const auto& getLabelPlaneParameterMap() const{ return label_plane_parameters_map_; }

    const int getNumberOfExtractedPlanes() const { return number_of_extracted_planes_; }

    //    void exportConvexPolygons(const std::string& path) const;

   private:
    double computeAverageErrorToPlane(const Eigen::Vector3d& normal_vector, const Eigen::Vector3d& support_vector,
                                      const std::vector<ransac_plane_extractor::PointWithNormal>& points_with_normal) const;

    void computeMapTransformation();

    void computePlaneParametersForLabel(int label);

    void extractPlaneParametersFromLabeledImage();

    auto runRansacRefinement(std::vector<ransac_plane_extractor::PointWithNormal>& points_with_normal) const;

    void runSegmentation();

    void runSlidingWindowDetector();

    void runSurfaceNormalCurvatureDetection();

    grid_map::GridMap& map_;
    std::string elevation_layer_;
    double resolution_;
    Eigen::Matrix2d transformation_xy_to_world_frame_;
    Eigen::Vector2d map_offset_;

    SlidingWindowPlaneExtractorParameters parameters_;
    ransac_plane_extractor::RansacPlaneExtractorParameters ransac_parameters_;
    cv::Mat binary_image_patch_;
    cv::Mat binary_image_angle_;
    cv::Mat labeled_image_;
    int number_of_extracted_planes_;
    std::map<int, convex_plane_extraction::PlaneParameters> label_plane_parameters_map_;
  };
}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_SLIDING_WINDOW_PLANE_EXTRACTOR_HPP_
