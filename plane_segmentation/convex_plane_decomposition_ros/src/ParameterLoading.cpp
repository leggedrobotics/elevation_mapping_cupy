//
// Created by rgrandia on 10.06.20.
//

#include "convex_plane_decomposition_ros/ParameterLoading.h"

namespace convex_plane_decomposition {

template <typename T>
bool loadParameter(const ros::NodeHandle& nodeHandle, const std::string& prefix, const std::string& param, T& value) {
  if (!nodeHandle.getParam(prefix + param, value)) {
    ROS_ERROR_STREAM("[ConvexPlaneExtractionROS] Could not read parameter `"
                     << param << "`. Setting parameter to default value : " << std::to_string(value));
    return false;
  } else {
    return true;
  }
}

PreprocessingParameters loadPreprocessingParameters(const ros::NodeHandle& nodeHandle, const std::string& prefix) {
  PreprocessingParameters preprocessingParameters;
  loadParameter(nodeHandle, prefix, "resolution", preprocessingParameters.resolution);
  loadParameter(nodeHandle, prefix, "kernelSize", preprocessingParameters.kernelSize);
  loadParameter(nodeHandle, prefix, "numberOfRepeats", preprocessingParameters.numberOfRepeats);
  return preprocessingParameters;
}

contour_extraction::ContourExtractionParameters loadContourExtractionParameters(const ros::NodeHandle& nodeHandle,
                                                                                const std::string& prefix) {
  contour_extraction::ContourExtractionParameters contourParams;
  loadParameter(nodeHandle, prefix, "marginSize", contourParams.marginSize);
  return contourParams;
}

ransac_plane_extractor::RansacPlaneExtractorParameters loadRansacPlaneExtractorParameters(const ros::NodeHandle& nodeHandle,
                                                                                          const std::string& prefix) {
  ransac_plane_extractor::RansacPlaneExtractorParameters ransacParams;
  loadParameter(nodeHandle, prefix, "probability", ransacParams.probability);
  loadParameter(nodeHandle, prefix, "min_points", ransacParams.min_points);
  loadParameter(nodeHandle, prefix, "epsilon", ransacParams.epsilon);
  loadParameter(nodeHandle, prefix, "cluster_epsilon", ransacParams.cluster_epsilon);
  loadParameter(nodeHandle, prefix, "normal_threshold", ransacParams.normal_threshold);
  return ransacParams;
}

sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters loadSlidingWindowPlaneExtractorParameters(
    const ros::NodeHandle& nodeHandle, const std::string& prefix) {
  sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters swParams;
  loadParameter(nodeHandle, prefix, "kernel_size", swParams.kernel_size);
  loadParameter(nodeHandle, prefix, "planarity_opening_filter", swParams.planarity_opening_filter);
  if (loadParameter(nodeHandle, prefix, "plane_inclination_threshold_degrees", swParams.plane_inclination_threshold)) {
    swParams.plane_inclination_threshold = std::cos(swParams.plane_inclination_threshold * M_PI / 180.0);
  }
  if (loadParameter(nodeHandle, prefix, "local_plane_inclination_threshold_degrees", swParams.local_plane_inclination_threshold)) {
    swParams.local_plane_inclination_threshold = std::cos(swParams.local_plane_inclination_threshold * M_PI / 180.0);
  }
  loadParameter(nodeHandle, prefix, "plane_patch_error_threshold", swParams.plane_patch_error_threshold);
  loadParameter(nodeHandle, prefix, "min_number_points_per_label", swParams.min_number_points_per_label);
  loadParameter(nodeHandle, prefix, "connectivity", swParams.connectivity);
  loadParameter(nodeHandle, prefix, "include_ransac_refinement", swParams.include_ransac_refinement);
  loadParameter(nodeHandle, prefix, "global_plane_fit_distance_error_threshold", swParams.global_plane_fit_distance_error_threshold);
  loadParameter(nodeHandle, prefix, "global_plane_fit_angle_error_threshold_degrees",
                swParams.global_plane_fit_angle_error_threshold_degrees);
  return swParams;
}

PostprocessingParameters loadPostprocessingParameters(const ros::NodeHandle& nodeHandle, const std::string& prefix) {
  PostprocessingParameters postprocessingParameters;
  loadParameter(nodeHandle, prefix, "extracted_planes_height_offset", postprocessingParameters.extracted_planes_height_offset);
  loadParameter(nodeHandle, prefix, "nonplanar_height_offset", postprocessingParameters.nonplanar_height_offset);
  loadParameter(nodeHandle, prefix, "nonplanar_horizontal_offset", postprocessingParameters.nonplanar_horizontal_offset);
  loadParameter(nodeHandle, prefix, "smoothing_dilation_size", postprocessingParameters.smoothing_dilation_size);
  loadParameter(nodeHandle, prefix, "smoothing_box_kernel_size", postprocessingParameters.smoothing_box_kernel_size);
  loadParameter(nodeHandle, prefix, "smoothing_gauss_kernel_size", postprocessingParameters.smoothing_gauss_kernel_size);
  return postprocessingParameters;
}

}  // namespace convex_plane_decomposition
