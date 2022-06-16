//
// Created by rgrandia on 10.06.20.
//

#pragma once

#include <ros/node_handle.h>

#include <convex_plane_decomposition/GridMapPreprocessing.h>
#include <convex_plane_decomposition/Postprocessing.h>
#include <convex_plane_decomposition/contour_extraction/ContourExtractionParameters.h>
#include <convex_plane_decomposition/ransac/RansacPlaneExtractorParameters.h>
#include <convex_plane_decomposition/sliding_window_plane_extraction/SlidingWindowPlaneExtractorParameters.h>

namespace convex_plane_decomposition {

PreprocessingParameters loadPreprocessingParameters(const ros::NodeHandle& nodeHandle, const std::string& prefix);

contour_extraction::ContourExtractionParameters loadContourExtractionParameters(const ros::NodeHandle& nodeHandle,
                                                                                const std::string& prefix);

ransac_plane_extractor::RansacPlaneExtractorParameters loadRansacPlaneExtractorParameters(const ros::NodeHandle& nodeHandle,
                                                                                          const std::string& prefix);

sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters loadSlidingWindowPlaneExtractorParameters(
    const ros::NodeHandle& nodeHandle, const std::string& prefix);

PostprocessingParameters loadPostprocessingParameters(const ros::NodeHandle& nodeHandle, const std::string& prefix);

}  // namespace convex_plane_decomposition
