#include "convex_plane_decomposition_ros/ConvexPlaneDecompositionRos.h"

#include <chrono>

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include <convex_plane_decomposition/GridMapPreprocessing.h>
#include <convex_plane_decomposition/Nan.h>
#include <convex_plane_decomposition/Postprocessing.h>
#include <convex_plane_decomposition/contour_extraction/ContourExtraction.h>
#include <convex_plane_decomposition/sliding_window_plane_extraction/SlidingWindowPlaneExtractor.h>

#include <convex_plane_decomposition_msgs/PlanarTerrain.h>

#include "convex_plane_decomposition_ros/MessageConversion.h"
#include "convex_plane_decomposition_ros/ParameterLoading.h"
#include "convex_plane_decomposition_ros/RosVisualizations.h"

namespace convex_plane_decomposition {

ConvexPlaneExtractionROS::ConvexPlaneExtractionROS(ros::NodeHandle& nodeHandle) : tfBuffer_(), tfListener_(tfBuffer_) {
  bool parametersLoaded = loadParameters(nodeHandle);

  if (parametersLoaded) {
    elevationMapSubscriber_ = nodeHandle.subscribe(elevationMapTopic_, 1, &ConvexPlaneExtractionROS::callback, this);
    filteredmapPublisher_ = nodeHandle.advertise<grid_map_msgs::GridMap>("filtered_map", 1);
    boundaryPublisher_ = nodeHandle.advertise<jsk_recognition_msgs::PolygonArray>("boundaries", 1);
    insetPublisher_ = nodeHandle.advertise<jsk_recognition_msgs::PolygonArray>("insets", 1);
    regionPublisher_ = nodeHandle.advertise<convex_plane_decomposition_msgs::PlanarTerrain>("planar_terrain", 1);
  }
}

ConvexPlaneExtractionROS::~ConvexPlaneExtractionROS() = default;

bool ConvexPlaneExtractionROS::loadParameters(const ros::NodeHandle& nodeHandle) {
  if (!nodeHandle.getParam("elevation_topic", elevationMapTopic_)) {
    ROS_ERROR("[ConvexPlaneExtractionROS] Could not read parameter `elevation_topic`.");
    return false;
  }
  if (!nodeHandle.getParam("target_frame_id", targetFrameId_)) {
    ROS_ERROR("[ConvexPlaneExtractionROS] Could not read parameter `target_frame_id`.");
    return false;
  }
  if (!nodeHandle.getParam("height_layer", elevationLayer_)) {
    ROS_ERROR("[ConvexPlaneExtractionROS] Could not read parameter `height_layer`.");
    return false;
  }
  if (!nodeHandle.getParam("submap/width", subMapWidth_)) {
    ROS_ERROR("[ConvexPlaneExtractionROS] Could not read parameter `submap/width`.");
    return false;
  }
  if (!nodeHandle.getParam("submap/length", subMapLength_)) {
    ROS_ERROR("[ConvexPlaneExtractionROS] Could not read parameter `submap/length`.");
    return false;
  }
  if (!nodeHandle.getParam("publish_to_controller", publishToController_)) {
    ROS_ERROR("[ConvexPlaneExtractionROS] Could not read parameter `publish_to_controller`.");
    return false;
  }

  const auto preprocessingParameters = loadPreprocessingParameters(nodeHandle, "preprocessing/");
  const auto contourExtractionParameters = loadContourExtractionParameters(nodeHandle, "contour_extraction/");
  const auto ransacPlaneExtractorParameters = loadRansacPlaneExtractorParameters(nodeHandle, "ransac_plane_refinement/");
  const auto slidingWindowPlaneExtractorParameters =
      loadSlidingWindowPlaneExtractorParameters(nodeHandle, "sliding_window_plane_extractor/");
  const auto postprocessingParameters = loadPostprocessingParameters(nodeHandle, "postprocessing/");

  std::lock_guard<std::mutex> lock(mutex_);
  preprocessing_ = std::make_unique<GridMapPreprocessing>(preprocessingParameters);
  slidingWindowPlaneExtractor_ = std::make_unique<sliding_window_plane_extractor::SlidingWindowPlaneExtractor>(
      slidingWindowPlaneExtractorParameters, ransacPlaneExtractorParameters);
  contourExtraction_ = std::make_unique<contour_extraction::ContourExtraction>(contourExtractionParameters);
  postprocessing_ = std::make_unique<Postprocessing>(postprocessingParameters);

  return true;
}

void ConvexPlaneExtractionROS::callback(const grid_map_msgs::GridMap& message) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Convert message to map.
  ROS_INFO("Reading input map...");
  grid_map::GridMap messageMap;
  std::vector<std::string> layers{elevationLayer_};
  grid_map::GridMapRosConverter::fromMessage(message, messageMap, layers, false, false);
  bool success;
  grid_map::GridMap elevationMap = messageMap.getSubmap(messageMap.getPosition(), Eigen::Array2d(subMapLength_, subMapWidth_), success);
  ROS_INFO("...done.");

  // Transform map if necessary
  if (targetFrameId_ != elevationMap.getFrameId()) {
    std::string errorMsg;
    if (tfBuffer_.canTransform(targetFrameId_, elevationMap.getFrameId(), ros::Time(0), &errorMsg)) {
      elevationMap = elevationMap.getTransformedMap(getTransformToTargetFrame(elevationMap.getFrameId()), elevationLayer_, targetFrameId_);
    } else {
      ROS_ERROR_STREAM("[ConvexPlaneExtractionROS] " << errorMsg);
      return;
    }
  }

  if (success) {
    auto t0 = std::chrono::high_resolution_clock::now();
    preprocessing_->preprocess(elevationMap, elevationLayer_);
    auto t1 = std::chrono::high_resolution_clock::now();
    ROS_INFO_STREAM("Preprocessing took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " [ms]");

    // Run pipeline.
    slidingWindowPlaneExtractor_->runExtraction(elevationMap, elevationLayer_);
    auto t2 = std::chrono::high_resolution_clock::now();
    ROS_INFO_STREAM("Sliding window took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " [ms]");

    PlanarTerrain planarTerrain;
    planarTerrain.planarRegions = contourExtraction_->extractPlanarRegions(slidingWindowPlaneExtractor_->getSegmentedPlanesMap());
    auto t3 = std::chrono::high_resolution_clock::now();
    ROS_INFO_STREAM("Contour extraction took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << " [ms]");

    // Add grid map to the terrain
    planarTerrain.gridMap = std::move(elevationMap);

    postprocessing_->postprocess(planarTerrain);
    auto t4 = std::chrono::high_resolution_clock::now();
    ROS_INFO_STREAM("Postprocessing took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " [ms]");

    // Publish terrain
    if (publishToController_) {
      regionPublisher_.publish(toMessage(planarTerrain));
    }

    // Visualize in Rviz.
    reapplyNans(planarTerrain.gridMap.get(elevationLayer_));
    planarTerrain.gridMap.add("segmentation");
    cv::cv2eigen(slidingWindowPlaneExtractor_->getSegmentedPlanesMap().labeledImage, planarTerrain.gridMap.get("segmentation"));
    grid_map_msgs::GridMap outputMessage;
    grid_map::GridMapRosConverter::toMessage(planarTerrain.gridMap, outputMessage);
    filteredmapPublisher_.publish(outputMessage);

    boundaryPublisher_.publish(convertBoundariesToRosPolygons(planarTerrain.planarRegions, planarTerrain.gridMap.getFrameId()));
    insetPublisher_.publish(convertInsetsToRosPolygons(planarTerrain.planarRegions, planarTerrain.gridMap.getFrameId()));
  } else {
    ROS_WARN("[ConvexPlaneExtractionROS] Could not extract submap");
  }
}

Eigen::Isometry3d ConvexPlaneExtractionROS::getTransformToTargetFrame(const std::string& sourceFrame) {
  geometry_msgs::TransformStamped transformStamped;
  try {
    transformStamped = tfBuffer_.lookupTransform(targetFrameId_, sourceFrame, ros::Time(0));
  } catch (tf2::TransformException& ex) {
    ROS_ERROR("[ConvexPlaneExtractionROS] %s", ex.what());
    return Eigen::Isometry3d();
  }

  Eigen::Isometry3d transformation;

  // Extract translation.
  transformation.translation().x() = transformStamped.transform.translation.x;
  transformation.translation().y() = transformStamped.transform.translation.y;
  transformation.translation().z() = transformStamped.transform.translation.z;

  // Extract rotation.
  Eigen::Quaterniond rotationQuaternion(transformStamped.transform.rotation.w, transformStamped.transform.rotation.x,
                                        transformStamped.transform.rotation.y, transformStamped.transform.rotation.z);
  transformation.linear() = rotationQuaternion.toRotationMatrix();
  return transformation;
}

}  // namespace convex_plane_decomposition
