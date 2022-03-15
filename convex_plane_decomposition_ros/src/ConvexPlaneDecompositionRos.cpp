#include "convex_plane_decomposition_ros/ConvexPlaneDecompositionRos.h"

#include <chrono>

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include <convex_plane_decomposition/GridMapPreprocessing.h>
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

ConvexPlaneExtractionROS::~ConvexPlaneExtractionROS() {
  std::stringstream infoStream;
  if (callbackTimer_.getNumTimedIntervals() > 0) {
    infoStream << "\n########################################################################\n";
    infoStream << "The benchmarking is computed over " << callbackTimer_.getNumTimedIntervals() << " iterations. \n";
    infoStream << "PlaneExtraction Benchmarking    : Average time [ms], Max time [ms]\n";
    auto printLine = [](std::string name, const Timer& timer) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2);
      ss << "\t" << name << "\t: " << std::setw(17) << timer.getAverageInMilliseconds() << ", " << std::setw(13)
         << timer.getMaxIntervalInMilliseconds() << "\n";
      return ss.str();
    };
    infoStream << printLine("Pre-process     ", preprocessTimer_);
    infoStream << printLine("Sliding window  ", slidingWindowTimer_);
    infoStream << printLine("Plane extraction", planeExtractionTimer_);
    infoStream << printLine("Post-process    ", postprocessTimer_);
    infoStream << printLine("Total callback  ", callbackTimer_);
  }
  std::cerr << infoStream.str() << std::endl;
}

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
  callbackTimer_.startTimer();

  // Convert message to map.
  grid_map::GridMap messageMap;
  std::vector<std::string> layers{elevationLayer_};
  grid_map::GridMapRosConverter::fromMessage(message, messageMap, layers, false, false);
  if (!containsFiniteValue(messageMap.get(elevationLayer_))) {
    ROS_WARN("[ConvexPlaneExtractionROS] map does not contain any values");
    callbackTimer_.endTimer();
    return;
  }

  // Transform map if necessary
  if (targetFrameId_ != messageMap.getFrameId()) {
    std::string errorMsg;
    ros::Time timeStamp = ros::Time(0); // Use Time(0) to get the latest transform.
    if (tfBuffer_.canTransform(targetFrameId_, messageMap.getFrameId(), timeStamp, &errorMsg)) {
      messageMap =
          messageMap.getTransformedMap(getTransformToTargetFrame(messageMap.getFrameId(), timeStamp), elevationLayer_, targetFrameId_);
    } else {
      ROS_ERROR_STREAM("[ConvexPlaneExtractionROS] " << errorMsg);
      callbackTimer_.endTimer();
      return;
    }
  }

  // Extract submap
  bool success;
  const grid_map::Position submapPosition = [&]() {
    // The map center might be between cells. Taking the submap there can result in changing submap dimensions.
    // project map center to an index and index to center s.t. we get the location of a cell.
    grid_map::Index centerIndex;
    grid_map::Position centerPosition;
    messageMap.getIndex(messageMap.getPosition(), centerIndex);
    messageMap.getPosition(centerIndex, centerPosition);
    return centerPosition;
  }();
  grid_map::GridMap elevationMap = messageMap.getSubmap(submapPosition, Eigen::Array2d(subMapLength_, subMapWidth_), success);
  if (!success) {
    ROS_WARN("[ConvexPlaneExtractionROS] Could not extract submap");
    callbackTimer_.endTimer();
    return;
  }
  const grid_map::Matrix elevationRaw = elevationMap.get(elevationLayer_);

  preprocessTimer_.startTimer();
  preprocessing_->preprocess(elevationMap, elevationLayer_);
  preprocessTimer_.endTimer();

  // Run pipeline.
  slidingWindowTimer_.startTimer();
  slidingWindowPlaneExtractor_->runExtraction(elevationMap, elevationLayer_);
  slidingWindowTimer_.endTimer();

  planeExtractionTimer_.startTimer();
  PlanarTerrain planarTerrain;
  planarTerrain.planarRegions = contourExtraction_->extractPlanarRegions(slidingWindowPlaneExtractor_->getSegmentedPlanesMap());
  planeExtractionTimer_.endTimer();

  // Add grid map to the terrain
  postprocessTimer_.startTimer();
  planarTerrain.gridMap = std::move(elevationMap);

  // Add binary map
  const std::string planeClassificationLayer{"plane_classification"};
  planarTerrain.gridMap.add(planeClassificationLayer);
  auto& traversabilityMask = planarTerrain.gridMap.get(planeClassificationLayer);
  cv::cv2eigen(slidingWindowPlaneExtractor_->getBinaryLabeledImage(), traversabilityMask);

  postprocessing_->postprocess(planarTerrain, elevationLayer_, planeClassificationLayer);
  postprocessTimer_.endTimer();

  // Publish terrain
  if (publishToController_) {
    regionPublisher_.publish(toMessage(planarTerrain));
  }

  // --- Visualize in Rviz --- Not published to the controller
  // Add raw map
  planarTerrain.gridMap.add("elevation_raw", elevationRaw);

  // Add surface normals
  slidingWindowPlaneExtractor_->addSurfaceNormalToMap(planarTerrain.gridMap, "normal");

  // Add surface normals
  planarTerrain.gridMap.add("segmentation");
  cv::cv2eigen(slidingWindowPlaneExtractor_->getSegmentedPlanesMap().labeledImage, planarTerrain.gridMap.get("segmentation"));

  grid_map_msgs::GridMap outputMessage;
  grid_map::GridMapRosConverter::toMessage(planarTerrain.gridMap, outputMessage);
  filteredmapPublisher_.publish(outputMessage);

  boundaryPublisher_.publish(convertBoundariesToRosPolygons(planarTerrain.planarRegions, planarTerrain.gridMap.getFrameId(),
                                                            planarTerrain.gridMap.getTimestamp()));
  insetPublisher_.publish(convertInsetsToRosPolygons(planarTerrain.planarRegions, planarTerrain.gridMap.getFrameId(),
                                                     planarTerrain.gridMap.getTimestamp()));

  callbackTimer_.endTimer();
}

Eigen::Isometry3d ConvexPlaneExtractionROS::getTransformToTargetFrame(const std::string& sourceFrame, const ros::Time& time) {
  geometry_msgs::TransformStamped transformStamped;
  try {
    transformStamped = tfBuffer_.lookupTransform(targetFrameId_, sourceFrame, time);
  } catch (tf2::TransformException& ex) {
    ROS_ERROR("[ConvexPlaneExtractionROS] %s", ex.what());
    return Eigen::Isometry3d::Identity();
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

ros::Time ConvexPlaneExtractionROS::getMessageTime(const grid_map_msgs::GridMap& message) const {
  try {
    ros::Time time;
    return time.fromNSec(message.info.header.stamp.toNSec());
  } catch (std::runtime_error& ex) {
    ROS_WARN("[ConvexPlaneExtractionROS::getMessageTime] %s", ex.what());
    return ros::Time::now();
  }
}

}  // namespace convex_plane_decomposition
