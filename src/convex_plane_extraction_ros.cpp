/*
 * FiltersDemo.cpp
 *
 *  Created on: Aug 16, 2017
 *      Author: Peter Fankhauser
 *   Institute: ETH Zurich, ANYbotics
 *
 */
#include <chrono>


#include  "geometry_msgs/Point32.h"
#include  "geometry_msgs/PolygonStamped.h"
#include  "jsk_recognition_msgs/PolygonArray.h"


#include "convex_plane_extraction_ros.hpp"

using namespace grid_map;

namespace convex_plane_extraction {

ConvexPlaneExtractionROS::ConvexPlaneExtractionROS(ros::NodeHandle& nodeHandle, bool& success)
    : nodeHandle_(nodeHandle)
{
  if (!readParameters()) {
    success = false;
    return;
  }

  subscriber_ = nodeHandle_.subscribe(inputTopic_, 1, &ConvexPlaneExtractionROS::callback, this);
  grid_map_publisher_ = nodeHandle_.advertise<grid_map_msgs::GridMap>("convex_plane_extraction", 1, true);
  convex_polygon_publisher_ = nodeHandle_.advertise<jsk_recognition_msgs::PolygonArray>("convex_polygons", 1);
  outer_contours_publisher_ = nodeHandle_.advertise<jsk_recognition_msgs::PolygonArray>("outer_contours", 1);
  hole_contours_publsiher_ = nodeHandle_.advertise<jsk_recognition_msgs::PolygonArray>("hole_contours", 1);
  success = true;
}

ConvexPlaneExtractionROS::~ConvexPlaneExtractionROS()
{
}

bool ConvexPlaneExtractionROS::readParameters()
{
  if (!nodeHandle_.getParam("input_topic", inputTopic_)) {
    ROS_ERROR("Could not read parameter `input_topic`.");
    return false;
  }
  if (!nodeHandle_.getParam("ransac_probability", ransac_parameters_.probability)) {
    ROS_ERROR("Could not read parameter `ransac_probability`. Setting parameter to default value.");
    ransac_parameters_.probability = 0.01;
  }
  if (!nodeHandle_.getParam("ransac_min_points", ransac_parameters_.min_points)) {
    ROS_ERROR("Could not read parameter `ransac_min_points`. Setting parameter to default value.");
    ransac_parameters_.min_points = 200;
  }
  if (!nodeHandle_.getParam("ransac_epsilon", ransac_parameters_.epsilon)) {
    ROS_ERROR("Could not read parameter `ransac_epsilon`. Setting parameter to default value.");
    ransac_parameters_.epsilon = 0.004;
  }
  if (!nodeHandle_.getParam("ransac_cluster_epsilon", ransac_parameters_.cluster_epsilon)) {
    ROS_ERROR("Could not read parameter `ransac_cluster_epsilon`. Setting parameter to default value.");
    ransac_parameters_.cluster_epsilon = 0.0282842712475;
  }
  if (!nodeHandle_.getParam("ransac_normal_threshold", ransac_parameters_.normal_threshold)) {
    ROS_ERROR("Could not read parameter `ransac_normal_threshold`. Setting parameter to default value.");
    ransac_parameters_.normal_threshold = 0.98;
  }
  if (!nodeHandle_.getParam("sliding_window_kernel_size", sliding_window_parameters_.kernel_size)) {
    ROS_ERROR("Could not read parameter `sliding_window_kernel_size`. Setting parameter to default value.");
    sliding_window_parameters_.kernel_size = 5;
  }
  if (!nodeHandle_.getParam("sliding_window_plane_error", sliding_window_parameters_.plane_error_threshold)) {
    ROS_ERROR("Could not read parameter `sliding_window_plane_error`. Setting parameter to default value.");
    sliding_window_parameters_.plane_error_threshold = 0.004;
  }
  std::string extractor_type;
  if (!nodeHandle_.getParam("plane_extractor_type", extractor_type)) {
    ROS_ERROR("Could not read parameter `plane_extractor_type`. Setting parameter to default value.");
    return false;
  }
  if (extractor_type.compare("ransac") == 0){
    plane_extractor_selector_ = kRansacExtractor;
  } else{
    plane_extractor_selector_ = kSlidingWindowExtractor;
  }
  return true;
}

void ConvexPlaneExtractionROS::callback(const grid_map_msgs::GridMap& message) {
  // Convert message to map.
  ROS_INFO("Reading input map...");
  GridMap messageMap;
  GridMapRosConverter::fromMessage(message, messageMap);
  bool success;
  GridMap inputMap = messageMap.getSubmap(messageMap.getPosition(), Eigen::Array2d(6, 6), success);
  Position3 position;
  inputMap.getPosition3("elevation", Index(0,0), position);
  std::cout << inputMap.getPosition() << std::endl;
  CHECK(success);
  ROS_INFO("...done.");
  applyMedianFilter(inputMap.get("elevation"), 5);
  // Compute planar region segmentation
  ROS_INFO("Initializing plane extractor...");
  PlaneExtractor extractor(inputMap, inputMap.getResolution(), "normal_vectors_", "elevation");
  ROS_INFO("...done.");
  jsk_recognition_msgs::PolygonArray ros_polygon_array;
  jsk_recognition_msgs::PolygonArray ros_polygon_outer_contours;
  jsk_recognition_msgs::PolygonArray ros_polygon_hole_contours;
  switch (plane_extractor_selector_) {
    case kRansacExtractor : {
      extractor.setRansacParameters(ransac_parameters_);
      extractor.runRansacPlaneExtractor();
      extractor.augmentMapWithRansacPlanes();
      break;
    }
    case kSlidingWindowExtractor : {
      extractor.setSlidingWindowParameters(sliding_window_parameters_);
      extractor.runSlidingWindowPlaneExtractor();
      extractor.augmentMapWithSlidingWindowPlanes();
      extractor.generatePlanes();
      extractor.visualizeConvexDecomposition(&ros_polygon_array);
      extractor.visualizePlaneContours(&ros_polygon_outer_contours, &ros_polygon_hole_contours);
      break;
      }
  }

  grid_map_msgs::GridMap outputMessage;
  GridMapRosConverter::toMessage(inputMap, outputMessage);
  grid_map_publisher_.publish(outputMessage);

  convex_polygon_publisher_.publish(ros_polygon_array);
  outer_contours_publisher_.publish(ros_polygon_outer_contours);
  hole_contours_publsiher_.publish(ros_polygon_hole_contours);

}

} /* namespace */
