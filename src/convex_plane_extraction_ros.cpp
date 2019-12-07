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
    : nodeHandle_(nodeHandle),
      it_(nodeHandle)
{
  if (!readParameters()) {
    success = false;
    return;
  }

  subscriber_ = nodeHandle_.subscribe(inputTopic_, 1, &ConvexPlaneExtractionROS::callback, this);
  ransacPublisher_ = it_.advertise("ransac_planes", 1);
  grid_map_publisher_ = nodeHandle_.advertise<grid_map_msgs::GridMap>("convex_plane_extraction", 1, true);

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
    return false;
  }
  if (!nodeHandle_.getParam("ransac_min_points", ransac_parameters_.min_points)) {
    ROS_ERROR("Could not read parameter `ransac_min_points`. Setting parameter to default value.");
    ransac_parameters_.min_points = 200;
    return false;
  }
  if (!nodeHandle_.getParam("ransac_epsilon", ransac_parameters_.epsilon)) {
    ROS_ERROR("Could not read parameter `ransac_epsilon`. Setting parameter to default value.");
    ransac_parameters_.epsilon = 0.004;
    return false;
  }
  if (!nodeHandle_.getParam("ransac_cluster_epsilon", ransac_parameters_.cluster_epsilon)) {
    ROS_ERROR("Could not read parameter `ransac_cluster_epsilon`. Setting parameter to default value.");
    ransac_parameters_.cluster_epsilon = 0.0282842712475;
    return false;
  }
  if (!nodeHandle_.getParam("ransac_normal_threshold", ransac_parameters_.normal_threshold)) {
    ROS_ERROR("Could not read parameter `ransac_cluster_epsilon`. Setting parameter to default value.");
    ransac_parameters_.normal_threshold = 0.98;
    return false;
  }
  std::string extractor_type;
  if (!nodeHandle_.getParam("plane_extractor_type", extractor_type)) {
    ROS_ERROR("Could not read parameter `ransac_cluster_epsilon`. Setting parameter to default value.");
    ransac_parameters_.normal_threshold = 0.98;
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
  CHECK(success);
  ROS_INFO("...done.");
  applyMedianFilter(inputMap.get("elevation"), 5);
  // Compute planar region segmentation
  ROS_INFO("Initializing plane extractor...");
  PlaneExtractor extractor(inputMap, inputMap.getResolution(), "normal_vectors_", "elevation");
  ROS_INFO("...done.");
  if (plane_extractor_selector_ == kRansacExtractor) {
    extractor.setRansacParameters(ransac_parameters_);
    extractor.runRansacPlaneExtractor();
    extractor.augmentMapWithRansacPlanes();
  }


  //Display RANSAC planes as image.
  cv_bridge::CvImage ransacImage;
  //grid_map::GridMapRosConverter::toCvImage(extractor.getMap(), "ransac_planes", "bgr8", ransacImage);

  cv::Mat coloredImage;
  // Apply the colormap:
  cv::applyColorMap(ransacImage.image, coloredImage, 2);
  ransacImage.image = coloredImage;
  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ransacImage.image).toImageMsg();

  // Publish filtered output grid map.
  ransacPublisher_.publish(img_msg);

  grid_map_msgs::GridMap outputMessage;
  GridMapRosConverter::toMessage(inputMap, outputMessage);
  grid_map_publisher_.publish(outputMessage);

  ROS_INFO("RANSAC planes published as image!");

}

} /* namespace */
