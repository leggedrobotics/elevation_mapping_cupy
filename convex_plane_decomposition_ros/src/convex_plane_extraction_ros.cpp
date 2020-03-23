#include <chrono>

#include "convex_plane_extraction_ros.hpp"

using namespace grid_map;

namespace convex_plane_extraction {

ConvexPlaneExtractionROS::ConvexPlaneExtractionROS(ros::NodeHandle& nodeHandle, bool& success)
    : nodeHandle_(nodeHandle) {
  if (!readParameters()) {
    success = false;
    return;
  }

  subscriber_ = nodeHandle_.subscribe(inputTopic_, 1, &ConvexPlaneExtractionROS::callback, this);
  grid_map_publisher_ = nodeHandle_.advertise<grid_map_msgs::GridMap>("convex_plane_extraction", 1, true);
  convex_polygon_publisher_ = nodeHandle_.advertise<jsk_recognition_msgs::PolygonArray>("convex_polygons", 1);
  outer_contours_publisher_ = nodeHandle_.advertise<jsk_recognition_msgs::PolygonArray>("outer_contours", 1);
  success = true;
}

ConvexPlaneExtractionROS::~ConvexPlaneExtractionROS(){}

bool ConvexPlaneExtractionROS::readParameters()
{
  if (!nodeHandle_.getParam("input_topic", inputTopic_)) {
    ROS_ERROR("Could not read parameter `input_topic`.");
    return false;
  }
  return true;
}

void ConvexPlaneExtractionROS::callback(const grid_map_msgs::GridMap& message) {
  auto start_total = std::chrono::system_clock::now();

  // Convert message to map.
  ROS_INFO("Reading input map...");
  GridMap messageMap;
  GridMapRosConverter::fromMessage(message, messageMap);
  bool success;
  GridMap inputMap = messageMap.getSubmap(messageMap.getPosition(), Eigen::Array2d(6, 6), success);
  CHECK(success);
  ROS_INFO("...done.");
  VLOG(1) << "Applying median filtering to map.";
  applyMedianFilter(inputMap.get("elevation"), 5);

  // Run pipeline.
  VLOG(1) << "Running pipeline ...";
  PipelineROS pipeline_ros(nodeHandle_, inputMap);
  VLOG(1) << "done.";
  // Visualize in Rviz.
  jsk_recognition_msgs::PolygonArray outer_plane_contours = pipeline_ros.getOuterPlaneContours();
  //jsk_recognition_msgs::PolygonArray convex_polygons = pipeline_ros.getConvexPolygons();
  pipeline_ros.augmentGridMapWithSegmentation(inputMap);

  grid_map_msgs::GridMap outputMessage;
  GridMapRosConverter::toMessage(inputMap, outputMessage);
  grid_map_publisher_.publish(outputMessage);

  //convex_polygon_publisher_.publish(convex_polygons);
  outer_contours_publisher_.publish(outer_plane_contours);

}

} /* namespace */
