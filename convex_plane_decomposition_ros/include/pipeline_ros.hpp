#ifndef CONVEX_PLANE_EXTRACTION_ROS_INCLUDE_PIPELINE_ROS_HPP_
#define CONVEX_PLANE_EXTRACTION_ROS_INCLUDE_PIPELINE_ROS_HPP_

#include  <geometry_msgs/Point32.h>
#include  <geometry_msgs/PolygonStamped.h>
#include  <glog/logging.h>
#include  <jsk_recognition_msgs/PolygonArray.h>
#include  <ros/ros.h>

#include "pipeline.hpp"
#include "ros_visualizations.hpp"

namespace convex_plane_extraction {

GridMapParameters loadGridMapParameters(ros::NodeHandle& nodeHandle, grid_map::GridMap& map);

PipelineParameters loadPipelineParameters(ros::NodeHandle& nodeHandle,  grid_map::GridMap& map);

class PipelineROS {
 public:
  PipelineROS(ros::NodeHandle& nodeHandle, grid_map::GridMap& map)
    : pipeline_(loadPipelineParameters(nodeHandle, map), loadGridMapParameters(nodeHandle, map)){}

  jsk_recognition_msgs::PolygonArray getConvexPolygons() const;

  jsk_recognition_msgs::PolygonArray getOuterPlaneContours() const;

  void augmentGridMapWithSegmentation(grid_map::GridMap& map);

 private:

  Pipeline pipeline_;
};

} // namespace convex_plane_extraction
#endif //CONVEX_PLANE_EXTRACTION_ROS__PIPELINE_ROS_HPP_
