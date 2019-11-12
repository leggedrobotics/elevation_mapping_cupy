/*
 * ConvexSetSequence.hpp
 *
 *  Created on: Nov 12, 2019
 *      Author: Marko Bjelonic
 */

#pragma once

// ros
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

// loco
#include <loco/common/typedefs.hpp>

// loco perception utils
#include <loco_perception_utils/ConvexSet.hpp>

// loco ros
#include <loco_ros/loco_ros.hpp>
#include <loco_ros/visualization/ModuleRos.hpp>

// robot utils
#include <robot_utils/geometry/geometry.hpp>

// stl
#include <string>


namespace loco_perception_ros {

class ConvexSetSequence {
 public:
  ConvexSetSequence();
  virtual ~ConvexSetSequence() = default;

  bool initialize(ros::NodeHandle& nodeHandle, const std::string& topic);
  bool shutdown();

  bool update(const std::vector<loco_perception_utils::ConvexSet>& convexSets);
  bool publish();

  void setColorVector(std::vector<loco::Vector, Eigen::aligned_allocator<loco::Vector>>& colors);
  void setColorVectorId(unsigned int colorId);

  unsigned int getNumSubscribers() const;

 protected:
  ros::NodeHandle nodeHandle_;
  visualization_msgs::MarkerArray convexSetSequence_;
  ros::Publisher publisher_;
  std::vector<loco::Vector, Eigen::aligned_allocator<loco::Vector>> colors_;
  unsigned int colorId_;
};

} /* namespace loco */
