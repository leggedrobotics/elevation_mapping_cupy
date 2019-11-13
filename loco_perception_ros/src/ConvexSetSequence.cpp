/*
 * ConvexSetSequence.cpp
 *
 *  Created on: Nov 12, 2019
 *      Author: Marko Bjelonic
 */


// loco perception ros
#include "loco_perception_ros/ConvexSetSequence.hpp"

namespace loco_perception_ros {

ConvexSetSequence::ConvexSetSequence()
  :  nodeHandle_(),
     convexSetSequence_(),
     publisher_(),
     colorId_(0u)
{

}

void ConvexSetSequence::setColorVector(std::vector<loco::Vector, Eigen::aligned_allocator<loco::Vector>>& colors) {
  colors_ = colors;
}

void ConvexSetSequence::setColorVectorId(unsigned int colorId) {
  colorId_ = colorId;
}

bool ConvexSetSequence::initialize(ros::NodeHandle& nodeHandle, const std::string& topic) {
  nodeHandle_ = nodeHandle;
  publisher_ = nodeHandle.advertise<visualization_msgs::MarkerArray>(topic, 100);

  return true;
}

bool ConvexSetSequence::shutdown() {
  publisher_.shutdown();
  return true;
}

bool ConvexSetSequence::update(const std::vector<loco_perception_utils::ConvexSet>& convexSets) {
  if (convexSets.size() == 0u) {
    return true;
  }

  if (getNumSubscribers() > 0u) {
    convexSetSequence_.markers.clear();

    unsigned int id = 0u;
    unsigned int offset = 0u;

    for (const auto& convexSet : convexSets) {
      // Number of vertices should not be zero.
      if (convexSet.getNumOfVertices() <= 0u) {
        ++offset;
        continue;
      }

      visualization_msgs::Marker convexSetSequence;
      convexSetSequence.pose.orientation.w = 1.0;
      convexSetSequence.action = visualization_msgs::Marker::ADD;
      convexSetSequence.type = visualization_msgs::Marker::LINE_STRIP;
      convexSetSequence.scale.x = 0.02000;
      convexSetSequence.color.a = 1.0;
      convexSetSequence.header.frame_id = "odom";

      const unsigned int colorId = robot_utils::intmod(colorId_, colors_.size());

      convexSetSequence.color.r = colors_[colorId].x();
      convexSetSequence.color.g = colors_[colorId].y();
      convexSetSequence.color.b = colors_[colorId].z();

      convexSetSequence.id = id;
      convexSetSequence.ns = "convex_set" + std::to_string(convexSetSequence.id);

      geometry_msgs::Point point;
      convexSetSequence.points.reserve(convexSet.getConvexSetInWorldFrame().getVertices().size());
      for (const auto& vertex : convexSet.getConvexSetInWorldFrame().getVertices()) {
        const loco::Position positionWorldToVertexInWorldFrame =
            loco::Position(vertex.x(), vertex.y(), 0.0);
        point.x = positionWorldToVertexInWorldFrame.x();
        point.y = positionWorldToVertexInWorldFrame.y();
        point.z = positionWorldToVertexInWorldFrame.z();
        convexSetSequence.points.push_back(point);
      }

      // Add first point again to reconnect the polygon.
      if (convexSetSequence.points.size() > 2) {
        convexSetSequence.points.push_back(convexSetSequence.points.front());
        convexSetSequence_.markers.push_back(convexSetSequence);
      }

      ++id;
    }
  }

  return true;
}

bool ConvexSetSequence::publish() {
  loco_ros::publishMsg(publisher_, convexSetSequence_);
  return true;
}

unsigned int ConvexSetSequence::getNumSubscribers() const {
  return publisher_.getNumSubscribers();
}

} /* namespace loco */
