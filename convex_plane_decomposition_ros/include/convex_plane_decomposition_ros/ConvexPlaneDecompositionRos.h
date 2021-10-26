#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Geometry>

#include <grid_map_msgs/GridMap.h>

namespace convex_plane_decomposition {

// Forward declarations
class GridMapPreprocessing;
class Postprocessing;
namespace sliding_window_plane_extractor {
class SlidingWindowPlaneExtractor;
}
namespace contour_extraction {
class ContourExtraction;
}

class ConvexPlaneExtractionROS {
 public:
  ConvexPlaneExtractionROS(ros::NodeHandle& nodeHandle);

  ~ConvexPlaneExtractionROS();

  bool loadParameters(const ros::NodeHandle& nodeHandle);

 private:
  /**
   * Callback method for the incoming grid map message.
   * @param message the incoming message.
   */
  void callback(const grid_map_msgs::GridMap& message);

  Eigen::Isometry3d getTransformToTargetFrame(const std::string& sourceFrame, const ros::Time& time);

  ros::Time getMessageTime(const grid_map_msgs::GridMap& message) const;

  // Parameters
  std::string elevationMapTopic_;
  std::string elevationLayer_;
  std::string targetFrameId_;
  double subMapWidth_;
  double subMapLength_;
  bool publishToController_;

  // ROS communication
  ros::Subscriber elevationMapSubscriber_;
  ros::Publisher filteredmapPublisher_;
  ros::Publisher boundaryPublisher_;
  ros::Publisher insetPublisher_;
  ros::Publisher regionPublisher_;
  tf2_ros::Buffer tfBuffer_;
  tf2_ros::TransformListener tfListener_;

  // Pipeline
  std::mutex mutex_;
  std::unique_ptr<GridMapPreprocessing> preprocessing_;
  std::unique_ptr<sliding_window_plane_extractor::SlidingWindowPlaneExtractor> slidingWindowPlaneExtractor_;
  std::unique_ptr<contour_extraction::ContourExtraction> contourExtraction_;
  std::unique_ptr<Postprocessing> postprocessing_;
};

}  // namespace convex_plane_decomposition
