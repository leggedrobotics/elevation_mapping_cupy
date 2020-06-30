#pragma once

#include <string>
#include <memory>
#include <mutex>

#include <ros/ros.h>

#include <grid_map_msgs/GridMap.h>

namespace convex_plane_decomposition {

// Forward declarations
class GridMapPreprocessing;
namespace sliding_window_plane_extractor {
class SlidingWindowPlaneExtractor;
}
namespace  contour_extraction {
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

  // Parameters
  std::string elevationMapTopic_;
  std::string elevationLayer_;
  double subMapWidth_;
  double subMapLength_;

  // ROS communication
  ros::Subscriber elevationMapSubscriber_;
  ros::Publisher filteredmapPublisher_;
  ros::Publisher boundaryPublisher_;
  ros::Publisher insetPublisher_;
  ros::Publisher regionPublisher_;

  // Pipeline
  std::mutex mutex_;
  std::unique_ptr<GridMapPreprocessing> preprocessing_;
  std::unique_ptr<sliding_window_plane_extractor::SlidingWindowPlaneExtractor> slidingWindowPlaneExtractor_;
  std::unique_ptr<contour_extraction::ContourExtraction> contourExtraction_;
};

}  // namespace convex_plane_decomposition
