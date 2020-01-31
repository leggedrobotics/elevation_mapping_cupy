#pragma once

#include <string>

#include <glog/logging.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>

#include <grid_map_ros/grid_map_ros.hpp>

#include "export_utils.hpp"
#include "grid_map_preprocessing.hpp"
#include "plane_extractor.hpp"
#include "ransac_plane_extractor.hpp"

namespace convex_plane_extraction {

/*!
 * Applies a chain of grid map filters to a topic and
 * republishes the resulting grid map.
 */
class ConvexPlaneExtractionROS
{
 public:

  /*!
   * Constructor.
   * @param nodeHandle the ROS node handle.
   * @param success signalizes if filter is configured ok or not.
   */
  ConvexPlaneExtractionROS(ros::NodeHandle& nodeHandle, bool& success);

  /*!
   * Destructor.
   */
  virtual ~ConvexPlaneExtractionROS();

  /*!
  * Reads and verifies the ROS parameters.
  * @return true if successful.
  */
  bool readParameters();

  /*!
   * Callback method for the incoming grid map message.
   * @param message the incoming message.
   */
  void callback(const grid_map_msgs::GridMap& message);

 private:

  //! ROS nodehandle.
  ros::NodeHandle& nodeHandle_;

  //! Name of the input grid map topic.
  std::string inputTopic_;

  //! Name of the output grid map topic.
  std::string outputTopic_;

  //! Grid map subscriber
  ros::Subscriber subscriber_;

  ros::Publisher grid_map_publisher_;

  ros::Publisher convex_polygon_publisher_;

  ros::Publisher outer_contours_publisher_;

  ros::Publisher hole_contours_publsiher_;

  //! Ransac plane extractor parameters.
  ransac_plane_extractor::RansacParameters ransac_parameters_;

  PlaneExtractorType plane_extractor_selector_;

  sliding_window_plane_extractor::SlidingWindowParameters sliding_window_parameters_;

};

} /* namespace */
