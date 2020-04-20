#ifndef CONVEX_PLANE_EXTRACTION_ROS_INCLUDE_CONVEX_PLANE_EXTRACTION_ROS_HPP_
#define CONVEX_PLANE_EXTRACTION_ROS_INCLUDE_CONVEX_PLANE_EXTRACTION_ROS_HPP_

#include <string>

#include <glog/logging.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

#include <grid_map_ros/grid_map_ros.hpp>

#include "grid_map_preprocessing.hpp"
#include "pipeline_ros.hpp"


namespace convex_plane_decomposition {

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

};

} /* namespace */
#endif