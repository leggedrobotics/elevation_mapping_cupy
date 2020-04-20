#include "convex_plane_extraction_ros.hpp"
#include <ros/ros.h>

#include <glog/logging.h>

int main(int argc, char** argv)
{
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_v = 1;
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "convex_plane_extraction_ros");
  ros::NodeHandle nodeHandle("~");
  bool success;
  convex_plane_extraction::ConvexPlaneExtractionROS convex_plane_extraction_ros(nodeHandle, success);
  if (success) ros::spin();
  return 0;
}
