#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
// Grid Map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"

using namespace elevation_mapping_cupy;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "elevation_mapping");
  ros::NodeHandle nh("~");

  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  ElevationMappingNode mapNode(nh);

  // Spin
  ros::AsyncSpinner spinner(1); // Use n threads
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
