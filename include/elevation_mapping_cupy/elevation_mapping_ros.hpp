#include <pybind11/embed.h> // everything needed for embedding
#include <iostream>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_srvs/Empty.h>
#include <std_srvs/SetBool.h>
#include <tf/transform_listener.h>
// Grid Map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_msgs/GetGridMap.h>
// PCL
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"

namespace py = pybind11;


namespace elevation_mapping_cupy{


class ElevationMappingNode {
  public:
    ElevationMappingNode(ros::NodeHandle& nh);
    ~ElevationMappingNode() = default;

  private:
    void readParameters();
    void pointcloudCallback(const sensor_msgs::PointCloud2& cloud);
    void poseCallback(const geometry_msgs::PoseWithCovarianceStamped& pose);
    void publishAsPointCloud();
    bool getSubmap(grid_map_msgs::GetGridMap::Request& request, grid_map_msgs::GetGridMap::Response& response);
    bool clearMap(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
    bool setPublishPoint(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response);
    ros::NodeHandle nh_;
    std::vector<ros::Subscriber> pointcloudSubs_;
    ros::Subscriber poseSub_;
    ros::Publisher mapPub_;
    ros::Publisher pointPub_;
    ros::ServiceServer rawSubmapService_;
    ros::ServiceServer clearMapService_;
    ros::ServiceServer setPublishPointService_;
    tf::TransformListener transformListener_;
    ElevationMappingWrapper map_;
    std::string mapFrameId_;
    grid_map::GridMap gridMap_;

    Eigen::Vector3d lowpassPosition_;
    Eigen::Vector4d lowpassOrientation_;
    double positionError_;
    double alpha_;
    bool enablePointCloudPublishing_;
};

}
