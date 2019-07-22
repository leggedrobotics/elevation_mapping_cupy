#include "elevation_mapping_cupy/elevation_mapping_ros.hpp"
#include <pybind11/embed.h> 
#include <pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/common/projection_matrix.h>
#include <tf_conversions/tf_eigen.h>
#include <ros/package.h>
#include <traversability_msgs/TraversabilityResult.h>

namespace elevation_mapping_cupy{


ElevationMappingNode::ElevationMappingNode(ros::NodeHandle& nh)
{
  nh_ = nh;
  map_.initialize(nh_);
  std::string pose_topic, map_frame;
  std::vector<std::string>pointcloud_topics;
  nh.param<std::vector<std::string>>("pointcloud_topics", pointcloud_topics, {"points"});
  nh.param<std::string>("pose_topic", pose_topic, "pose");
  nh.param<std::string>("map_frame", mapFrameId_, "map");
  poseSub_ = nh_.subscribe(pose_topic, 1, &ElevationMappingNode::poseCallback, this);
  for (const auto& pointcloud_topic: pointcloud_topics) {
    ros::Subscriber sub = nh_.subscribe(pointcloud_topic, 1, &ElevationMappingNode::pointcloudCallback, this);
    pointcloudSubs_.push_back(sub);
  }
  mapPub_ = nh_.advertise<grid_map_msgs::GridMap>("elevation_map_raw", 1);
  gridMap_.setFrameId(mapFrameId_);
  rawSubmapService_ = nh_.advertiseService("get_raw_submap", &ElevationMappingNode::getSubmap, this);
  clearMapService_ = nh_.advertiseService("clear_map", &ElevationMappingNode::clearMap, this);
  footprintPathService_ = nh_.advertiseService("check_footprint_path", &ElevationMappingNode::checkFootprintPath, this);
  ROS_INFO("[ElevationMappingCupy] finish initialization");
}


void ElevationMappingNode::pointcloudCallback(const sensor_msgs::PointCloud2& cloud)
{
  auto start = ros::Time::now();
  pcl::PCLPointCloud2 pcl_pc;
  pcl_conversions::toPCL(cloud, pcl_pc);

  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(pcl_pc, *pointCloud);
  tf::StampedTransform transformTf;
  std::string sensorFrameId = cloud.header.frame_id;
  auto timeStamp = cloud.header.stamp;
  Eigen::Affine3d transformationSensorToMap;
  try {
    transformListener_.waitForTransform(mapFrameId_, sensorFrameId, timeStamp, ros::Duration(1.0));
    transformListener_.lookupTransform(mapFrameId_, sensorFrameId, timeStamp, transformTf);
    poseTFToEigen(transformTf, transformationSensorToMap);
  }
  catch (tf::TransformException &ex) {
    ROS_ERROR("%s", ex.what());
    return;
  }
  map_.input(pointCloud,
             transformationSensorToMap.rotation(),
             transformationSensorToMap.translation());
  map_.get_grid_map(gridMap_);
  gridMap_.setTimestamp(ros::Time::now().toSec());
  grid_map_msgs::GridMap msg;
  grid_map::GridMapRosConverter::toMessage(gridMap_, msg);
  mapPub_.publish(msg);

  ROS_INFO_THROTTLE(1.0, "ElevationMap processed a point cloud (%i points) in %f sec.", static_cast<int>(pointCloud->size()), (ros::Time::now() - start).toSec());
}

void ElevationMappingNode::poseCallback(const geometry_msgs::PoseWithCovarianceStamped& pose)
{
  Eigen::Vector2d position(pose.pose.pose.position.x, pose.pose.pose.position.y);
  map_.move_to(position);
}

bool ElevationMappingNode::getSubmap(grid_map_msgs::GetGridMap::Request& request, grid_map_msgs::GetGridMap::Response& response)
{
  grid_map::Position requestedSubmapPosition(request.position_x, request.position_y);
  grid_map::Length requestedSubmapLength(request.length_x, request.length_y);
  ROS_DEBUG("Elevation submap request: Position x=%f, y=%f, Length x=%f, y=%f.", requestedSubmapPosition.x(), requestedSubmapPosition.y(), requestedSubmapLength(0), requestedSubmapLength(1));

  bool isSuccess;
  grid_map::Index index;
  grid_map::GridMap subMap = gridMap_.getSubmap(requestedSubmapPosition, requestedSubmapLength, index, isSuccess);

  if (request.layers.empty()) {
    grid_map::GridMapRosConverter::toMessage(subMap, response.map);
  } else {
    std::vector<std::string> layers;
    for (const auto& layer : request.layers) {
      layers.push_back(layer);
    }
    grid_map::GridMapRosConverter::toMessage(subMap, layers, response.map);
  }
  return isSuccess;
}

bool ElevationMappingNode::clearMap(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
  ROS_INFO("Clearing map.");
  map_.clear();
  return true;
}

bool ElevationMappingNode::checkFootprintPath(traversability_msgs::CheckFootprintPath::Request& request, traversability_msgs::CheckFootprintPath::Response& response) {

  for (auto& path_elem: request.path) {
    std::vector<Eigen::Vector2d> polygon;
    Eigen::Vector3d result;
    for (auto& p: path_elem.footprint.polygon.points) {
      polygon.push_back(Eigen::Vector2d(p.x, p.y));
    }
    double traversability = map_.get_polygon_traversability(polygon, result);
    traversability_msgs::TraversabilityResult traversability_result;
    traversability_result.is_safe = bool(result[0] > 0.5);
    traversability_result.traversability = result[1];
    traversability_result.area = result[2];
    response.result.push_back(traversability_result);
  }
  return true;
}

}
