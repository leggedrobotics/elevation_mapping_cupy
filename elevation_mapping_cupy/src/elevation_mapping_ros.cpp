#include "elevation_mapping_cupy/elevation_mapping_ros.hpp"
#include <pybind11_catkin/pybind11/embed.h> 
#include <pybind11_catkin/pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/common/projection_matrix.h>
#include <tf_conversions/tf_eigen.h>
#include <ros/package.h>
// #include <traversability_msgs/TraversabilityResult.h>
#include <geometry_msgs/Point32.h>
#include <elevation_map_msgs/CheckSafety.h>

namespace elevation_mapping_cupy{


ElevationMappingNode::ElevationMappingNode(ros::NodeHandle& nh) :
  lowpassPosition_(0, 0, 0),
  lowpassOrientation_(0, 0, 0, 1),
  positionError_(0),
  orientationError_(0),
  positionAlpha_(0.1),
  orientationAlpha_(0.1),
  enablePointCloudPublishing_(false),
  recordableFps_(0.0)
{
  nh_ = nh;
  map_.initialize(nh_);
  std::string pose_topic, map_frame;
  std::vector<std::string>pointcloud_topics;
  nh.param<std::vector<std::string>>("pointcloud_topics", pointcloud_topics, {"points"});
  nh.param<std::vector<std::string>>("recordable_map_layers", recordable_map_layers_, {"elevation"});
  nh.param<std::string>("pose_topic", pose_topic, "pose");
  nh.param<std::string>("map_frame", mapFrameId_, "map");
  nh.param<double>("position_lowpass_alpha", positionAlpha_, 0.2);
  nh.param<double>("orientation_lowpass_alpha", orientationAlpha_, 0.2);
  nh.param<double>("recordable_fps", recordableFps_, 3.0);
  nh.param<bool>("enable_pointcloud_publishing", enablePointCloudPublishing_, false);
  poseSub_ = nh_.subscribe(pose_topic, 1, &ElevationMappingNode::poseCallback, this);
  for (const auto& pointcloud_topic: pointcloud_topics) {
    ros::Subscriber sub = nh_.subscribe(pointcloud_topic, 1, &ElevationMappingNode::pointcloudCallback, this);
    pointcloudSubs_.push_back(sub);
  }
  mapPub_ = nh_.advertise<grid_map_msgs::GridMap>("elevation_map_raw", 1);
  recordablePub_ = nh_.advertise<grid_map_msgs::GridMap>("elevation_map_recordable", 1);
  pointPub_ = nh_.advertise<sensor_msgs::PointCloud2>("elevation_map_points", 1);
  alivePub_ = nh_.advertise<std_msgs::Empty>("alive", 1);
  polygonPub_ = nh_.advertise<geometry_msgs::PolygonStamped>("foot_print", 1);
  gridMap_.setFrameId(mapFrameId_);
  rawSubmapService_ = nh_.advertiseService("get_raw_submap", &ElevationMappingNode::getSubmap, this);
  clearMapService_ = nh_.advertiseService("clear_map", &ElevationMappingNode::clearMap, this);
  setPublishPointService_ = nh_.advertiseService("set_publish_points", &ElevationMappingNode::setPublishPoint, this);
  checkSafetyService_ = nh_.advertiseService("check_safety", &ElevationMappingNode::checkSafety, this);
  if (recordableFps_ > 0) {
    double duration = 1.0 / (recordableFps_ + 0.00001);
    recordableTimer_= nh_.createTimer(ros::Duration(duration),
                                      &ElevationMappingNode::publishRecordableMap, this, false, true);
  }
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
             transformationSensorToMap.translation(),
             positionError_,
             orientationError_);
  boost::recursive_mutex::scoped_lock scopedLockForGridMap(mapMutex_);
  map_.get_grid_map(gridMap_);
  gridMap_.setTimestamp(ros::Time::now().toNSec());
  grid_map_msgs::GridMap msg;
  grid_map::GridMapRosConverter::toMessage(gridMap_, msg);
  scopedLockForGridMap.unlock();
  mapPub_.publish(msg);
  alivePub_.publish(std_msgs::Empty());

  if (enablePointCloudPublishing_) {
    publishAsPointCloud();
  }

  ROS_DEBUG_THROTTLE(1.0, "ElevationMap processed a point cloud (%i points) in %f sec.", static_cast<int>(pointCloud->size()), (ros::Time::now() - start).toSec());
  ROS_DEBUG_THROTTLE(1.0, "positionError: %f ", positionError_);
  ROS_DEBUG_THROTTLE(1.0, "orientationError: %f ", orientationError_);
}

void ElevationMappingNode::poseCallback(const geometry_msgs::PoseWithCovarianceStamped& pose)
{
  Eigen::Vector2d position(pose.pose.pose.position.x, pose.pose.pose.position.y);
  map_.move_to(position);
  Eigen::Vector3d position3(pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z);
  Eigen::Vector4d orientation(pose.pose.pose.orientation.x, pose.pose.pose.orientation.y,
                              pose.pose.pose.orientation.z, pose.pose.pose.orientation.w);
  lowpassPosition_ = positionAlpha_ * position3 + (1 - positionAlpha_) * lowpassPosition_;
  lowpassOrientation_ = orientationAlpha_ * orientation + (1 - orientationAlpha_) * lowpassOrientation_;
  positionError_ = (position3 - lowpassPosition_).norm();
  orientationError_ = (orientation - lowpassOrientation_).norm();
}

void ElevationMappingNode::publishAsPointCloud() {
  sensor_msgs::PointCloud2 msg;
  grid_map::GridMapRosConverter::toPointCloud(gridMap_, "elevation", msg);
  pointPub_.publish(msg);
}

bool ElevationMappingNode::getSubmap(grid_map_msgs::GetGridMap::Request& request, grid_map_msgs::GetGridMap::Response& response)
{
  grid_map::Position requestedSubmapPosition(request.position_x, request.position_y);
  grid_map::Length requestedSubmapLength(request.length_x, request.length_y);
  ROS_DEBUG("Elevation submap request: Position x=%f, y=%f, Length x=%f, y=%f.", requestedSubmapPosition.x(), requestedSubmapPosition.y(), requestedSubmapLength(0), requestedSubmapLength(1));

  bool isSuccess;
  grid_map::Index index;
  grid_map::GridMap subMap = gridMap_.getSubmap(requestedSubmapPosition, requestedSubmapLength, index, isSuccess);
  const auto& length = subMap.getLength();
  Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(length(0), length(1));
  subMap.add("horizontal_variance_x", zero.cast<float>());
  subMap.add("horizontal_variance_y", zero.cast<float>());
  subMap.add("horizontal_variance_xy", zero.cast<float>());
  subMap.add("time", zero.cast<float>());
  subMap.add("color", zero.cast<float>());
  subMap.add("lowest_scan_point", zero.cast<float>());
  subMap.add("sensor_x_at_lowest_scan", zero.cast<float>());
  subMap.add("sensor_y_at_lowest_scan", zero.cast<float>());
  subMap.add("sensor_z_at_lowest_scan", zero.cast<float>());

  if (request.layers.empty()) {
    grid_map::GridMapRosConverter::toMessage(subMap, response.map);
  }
  else {
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

bool ElevationMappingNode::checkSafety(elevation_map_msgs::CheckSafety::Request& request,
                                       elevation_map_msgs::CheckSafety::Response& response) {

  for (auto& polygonstamped: request.polygons) {
    std::vector<Eigen::Vector2d> polygon;
    std::vector<Eigen::Vector2d> untraversable_polygon;
    Eigen::Vector3d result;
    result.setZero();
    const auto& polygonFrameId = polygonstamped.header.frame_id;
    const auto& timeStamp = polygonstamped.header.stamp;
    double polygon_z = polygonstamped.polygon.points[0].z;

    // Get tf from map frame to polygon frame
    if (mapFrameId_ != polygonFrameId) {
      Eigen::Affine3d transformationBaseToMap;
      tf::StampedTransform transformTf;
      try {
        transformListener_.waitForTransform(mapFrameId_, polygonFrameId, timeStamp, ros::Duration(1.0));
        transformListener_.lookupTransform(mapFrameId_, polygonFrameId, timeStamp, transformTf);
        poseTFToEigen(transformTf, transformationBaseToMap);
      }
      catch (tf::TransformException &ex) {
        ROS_ERROR("%s", ex.what());
        return false;
      }
      for (const auto& p: polygonstamped.polygon.points) {
        const auto& pvector = Eigen::Vector3d(p.x, p.y, p.z);
        const auto transformed_p = transformationBaseToMap * pvector;
        polygon.push_back(Eigen::Vector2d(transformed_p.x(), transformed_p.y()));
      }
    }
    else {
      for (const auto& p: polygonstamped.polygon.points) {
        polygon.push_back(Eigen::Vector2d(p.x, p.y));
      }
    }

    map_.get_polygon_traversability(polygon, result, untraversable_polygon);

    geometry_msgs::PolygonStamped untraversable_polygonstamped;
    untraversable_polygonstamped.header.stamp = ros::Time::now();
    untraversable_polygonstamped.header.frame_id = mapFrameId_;
    for (const auto& p: untraversable_polygon) {
      geometry_msgs::Point32 point;
      point.x = p.x();
      point.y = p.y();
      point.z = polygon_z;
      untraversable_polygonstamped.polygon.points.push_back(point);
    }
    // elevation_map_msgs::C traversability_result;
    response.is_safe.push_back(bool(result[0] > 0.5));
    response.traversability.push_back(result[1]);
    response.untraversable_polygons.push_back(untraversable_polygonstamped);
  }
  return true;
}

bool ElevationMappingNode::setPublishPoint(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response) {
  enablePointCloudPublishing_ = request.data;
  response.success = true;
  return true;
}


void ElevationMappingNode::publishRecordableMap(const ros::TimerEvent&) {
  grid_map_msgs::GridMap msg;
  std::vector<std::string> layers;
  for (const auto& layer: recordable_map_layers_) {
    if (gridMap_.exists(layer)) {
      layers.push_back(layer);
    }
  }
  if (layers.size() == 0) 
    return;
  boost::recursive_mutex::scoped_lock scopedLockForGridMap(mapMutex_);
  grid_map::GridMapRosConverter::toMessage(gridMap_, layers, msg);
  scopedLockForGridMap.unlock();
  msg.basic_layers = layers;
  recordablePub_.publish(msg);
  return;
}

}
