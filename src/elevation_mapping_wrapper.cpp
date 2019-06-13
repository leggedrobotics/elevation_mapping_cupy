#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"
#include <pybind11/embed.h> 
#include <pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/common/projection_matrix.h>
#include <tf_conversions/tf_eigen.h>

namespace elevation_mapping_cupy{
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

ElevationMappingWrapper::ElevationMappingWrapper() {
    auto elevation_mapping = py::module::import("elevation_mapping");
    py::object param = elevation_mapping.attr("Parameter")();
    // param.attr("use_cupy") = false;
    param.attr("load_weights")("/home/takahiro/catkin_ws/src/elevation_mapping_cupy/config/weights.yaml");
    map_ = elevation_mapping.attr("ElevationMap")(param);
    resolution_ = map_.attr("get_resolution")().cast<double>();
    map_length_ = map_.attr("get_length")().cast<double>();
}


void ElevationMappingWrapper::input(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, const RowMatrixXd& R, const Eigen::VectorXd& t) {
  
  RowMatrixXd points;
  pointCloudToMatrix(pointCloud, points);
  map_.attr("input")(static_cast<Eigen::Ref<const RowMatrixXd>>(points),
                     static_cast<Eigen::Ref<const RowMatrixXd>>(R),
                     static_cast<Eigen::Ref<const Eigen::VectorXd>>(t));
}


void ElevationMappingWrapper::move_to(const Eigen::VectorXd& p) {
  map_.attr("move_to")(static_cast<Eigen::Ref<const Eigen::VectorXd>>(p));
}


void ElevationMappingWrapper::get_maps(std::vector<Eigen::MatrixXd>& maps) {

  RowMatrixXd elevation(500, 500);
  RowMatrixXd variance(500, 500);
  RowMatrixXd traversability(500, 500);
  map_.attr("get_maps_ref")(static_cast<Eigen::Ref<RowMatrixXd>>(elevation),
                            static_cast<Eigen::Ref<RowMatrixXd>>(variance),
                            static_cast<Eigen::Ref<RowMatrixXd>>(traversability));
  maps.clear();
  maps.push_back(elevation);
  maps.push_back(variance);
  maps.push_back(traversability);
  return;
}


void ElevationMappingWrapper::get_grid_map(grid_map::GridMap& gridMap) {
  RowMatrixXd pos(1, 2);
  map_.attr("get_position")(static_cast<Eigen::Ref<RowMatrixXd>>(pos));

  grid_map::Position position(pos(0, 1), pos(0, 0));
  grid_map::Length length(map_length_, map_length_);
  gridMap.setGeometry(length, resolution_, position);
  std::vector<Eigen::MatrixXd> maps;
  get_maps(maps);
  std::vector<std::string> layerNames = {"elevation", "variance", "traversability"};
  for(int i = 0; i < maps.size() ; ++i) {
    gridMap.add(layerNames[i], maps[i].cast<float>());
  }
}



void ElevationMappingWrapper::pointCloudToMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud,
                                                 RowMatrixXd& points)
{
  points = RowMatrixXd(pointCloud->size(), 3);
  for (unsigned int i = 0; i < pointCloud->size(); ++i) {
    const auto& point = pointCloud->points[i];
    points(i, 0) = static_cast<double>(point.x);
    points(i, 1) = static_cast<double>(point.y);
    points(i, 2) = static_cast<double>(point.z);
  }
  return;
}


ElevationMappingNode::ElevationMappingNode(ros::NodeHandle& nh)
{
  nh_ = nh;
  poseSub_ = nh_.subscribe("/state_estimator/pose_in_odom", 1, &ElevationMappingNode::poseCallback, this);
  pointcloudSub_ = nh_.subscribe("/realsense_d435_front/depth/color/points", 1, &ElevationMappingNode::pointcloudCallback, this);
  mapPub_ = nh_.advertise<grid_map_msgs::GridMap>("elevation_map", 1);
  mapFrameId_ = "map";
  gridMap_.setFrameId("map");
  ROS_INFO("finish initialization");
}


void ElevationMappingNode::pointcloudCallback(const sensor_msgs::PointCloud2& cloud)
{
  std::cout << "got point" << std::endl;
  auto start = ros::Time::now();
  pcl::PCLPointCloud2 pcl_pc;
  pcl_conversions::toPCL(cloud, pcl_pc);

  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(pcl_pc, *pointCloud);
  tf::StampedTransform transformTf;
  // std::string sensorFrameId = cloud.header.frame_id;
  std::string sensorFrameId = "ghost_desired/realsense_d435_front_depth_optical_frame";
  auto timeStamp = cloud.header.stamp;
  ROS_INFO_STREAM("sensorFrameId " << sensorFrameId);
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
  ROS_INFO_STREAM("input " << ros::Time::now() - start);

  start = ros::Time::now();
  map_.get_grid_map(gridMap_);
  gridMap_.setTimestamp(ros::Time::now().toSec());
  grid_map_msgs::GridMap msg;
  grid_map::GridMapRosConverter::toMessage(gridMap_, msg);
  ROS_INFO_STREAM("get " << ros::Time::now() - start);
  start = ros::Time::now();
  mapPub_.publish(msg);
  ROS_INFO_STREAM("publish " << ros::Time::now() - start);

  ROS_INFO("ElevationMap received a point cloud (%i points) for elevation mapping.", static_cast<int>(pointCloud->size()));
}

void ElevationMappingNode::poseCallback(const geometry_msgs::PoseWithCovarianceStamped& pose)
{
  Eigen::Vector2d position(pose.pose.pose.position.x, pose.pose.pose.position.y);
  map_.move_to(position);
}

}
