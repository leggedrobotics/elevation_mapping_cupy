#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"
#include <pybind11/embed.h> 
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
    param.attr("load_weights")("/home/tamiki/catkin_ws/src/elevation_mapping_cupy/config/weights.yaml");
    map_ = elevation_mapping.attr("ElevationMap")(param);
    std::cout << "wrapper " << std::endl;
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


int ElevationMappingWrapper::get_maps(std::vector<Eigen::MatrixXd>& maps) {
  return 0;
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
  // std::cout << points << std::endl;
  return;
}


ElevationMappingNode::ElevationMappingNode(ros::NodeHandle& nh)
{
  nh_ = nh;
  poseSub_ = nh_.subscribe("/state_estimator/pose_in_odom", 1, &ElevationMappingNode::poseCallback, this);
  pointcloudSub_ = nh_.subscribe("/realsense_d435_front/depth/color/points", 1, &ElevationMappingNode::pointcloudCallback, this);
  // map_ = ElevationMappingWrapper();
  ROS_INFO("finish initialization");
  // ros::spin();
}


void ElevationMappingNode::pointcloudCallback(const sensor_msgs::PointCloud2& cloud)
{
  std::cout << "got point" << std::endl;
  pcl::PCLPointCloud2 pcl_pc;
  pcl_conversions::toPCL(cloud, pcl_pc);

  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(pcl_pc, *pointCloud);
  RowMatrixXd matrix;
  map_.pointCloudToMatrix(pointCloud, matrix);
  tf::StampedTransform transformTf;
  std::string sensorFrameId = cloud.header.frame_id;
  auto timeStamp = cloud.header.stamp;
  transformListener_.lookupTransform(mapFrameId_, sensorFrameId, timeStamp, transformTf);
  Eigen::Affine3d transformationSensorToMap;
  poseTFToEigen(transformTf, transformationSensorToMap);
  // std::cout << transformTf << std::endl;
  std::cout << transformationSensorToMap.rotation() << std::endl;
  std::cout << transformationSensorToMap.translation() << std::endl;
  // auto const m = pointCloud.getMatrixXfMap();
  // std::cout << m.rows() << " x " << m.cols() << std::endl;
  // std::cout << m << std::endl;

  ROS_INFO("ElevationMap received a point cloud (%i points) for elevation mapping.", static_cast<int>(pointCloud->size()));
}

void ElevationMappingNode::poseCallback(const geometry_msgs::PoseWithCovarianceStamped& pose)
{
  ROS_INFO("got pose callback");
}

}
