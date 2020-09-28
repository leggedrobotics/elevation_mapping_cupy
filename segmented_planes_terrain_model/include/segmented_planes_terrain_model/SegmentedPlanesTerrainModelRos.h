//
// Created by rgrandia on 24.06.20.
//

#pragma once

#include <mutex>

#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <convex_plane_decomposition_msgs/PlanarTerrain.h>

#include "SegmentedPlanesTerrainModel.h"

namespace switched_model {

class SegmentedPlanesTerrainModelRos {
 public:
  std::string frameId_ = "odom";

  SegmentedPlanesTerrainModelRos(ros::NodeHandle& nodehandle);

  /// Updates the terrain if a new one is available. Return if an update was made
  bool update(std::unique_ptr<SegmentedPlanesTerrainModel>& terrainPtr);

  void createSignedDistanceBetween(const Eigen::Vector3d& minCoordinates, const Eigen::Vector3d& maxCoordinates);

  void publish();

 private:
  void callback(const convex_plane_decomposition_msgs::PlanarTerrain::ConstPtr& msg);

  ros::Subscriber terrainSubscriber_;
  ros::Publisher distanceFieldPublisher_;

  std::mutex updateMutex_;
  std::atomic_bool terrainUpdated_;
  std::unique_ptr<SegmentedPlanesTerrainModel> terrainPtr_;

  std::mutex updateCoordinatesMutex_;
  Eigen::Vector3d minCoordinates_;
  Eigen::Vector3d maxCoordinates_;
  bool createSignedDistance_ = false;

  std::mutex pointCloudMutex_;
  pcl::PointCloud<pcl::PointXYZI> pointCloud_;
};

}  // namespace switched_model
