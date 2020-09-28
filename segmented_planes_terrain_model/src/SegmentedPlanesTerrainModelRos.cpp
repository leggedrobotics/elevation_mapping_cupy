//
// Created by rgrandia on 24.06.20.
//

#include "segmented_planes_terrain_model/SegmentedPlanesTerrainModelRos.h"

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <convex_plane_decomposition_ros/MessageConversion.h>
#include <ocs2_switched_model_interface/visualization/VisualizationHelpers.h>

namespace switched_model {

SegmentedPlanesTerrainModelRos::SegmentedPlanesTerrainModelRos(ros::NodeHandle& nodehandle)
    : terrainUpdated_(false), minCoordinates_(Eigen::Vector3d::Zero()), maxCoordinates_(Eigen::Vector3d::Zero()) {
  terrainSubscriber_ =
      nodehandle.subscribe("/convex_plane_decomposition_ros/planar_terrain", 1, &SegmentedPlanesTerrainModelRos::callback, this);
  distanceFieldPublisher_ = nodehandle.advertise<sensor_msgs::PointCloud2>("/convex_plane_decomposition_ros/signed_distance_field", 1);
}

bool SegmentedPlanesTerrainModelRos::update(std::unique_ptr<SegmentedPlanesTerrainModel>& terrainPtr) {
  // Avoid locking the mutex in the common case that the terrain is not updated.
  if (terrainUpdated_) {
    std::lock_guard<std::mutex> lock(updateMutex_);
    terrainPtr = std::move(terrainPtr_);
    terrainPtr_ = nullptr;
    terrainUpdated_ = false;
    return true;
  } else {
    return false;
  }
}

void SegmentedPlanesTerrainModelRos::createSignedDistanceBetween(const Eigen::Vector3d& minCoordinates,
                                                                 const Eigen::Vector3d& maxCoordinates) {
  std::lock_guard<std::mutex> lock(updateCoordinatesMutex_);
  minCoordinates_ = minCoordinates;
  maxCoordinates_ = maxCoordinates;
  createSignedDistance_ = true;
}

void SegmentedPlanesTerrainModelRos::publish() {
  sensor_msgs::PointCloud2 pointCloud2Msg;
  {
    std::lock_guard<std::mutex> lock(pointCloudMutex_);
    pcl::toROSMsg(pointCloud_, pointCloud2Msg);
  }

  pointCloud2Msg.header = switched_model::getHeaderMsg(frameId_, ros::Time::now());
  distanceFieldPublisher_.publish(pointCloud2Msg);
}

void SegmentedPlanesTerrainModelRos::callback(const convex_plane_decomposition_msgs::PlanarTerrain::ConstPtr& msg) {
  // Read terrain
  auto terrainPtr = std::make_unique<SegmentedPlanesTerrainModel>(convex_plane_decomposition::fromMessage(*msg));

  // Extract coordinates for signed distance field
  bool createSignedDistance = false;
  Eigen::Vector3d minCoordinates;
  Eigen::Vector3d maxCoordinates;
  {
    std::lock_guard<std::mutex> lock(updateCoordinatesMutex_);
    minCoordinates = minCoordinates_;
    maxCoordinates = maxCoordinates_;
    createSignedDistance = createSignedDistance_;
  }

  if (createSignedDistance) {
    // Build signed distance field
    terrainPtr->createSignedDistanceBetween(minCoordinates_, maxCoordinates_);

    // Create pointcloud
    {
      std::lock_guard<std::mutex> lock(pointCloudMutex_);
      pointCloud_ = terrainPtr->getSignedDistanceField()->asPointCloud(1, [](float val) { return -0.05F <= val && val <= 0.0F; });
    }
  }

  // Move to storage under the lock
  {
    std::lock_guard<std::mutex> lock(updateMutex_);
    terrainPtr_ = std::move(terrainPtr);
    terrainUpdated_ = true;
  }
}

}  // namespace switched_model
