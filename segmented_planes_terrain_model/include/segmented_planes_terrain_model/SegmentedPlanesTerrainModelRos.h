//
// Created by rgrandia on 24.06.20.
//

#pragma once

#include <mutex>

#include <ros/ros.h>

#include <convex_plane_decomposition_msgs/PlanarTerrain.h>

#include "SegmentedPlanesTerrainModel.h"

namespace switched_model {

class SegmentedPlanesTerrainModelRos {
 public:
  SegmentedPlanesTerrainModelRos(ros::NodeHandle& nodehandle);

  /// Updates the terrain if a new one is available. Return if an update was made
  bool update(std::unique_ptr<SegmentedPlanesTerrainModel>& terrainPtr);

 private:
  void callback(const convex_plane_decomposition_msgs::PlanarTerrain::ConstPtr& msg);

  ros::Subscriber terrainSubscriber_;
  std::mutex updateMutex_;
  std::unique_ptr<SegmentedPlanesTerrainModel> terrainPtr_;
};

}  // namespace switched_model
