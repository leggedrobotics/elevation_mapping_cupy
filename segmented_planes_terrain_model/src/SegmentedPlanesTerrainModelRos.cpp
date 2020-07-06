//
// Created by rgrandia on 24.06.20.
//

#include "segmented_planes_terrain_model/SegmentedPlanesTerrainModelRos.h"

#include <convex_plane_decomposition_ros/MessageConversion.h>

namespace switched_model {

SegmentedPlanesTerrainModelRos::SegmentedPlanesTerrainModelRos(ros::NodeHandle& nodehandle) : terrainUpdated_(false) {
  terrainSubscriber_ =
      nodehandle.subscribe("/convex_plane_decomposition_ros/planar_terrain", 1, &SegmentedPlanesTerrainModelRos::callback, this);
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

void SegmentedPlanesTerrainModelRos::callback(const convex_plane_decomposition_msgs::PlanarTerrain::ConstPtr& msg) {
  std::lock_guard<std::mutex> lock(updateMutex_);
  terrainPtr_ = std::make_unique<SegmentedPlanesTerrainModel>(convex_plane_decomposition::fromMessage(*msg));
  terrainUpdated_ = true;
}

}  // namespace switched_model
