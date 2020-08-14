//
// Created by rgrandia on 24.06.20.
//

#include "segmented_planes_terrain_model/SegmentedPlanesTerrainModelRos.h"

#include <chrono>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <grid_map_msgs/GridMap.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <signed_distance_field/GridmapSignedDistanceField.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_sdf/SignedDistanceField.hpp>

#include <ocs2_quadruped_interface/QuadrupedVisualizationHelpers.h>

#include "segmented_planes_terrain_model/SegmentedPlanesTerrainVisualization.h"

const std::string frameId_ = "odom";
std::unique_ptr<grid_map::GridMap> messageMap;

visualization_msgs::MarkerArray toMarker(const switched_model::ConvexTerrain& convexTerrain) {
  visualization_msgs::MarkerArray markerArray =
      switched_model::getConvexTerrainMarkers(convexTerrain, switched_model::Color::orange, 0.02, 0.005, 0.1);

  // Add headers and Id
  const ros::Time timeStamp = ros::Time::now();
  switched_model::assignHeader(markerArray.markers.begin(), markerArray.markers.end(),
                               switched_model::getHeaderMsg(frameId_, timeStamp));
  switched_model::assignIncreasingId(markerArray.markers.begin(), markerArray.markers.end());

  return markerArray;
}

visualization_msgs::Marker toMarker(const switched_model::vector3_t& position) {
  auto marker = switched_model::getSphereMsg(position, switched_model::Color::green, 0.02);

  const ros::Time timeStamp = ros::Time::now();
  marker.header = switched_model::getHeaderMsg(frameId_, timeStamp);
  return marker;
}

void elevationMappingCallback(const grid_map_msgs::GridMap& message) {
  if (!messageMap) {
    messageMap = std::make_unique<grid_map::GridMap>();
  }
  grid_map::GridMapRosConverter::fromMessage(message, *messageMap, {"elevation"}, false, false);
}

float randomFloat(float a, float b) {
  float random = ((float)rand()) / (float)RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "segmented_planes_demo_node");
  ros::NodeHandle nodeHandle("~");

  // Subscription to terrain
  switched_model::SegmentedPlanesTerrainModelRos segmentedPlanesTerrainModelRos(nodeHandle);
  std::unique_ptr<switched_model::SegmentedPlanesTerrainModel> terrainModel;

  // Elevation subscription

  // Publishers for visualization
  auto positionPublisher_ = nodeHandle.advertise<visualization_msgs::Marker>("queryPosition", 1);
  auto convexTerrainPublisher_ = nodeHandle.advertise<visualization_msgs::MarkerArray>("convex_terrain", 1);
  auto regionBoundaryPublisher_ = nodeHandle.advertise<visualization_msgs::MarkerArray>("regionBoundary", 1);
  auto distanceFieldPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("signed_distance_field", 1);
  auto elevationMapSubscriber = nodeHandle.subscribe("/convex_plane_decomposition_ros/filtered_map", 1, &elevationMappingCallback);

  // Node loop
  ros::Rate rate(1./5.);
  while (ros::ok()) {
    if (segmentedPlanesTerrainModelRos.update(terrainModel)) {
      ROS_INFO("Terrain model updated!!");
    }

    if (terrainModel && messageMap) {
      switched_model::vector3_t positionInWorld{randomFloat(-2.0, 2.0), randomFloat(-2.0, 2.0), randomFloat(0.0, 1.0)};
      positionPublisher_.publish(toMarker(positionInWorld));
      std::cout << "Query position: " << positionInWorld.transpose() << std::endl;

      auto t0 = std::chrono::high_resolution_clock::now();
      const auto convexTerrain = terrainModel->getConvexTerrainAtPositionInWorld(positionInWorld);
      auto t1 = std::chrono::high_resolution_clock::now();
      std::cout << "Query took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " [ms]\n";

      positionPublisher_.publish(toMarker(positionInWorld));
      convexTerrainPublisher_.publish(toMarker(convexTerrain));

      double heightClearance = 0.35;
      double width = 1.5;
      double length = 2.0;
      bool success;
      grid_map::GridMap localMap = messageMap->getSubmap({convexTerrain.plane.positionInWorld.x(), convexTerrain.plane.positionInWorld.y()}, Eigen::Array2d(width, length), success);
      auto t2 = std::chrono::high_resolution_clock::now();
      signed_distance_field::GridmapSignedDistanceField sdf(localMap, "elevation", convexTerrain.plane.positionInWorld.z() - heightClearance, convexTerrain.plane.positionInWorld.z() + heightClearance);
      auto t3 = std::chrono::high_resolution_clock::now();
      std::cout << "Sdf computation took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << " [ms]\n";

      auto t4 = std::chrono::high_resolution_clock::now();
      auto sdfClone = std::unique_ptr<signed_distance_field::GridmapSignedDistanceField>(sdf.clone());
      auto t5 = std::chrono::high_resolution_clock::now();
      std::cout << "Sdf.clone() computation took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() << " [ms]\n";

      sensor_msgs::PointCloud2 pointCloud2Msg;
      pcl::toROSMsg(sdf.obstaclePointCloud(4), pointCloud2Msg);
      pointCloud2Msg.header = switched_model::getHeaderMsg(frameId_, ros::Time::now());

      distanceFieldPublisher.publish(pointCloud2Msg);
    }

    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}