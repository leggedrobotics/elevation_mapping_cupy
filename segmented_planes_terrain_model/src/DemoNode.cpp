//
// Created by rgrandia on 24.06.20.
//

#include "segmented_planes_terrain_model/SegmentedPlanesTerrainModelRos.h"

#include <chrono>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <ocs2_quadruped_interface/QuadrupedVisualizationHelpers.h>

#include <convex_plane_decomposition/ConvexRegionGrowing.h>
#include <convex_plane_decomposition/GeometryUtils.h>

const std::string originFrameId_ = "map";

visualization_msgs::MarkerArray toMarker(const switched_model::ConvexTerrain& convexTerrain) {
  visualization_msgs::MarkerArray markerArray;

  // Polygon message
  std::vector<geometry_msgs::Point> boundary;
  boundary.reserve(convexTerrain.boundary.size() + 1);
  for (const auto& point : convexTerrain.boundary) {
    const auto& pointInWorldFrame = positionInWorldFrameFromPositionInTerrain({point.x(), point.y(), 0.0}, convexTerrain.plane);
    boundary.emplace_back(switched_model::getPointMsg(pointInWorldFrame));
  }
  // Close the polygon
  const auto& pointInWorldFrame = positionInWorldFrameFromPositionInTerrain({convexTerrain.boundary.front().x(), convexTerrain.boundary.front().y(), 0.0}, convexTerrain.plane);
  boundary.emplace_back(switched_model::getPointMsg(pointInWorldFrame));

  markerArray.markers.emplace_back(switched_model::getSphereMsg(convexTerrain.plane.positionInWorld, switched_model::Color::orange, 0.02));
  markerArray.markers.emplace_back(switched_model::getLineMsg(std::move(boundary), switched_model::Color::orange, 0.005));

  // Add headers and Id
  const ros::Time timeStamp = ros::Time::now();
  switched_model::assignHeader(markerArray.markers.begin(), markerArray.markers.end(), switched_model::getHeaderMsg(originFrameId_, timeStamp));
  switched_model::assignIncreasingId(markerArray.markers.begin(), markerArray.markers.end());

  return markerArray;
}

visualization_msgs::Marker toMarker(const switched_model::vector3_t& position) {
  auto marker = switched_model::getSphereMsg(position, switched_model::Color::green, 0.02);

  const ros::Time timeStamp = ros::Time::now();
  marker.header =  switched_model::getHeaderMsg(originFrameId_, timeStamp);
  return marker;
}

switched_model::ConvexTerrain toConvexTerrain(const convex_plane_decomposition::CgalPoint2d& centerInTerrainFrame, const convex_plane_decomposition::CgalPolygon2d& polygon, const switched_model::TerrainPlane& plane) {
  // Return convex region with origin at the seedpoint
  switched_model::ConvexTerrain convexTerrain;
  const auto& seedpointInWorldFrame = positionInWorldFrameFromPositionInTerrain({centerInTerrainFrame.x(), centerInTerrainFrame.y(), 0.0}, plane);
  convexTerrain.plane = {seedpointInWorldFrame, plane.orientationWorldToTerrain}; // Origin is at the seedpoint
  convexTerrain.boundary.reserve(polygon.size());
  for (const auto& point : polygon) {
    convexTerrain.boundary.emplace_back(point.x() - centerInTerrainFrame.x(), point.y() - centerInTerrainFrame.y()); // Shift points to new origin
  }
  return convexTerrain;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "segmented_planes_demo_node");
  ros::NodeHandle nodeHandle("~");

  // Subscription to terrain
  switched_model::SegmentedPlanesTerrainModelRos segmentedPlanesTerrainModelRos(nodeHandle);
  std::unique_ptr<switched_model::SegmentedPlanesTerrainModel> terrainModel;

  // Publishers for visualization
  auto positionPublisher_ = nodeHandle.advertise<visualization_msgs::Marker>("queryPosition", 1);
  auto convexTerrainPublisher_ = nodeHandle.advertise<visualization_msgs::MarkerArray>("convex_terrain", 1);
  auto regionBoundaryPublisher_ = nodeHandle.advertise<visualization_msgs::MarkerArray>("regionBoundary", 1);

  // Node loop
  ros::Rate rate(2.);
  while (ros::ok()) {
    if (segmentedPlanesTerrainModelRos.update(terrainModel)) {
      ROS_INFO("Terrain model updated!!");
    }

    if (terrainModel) {
      const auto RandomFloat = [](float a, float b) {
        float random = ((float) rand()) / (float) RAND_MAX;
        float diff = b - a;
        float r = random * diff;
        return a + r;
      };

      switched_model::vector3_t positionInWorld{RandomFloat(-2.0, 2.0),  RandomFloat(-2.0, 2.0), RandomFloat(0.0, 1.0)};
      positionPublisher_.publish(toMarker(positionInWorld));
      std::cout << "Query position: " << positionInWorld.transpose() << std::endl;

      auto t0 = std::chrono::high_resolution_clock::now();
      const auto convexTerrain = terrainModel->getConvexTerrainAtPositionInWorld(positionInWorld);
      auto t1 = std::chrono::high_resolution_clock::now();
      std::cout << "Query took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count() << " [ms]\n";

      positionPublisher_.publish(toMarker(positionInWorld));
      convexTerrainPublisher_.publish(toMarker(convexTerrain));
    }

    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}