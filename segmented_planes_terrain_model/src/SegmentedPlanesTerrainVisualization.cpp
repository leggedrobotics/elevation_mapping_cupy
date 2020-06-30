//
// Created by rgrandia on 24.06.20.
//

#include "segmented_planes_terrain_model/SegmentedPlanesTerrainVisualization.h"

namespace switched_model {

visualization_msgs::MarkerArray getConvexTerrainMarkers(const ConvexTerrain& convexTerrain, Color color, double diameter, double linewidth,
                                                        double normalLength) {
  visualization_msgs::MarkerArray markerArray;
  markerArray.markers.reserve(3);

  // Mark the center point
  markerArray.markers.emplace_back(getSphereMsg(convexTerrain.plane.positionInWorld, color, diameter));

  // Mark the surface normal
  const vector3_t surfaceNormal = normalLength * surfaceNormalInWorld(convexTerrain.plane);
  markerArray.markers.emplace_back(getArrowAtPointMsg(surfaceNormal, convexTerrain.plane.positionInWorld, color));

  // Polygon message
  if (!convexTerrain.boundary.empty()) {
    std::vector<geometry_msgs::Point> boundary;
    boundary.reserve(convexTerrain.boundary.size() + 1);
    for (const auto& point : convexTerrain.boundary) {
      const auto& pointInWorldFrame = positionInWorldFrameFromPositionInTerrain({point.x(), point.y(), 0.0}, convexTerrain.plane);
      boundary.emplace_back(getPointMsg(pointInWorldFrame));
    }
    // Close the polygon
    const auto& pointInWorldFrame = positionInWorldFrameFromPositionInTerrain(
        {convexTerrain.boundary.front().x(), convexTerrain.boundary.front().y(), 0.0}, convexTerrain.plane);
    boundary.emplace_back(getPointMsg(pointInWorldFrame));

    markerArray.markers.emplace_back(getLineMsg(std::move(boundary), color, linewidth));
  } else {
  }

  return markerArray;
}

}  // namespace switched_model
