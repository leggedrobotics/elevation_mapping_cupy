#include "convex_plane_decomposition_ros/RosVisualizations.h"

#include <geometry_msgs/Point32.h>

namespace convex_plane_decomposition {

geometry_msgs::PolygonStamped to3dRosPolygon(const CgalPolygon2d& polygon, const Eigen::Isometry3d& transformPlaneToWorld,
                                             const std_msgs::Header& header) {
  geometry_msgs::PolygonStamped polygon3d;
  polygon3d.header = header;
  polygon3d.polygon.points.reserve(polygon.size());
  for (const auto& point : polygon) {
    geometry_msgs::Point32 point_ros;
    const auto pointInWorld = positionInWorldFrameFromPosition2dInPlane(point, transformPlaneToWorld);
    point_ros.x = static_cast<float>(pointInWorld.x());
    point_ros.y = static_cast<float>(pointInWorld.y());
    point_ros.z = static_cast<float>(pointInWorld.z());
    polygon3d.polygon.points.push_back(point_ros);
  }
  return polygon3d;
}

std::vector<geometry_msgs::PolygonStamped> to3dRosPolygon(const CgalPolygonWithHoles2d& polygonWithHoles,
                                                          const Eigen::Isometry3d& transformPlaneToWorld, const std_msgs::Header& header) {
  std::vector<geometry_msgs::PolygonStamped> polygons;

  polygons.reserve(polygonWithHoles.number_of_holes() + 1);
  polygons.emplace_back(to3dRosPolygon(polygonWithHoles.outer_boundary(), transformPlaneToWorld, header));

  for (const auto& hole : polygonWithHoles.holes()) {
    polygons.emplace_back(to3dRosPolygon(hole, transformPlaneToWorld, header));
  }
  return polygons;
}

jsk_recognition_msgs::PolygonArray convertBoundariesToRosPolygons(const std::vector<PlanarRegion>& planarRegions,
                                                                  const std::string& frameId, grid_map::Time time) {
  jsk_recognition_msgs::PolygonArray polygon_buffer;
  std_msgs::Header header;
  header.stamp.fromNSec(time);
  header.frame_id = frameId;

  polygon_buffer.header = header;
  polygon_buffer.polygons.reserve(planarRegions.size());  // lower bound
  polygon_buffer.labels.reserve(planarRegions.size());
  uint32_t label = 0;
  for (const auto& planarRegion : planarRegions) {
    auto boundaries = to3dRosPolygon(planarRegion.boundaryWithInset.boundary, planarRegion.transformPlaneToWorld, header);
    std::move(boundaries.begin(), boundaries.end(), std::back_inserter(polygon_buffer.polygons));
    for (size_t i = 0; i < boundaries.size(); ++i) {
      polygon_buffer.labels.push_back(label);
    }
    ++label;
  }

  return polygon_buffer;
}

jsk_recognition_msgs::PolygonArray convertInsetsToRosPolygons(const std::vector<PlanarRegion>& planarRegions, const std::string& frameId,
                                                              grid_map::Time time) {
  jsk_recognition_msgs::PolygonArray polygon_buffer;
  std_msgs::Header header;
  header.stamp.fromNSec(time);
  header.frame_id = frameId;

  polygon_buffer.header = header;
  polygon_buffer.polygons.reserve(planarRegions.size());  // lower bound
  polygon_buffer.labels.reserve(planarRegions.size());
  uint32_t label = 0;
  for (const auto& planarRegion : planarRegions) {
    for (const auto& inset : planarRegion.boundaryWithInset.insets) {
      auto insets = to3dRosPolygon(inset, planarRegion.transformPlaneToWorld, header);
      std::move(insets.begin(), insets.end(), std::back_inserter(polygon_buffer.polygons));
      for (size_t i = 0; i < insets.size(); ++i) {
        polygon_buffer.labels.push_back(label);
      }
    }
    ++label;
  }
  return polygon_buffer;
}

}  // namespace convex_plane_decomposition
