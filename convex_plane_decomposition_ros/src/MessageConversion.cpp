//
// Created by rgrandia on 14.06.20.
//

#include "convex_plane_decomposition_ros/MessageConversion.h"

namespace convex_plane_decomposition {

CgalBbox2d fromMessage(const convex_plane_decomposition_msgs::BoundingBox2d& msg) {
  return {msg.min_x, msg.min_y, msg.max_x, msg.max_y};
}

convex_plane_decomposition_msgs::BoundingBox2d toMessage(const CgalBbox2d& bbox2d) {
  convex_plane_decomposition_msgs::BoundingBox2d msg;
  msg.min_x = bbox2d.xmin();
  msg.min_y = bbox2d.ymin();
  msg.max_x = bbox2d.xmax();
  msg.max_y = bbox2d.ymax();
  return msg;
}

PlanarRegion fromMessage(const convex_plane_decomposition_msgs::PlanarRegion& msg) {
  PlanarRegion planarRegion;
  planarRegion.planeParameters = fromMessage(msg.plane_parameters);
  planarRegion.boundaryWithInset.boundary = fromMessage(msg.boundary);
  planarRegion.boundaryWithInset.insets.reserve(msg.insets.size());
  for (const auto& inset : msg.insets) {
    planarRegion.boundaryWithInset.insets.emplace_back(fromMessage(inset));
  }
  planarRegion.bbox2d = fromMessage(msg.bbox2d);
  return planarRegion;
}

convex_plane_decomposition_msgs::PlanarRegion toMessage(const PlanarRegion& planarRegion) {
  convex_plane_decomposition_msgs::PlanarRegion msg;
  msg.plane_parameters = toMessage(planarRegion.planeParameters);
  msg.boundary = toMessage(planarRegion.boundaryWithInset.boundary);
  msg.insets.reserve(planarRegion.boundaryWithInset.insets.size());
  for (const auto& inset : planarRegion.boundaryWithInset.insets) {
    msg.insets.emplace_back(toMessage(inset));
  }
  msg.bbox2d = toMessage(planarRegion.bbox2d);
  return msg;
}

PlanarTerrain fromMessage(const convex_plane_decomposition_msgs::PlanarTerrain& msg) {
  PlanarTerrain planarTerrain;
  planarTerrain.reserve(msg.planarRegions.size());
  for (const auto& planarRegion : msg.planarRegions) {
    planarTerrain.emplace_back(fromMessage(planarRegion));
  }
  return planarTerrain;
}

convex_plane_decomposition_msgs::PlanarTerrain toMessage(const PlanarTerrain& planarTerrain) {
  convex_plane_decomposition_msgs::PlanarTerrain msg;
  msg.planarRegions.reserve(planarTerrain.size());
  for (const auto& planarRegion : planarTerrain) {
    msg.planarRegions.emplace_back(toMessage(planarRegion));
  }
  return msg;
}

TerrainPlane fromMessage(const geometry_msgs::Pose& msg) {
  TerrainPlane plane;
  plane.positionInWorld.x() = msg.position.x;
  plane.positionInWorld.y() = msg.position.y;
  plane.positionInWorld.z() = msg.position.z;
  Eigen::Quaterniond terrainOrientation;
  terrainOrientation.x() = msg.orientation.x;
  terrainOrientation.y() = msg.orientation.y;
  terrainOrientation.z() = msg.orientation.z;
  terrainOrientation.w() = msg.orientation.w;
  plane.orientationWorldToTerrain = terrainOrientation.toRotationMatrix().transpose();
  return plane;
}

geometry_msgs::Pose toMessage(const TerrainPlane& plane) {
  geometry_msgs::Pose pose;
  pose.position.x = plane.positionInWorld.x();
  pose.position.y = plane.positionInWorld.y();
  pose.position.z = plane.positionInWorld.z();
  Eigen::Quaterniond terrainOrientation(plane.orientationWorldToTerrain.transpose());
  pose.orientation.x = terrainOrientation.x();
  pose.orientation.y = terrainOrientation.y();
  pose.orientation.z = terrainOrientation.z();
  pose.orientation.w = terrainOrientation.w();
  return pose;
}

CgalPoint2d fromMessage(const convex_plane_decomposition_msgs::Point2d& msg) {
  return {msg.x, msg.y};
}

convex_plane_decomposition_msgs::Point2d toMessage(const CgalPoint2d& point2d) {
  convex_plane_decomposition_msgs::Point2d msg;
  msg.x = point2d.x();
  msg.y = point2d.y();
  return msg;
}

CgalPolygon2d fromMessage(const convex_plane_decomposition_msgs::Polygon2d& msg) {
  CgalPolygon2d polygon2d;
  polygon2d.container().reserve(msg.points.size());
  for (const auto& point : msg.points) {
    polygon2d.container().emplace_back(fromMessage(point));
  }
  return polygon2d;
}

convex_plane_decomposition_msgs::Polygon2d toMessage(const CgalPolygon2d& polygon2d) {
  convex_plane_decomposition_msgs::Polygon2d msg;
  msg.points.reserve(polygon2d.container().size());
  for (const auto& point : polygon2d) {
    msg.points.emplace_back(toMessage(point));
  }
  return msg;
}

CgalPolygonWithHoles2d fromMessage(const convex_plane_decomposition_msgs::PolygonWithHoles2d& msg) {
  CgalPolygonWithHoles2d polygonWithHoles2d;
  polygonWithHoles2d.outer_boundary() = fromMessage(msg.outer_boundary);
  for (const auto& hole : msg.holes) {
    polygonWithHoles2d.add_hole(fromMessage(hole));
  }
  return polygonWithHoles2d;
}

convex_plane_decomposition_msgs::PolygonWithHoles2d toMessage(const CgalPolygonWithHoles2d& polygonWithHoles2d) {
  convex_plane_decomposition_msgs::PolygonWithHoles2d msg;
  msg.outer_boundary = toMessage(polygonWithHoles2d.outer_boundary());
  msg.holes.reserve(polygonWithHoles2d.number_of_holes());
  for (const auto& hole : polygonWithHoles2d.holes()) {
    msg.holes.emplace_back(toMessage(hole));
  }
  return msg;
}

}  // namespace convex_plane_decomposition
