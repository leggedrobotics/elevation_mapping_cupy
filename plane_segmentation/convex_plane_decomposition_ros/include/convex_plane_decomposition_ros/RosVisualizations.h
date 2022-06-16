#pragma once

#include <geometry_msgs/PolygonStamped.h>

#include <jsk_recognition_msgs/PolygonArray.h>

#include <convex_plane_decomposition/PlanarRegion.h>

namespace convex_plane_decomposition {

geometry_msgs::PolygonStamped to3dRosPolygon(const CgalPolygon2d& polygon, const Eigen::Isometry3d& transformPlaneToWorld,
                                   const std_msgs::Header& header);

std::vector<geometry_msgs::PolygonStamped> to3dRosPolygon(const CgalPolygonWithHoles2d& polygonWithHoles,
                                                const Eigen::Isometry3d& transformPlaneToWorld, const std_msgs::Header& header);

jsk_recognition_msgs::PolygonArray convertBoundariesToRosPolygons(const std::vector<PlanarRegion>& planarRegions,
                                                                  const std::string& frameId, grid_map::Time time);

jsk_recognition_msgs::PolygonArray convertInsetsToRosPolygons(const std::vector<PlanarRegion>& planarRegions, const std::string& frameId,
                                                              grid_map::Time time);

}  // namespace convex_plane_decomposition
