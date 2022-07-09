#pragma once

#include <geometry_msgs/PolygonStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <convex_plane_decomposition/PlanarRegion.h>

namespace convex_plane_decomposition {

geometry_msgs::PolygonStamped to3dRosPolygon(const CgalPolygon2d& polygon, const Eigen::Isometry3d& transformPlaneToWorld,
                                             const std_msgs::Header& header);

std::vector<geometry_msgs::PolygonStamped> to3dRosPolygon(const CgalPolygonWithHoles2d& polygonWithHoles,
                                                          const Eigen::Isometry3d& transformPlaneToWorld, const std_msgs::Header& header);

visualization_msgs::MarkerArray convertBoundariesToRosMarkers(const std::vector<PlanarRegion>& planarRegions, const std::string& frameId,
                                                              grid_map::Time time, double lineWidth);

visualization_msgs::MarkerArray convertInsetsToRosMarkers(const std::vector<PlanarRegion>& planarRegions, const std::string& frameId,
                                                          grid_map::Time time, double lineWidth);

}  // namespace convex_plane_decomposition
