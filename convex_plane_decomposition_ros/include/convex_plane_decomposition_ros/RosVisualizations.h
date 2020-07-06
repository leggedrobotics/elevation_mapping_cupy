#pragma once

#include <jsk_recognition_msgs/PolygonArray.h>

#include <convex_plane_decomposition/PlanarRegion.h>

namespace convex_plane_decomposition {

jsk_recognition_msgs::PolygonArray convertBoundariesToRosPolygons(const std::vector<PlanarRegion>& planarRegions, const std::string& frameId);

jsk_recognition_msgs::PolygonArray convertInsetsToRosPolygons(const std::vector<PlanarRegion>& planarRegions, const std::string& frameId);

}  // namespace convex_plane_decomposition
