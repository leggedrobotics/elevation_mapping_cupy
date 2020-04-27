#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_EXPORT_UTILS_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_EXPORT_UTILS_HPP_

#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <grid_map_core/GridMap.hpp>

namespace convex_plane_decomposition {

bool exportPointsWithNormalsToCsv(grid_map::GridMap& map, const std::string& normals_layer_prefix, const std::string& layer_height,
                                  const std::string& path);

}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_EXPORT_UTILS_HPP_
