#include "convex_plane_decomposition/export_utils.hpp"

#include <fstream>

#include <grid_map_core/iterators/GridMapIterator.hpp>

namespace convex_plane_decomposition {

bool exportPointsWithNormalsToCsv(grid_map::GridMap& map, const std::string& normals_layer_prefix, const std::string& layer_height,
                                  const std::string& path) {
  std::ofstream output_file;
  output_file.open(path + "points_and_normals.csv");
  for (grid_map::GridMapIterator iterator(map); !iterator.isPastEnd(); ++iterator) {
    grid_map::Position3 position3;
    Eigen::Vector3d point;
    Eigen::Vector3d normal;
    map.getPosition3(layer_height, *iterator, point);
    map.getVector(normals_layer_prefix, *iterator, normal);
    output_file << point(0) << ", " << point(1) << ", " << point(2) << ", " << normal(0) << ", " << normal(1) << ", " << normal(2) << "\n";
    }
    return true;
  }

}