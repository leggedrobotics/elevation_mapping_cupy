//
// Created by rgrandia on 10.07.20.
//

#pragma once

#include <vector>

#include <grid_map_core/TypeDefs.hpp>

#include "Utils.h"

namespace signed_distance_field {

inline Eigen::Matrix<bool, -1, -1> occupancyAtHeight(const grid_map::Matrix& elevationMap, float height) {
  Eigen::Matrix<bool, -1, -1> occupany = elevationMap.unaryExpr([=](float val) { return val > height; });
  return occupany;
}

grid_map::Matrix signedDistanceAtHeight(const grid_map::Matrix& elevationMap, float height, float resolution);

grid_map::Matrix signedDistanceFromOccupancy(const Eigen::Matrix<bool, -1, -1>& occupancyGrid, float resolution);

}  // namespace signed_distance_field