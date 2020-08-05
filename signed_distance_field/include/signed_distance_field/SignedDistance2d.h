//
// Created by rgrandia on 10.07.20.
//

#pragma once

#include <vector>

#include <grid_map_core/TypeDefs.hpp>

#include "Utils.h"

namespace signed_distance_field {

struct SparsityInfo {
  bool hasObstacles;
  bool hasFreeSpace;
  std::vector<bool> rowsIsHomogenous;
  std::vector<bool> colsIsHomogenous;
  int numRowsHomogeneous;
  int numColsHomogeneous;
};

SparsityInfo collectSparsityInfo(const grid_map::Matrix& elevationMap, float height);

grid_map::Matrix computeSignedDistanceAtHeight(const grid_map::Matrix& elevationMap, float height, float resolution);

}