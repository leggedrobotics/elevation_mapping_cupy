//
// Created by rgrandia on 10.07.20.
//

#pragma once

#include <vector>

#include <grid_map_core/TypeDefs.hpp>

#include "Utils.h"

namespace signed_distance_field {

/**
 * Computes the signed distance field at a specified height for a given elevation map.
 *
 * @param elevationMap : elevation data.
 * @param height : height to generate the signed distance at.
 * @param resolution : resolution of the elevation map. (The true distance [m] between cells in world frame)
 * @param minHeight : the lowest height contained in elevationMap
 * @param maxHeight : the maximum height contained in elevationMap
 * @return The signed distance field at the query height.
 */
grid_map::Matrix signedDistanceAtHeight(const grid_map::Matrix& elevationMap, float height, float resolution, float minHeight,
                                        float maxHeight);

/**
 * Same as above, but returns the sdf in transposed form.
 */
grid_map::Matrix signedDistanceAtHeightTranspose(const grid_map::Matrix& elevationMap, float height, float resolution, float minHeight,
                                        float maxHeight);

grid_map::Matrix signedDistanceFromOccupancy(const Eigen::Matrix<bool, -1, -1>& occupancyGrid, float resolution);

}  // namespace signed_distance_field