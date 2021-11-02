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
 * Also takes temporary variables from outside to prevent memory allocation.
 *
 * @param elevationMap : elevation data.
 * @param sdfTranspose : [output] resulting sdf in transposed form (automatically allocated if of wrong size)
 * @param tmp : temporary of size elevationMap (automatically allocated if of wrong size)
 * @param tmpTranspose : temporary of size elevationMap transpose (automatically allocated if of wrong size)
 * @param height : height to generate the signed distance at.
 * @param resolution : resolution of the elevation map. (The true distance [m] between cells in world frame)
 * @param minHeight : the lowest height contained in elevationMap
 * @param maxHeight : the maximum height contained in elevationMap
 */
void signedDistanceAtHeightTranspose(const grid_map::Matrix& elevationMap, grid_map::Matrix& sdfTranspose, grid_map::Matrix& tmp,
                                     grid_map::Matrix& tmpTranspose, float height, float resolution, float minHeight, float maxHeight);

grid_map::Matrix signedDistanceFromOccupancy(const Eigen::Matrix<bool, -1, -1>& occupancyGrid, float resolution);

}  // namespace signed_distance_field