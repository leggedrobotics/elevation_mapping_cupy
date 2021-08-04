/**
 * @file        processing.hpp
 * @authors     Fabian Jenelten
 * @date        04.08, 2021
 * @affiliation ETH RSL
 * @brief       Processing filter (everything that is not smoothing or inpainting).
 */

#pragma once

// grid map.
#include <grid_map_core/grid_map_core.hpp>

namespace grid_map {
namespace processing {

/**
 * @brief Replaces values by max in region (open-cv function). In-place operation (layerIn = layerOut) is supported.
 * @param map           grid map
 * @param layerIn       reference layer (filter is applied wrt this layer)
 * @param layerOut      output layer (filtered map is written into this layer)
 * @param kernelSize    vicinity considered by filter (should be odd).
 */
void dilate(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize = 10);
}  // namespace processing
}  // namespace grid_map
