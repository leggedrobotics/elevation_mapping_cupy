/**
 * @file        inpainting.hpp
 * @authors     Fabian Jenelten
 * @date        18.05, 2021
 * @affiliation ETH RSL
 * @brief       Inpainting filters.
 */

#pragma once

// grid map.
#include <grid_map_core/grid_map_core.hpp>

namespace grid_map {
namespace inpainting {

/**
 * Inpaint missing data using min value in neighborhood. In-place operation (layerIn = layerOut) is NOT supported.
 * @param map           grid map
 * @param layerIn       reference layer (filter is applied wrt this layer)
 * @param layerOut      output layer (filtered map is written into this layer)
 */
void minValues(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut);

/**
 * Inpaint missing data using bi-linear interpolation. In-place operation (layerIn = layerOut) is NOT supported.
 * @param map           grid map
 * @param layerIn       reference layer (filter is applied wrt this layer)
 * @param layerOut      output layer (filtered map is written into this layer)
 */
void biLinearInterpolation(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut);

/**
 * nonlinear interpolation (open-cv function). In-place operation (layerIn = layerOut) is supported.
 * @param map           grid map
 * @param layerIn       reference layer (filter is applied wrt this layer)
 * @param layerOut      output layer (filtered map is written into this layer)
 * @param inpaintRadius vicinity considered by inpaint filter.
 */
void nonlinearInterpolation(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, double inpaintRadius = 0.05);
}  // namespace inpainting
}  // namespace grid_map
