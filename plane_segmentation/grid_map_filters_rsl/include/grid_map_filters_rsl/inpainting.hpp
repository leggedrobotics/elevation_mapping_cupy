/**
 * @file        inpainting.hpp
 * @authors     Fabian Jenelten
 * @date        18.05, 2021
 * @affiliation ETH RSL
 * @brief       Inpainting filter (extrapolate nan values from surrounding data).
 */

#pragma once

// grid map.
#include <grid_map_core/grid_map_core.hpp>

namespace grid_map {
namespace inpainting {

/**
 * @brief Inpaint missing data using min-value in neighborhood. The neighborhood search is performed along the contour of nan-patches.
 * In-place operation (layerIn = layerOut) is NOT supported.
 * @param map           grid map
 * @param layerIn       reference layer (filter is applied wrt this layer)
 * @param layerOut      output layer (filtered map is written into this layer)
 */
void minValues(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut);

/**
 * @brief Up- or down-sample elevation map (open-cv function). In-place operation only. Only the layer with name "layer" is resampled, while
 * all other layers (if there are any) are left untouched (exception if layer="all", which applies filter to all layers).
 * @param map         grid map
 * @param layer       resampling is done based in this layer. If "all", resamples all layers
 * @param newRes      new resolution.
 */
void resample(grid_map::GridMap& map, const std::string& layer, double newRes);

}  // namespace inpainting
}  // namespace grid_map
