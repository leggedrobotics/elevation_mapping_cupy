/**
 * @file        smoothing.hpp
 * @authors     Fabian Jenelten
 * @date        18.05, 2021
 * @affiliation ETH RSL
 * @brief       Smoothing and outlier rejection filters.
 */

#pragma once

// grid map.
#include <grid_map_core/grid_map_core.hpp>

namespace grid_map {
namespace smoothing {

/**
 * @brief Sequential median filter (open-cv function). In-place operation (layerIn = layerOut) is supported.
 * @param map               grid map
 * @param layerIn           reference layer (filter is applied wrt this layer)
 * @param layerOut          output layer (filtered map is written into this layer)
 * @param kernelSize        size of the smoothing window (must be an odd number)
 * @param deltaKernelSize   kernel size is increased by this value, if numberOfRepeats > 1
 * @param numberOfRepeats   number of sequentially applied median filters (approaches to gaussian blurring if increased)
 */
void median(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, int deltaKernelSize = 2,
            int numberOfRepeats = 1);

/**
 * @brief Sequential box blur filter (open cv-function). In-place operation (layerIn = layerOut) is supported.
 * @param map               grid map
 * @param layerIn           reference layer (filter is applied wrt this layer)
 * @param layerOut          output layer (filtered map is written into this layer)
 * @param kernelSize        size of the smoothing window (should be an odd number, otherwise, introduces offset)
 * @param numberOfRepeats   number of sequentially applied blurring filters (approaches to gaussian blurring if increased)
 */
void boxBlur(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, int numberOfRepeats = 1);

/**
 * @brief Gaussian blur filter (open cv-function). In-place operation (layerIn = layerOut) is supported.
 * @param map               grid map
 * @param layerIn           reference layer (filter is applied wrt this layer)
 * @param layerOut          output layer (filtered map is written into this layer)
 * @param kernelSize        size of the smoothing window (should be an odd number, otherwise, introduces offset)
 * @param sigma             variance
 */
void gaussianBlur(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, double sigma);

/**
 * @brief Non-Local means denoising filter (open-cv function). In-place operation (layerIn = layerOut) is supported. Attention: slow (~5ms)!
 * @param map               grid map
 * @param layerIn           reference layer (filter is applied wrt this layer)
 * @param layerOut          output layer (filtered map is written into this layer)
 * @param kernelSize        size of the smoothing window (must be an odd number)
 * @param searchWindow      search window (must be an odd number and larger than kernelSize)
 * @param w                 filter strength
 */
void nlm(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize = 3, int searchWindow = 7,
         float w = 45.F);

/**
 * @brief Performs image denoising using the Block-Matching and 3D-filtering algorithm (open-cv function). In-place operation (layerIn =
 * layerOut) is supported. Attention: very slow (~30ms)!
 * @param map               grid map
 * @param layerIn           reference layer (filter is applied wrt this layer)
 * @param layerOut          output layer (filtered map is written into this layer)
 * @param kernelSize        size of the smoothing window (must be power of 2)
 * @param searchWindow      search window (must be larger than kernelSize)
 * @param w                 filter strength
 */
void bm3d(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize = 4, int searchWindow = 6,
          float w = 25.F);

/**
 * @brief Bilateral filter (open-cv function). In-place operation (layerIn = layerOut) is supported. Attention:  slow (~0.3ms)!
 * @param map               grid map
 * @param layerIn           reference layer (filter is applied wrt this layer)
 * @param layerOut          output layer (filtered map is written into this layer)
 * @param kernelSize        size of the smoothing window (must be an even number)
 * @param w                 filter strength
 */
void bilateralFilter(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize = 0, double w = 0.2);

/**
 * @brief Optimization based (open-cv function). In-place operation (layerIn = layerOut) is supported. Attention: slow (~15ms)!
 * @param map               grid map
 * @param layerIn           reference layer (filter is applied wrt this layer)
 * @param layerOut          output layer (filtered map is written into this layer)
 * @param lambda            the smaller the value, the smoother the image
 * @param n                 number of optimization iterations
 */
void tvL1(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, double lambda = 1.0, int n = 60);

}  // namespace smoothing
}  // namespace grid_map
