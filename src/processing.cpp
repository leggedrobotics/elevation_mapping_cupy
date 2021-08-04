/**
 * @file        processing.cpp
 * @authors     Fabian Jenelten
 * @date        04.08, 2021
 * @affiliation ETH RSL
 * @brief       Processing filter (everything that is not smoothing or inpainting).
 */

// grid map filters rsl.
#include <grid_map_filters_rsl/processing.hpp>

// open cv.
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace grid_map {
namespace processing {

void dilate(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Convert to image.
  cv::Mat elevationImage;
  cv::eigen2cv(map.get(layerIn), elevationImage);

  // Box blur.
  cv::Size kernelSize2D(kernelSize, kernelSize);
  cv::dilate(elevationImage, elevationImage, cv::getStructuringElement(0, cv::Size(kernelSize, kernelSize)));

  // Set output layer.
  cv::cv2eigen(elevationImage, map.get(layerOut));
}
}  // namespace processing
}  // namespace grid_map
