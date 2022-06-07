/**
 * @file        inpainting.cpp
 * @authors     Fabian Jenelten
 * @date        18.05, 2021
 * @affiliation ETH RSL
 * @brief      Inpainting filter (extrapolate nan values from surrounding data).
 */

// grid map filters rsl.
#include <grid_map_filters_rsl/inpainting.hpp>
#include <grid_map_filters_rsl/processing.hpp>

namespace grid_map {
namespace inpainting {

void minValues(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut) {
  // Create new layer if missing
  if (!map.exists(layerOut)) {
    map.add(layerOut, map.get(layerIn));
  }

  // Reference to in, and out maps, initialize with copy
  const grid_map::Matrix& H_in = map.get(layerIn);
  grid_map::Matrix& H_out = map.get(layerOut);
  H_out = H_in;

  // Some constant
  const int numCols = H_in.cols();
  const int maxColId = numCols - 1;
  const int numRows = H_in.rows();
  const int maxRowId = numRows - 1;

  // Common operation of updating the minimum and keeping track if the minimum was updated.
  auto compareAndStoreMin = [](float newValue, float& currentMin, bool& changedValue) {
    if (!std::isnan(newValue)) {
      if (newValue < currentMin || std::isnan(currentMin)) {
        currentMin = newValue;
        changedValue = true;
      }
    }
  };

  /*
   * Fill each cell that needs inpainting with the min of its neighbours until the map doesn't change anymore.
   * This way each inpainted area gets the minimum value along its contour.
   *
   * We will be reading and writing to H_out during iteration. However, the aliasing does not break the correctness of the algorithm.
   */
  bool hasAtLeastOneValue = true;
  bool changedValue = true;
  while (changedValue && hasAtLeastOneValue) {
    hasAtLeastOneValue = false;
    changedValue = false;
    for (int colId = 0; colId < numCols; ++colId) {
      for (int rowId = 0; rowId < numRows; ++rowId) {
        if (std::isnan(H_in(rowId, colId))) {
          auto& middleValue = H_out(rowId, colId);

          // left
          if (colId > 0) {
            const auto leftValue = H_out(rowId, colId - 1);
            compareAndStoreMin(leftValue, middleValue, changedValue);
          }
          // right
          if (colId < maxColId) {
            const auto rightValue = H_out(rowId, colId + 1);
            compareAndStoreMin(rightValue, middleValue, changedValue);
          }
          // top
          if (rowId > 0) {
            const auto topValue = H_out(rowId - 1, colId);
            compareAndStoreMin(topValue, middleValue, changedValue);
          }
          // bottom
          if (rowId < maxRowId) {
            const auto bottomValue = H_out(rowId + 1, colId);
            compareAndStoreMin(bottomValue, middleValue, changedValue);
          }
        } else {
          hasAtLeastOneValue = true;
        }
      }
    }
  }
}

void resample(grid_map::GridMap& map, const std::string& layer, double newRes) {
  // Original map info
  const auto oldPos = map.getPosition();
  const auto oldSize = map.getSize();
  const auto oldRes = map.getResolution();

  if (oldRes == newRes) {
    return;
  }

  // Layers to be resampled.
  std::vector<std::string> layer_names;
  if (layer == "all") {
    layer_names = map.getLayers();
  } else {
    layer_names.push_back(layer);
  }

  for (const auto& layer_name : layer_names) {
    Eigen::MatrixXf elevationMap = std::move(map.get(layer_name));

    // Convert elevation map ro open-cv image.
    cv::Mat elevationImage;
    cv::eigen2cv(elevationMap, elevationImage);

    // Compute new dimensions.
    const double scaling = oldRes / newRes;
    int width = int(elevationImage.size[1] * scaling);
    int height = int(elevationImage.size[0] * scaling);
    cv::Size dim{width, height};

    // Resize image
    cv::Mat resizedImage;
    cv::resize(elevationImage, resizedImage, dim, 0, 0, cv::INTER_LINEAR);
    cv::cv2eigen(resizedImage, elevationMap);

    // Compute true new resolution. Might be slightly different due to rounding. Take average of both dimensions.
    grid_map::Size newSize = {elevationMap.rows(), elevationMap.cols()};
    newRes = 0.5 * ((oldSize[0] * oldRes) / newSize[0] + (oldSize[1] * oldRes) / newSize[1]);

    // Store new map.
    map.setGeometry({newSize[0] * newRes, newSize[1] * newRes}, newRes, oldPos);
    map.get(layer_name) = std::move(elevationMap);
  }
}

}  // namespace inpainting
}  // namespace grid_map
