#pragma once

#include <Eigen/Core>

#include <grid_map_core/GridMap.hpp>

namespace convex_plane_decomposition {

struct PreprocessingParameters {
  /// Kernel size of the median filter
  int kernelSize = 5;
  /// Number of times the image is filtered
  int numberOfRepeats = 1;
  /// If the kernel size should increase each filter step.
  bool increasing = false;
  /// [m] radius used for inpainting
  double inpaintRadius = 0.05;
};

class GridMapPreprocessing {
 public:
  GridMapPreprocessing(const PreprocessingParameters& parameters);

  void preprocess(grid_map::GridMap& gridMap, const std::string& layer);

 private:
  void denoise(grid_map::GridMap& gridMap, const std::string& layer);
  void inpaint(grid_map::GridMap& gridMap, const std::string& layer, float minValue, float maxValue);

  PreprocessingParameters parameters_;
};

}  // namespace convex_plane_decomposition
