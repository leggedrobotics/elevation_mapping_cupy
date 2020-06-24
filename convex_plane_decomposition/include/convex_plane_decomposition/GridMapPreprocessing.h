#pragma once

#include <Eigen/Core>

#include <grid_map_core/GridMap.hpp>

namespace convex_plane_decomposition {

struct PreprocessingParameters {
  int kernelSize = 5; // Kernel size of the median filter
  int numberOfRepeats = 1; // Number of times the image is filtered
  bool increasing = false; // If the kernel size should increase each filter step.
  double inpaintRadius = 0.05; // [in m]
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

