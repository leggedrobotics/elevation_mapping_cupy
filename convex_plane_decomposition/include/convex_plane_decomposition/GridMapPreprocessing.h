#pragma once

#include <Eigen/Core>

#include <grid_map_core/GridMap.hpp>

namespace convex_plane_decomposition {

struct PreprocessingParameters {
  /// Resample to this resolution, set to negative values to skip
  double resolution = -1.0;
  /// Kernel size of the median filter, either 3 or 5
  int kernelSize = 5;
  /// Number of times the image is filtered
  int numberOfRepeats = 1;
  /// If the kernel size should increase each filter step.
  bool increasing = false;
};

class GridMapPreprocessing {
 public:
  GridMapPreprocessing(const PreprocessingParameters& parameters);

  void preprocess(grid_map::GridMap& gridMap, const std::string& layer) const;

 private:
  void denoise(grid_map::GridMap& gridMap, const std::string& layer) const;
  void changeResolution(grid_map::GridMap& gridMap, const std::string& layer) const;
  void inpaint(grid_map::GridMap& gridMap, const std::string& layer) const;

  PreprocessingParameters parameters_;
};

}  // namespace convex_plane_decomposition
