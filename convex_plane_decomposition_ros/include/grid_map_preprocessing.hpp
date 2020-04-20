#ifndef CONVEX_PLANE_EXTRACTION_ROS_INCLUDE_GRID_MAP_PREPROCESSING_HPP_
#define CONVEX_PLANE_EXTRACTION_ROS_INCLUDE_GRID_MAP_PREPROCESSING_HPP_

#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"

#include <grid_map_ros/grid_map_ros.hpp>

namespace convex_plane_decomposition{
  using namespace grid_map;

  void applyMedianFilter(Eigen::MatrixXf& elevation_map, int kernel_size);
}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_GRID_MAP_PREPROCESSING_HPP_
