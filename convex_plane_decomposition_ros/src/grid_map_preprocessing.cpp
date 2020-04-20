
#include "grid_map_preprocessing.hpp"

namespace convex_plane_decomposition{

  void applyMedianFilter(Eigen::MatrixXf& elevation_map, int kernel_size) {
    cv::Mat elevation_image;
    cv::eigen2cv(elevation_map, elevation_image);
    cv::Mat blurred_image;
    cv::medianBlur(elevation_image, blurred_image, kernel_size);
    cv::cv2eigen(blurred_image, elevation_map);
  }

}