#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGONIZER_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGONIZER_HPP_

#include <vector>

#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#include "plane.hpp"
#include "polygon.hpp"

namespace convex_plane_extraction {

struct PolygonizerParameters{
  double resolution = 0.02;
  int upsampling_factor = 3;
  bool activate_long_edge_upsampling = false;
  bool activate_contour_approximation = false;
  double hole_area_threshold_squared_meters = 2e-3;
};

class Polygonizer {
 public:

  Polygonizer(const PolygonizerParameters parameters = PolygonizerParameters())
  :parameters_(parameters){};

  PolygonWithHoles extractPolygonsFromBinaryImage(const cv::Mat& binary_image) const;

  CgalPolygon2d resolveHoles(PolygonWithHoles& polygon_with_holes) const;

 private:

  PolygonizerParameters parameters_;

};


} // namespace convex_plane_extraction

#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGONIZER_HPP_
