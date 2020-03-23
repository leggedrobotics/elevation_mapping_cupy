#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGONIZER_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGONIZER_HPP_

#include <vector>

#include <boost/shared_ptr.hpp>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/connect_holes.h>
#include <CGAL/create_offset_polygons_from_polygon_with_holes_2.h>
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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
  double contour_approximation_relative_area_threshold = 0.01;
  double contour_approximation_absolute_area_threshold_squared_meters = 0.04;
  int max_number_of_iterations = 5;
};

class Polygonizer {
 public:

  explicit Polygonizer(const PolygonizerParameters parameters = PolygonizerParameters())
  :parameters_(parameters){};

  PolygonWithHoles extractPolygonsFromBinaryImage(const cv::Mat& binary_image) const;

  void removeAreasNotContainedInOuterContourFromHoles(const CgalPolygon2d& outer_polygon, std::vector<CgalPolygon2d>& holes) const;

  CgalPolygon2d resolveHolesWithVerticalConnection(PolygonWithHoles& polygon_with_holes) const;

  void approximatePolygon(CgalPolygon2d& polygon) const;

  CgalPolygon2d runPolygonizationOnBinaryImage(const cv::Mat& binary_image) const;

  bool addHoleToOuterContourAtVertexIndex(int segment_target_vertex_index, const CgalPolygon2d& hole, CgalPolygon2d& outer_contour) const;

  CgalPolygon2d resolveHolesUsingSlConnection(const PolygonWithHoles& polygon_with_holes) const;

  void resolveHolesInBinaryImage(cv::Mat& binary_image, const PolygonWithHoles& contour_with_holes) const;

  PolygonWithHoles extractContoursFromBinaryImage(cv::Mat& binary_image) const;

 private:

  PolygonizerParameters parameters_;

};


} // namespace convex_plane_extraction

#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGONIZER_HPP_
