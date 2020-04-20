#include "polygonizer.hpp"

using namespace convex_plane_extraction;

PolygonWithHoles Polygonizer::extractPolygonsFromBinaryImage(const cv::Mat& binary_image) const{
  PolygonWithHoles plane_polygons;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::Mat binary_image_upsampled;
  cv::resize(binary_image,
             binary_image_upsampled,
             cv::Size(parameters_.upsampling_factor * binary_image.size().height,
                      parameters_.upsampling_factor * binary_image.size().width));

  findContours(binary_image_upsampled, contours, hierarchy,
               CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
  std::vector<std::vector<cv::Point>> approx_contours;
  int hierachy_it = 0;
  for (const auto& contour : contours) {
    ++hierachy_it;
    if (contour.size() <= 2) {
      LOG_IF(WARNING, hierarchy[hierachy_it - 1][3] < 0 && contour.size() > 3)
          << "Removing parental polygon since too few vertices!";
      continue;
    }
    CgalPolygon2d polygon = convex_plane_extraction::createCgalPolygonFromOpenCvPoints(
        contour.begin(), contour.end(),
        parameters_.resolution / static_cast<double>(parameters_.upsampling_factor));
    CHECK(polygon.is_simple()) << "Contour extraction from binary image caused intersection!";
    constexpr int kParentFlagIndex = 3;
    if (hierarchy[hierachy_it - 1][kParentFlagIndex] < 0) {
      if (parameters_.activate_long_edge_upsampling) {
        upSampleLongEdges(&polygon);
      }
      if (parameters_.activate_contour_approximation) {
        approximateContour(&polygon, parameters_.max_number_of_iterations, parameters_.contour_approximation_relative_local_area_threshold,
            parameters_.contour_approximation_absolute_local_area_threshold_squared_meters,
            parameters_.contour_approximation_relative_total_area_threshold, parameters_.contour_approximation_absolute_total_area_threshold_squared_meters);
        VLOG(1) << "Contour approximated!";
      }
      plane_polygons.outer_contour = polygon;
    } else {
      plane_polygons.holes.push_back(polygon);
    }
  }
  return plane_polygons;
}

PolygonWithHoles Polygonizer::extractContoursFromBinaryImage(cv::Mat& binary_image) const{
  PolygonWithHoles plane_polygons;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  findContours(binary_image, contours, hierarchy,
               CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
  std::vector<std::vector<cv::Point>> approx_contours;
  int hierachy_it = 0;
  for (auto& contour : contours) {
    ++hierachy_it;
    CHECK_GE(contour.size(), 2);
    CgalPolygon2d polygon = convex_plane_extraction::createCgalPolygonFromOpenCvPoints(
        contour.begin(), contour.end(), 1);
    constexpr int kParentFlagIndex = 3;
    if (hierarchy[hierachy_it - 1][kParentFlagIndex] < 0) {
      plane_polygons.outer_contour = polygon;
    } else {
      plane_polygons.holes.push_back(polygon);
    }
  }
  return plane_polygons;
}

// Deprecated: Due to finite resolution in image, complicated contours are extracted in the end.
void Polygonizer::resolveHolesInBinaryImage(cv::Mat& binary_image, const PolygonWithHoles& contour_with_holes) const{
  for(const auto& hole : contour_with_holes.holes){
    if (hole.area() > 36){
      continue;
    }
    std::pair<int, int> slConcavityVertices =  getVertexPositionsWithHighestHoleSlConcavityMeasure(hole);
    const CgalPoint2d first_sl_point = hole.container().at(slConcavityVertices.first);
    const CgalPoint2d second_sl_point = hole.container().at(slConcavityVertices.second);
    std::pair<int, CgalPoint2d> first_connection_candidate =
        getClosestPointAndSegmentOnPolygonToPoint(first_sl_point, contour_with_holes.outer_contour);
    std::pair<int, CgalPoint2d> second_connection_candidate =
        getClosestPointAndSegmentOnPolygonToPoint(second_sl_point, contour_with_holes.outer_contour);
    if ((first_connection_candidate.second - first_sl_point).squared_length() <
        (second_connection_candidate.second - second_sl_point).squared_length()){
      cv::Point first_point(first_connection_candidate.second.y(), first_connection_candidate.second.x());
      cv::Point second_point(first_sl_point.y(), first_sl_point.x());
      cv::line(binary_image, first_point, second_point, CV_RGB(0,0,0),2);
    } else{
      cv::Point first_point(second_connection_candidate.second.y(), second_connection_candidate.second.x());
      cv::Point second_point(second_sl_point.y(), second_sl_point.x());
      cv::line(binary_image, first_point, second_point, CV_RGB(0,0,0), 2);
    }
  }
}

void Polygonizer::removeAreasNotContainedInOuterContourFromHoles(const CgalPolygon2d& outer_polygon, std::vector<CgalPolygon2d>& holes) const{
  auto hole_it = holes.begin();
  while(hole_it != holes.end()) {
    std::vector<CgalPolygonWithHoles2d> buffer;
    CGAL::intersection(outer_polygon, *hole_it, std::back_inserter(buffer));
    if (buffer.empty()){
      hole_it = holes.erase(hole_it);
    } else {
      CHECK_EQ(buffer.size(), 1) << "Intersection of hole and outer contour cannot have multiple polygons!";
      *hole_it = CgalPolygon2d(buffer.begin()->outer_boundary().vertices_begin(), buffer.begin()->outer_boundary().vertices_end());
      ++hole_it;
    }
  }
}

bool Polygonizer::addHoleToOuterContourAtVertexIndex(int segment_target_vertex_index, const CgalPolygon2d& hole, CgalPolygon2d& outer_contour) const{
  CHECK_EQ(outer_contour.orientation(), CGAL::COUNTERCLOCKWISE);
  CHECK_EQ(hole.orientation(), CGAL::COUNTERCLOCKWISE);
  auto outer_vertex_it = outer_contour.vertices_begin();
  std::advance(outer_vertex_it, segment_target_vertex_index);
  const auto& hole_vertex_buffer = hole.container();
  // TODO
  return false;
}

// Deprecated: This method produces non-simple polygons.
CgalPolygon2d Polygonizer::resolveHolesWithVerticalConnection(PolygonWithHoles& polygon_with_holes) const{
  CgalPolygon2dContainer& holes = polygon_with_holes.holes;
  // Get hole part which is entirely contained in outer contour.
  // In the contour simplification stage intersections may have been introduced.
  removeAreasNotContainedInOuterContourFromHoles(polygon_with_holes.outer_contour, holes);
  if (holes.empty()) {
    return polygon_with_holes.outer_contour;
  }
  auto hole_it = holes.begin();
  while(hole_it!= holes.end()) {
    if (hole_it->size() < 3) {
      hole_it = holes.erase(hole_it);
    } else if (abs(hole_it->area()) < parameters_.hole_area_threshold_squared_meters) {
      hole_it = holes.erase(hole_it);
    } else {
      ++hole_it;
    }
  }
  CgalPolygonWithHoles2d temp_polygon_with_holes(polygon_with_holes.outer_contour,
                                                 polygon_with_holes.holes.begin(), polygon_with_holes.holes.end());
  CgalPolygon2d resolved_polygon;
  connect_holes(temp_polygon_with_holes, std::back_inserter (resolved_polygon.container()));
  // Create offset polygon to return
  constexpr double kSmallOffsetMeters = 1e-12;
  std::vector<boost::shared_ptr<CgalPolygon2d>> offset_polygons = CGAL::create_interior_skeleton_and_offset_polygons_2(1e-12, resolved_polygon);
  CHECK_EQ(offset_polygons.size(), 1);
  return **offset_polygons.begin();
}

CgalPolygon2d Polygonizer::runPolygonizationOnBinaryImage(const cv::Mat& binary_image) const{
  PolygonWithHoles polygon_with_holes = extractPolygonsFromBinaryImage(binary_image);
  //CgalPolygon2d polygon = resolveHolesWithVerticalConnection(polygon_with_holes);
  return polygon_with_holes.outer_contour;
}

// TODO: complete implementation!
CgalPolygon2d Polygonizer::resolveHolesUsingSlConnection(const PolygonWithHoles& polygon_with_holes) const{
  CgalPolygon2dContainer holes = polygon_with_holes.holes;
  // Get hole part which is entirely contained in outer contour.
  // In the contour simplification stage intersections may have been introduced.
  removeAreasNotContainedInOuterContourFromHoles(polygon_with_holes.outer_contour, holes);
  if (holes.empty()) {
    return polygon_with_holes.outer_contour;
  }
  auto hole_it = holes.begin();
  while(hole_it!= holes.end()) {
    if (hole_it->size() < 3) {
      hole_it = holes.erase(hole_it);
    } else if (abs(hole_it->area()) < parameters_.hole_area_threshold_squared_meters) {
      hole_it = holes.erase(hole_it);
    } else {
      ++hole_it;
    }
  }
  std::vector<CgalPolygonWithHoles2d> difference_polygons;
  difference_polygons.emplace_back(polygon_with_holes.outer_contour);
  for (const auto& hole : holes) {
    std::vector<CgalPolygonWithHoles2d> temp_polygon_buffer;
    for (const auto& polygon : difference_polygons) {
      CGAL::difference(polygon, hole, std::back_inserter(temp_polygon_buffer));
    }
    difference_polygons.clear();
    std::move(temp_polygon_buffer.begin(), temp_polygon_buffer.end(), std::back_inserter(difference_polygons));
  }
  CgalPolygon2d outer_contour = difference_polygons.rbegin()->outer_boundary();
  holes.clear();
  std::move(difference_polygons.rbegin()->holes_begin(), difference_polygons.rbegin()->holes_end(), std::back_inserter(holes));
  for (const auto& hole : holes){
    if (CGAL::do_intersect(hole, outer_contour)){
      // At this point intersections may only occur in single points. Overlapping edges are not possible.

    }
  }
}