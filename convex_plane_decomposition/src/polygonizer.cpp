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
  for (auto &contour : contours) {
    ++hierachy_it;
    std::vector<cv::Point> approx_contour;
    if (contour.size() <= 10) {
      approx_contour = contour;
    } else {
      cv::approxPolyDP(contour, approx_contour, 9, true);
    }
    if (approx_contour.size() <= 2) {
      LOG_IF(WARNING, hierarchy[hierachy_it - 1][3] < 0 && contour.size() > 3)
          << "Removing parental polygon since too few vertices!";
      continue;
    }
    CgalPolygon2d polygon = convex_plane_extraction::createCgalPolygonFromOpenCvPoints(
        approx_contour.begin(),
        approx_contour.end(),
        parameters_.resolution / parameters_.upsampling_factor);

    if (!polygon.is_simple()) {
      convex_plane_extraction::Vector2i index;
      index << (*polygon.begin()).x(), (*polygon.begin()).y();
      LOG(WARNING) << "Polygon starting at " << index << " is not simple, will be ignored!";
      continue;
    }
    constexpr int kParentFlagIndex = 3;
    if (hierarchy[hierachy_it - 1][kParentFlagIndex] < 0) {
      if (parameters_.activate_long_edge_upsampling) {
        upSampleLongEdges(&polygon);
      }
      if (parameters_.activate_contour_approximation) {
        approximateContour(&polygon);
      }
      plane_polygons.outer_contour = polygon;
    } else {
      plane_polygons.holes.push_back(polygon);
    }
  }
  return plane_polygons;
}

CgalPolygon2d Polygonizer::resolveHoles(PolygonWithHoles& polygon_with_holes) const{
  CgalPolygon2dContainer& holes = polygon_with_holes.holes;
  if (holes.empty()) {
    return polygon_with_holes.outer_contour;
  }
  auto hole_it = holes.begin();
  while(!holes.empty()) {
    if (hole_it == holes.end()){
      hole_it = holes.begin();
    }
    // Compute distance to outer contour for each hole.
    std::vector<double> distance_to_outer_polygon;
    std::vector<int> outer_contour_connection_vertex_positions; // closest outer contour vertex for each hole
    std::vector<int> hole_connection_vertex_positions;
    if (hole_it->size() < 3) {
      ++hole_it;
      holes.erase(std::prev(hole_it));
    } else if (abs(hole_it->area()) < parameters_.hole_area_threshold_squared_meters) {
      ++hole_it;
      holes.erase(std::prev(hole_it));
    }  else {
      std::cout << "List size: " << holes.size() << std::endl;
      std::cout << "Outer: " << std::endl;
      printPolygon(polygon_with_holes.outer_contour);
      std::cout << "Hole: " << std::endl;
      printPolygon(*holes.begin());
      std::multimap<double, std::pair<int, int>> connection_candidates;
      slConcavityHoleVertexSorting(*hole_it, &connection_candidates);
      CHECK(connection_candidates.size() == hole_it->size()*outer_polygon_.size());
      // Instead of extracting only the 2 SL-concavity points, sort vertices according to SL-concavity measure.
      // Then iterate over sorted vertices until a connection to the outer contour is not intersecting the polyogn.
      // Implement function that checks for intersections with existing polygon contour.
      int hole_connection_vertex_position;
      int outer_polygon_connection_vertex_position;
      bool valid_connection = true;
      int i = 0;
      for (const auto& position : connection_candidates){
        auto hole_vertex_it = hole_it->vertices_begin();
        std::advance(hole_vertex_it, position.second.first);
        bool print_flag = false;
        if (position.second.first == 7){
          print_flag = true;
        }
        CgalPoint2d hole_vertex = *hole_vertex_it;
        // Get outer contour connection vertices sorted according to distance to hole vertex.
        auto outer_vertex_it = outer_polygon_.vertices_begin();
        std::advance(outer_vertex_it, position.second.second);
        CgalVector2d direction = *outer_vertex_it - hole_vertex;
        direction = (1.0/direction.squared_length()) * direction;
        CgalSegment2d connection(hole_vertex + 0.0001 * direction, *outer_vertex_it - 0.0001 * direction);
        if (!doPolygonAndSegmentIntersect(outer_polygon_, connection, print_flag)){
          valid_connection = true;
          for (auto hole_iterator = hole_polygon_list_.begin(); hole_iterator != hole_polygon_list_.end(); ++hole_iterator){
            if (hole_iterator->size() < 3){
              continue;
            }
            if (doPolygonAndSegmentIntersect(*hole_iterator, connection, print_flag)) {
              valid_connection = false;
              //std::cout << "Hole and connection intersect!" << std::endl;
              ++i;
              break;
            }
          }
        } else {
          //std::cout << "Outer contour and connection intersect!" << std::endl;
          ++i;
          valid_connection = false;
        }
        if (valid_connection){
          hole_connection_vertex_position = position.second.first;
          outer_polygon_connection_vertex_position = position.second.second;
          break;
        }
      }
      if (!valid_connection){
        CHECK_EQ(i, hole_it->size()*outer_polygon_.size());
        LOG(WARNING) << "No valid connection found! " << i << " iterations.";
        ++hole_it;
        continue;
      }
      CHECK(valid_connection);
      // Perform the integration of the hole into the outer contour.
      auto outer_contour_connection_vertex = outer_polygon_.vertices_begin();
      std::advance(outer_contour_connection_vertex, outer_polygon_connection_vertex_position);
      auto hole_contour_connection_vertex = hole_it->vertices_begin();
      std::advance(hole_contour_connection_vertex, hole_connection_vertex_position);
      CgalPolygon2d new_polygon;
      // Start filling new polygon with outer contour.
      for (auto vertex_it = outer_polygon_.vertices_begin(); vertex_it != outer_contour_connection_vertex; ++vertex_it){
        new_polygon.push_back(*vertex_it);
      }
      new_polygon.push_back(*outer_contour_connection_vertex);
      auto hole_vertex_it = hole_contour_connection_vertex;
      do {
        new_polygon.push_back(*hole_vertex_it);
        if (hole_vertex_it != std::prev(hole_it->vertices_end())){
          hole_vertex_it = std::next(hole_vertex_it);
        } else {
          hole_vertex_it = hole_it->vertices_begin();
        }
      } while(hole_vertex_it != hole_contour_connection_vertex);
      // Create new vertices next to connection points to avoid same enter and return path.
      CgalSegment2d enter_connection(*outer_contour_connection_vertex, *hole_contour_connection_vertex);
      Eigen::Vector2d normal_vector;
      getSegmentNormalVector(enter_connection, &normal_vector);
      // Check for possible intersections.
      Eigen::Vector2d line_start_point(hole_contour_connection_vertex->x(), hole_contour_connection_vertex->y());
      Eigen::Vector2d direction_vector(outer_contour_connection_vertex->x() - hole_contour_connection_vertex->x(),
                                       outer_contour_connection_vertex->y() - hole_contour_connection_vertex->y());
      auto next_vertex = std::next(outer_contour_connection_vertex);
      if (next_vertex == outer_polygon_.vertices_end()){
        next_vertex = outer_polygon_.begin();
      }
      Eigen::Vector2d test_point(next_vertex->x(), next_vertex->y());
      constexpr double kPointOffset = 0.0001;
      Eigen::Vector2d outer_point(outer_contour_connection_vertex->x(), outer_contour_connection_vertex->y());
      if (isPointOnLeftSide(line_start_point, direction_vector, test_point)){
        Eigen::Vector2d translation_vector(next_vertex->x() - outer_contour_connection_vertex->x(),
                                           next_vertex->y() - outer_contour_connection_vertex->y());
        translation_vector.normalize();
        outer_point += kPointOffset * translation_vector.normalized();
      } else {
        outer_point += kPointOffset * normal_vector.normalized();
      }
      next_vertex = hole_contour_connection_vertex;
      if (next_vertex == hole_it->vertices_begin()){
        next_vertex = hole_it->vertices_end();
        next_vertex = std::prev(next_vertex);
      } else {
        next_vertex = std::prev(next_vertex);
      }
      Eigen::Vector2d hole_point(hole_contour_connection_vertex->x(), hole_contour_connection_vertex->y());
      test_point << next_vertex->x(), next_vertex->y();
      constexpr double kPiHalf = 3.1416 / 2;
      if (isPointOnLeftSide(line_start_point, direction_vector, test_point)
          && computeAngleBetweenVectors(test_point - line_start_point, direction_vector) < kPiHalf){
        Eigen::Vector2d translation_vector(next_vertex->x() - hole_contour_connection_vertex->x(),
                                           next_vertex->y() - hole_contour_connection_vertex->y());
        translation_vector.normalize();
        hole_point += kPointOffset * translation_vector;
      } else {
        hole_point += kPointOffset * normal_vector;
      }
      Vector2d new_edge_direction = outer_point - hole_point;
      new_edge_direction.normalize();
      constexpr double kShiftFactor = 0.0001;
      bool intersection_caused = false;
      CgalSegment2d new_edge(CgalPoint2d(hole_point.x() + kShiftFactor * new_edge_direction.x(),
                                         hole_point.y() + kShiftFactor * new_edge_direction.y()),CgalPoint2d(outer_point.x() -
          kShiftFactor * new_edge_direction.x(), outer_point.y() - kShiftFactor * new_edge_direction.y()));
      if (!doPolygonAndSegmentIntersect(outer_polygon_, new_edge, false)){
        for (auto hole_iterator = hole_polygon_list_.begin(); hole_iterator != hole_polygon_list_.end(); ++hole_iterator){
          if (hole_iterator->size() < 3){
            continue;
          }
          if (doPolygonAndSegmentIntersect(*hole_iterator, new_edge, false)) {
            intersection_caused = true;
            break;
          }
        }
      } else {
        //std::cout << "Outer contour and connection intersect!" << std::endl;
        intersection_caused = true;
      }
      if (intersection_caused){
        ++hole_it;
        continue;
      }
      // Add new vertices to outer contour.
      new_polygon.push_back(CgalPoint2d(hole_point.x(), hole_point.y()));
      new_polygon.push_back(CgalPoint2d(outer_point.x(), outer_point.y()));
      // Add remaining outer contour vertices to new polygon.
      auto vertex_it = outer_contour_connection_vertex;
      vertex_it = std::next(vertex_it);
      for (; vertex_it != outer_polygon_.vertices_end(); ++vertex_it){
        new_polygon.push_back(*vertex_it);
      }
      if (!new_polygon.is_simple()){
        std::cout << "Hole orientation: " << hole_it->orientation() << std::endl;
        std::cout <<"Hole vertex: " << *hole_contour_connection_vertex << std::endl;
        std::cout <<"Outer vertex: " << *outer_contour_connection_vertex << std::endl;
        printPolygon(new_polygon);
      }
      CHECK(new_polygon.is_simple());
      outer_polygon_ = new_polygon;
      ++hole_it;
      hole_polygon_list_.erase(std::prev(hole_it));
    }
  }
  //approximateContour(&outer_polygon_);
  CHECK(outer_polygon_.is_simple());
}