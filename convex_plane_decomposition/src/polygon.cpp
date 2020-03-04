#include "polygon.hpp"

namespace convex_plane_extraction {

  bool doPolygonAndSegmentIntersect(const CgalPolygon2d& polygon, const CgalSegment2d& segment, bool print_flag){
    for (auto vertex_it = polygon.vertices_begin(); vertex_it != polygon.vertices_end(); ++vertex_it){
      auto next_vertex_it = std::next(vertex_it);
      if (next_vertex_it == polygon.vertices_end()){
        next_vertex_it = polygon.vertices_begin();
      }
      CgalSegment2d test_segment(*vertex_it, *next_vertex_it);
      CGAL::cpp11::result_of<Intersect_2(CgalSegment2d, CgalSegment2d)>::type
          result = intersection(test_segment, segment);
      if (result) {
        if (const CgalSegment2d *s = boost::get<CgalSegment2d>(&*result)) {
          std::cout << *s << std::endl;
          return true;
        } else {
          const CgalPoint2d *p = boost::get<CgalPoint2d>(&*result);
          if (print_flag) {
            std::cout << *p << ";" << std::endl;
          }
          return true;
        }
      }
//      Vector2d test_segment_source(vertex_it->x(), vertex_it->y());
//      Vector2d test_segment_target(next_vertex_it->x(), next_vertex_it->y());
//      Vector2d segment_source(segment.source().x(), segment.source().y());
//      Vector2d segment_target(segment.target().x(), segment.target().y());
//      Vector2d intersection_point;
//      if (intersectLineSegmentWithLineSegment(test_segment_source, test_segment_target,
//      segment_source, segment_target, &intersection_point)){
//        std::cout << intersection_point.x() << ", " << intersection_point.y() << ";" << std::endl;
//        return true;
//      }
//      if (do_intersect(test_segment, segment)){
//        return true;
//      }
    }
    return false;
  }

  int getClosestPolygonVertexPosition(const CgalPolygon2d& polygon, const CgalPoint2d& point){
    int closest_vertex_position = 0;
    double smallest_distance = std::numeric_limits<double>::infinity();
    int position = 0;
    for(auto vertex_it = polygon.vertices_begin(); vertex_it != polygon.vertices_end(); ++vertex_it){
      double temp_distance = squared_distance(*vertex_it, point);
      if (temp_distance < smallest_distance){
        smallest_distance = temp_distance;
        closest_vertex_position = position;
      }
      ++position;
    }
    return closest_vertex_position;
  }

  void getVertexPositionsInAscendingDistanceToPoint(const CgalPolygon2d& polygon, const CgalPoint2d& point,
      std::multimap<double, int>* vertex_positions){
    CHECK_NOTNULL(vertex_positions);
    int position = 0;
    for (auto vertex_it = polygon.vertices_begin(); vertex_it != polygon.vertices_end(); ++vertex_it) {
      vertex_positions->insert(std::pair<double, int>(squared_distance(*vertex_it, point), position));
      ++position;
    }
  }

  void getSegmentNormalVector(const CgalSegment2d& segment, Eigen::Vector2d* normal_vector){
    CHECK_NOTNULL(normal_vector);
    const auto first_vertex = segment.source();
    const auto second_vertex = segment.target();
    double d_x = second_vertex.x() - first_vertex.x();
    double d_y = second_vertex.y() - first_vertex.y();
    Eigen::Vector2d normal(d_y, -d_x);
    normal.normalize();
    *normal_vector = normal;
  }

  void approximateContour(CgalPolygon2d* polygon){
    CHECK_NOTNULL(polygon);
    if (polygon->size() < 4){
      return;
    }
    CHECK(polygon->orientation() == CGAL::COUNTERCLOCKWISE);
    int old_size;
    auto first_vertex_it = polygon->vertices_begin();
    auto second_vertex_it = std::next(first_vertex_it);
    auto third_vertex_it = std::next(second_vertex_it);
    double area = polygon->area();
    do {
      old_size = polygon->size();
      for (int vertex_position = 0; vertex_position < old_size; ++vertex_position){
        if (polygon->size() < 4){
          break;
        }
        Vector2d first_point(first_vertex_it->x(), first_vertex_it->y());
        Vector2d second_point(second_vertex_it->x(), second_vertex_it->y());
        Vector2d third_point(third_vertex_it->x(), third_vertex_it->y());
        if (isPointOnRightSide(first_point, third_point - first_point, second_point)) {
          LOG(INFO) << "Point on right side!";
          double a = (third_point - first_point).norm();
          double b = (second_point - third_point).norm();
          double c = (first_point - second_point).norm();
          constexpr double areaThresholdFactor = 0.025;
          if (computeTriangleArea(a, b, c) < areaThresholdFactor * area) {
            LOG(INFO) << "Area sufficiently small!";
            CgalPolygon2d new_polygon(*polygon);
            int vertex_position_offset = std::distance(polygon->vertices_begin(), second_vertex_it);
            CgalPolygon2dVertexIterator tmp_iterator = new_polygon.vertices_begin();
            std::advance(tmp_iterator, vertex_position_offset);
            LOG(INFO) << "Before erase call!";
            new_polygon.erase(tmp_iterator);
            LOG(INFO) << "After ease call!";
            if (new_polygon.is_simple()) {
              first_vertex_it = third_vertex_it;
              polygon->erase(second_vertex_it);
              second_vertex_it = next(first_vertex_it, *polygon);
              third_vertex_it = next(second_vertex_it, *polygon);
              LOG(INFO) << "Removed one vertex!";
              continue;
            }
          }
        }
        first_vertex_it = second_vertex_it;
        second_vertex_it = third_vertex_it;
        third_vertex_it = next(second_vertex_it, *polygon);
      }
    } while (polygon->size() < old_size);
  }

  double computeTriangleArea(double side_length_a, double side_length_b, double side_length_c){
    double s = (side_length_a + side_length_b + side_length_c) / 2.0;
    return sqrt(s * (s-side_length_a) * (s - side_length_b) * (s - side_length_c));
  }

  CgalPolygon2dVertexIterator next(const CgalPolygon2dVertexIterator& iterator, const CgalPolygon2d& polygon){
    if (std::next(iterator) == polygon.vertices_end()){
      return polygon.vertices_begin();
    } else {
      return std::next(iterator);
    }
  }

  CgalPolygon2dVertexIterator previous(const CgalPolygon2dVertexIterator& iterator, const CgalPolygon2d& polygon){
    if (iterator == polygon.vertices_begin()){
      return std::prev(polygon.vertices_end());
    } else {
      return std::prev(iterator);
    }
  }

  void upSampleLongEdges(CgalPolygon2d* polygon){
    CHECK_NOTNULL(polygon);
    for (auto vertex_it = polygon->vertices_begin(); vertex_it != polygon->vertices_end(); ++vertex_it) {
      double edge_length = getEdgeLength(vertex_it, *polygon);
      CHECK_GT(edge_length, 0);
      double kEdgeLengthThreshold = 0.5;
      if (edge_length > 2 * kEdgeLengthThreshold){
        int number_of_intervals = static_cast<int>(round(ceil(edge_length / kEdgeLengthThreshold)));
        CgalPoint2d source_point = *vertex_it;
        CgalPoint2d destination_point = *(next(vertex_it, *polygon));
        CHECK_NE(number_of_intervals , 0);
        CgalPoint2d direction_vector = CgalPoint2d((destination_point.x() - source_point.x()) /
            static_cast<double>(number_of_intervals),(destination_point.y() - source_point.y()) /
            static_cast<double>(number_of_intervals));

        CHECK_LE(abs(sqrt(pow(direction_vector.x(),2) + pow(direction_vector.y(),2))),edge_length);
        std::cout << "Direction vector: " << direction_vector << std::endl;
        auto inserter_it = next(vertex_it, *polygon);
        for (int i = 0; i < number_of_intervals-1; ++i){
          CgalVector2d new_point = *inserter_it - direction_vector;
          inserter_it = polygon->insert(inserter_it, CgalPoint2d(new_point.x(), new_point.y()));
        }
        std::advance(vertex_it, number_of_intervals - 1);
      }
    }
  }

  double getEdgeLength(const CgalPolygon2dVertexIterator& source, const CgalPolygon2d& polygon) {
    CgalPoint2d source_point = *source;
    CgalPoint2d destination_point = *(next(source, polygon));
    return abs(sqrt((destination_point.x() - source_point.x()) * (destination_point.x() - source_point.x())
        + (destination_point.y() - source_point.y()) * (destination_point.y() - source_point.y())));
  }

  // Counter-clockwise orientation: ray source vertex is previous vertex in counter-clockwise polygon orientation.
  bool intersectPolygonWithRay(int ray_target_location, CGAL::Orientation orientation, const CgalPolygon2d& polygon,
                               Intersection* intersection){
    CHECK_NOTNULL(intersection);
    CHECK(orientation == CGAL::COUNTERCLOCKWISE || orientation == CGAL::CLOCKWISE);
    CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
    Intersection intersection_tmp;
    constexpr double kLargeDistanceValue = 1000;
    double min_distance = kLargeDistanceValue;
    bool one_intersection_at_least = false;
    auto vertex_it = polygon.vertices_begin();
    std::advance(vertex_it, ray_target_location);
    auto ray_source_it = vertex_it;
    CgalPolygon2dVertexIterator ray_target_it = vertex_it;
    if (orientation == CGAL::COUNTERCLOCKWISE) {
      ray_source_it = previous(vertex_it, polygon);
    } else {
      vertex_it = next(vertex_it, polygon);
      ray_source_it = vertex_it;
    }
    Vector2d ray_source = Vector2d(ray_source_it->x(), ray_source_it->y());
    LOG(INFO) << "Source ray:" << ray_source;
    Vector2d ray_target = Vector2d(ray_target_it->x(), ray_target_it->y());
    Vector2d ray_direction = ray_target - ray_source;
    vertex_it = next(vertex_it, polygon);
    // Do not intersect with adjacent edges since the intersect already in common vertex.
    auto condition_it = ray_source_it;
    if (orientation == CGAL::CLOCKWISE){
      condition_it = ray_target_it;
    }
    while(next(vertex_it, polygon) != condition_it){
      Vector2d segment_source = Vector2d(vertex_it->x(), vertex_it->y());
      auto segment_target_it = next(vertex_it, polygon);
      Vector2d segment_target = Vector2d(segment_target_it->x(), segment_target_it->y());
      Vector2d intersection_point;
      if (intersectRayWithLineSegment(ray_source, ray_direction, segment_source, segment_target,
          &intersection_point)){
        LOG(INFO) << "Intersection point:" << intersection_point;
        LOG(INFO) << segment_source;
        if ((ray_target - intersection_point).norm() < 0.001){
          LOG(INFO) << segment_source;
          vertex_it = next(vertex_it, polygon);
          continue;
        }
        double current_distance = distanceBetweenPoints(ray_target, intersection_point);
        // Take first intersection on ray.
        if (current_distance < min_distance){
          one_intersection_at_least = true;
          min_distance = current_distance;
          intersection_tmp.setAllMembers(std::distance(polygon.vertices_begin(), vertex_it),
          std::distance(polygon.vertices_begin(), segment_target_it), CgalPoint2d(intersection_point.x(), intersection_point.y()));
        }
      }
      vertex_it = next(vertex_it, polygon);
    }

    if (!one_intersection_at_least){
      return false;
    }
    intersection->setAllMembers(intersection_tmp.edge_source_location_, intersection_tmp.edge_target_location_,
        intersection_tmp.intersection_point_);
    return true;
  }

  // Erases vertices [first, last), takes warp around into account.
  CgalPolygon2dVertexIterator erase(CgalPolygon2dVertexIterator first, CgalPolygon2dVertexIterator last,
      CgalPolygon2d* polygon){
    CHECK_NOTNULL(polygon);
    CHECK(first != last);
    if ((std::distance(polygon->vertices_begin(), last) - std::distance(polygon->vertices_begin(), first)) > 0){
      // std::cout << (std::distance(polygon->vertices_begin(), last) - std::distance(polygon->vertices_begin(), first)) << std::endl;
      return polygon->erase(first, last);
    } else {
      int last_iterator_distance = std::distance(polygon->vertices_begin(), last);
      polygon->erase(first, polygon->vertices_end());
      auto erase_iterator = polygon->vertices_begin();
      std::advance(erase_iterator, last_iterator_distance);
      return polygon->erase(polygon->vertices_begin(), erase_iterator);
    }
  }

  // [first last) are copied to new_polygon before the position indicated by insert_position.
  // Takes wrap around into account
  void copyVertices(const CgalPolygon2d& old_polygon, const CgalPolygon2dVertexIterator first, const CgalPolygon2dVertexIterator last,
      CgalPolygon2d* new_polygon, const CgalPolygon2dVertexIterator insert_position){
    CHECK_NOTNULL(new_polygon);
    CHECK(first != last);
    if (std::distance(old_polygon.vertices_begin(), last) - std::distance(old_polygon.vertices_begin(), first) > 0){
      return new_polygon->insert(insert_position, first, last);
    } else {
      int insert_position_distance = std::distance(new_polygon->vertices_begin(), insert_position);
      int number_of_inserted_elements = std::distance(first, old_polygon.vertices_end());
      new_polygon->insert(insert_position, first, old_polygon.vertices_end());
      CgalPolygon2dVertexIterator insert_position_mutable = new_polygon->vertices_begin();
      std::advance(insert_position_mutable, insert_position_distance + number_of_inserted_elements);
      new_polygon->insert(insert_position_mutable, old_polygon.vertices_begin(), last);
      return;
    }
  }

  void printPolygon(const CgalPolygon2d& polygon){
    std::cout << "Polygon has " << polygon.size() << " vertices." << std::endl;
    for (const CgalPoint2d& vertex : polygon){
      std::cout << vertex.x() << " , " << vertex.y() << " ; " << std::endl;
    }
  }

//  void slConcavityHoleVertexSorting(const CgalPolygon2d& hole, std::multimap<double, std::pair<int, int>>* concavity_positions){
//    CHECK_NOTNULL(concavity_positions);
//    for (auto vertex_it = hole.vertices_begin(); vertex_it != hole.vertices_end(); ++vertex_it){
//      std::multimap<double, int> outer_polygon_vertices;
//      getVertexPositionsInAscendingDistanceToPoint(outer_polygon_, *vertex_it, &outer_polygon_vertices);
//      for (const auto& outer_distance_vertex_pair : outer_polygon_vertices) {
//        concavity_positions->insert(std::pair<double, std::pair<int, int>>(outer_distance_vertex_pair.first,
//                                                                           std::pair<int, int>(std::distance(hole.vertices_begin(), vertex_it),outer_distance_vertex_pair.second)));
//      }
//    }
//  }

  std::pair<int, int> getIndicesOfClosestVertexPair(const CgalPolygon2d& first_polygon, const CgalPolygon2d& second_polygon){
    CHECK_GT(first_polygon.size(), 0);
    CHECK_GT(second_polygon.size(), 0);
    std::multimap<double, std::pair<int, int>> buffer = getClosestVertexPairsOrdered(first_polygon, second_polygon);
    CHECK(!buffer.empty());
    return buffer.begin()->second;
  }

  std::multimap<double, std::pair<int, int>> getClosestVertexPairsOrdered(const CgalPolygon2d& first_polygon, const CgalPolygon2d& second_polygon){
    CHECK_GT(first_polygon.size(), 0);
    CHECK_GT(second_polygon.size(), 0);
    std::multimap<double, std::pair<int, int>> buffer;
    for (auto vertex_it = first_polygon.vertices_begin(); vertex_it != first_polygon.vertices_end(); ++vertex_it){
      int first_vertex_index = std::distance(first_polygon.vertices_begin(), vertex_it);
      double distance = std::numeric_limits<double>::max();
      int second_vertex_index = 0;
      for (auto second_vertex_it = second_polygon.vertices_begin(); second_vertex_it != second_polygon.vertices_begin(); ++second_vertex_it){
        const double temp_distance = sqrt(static_cast<CgalVector2d>(*vertex_it - *second_vertex_it).squared_length());
        if (temp_distance < distance){
          distance = temp_distance;
          second_vertex_index = std::distance(second_polygon.vertices_begin(), second_vertex_it);
        }
      }
      buffer.insert(std::make_pair(distance, std::make_pair(first_vertex_index, second_vertex_index)));
    }
    return buffer;
  }

  CgalPoint2d getPolygonVertexAtIndex(const CgalPolygon2d& polygon, int index){
    CHECK_GT(index, 0);
    CHECK_LT(index, polygon.size());
    auto vertex_it = polygon.vertices_begin();
    std::advance(vertex_it, index);
    return *vertex_it;
  }

  bool doPointAndPolygonIntersect(const CgalPolygon2d& polygon, const CgalPoint2d& point, int& segment_target_vertex_index){
    for (auto vertex_it = polygon.vertices_begin(); vertex_it != polygon.vertices_end(); ++vertex_it) {
      auto next_vertex_it = std::next(vertex_it);
      if (next_vertex_it == polygon.vertices_end()) {
        next_vertex_it = polygon.vertices_begin();
      }
      CgalSegment2d test_segment(*vertex_it, *next_vertex_it);
      if (test_segment.has_on(point)){
        segment_target_vertex_index = std::distance(polygon.vertices_begin(), next_vertex_it);
        return true;
      }
    }
    return false;
  }

}
