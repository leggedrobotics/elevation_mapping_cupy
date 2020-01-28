#include "polygon.hpp"

namespace convex_plane_extraction {

  void performConvexDecomposition(const CgalPolygon2d& polygon, CgalPolygon2dListContainer* output_polygon_list){
    CHECK_GE(polygon.size(), 3);
    CHECK(polygon.is_simple());
    LOG(INFO) << "Started convex decomposition...";
    size_t old_list_size = output_polygon_list->size();
//    CGAL::optimal_convex_partition_2(polygon.vertices_begin(),
//                                     polygon.vertices_end(),
//                                     std::back_inserter(*output_polygon_list));
//
//    assert(CGAL::partition_is_valid_2(polygon.vertices_begin(),
//                                      polygon.vertices_end(),
//                                      polygon_list.begin(),
//                                      polygon_list.end()));
    *output_polygon_list = decomposeInnerApproximation(polygon);
    CHECK_GT(output_polygon_list->size(), old_list_size);
    LOG(INFO) << "done.";
  }

  bool doPolygonAndSegmentIntersect(const CgalPolygon2d& polygon, const CgalSegment2d& segment){
    for (auto vertex_it = polygon.vertices_begin(); vertex_it != polygon.vertices_end(); ++vertex_it){
      auto next_vertex_it = std::next(vertex_it);
      if (next_vertex_it == polygon.vertices_end()){
        next_vertex_it = polygon.vertices_begin();
      }
      CgalSegment2d test_segment(*vertex_it, *next_vertex_it);
      if (do_intersect(test_segment, segment)){
        return true;
      }
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
          constexpr double areaThresholdFactor = 0.01;
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

  void detectDentLocations(std::vector<int>* dent_locations, const CgalPolygon2d& polygon){
    CHECK_NOTNULL(dent_locations);
    CHECK(dent_locations->empty());
    CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
    for (auto source_it = polygon.vertices_begin(); source_it != polygon.vertices_end(); ++source_it){
      auto dent_it = next(source_it, polygon);
      auto destination_it = next(dent_it, polygon);
      Vector2d source_point = Vector2d(source_it->x(), source_it->y());
      Vector2d direction_vector = Vector2d(destination_it->x() - source_it->x(), destination_it->y() - source_it->y());
      Vector2d test_point = Vector2d(dent_it->x(), dent_it->y());
      if(isPointOnLeftSide(source_point, direction_vector, test_point)){
        dent_locations->push_back(std::distance(polygon.vertices_begin(), dent_it));
      }
    }
  }

  std::list<CgalPolygon2d> decomposeInnerApproximation(const CgalPolygon2d& polygon){
    CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
    std::list<CgalPolygon2d> return_list;
    // If polygon ceonvex terminate recursion.
    if(polygon.is_convex()){
      return_list.push_back(polygon);
      return return_list;
    }
    std::vector<int> dent_locations;
    detectDentLocations(&dent_locations, polygon);
    CHECK(!dent_locations.empty()); // If no dents, polygon would be convex, which should have terminated recursion.
    std::cout << "Dent locations:" << std::endl;
    for (int location : dent_locations){
      std::cout << location << std::endl;
    }
    std::cout << "Polygon:" << std::endl;
    std::cout << polygon << std::endl;
    Intersection intersection_clockwise;
    Intersection intersection_counterclockwise;
    intersectPolygonWithRay(dent_locations.at(0), CGAL::COUNTERCLOCKWISE,
        polygon, &intersection_counterclockwise);
    intersectPolygonWithRay(dent_locations.at(0), CGAL::CLOCKWISE,
        polygon, &intersection_clockwise);
    // Generate resulting polygons from cut.
    // Resulting cut from counter clockwise ray intersection.
    CgalPolygon2d polygon_counterclockwise_1 = polygon;
    CgalPolygon2d polygon_counterclockwise_2;
    auto first_vertex_to_erase_it = polygon_counterclockwise_1.vertices_begin();
    std::advance(first_vertex_to_erase_it, dent_locations.at(0));
    polygon_counterclockwise_2.push_back(*first_vertex_to_erase_it);
    first_vertex_to_erase_it = next(first_vertex_to_erase_it, polygon_counterclockwise_1);
//    for (int i = 0; i < 2; ++i){
//      first_vertex_to_erase_it = previous(first_vertex_to_erase_it, polygon);
//    }
    auto last_vertex_to_erase_it = polygon_counterclockwise_1.vertices_begin();
    std::advance(last_vertex_to_erase_it, intersection_counterclockwise.edge_source_location_);
//    auto last_vertex_to_erase_it = polygon.vertices_begin();
//    std::advance(last_vertex_to_erase_it, intersection_counterclockwise.edge_source_location_);
    last_vertex_to_erase_it = next(last_vertex_to_erase_it, polygon_counterclockwise_1);
    copyVertices(polygon_counterclockwise_1,first_vertex_to_erase_it,last_vertex_to_erase_it,
    &polygon_counterclockwise_2, polygon_counterclockwise_2.vertices_end());
    std::cout << "Copy succeeded without problems!" << std::endl;
//    polygon_counterclockwise_2.insert(polygon_counterclockwise_2.vertices_end(),
//        first_vertex_to_erase_it, last_vertex_to_erase_it);
    polygon_counterclockwise_2.push_back(intersection_counterclockwise.intersection_point_);
//    auto element_behind_deleted_it = polygon_counterclockwise_1.erase(first_vertex_to_erase_it, std::next(last_vertex_to_erase_it));
    CgalPolygon2dVertexIterator element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_counterclockwise_1);
    polygon_counterclockwise_1.insert(element_behind_deleted_it, intersection_counterclockwise.intersection_point_);
    std::cout << "First polygon counter-clockwise: " << polygon_counterclockwise_1 << std::endl;
    std::cout << "Second polygon counter-clockwise: " << polygon_counterclockwise_2 << std::endl;
    // Resulting cut from clockwise ray intersection.
    CgalPolygon2d polygon_clockwise_1 = polygon;
    CgalPolygon2d polygon_clockwise_2;
    polygon_clockwise_2.push_back(intersection_clockwise.intersection_point_);
    first_vertex_to_erase_it = polygon_clockwise_1.vertices_begin();
    std::advance(first_vertex_to_erase_it, intersection_clockwise.edge_target_location_);
    last_vertex_to_erase_it = polygon_clockwise_1.vertices_begin();
    std::advance(last_vertex_to_erase_it, dent_locations.at(0));
    copyVertices(polygon_clockwise_1, first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_2,
                 polygon_clockwise_2.vertices_end());
    erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_1);
    polygon_clockwise_2.push_back(*last_vertex_to_erase_it);
//    std::advance(first_element_to_insert_it, intersection_clockwise.edge_target_location_);
//    copyVertices(polygon, first_element_to_insert_it, first_vertex_to_erase_it, &polygon_clockwise_2,
//        polygon_clockwise_2.vertices_end());

    // Take the cut with smallest min. area of resulting polygons.
    double area[4] = {polygon_counterclockwise_1.area(), polygon_counterclockwise_2.area(),
                      polygon_clockwise_1.area(), polygon_clockwise_2.area()};
    double* min_area_strategy = std::min_element(area, area+4);
    std::list<CgalPolygon2d> recursion_1;
    std::list<CgalPolygon2d> recursion_2;
    if ((min_area_strategy - area) < 2){
      // In this case the counter clockwise intersection leads to less loss in area.
      // Perform recursion with this split.
      if(polygon_counterclockwise_1.orientation() != CGAL::COUNTERCLOCKWISE){
        LOG(FATAL) << polygon_counterclockwise_1;
      }
      if(polygon_counterclockwise_2.orientation() != CGAL::COUNTERCLOCKWISE){
        LOG(FATAL) << polygon_counterclockwise_2;
      }
      recursion_1 = decomposeInnerApproximation(polygon_counterclockwise_1);
      recursion_2 = decomposeInnerApproximation(polygon_counterclockwise_2);
    } else {
      // In this case the clockwise intersection leads to less loss in area.
      if(polygon_clockwise_1.orientation() != CGAL::COUNTERCLOCKWISE){
        LOG(FATAL) << polygon_clockwise_1;
      }
      if(polygon_clockwise_2.orientation() == CGAL::COUNTERCLOCKWISE){
        LOG(FATAL) << polygon_clockwise_2;
      }
      recursion_1 = decomposeInnerApproximation(polygon_clockwise_1);
      recursion_2 = decomposeInnerApproximation(polygon_clockwise_2);
    }
    return_list.splice(return_list.end(), recursion_1);
    return_list.splice(return_list.end(), recursion_2);
    // Check that returned decomposition is in fact convex.
    for (const CgalPolygon2d& polygon_under_test : return_list){
      CHECK(polygon_under_test.is_convex());
    }
    return return_list;
  }

  void intersectPolygonWithRay(int ray_source_location, CGAL::Orientation orientation, const CgalPolygon2d& polygon,
                               Intersection* intersection){
    CHECK_NOTNULL(intersection);
    CHECK(orientation == CGAL::COUNTERCLOCKWISE || orientation == CGAL::CLOCKWISE);
    CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
    Intersection intersection_tmp;
    constexpr double kLargeDistanceValue = 1000;
    double min_distance = kLargeDistanceValue;
    bool one_intersection_at_least = false;
    auto vertex_it = polygon.vertices_begin();
    std::advance(vertex_it, ray_source_location);
    auto ray_source_it = vertex_it;
    CgalPolygon2dVertexIterator ray_target_it = vertex_it;
    if (orientation == CGAL::COUNTERCLOCKWISE) {
      ray_source_it = previous(vertex_it, polygon);
    } else {
      vertex_it = next(vertex_it, polygon);
      ray_source_it = vertex_it;
    }
    Vector2d ray_source = Vector2d(ray_source_it->x(), ray_source_it->y());
    Vector2d ray_target = Vector2d(ray_target_it->x(), ray_target_it->y());
    Vector2d ray_direction = ray_target - ray_source;
    std::cout << "Ray direction: " << ray_direction << std::endl;
    vertex_it = next(vertex_it, polygon);
    auto condition_it = ray_source_it;
    if (orientation == CGAL::CLOCKWISE){
      //vertex_it = next(vertex_it, polygon); // Compensate for previous above.
      condition_it = ray_target_it;
    }
    while(next(vertex_it, polygon) != condition_it){
      Vector2d segment_source = Vector2d(vertex_it->x(), vertex_it->y());
      auto segment_target_it = next(vertex_it, polygon);
      Vector2d segment_target = Vector2d(segment_target_it->x(), segment_target_it->y());
      Vector2d intersection_point;
      if (intersectRayWithLineSegment(ray_source, ray_direction, segment_source, segment_target,
          &intersection_point)){
        std::cout << "Intersection succeeded with: " << std::endl;
        std::cout << "Segment source: " << segment_source << std::endl;
        std::cout << "Segment target: " << segment_target << std::endl;
        std::cout << "Intersection point: " << intersection_point << std::endl;
          double current_distance = distanceBetweenPoints(ray_target, intersection_point);
        if (current_distance < min_distance){
          one_intersection_at_least = true;
          min_distance = current_distance;
          intersection_tmp.setAllMembers(std::distance(polygon.vertices_begin(), vertex_it),
          std::distance(polygon.vertices_begin(), segment_target_it), CgalPoint2d(intersection_point.x(), intersection_point.y()));
        }
      }
      vertex_it = next(vertex_it, polygon);
    }
    CHECK(one_intersection_at_least);
    intersection->setAllMembers(intersection_tmp.edge_source_location_, intersection_tmp.edge_target_location_,
        intersection_tmp.intersection_point_);
  }

  CgalPolygon2dVertexIterator erase(CgalPolygon2dVertexIterator first, CgalPolygon2dVertexIterator last,
      CgalPolygon2d* polygon){
    CHECK_NOTNULL(polygon);
    CHECK(first != last);
    std::cout << "Starting erasing!" << std::endl;
    if ((std::distance(polygon->vertices_begin(), last) - std::distance(polygon->vertices_begin(), first)) > 0){
      std::cout << (std::distance(polygon->vertices_begin(), last) - std::distance(polygon->vertices_begin(), first)) << std::endl;
      return polygon->erase(first, last);
    } else {
      std::cout << "Erasing overflow!" << std::endl;
      int last_iterator_distance = std::distance(polygon->vertices_begin(), last);
      polygon->erase(first, polygon->vertices_end());
      auto erase_iterator = polygon->vertices_begin();
      std::advance(erase_iterator, last_iterator_distance);
      return polygon->erase(polygon->vertices_begin(), erase_iterator);
    }
  }
  // Solves the wrap around issue.
  // [first last) are copied to new_polygon before the position indicated ny insert_position.
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

}
