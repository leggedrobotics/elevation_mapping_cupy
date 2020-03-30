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
          if (print_flag) {
            VLOG(1) << "Intersection over segment: " << *s;
          }
          return true;
        } else {
          const CgalPoint2d *p = boost::get<CgalPoint2d>(&*result);
          if (print_flag) {
            VLOG(1) << "Intersection in in point: " << *p;
          }
          return true;
        }
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

  void approximateContour(CgalPolygon2d* polygon, int max_number_of_iterations, double relative_local_area_threshold,
                          double absolute_local_area_threshold, double relative_total_area_threshold,
                          double absolute_total_area_threshold) {
    CHECK_NOTNULL(polygon);
    if (polygon->size() < 4) {
      return;
    }
    CHECK(polygon->orientation() == CGAL::COUNTERCLOCKWISE);
    int old_size;
    auto first_vertex_it = polygon->vertices_begin();
    auto second_vertex_it = std::next(first_vertex_it);
    auto third_vertex_it = std::next(second_vertex_it);
    double area = polygon->area();
    int number_of_iterations = 0;
    while (number_of_iterations < max_number_of_iterations ) {
      old_size = polygon->size();
      if (polygon->size() < 4) {
        break;
      }
      Vector2d first_point(first_vertex_it->x(), first_vertex_it->y());
      Vector2d second_point(second_vertex_it->x(), second_vertex_it->y());
      Vector2d third_point(third_vertex_it->x(), third_vertex_it->y());
      VLOG(2) << "Got here!";
      if (isPointOnRightSideOfLine(first_point, third_point - first_point, second_point)) {
        VLOG(2) << "Point on right side!";
        double a = (third_point - first_point).norm();
        double b = (second_point - third_point).norm();
        double c = (first_point - second_point).norm();
        const double triangle_area = computeTriangleArea(a, b, c);
        CHECK(isfinite(triangle_area)) << "Area: " << triangle_area << ", a: " << a << ", b: " << b << ", c: " << c;
        CHECK_GE(triangle_area, 0.0);
        if ((triangle_area < relative_local_area_threshold * area) && (triangle_area < absolute_local_area_threshold)) {
          VLOG(2) << "Area sufficiently small!";
          CgalPolygon2d new_polygon(*polygon);
          int vertex_position_offset = std::distance(polygon->vertices_begin(), second_vertex_it);
          CgalPolygon2dVertexIterator tmp_iterator = new_polygon.vertices_begin();
          std::advance(tmp_iterator, vertex_position_offset);
          VLOG(2) << "Before erase call!";
          new_polygon.erase(tmp_iterator);
          VLOG(2) << "After ease call!";
          if (new_polygon.is_simple() && (new_polygon.orientation() == CGAL::COUNTERCLOCKWISE) &&
              abs(new_polygon.area() - area) < relative_total_area_threshold * area &&
              abs(new_polygon.area() - area) < absolute_total_area_threshold) {
            CHECK_LE(triangle_area, area);
            first_vertex_it = polygon->erase(second_vertex_it);
            if (first_vertex_it == polygon->vertices_end()) {
              first_vertex_it = polygon->vertices_begin();
            }
            second_vertex_it = next(first_vertex_it, *polygon);
            third_vertex_it = next(second_vertex_it, *polygon);
            if ((std::distance(polygon->begin(), first_vertex_it) >= std::distance(polygon->begin(), second_vertex_it)) ||
                (std::distance(polygon->begin(), first_vertex_it) >= std::distance(polygon->begin(), third_vertex_it)) ||
                (std::distance(polygon->begin(), second_vertex_it) >= std::distance(polygon->begin(), third_vertex_it))) {
              ++number_of_iterations;
            }
            VLOG(2) << "Removed one vertex!";
            continue;
          }
        }
      }
      first_vertex_it = second_vertex_it;
      second_vertex_it = third_vertex_it;
      third_vertex_it = next(second_vertex_it, *polygon);
      VLOG(2) << "Got to bottom! Number of iterations: " << std::distance(polygon->begin(), first_vertex_it) << " "
              << std::distance(polygon->begin(), second_vertex_it) << " " << std::distance(polygon->begin(), third_vertex_it);
      if (std::distance(polygon->begin(), first_vertex_it) >= std::distance(polygon->begin(), second_vertex_it)) {
        ++number_of_iterations;
      }
    }
  }

  double computeTriangleArea(double side_length_a, double side_length_b, double side_length_c){
    const double s = (side_length_a + side_length_b + side_length_c) / 2.0;
    CHECK_GT(s, 0.0);
    double sqrt_argument = s * (s-side_length_a) * (s - side_length_b) * (s - side_length_c);
    if (abs(sqrt_argument) < 1e-6){
      sqrt_argument = 0.0;
    }
    CHECK_GE(sqrt_argument, 0.0);
    return sqrt(sqrt_argument);
  }

  CgalPolygon2dVertexIterator next(const CgalPolygon2dVertexIterator& iterator, const CgalPolygon2d& polygon){
    if (std::next(iterator) == polygon.vertices_end()){
      return polygon.vertices_begin();
    } else {
      return std::next(iterator);
    }
  }

  // Retrieves the following vertex in polygon, including wrap around, when last polygon reached.
  // In this case the output flag is set to true.
  bool next(const CgalPolygon2dVertexIterator& iterator, CgalPolygon2dVertexIterator& output_iterator, const CgalPolygon2d& polygon){
    if (std::next(iterator) == polygon.vertices_end()){
      output_iterator = polygon.vertices_begin();
      return true;
    } else {
      output_iterator = std::next(iterator);
      return false;
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
                               Intersection* output_intersection) {
    CHECK_NOTNULL(output_intersection);
    CHECK(orientation == CGAL::COUNTERCLOCKWISE || orientation == CGAL::CLOCKWISE);
    CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
    Intersection intersection_tmp;
    constexpr double kLargeDistanceValue = 1000;
    double min_distance = kLargeDistanceValue;
    bool one_intersection_at_least = false;
    auto vertex_it = polygon.vertices_begin();
    std::advance(vertex_it, ray_target_location);
    auto ray_source_it = vertex_it;
    auto ray_target_it = vertex_it;
    if (orientation == CGAL::COUNTERCLOCKWISE) {
      ray_source_it = previous(vertex_it, polygon);
    } else {
      vertex_it = next(vertex_it, polygon);
      ray_source_it = vertex_it;
    }
    vertex_it = next(vertex_it, polygon);
    const CgalPoint2d ray_source = *ray_source_it;
    VLOG(1) << "Source ray:" << ray_source;
    const CgalPoint2d ray_target = *ray_target_it;
    const CgalRay2d ray(*ray_target_it, *ray_target_it - *ray_source_it);
    VLOG(1) << "Ray target: " << ray.source();
    VLOG(1) << "Ray direction: " << ray.direction();
    CHECK(!ray.is_degenerate());
    // Do not intersect with adjacent edges since they intersect already in common vertex.
    auto condition_it = ray_source_it;
    if (orientation == CGAL::CLOCKWISE) {
      condition_it = ray_target_it;
    }
    int number_of_polygon_edges_visited = 0;
    while (next(vertex_it, polygon) != condition_it) {
      ++number_of_polygon_edges_visited;
      const CgalPoint2d segment_source = *vertex_it;
      const auto segment_target_it = next(vertex_it, polygon);
      const CgalPoint2d segment_target = *segment_target_it;
      VLOG(1) << "Segment under test: Source: " << segment_source << " Target: " << segment_target;
      const CgalSegment2d segment(*vertex_it, *segment_target_it);
      RaySegmentIntersection intersection;
      if (doRayAndSegmentIntersect(ray, segment, &intersection)) {
        VLOG(1) << "Intersection type: " << static_cast<int>(intersection.intersection_location);
        VLOG(1) << "Intersection point:" << intersection.intersection_point;
        VLOG(1) << segment_source;
        const double current_distance = sqrt((ray_target - intersection.intersection_point).squared_length());
        // Take first intersection on ray.
        if (current_distance < min_distance) {
          one_intersection_at_least = true;
          min_distance = current_distance;
          intersection_tmp.setAllMembers(std::distance(polygon.vertices_begin(), vertex_it),
                                         std::distance(polygon.vertices_begin(), segment_target_it), intersection.intersection_point,
                                         intersection.intersection_location);
        }
      }
      vertex_it = next(vertex_it, polygon);
    }
    CHECK_EQ(number_of_polygon_edges_visited, polygon.size() - 3);
    if (!one_intersection_at_least) {
      return false;
    }
    *output_intersection = intersection_tmp;
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

  std::pair<int, int> getVertexPositionsWithHighestHoleSlConcavityMeasure(const CgalPolygon2d& polygon){
    CHECK(polygon.size() > 2);
    std::multimap<double,std::pair<int, int>> slConcavityScoreMap;
    const int number_of_vertices = polygon.size();
    for (int first_vertex_counter = 0; first_vertex_counter < number_of_vertices - 1; ++first_vertex_counter){
      const auto& vertex_container = polygon.container();
      const CgalPoint2d& first_vertex = vertex_container.at(first_vertex_counter);
      for (int second_vertex_counter = first_vertex_counter + 1; second_vertex_counter < number_of_vertices; ++second_vertex_counter){
        const CgalPoint2d& second_vertex = vertex_container.at(second_vertex_counter);
        const double vertex_squared_distance = (first_vertex - second_vertex).squared_length();
        slConcavityScoreMap.insert(std::make_pair(vertex_squared_distance, std::make_pair(first_vertex_counter, second_vertex_counter)));
      }
    }
    return slConcavityScoreMap.begin()->second;
  }

  // Attention, if point overlaps with existing vertex.
  std::pair<int, CgalPoint2d> getClosestPointAndSegmentOnPolygonToPoint(const CgalPoint2d& point, const CgalPolygon2d& polygon){
    CHECK_GT(polygon.size(), 2);
    std::multimap<double, std::pair<int, Vector2d>> distance_to_segment_closest_point_map;
    int edge_counter = 0;
    for(auto edge_it = polygon.edges_begin(); edge_it != polygon.edges_end(); ++edge_it){
      if (edge_it->has_on(point)){
        distance_to_segment_closest_point_map.insert(std::make_pair(0.0, std::make_pair(edge_counter, Vector2d(point.x(), point.y()))));
        ++edge_counter;
        continue;
      }
      const Vector2d source(edge_it->source().x(), edge_it->source().y());
      const Vector2d target(edge_it->target().x(), edge_it->target().y());
      const Vector2d test_point(point.x(), point.y());
      std::pair<double, Vector2d> distance_closest_point = getDistanceAndClosestPointOnLineSegment(test_point, source, target);
      distance_to_segment_closest_point_map.insert(std::make_pair(distance_closest_point.first, std::make_pair(edge_counter, distance_closest_point.second)));
      ++edge_counter;
    }
    const int closest_edge_index = distance_to_segment_closest_point_map.begin()->second.first;
    const CgalPoint2d closest_point(distance_to_segment_closest_point_map.begin()->second.second.x(), distance_to_segment_closest_point_map.begin()->second.second.y());
    return std::make_pair(closest_edge_index, closest_point);
  }

  std::vector<std::pair<int, int>> getCommonVertexPairIndices(const CgalPolygon2d& first_polygon, const CgalPolygon2d& second_polygon){
    std::vector<std::pair<int, int>> return_buffer;
    for(int first_polygon_vertex_index = 0; first_polygon_vertex_index < first_polygon.container().size(); ++first_polygon_vertex_index){
      for(int second_polygon_vertex_index = 0; second_polygon_vertex_index < second_polygon.container().size(); ++second_polygon_vertex_index){
        if (first_polygon.container().at(first_polygon_vertex_index) == second_polygon.container().at(second_polygon_vertex_index)){
          return_buffer.push_back(std::make_pair(first_polygon_vertex_index, second_polygon_vertex_index));
        }
      }
    }
    return return_buffer;
  }

  std::vector<int> getVertexIndicesOfFirstPolygonContainedInSecondPolygonContour(const CgalPolygon2d& first_polygon,
                                                                                 const CgalPolygon2d& second_polygon) {
    std::vector<int> return_buffer;
    for (int vertex_index = 0; vertex_index < first_polygon.container().size(); ++vertex_index) {
      if (second_polygon.has_on_boundary(first_polygon.container().at(vertex_index))) {
        return_buffer.push_back(vertex_index);
      }
    }
    return return_buffer;
  }

  bool doRayAndSegmentIntersect(const CgalRay2d& ray, const CgalSegment2d& segment, RaySegmentIntersection* intersection) {
    constexpr double kSquaredToleranceMeters = 1e-8;
    CGAL::Cartesian_converter<K, EKernel> to_exact;
    CGAL::Cartesian_converter<EKernel, K> to_inexact;
    CGAL::cpp11::result_of<EKernel::Intersect_2(EKernel::Ray_2, EKernel::Segment_2)>::type result =
        CGAL::intersection(to_exact(ray), to_exact(segment));

    if (result) {
      if (const EKernel::Segment_2* ek_segment = boost::get<EKernel::Segment_2>(&*result)) {
        VLOG(1) << "Segment intersection.";
        VLOG(1) << "Segment source: " << segment.source();
        VLOG(1) << "Segment target: " << segment.target();
        VLOG(1) << "Ray point: " << ray.source();
        VLOG(1) << "Ray direction: " << ray.direction();
        if ((ray.source() - segment.source()).squared_length() < (ray.source() - segment.target()).squared_length()) {
          intersection->intersection_location = SegmentIntersectionType::kSource;
          intersection->intersection_point = segment.source();
        } else {
          intersection->intersection_location = SegmentIntersectionType::kTarget;
          intersection->intersection_point = segment.target();
        }
      } else {
        VLOG(1) << "Point intersection.";
        VLOG(1) << "Segment source: " << segment.source();
        VLOG(1) << "Segment target: " << segment.target();
        VLOG(1) << "Ray point: " << ray.source();
        VLOG(1) << "Ray direction: " << ray.direction();
        const EKernel::Point_2* ek_point = boost::get<EKernel::Point_2>(&*result);
        const CgalPoint2d intersection_point = to_inexact(*ek_point);
        if ((intersection_point - segment.source()).squared_length() <= kSquaredToleranceMeters) {
          intersection->intersection_location = SegmentIntersectionType::kSource;
          intersection->intersection_point = segment.source();
        } else if ((intersection_point - segment.target()).squared_length() <= kSquaredToleranceMeters) {
          intersection->intersection_location = SegmentIntersectionType::kTarget;
          intersection->intersection_point = segment.target();
        } else {
          intersection->intersection_location = SegmentIntersectionType::kInterior;
          intersection->intersection_point = intersection_point;
        }
      }
      return true;
    }
    constexpr double kToleranceMeters = 1e-6;
    const Vector2d ray_source = Vector2d(ray.source().x(), ray.source().y());
    const Vector2d ray_direction = Vector2d(ray.direction().dx(), ray.direction().dy()).normalized();
    const Vector2d segment_source = Vector2d(segment.source().x(), segment.source().y());
    const Vector2d segment_target = Vector2d(segment.target().x(), segment.target().y());
    const Vector2d p_ray_source__segment_source = segment_source - ray_source;
    const Vector2d p_ray_source__segment_target = segment_target - ray_source;
    const double segment_source_projection = p_ray_source__segment_source.dot(ray_direction);
    const double segment_target_projection = p_ray_source__segment_target.dot(ray_direction);
    const Vector2d segment_source_projection_point = ray_source + segment_source_projection * ray_direction;
    const Vector2d segment_target_projection_point = ray_source + segment_target_projection * ray_direction;
    if (segment_source_projection >= 0.0 && segment_target_projection >= 0.0) {
      if ((segment_source_projection_point - segment_source).norm() < kToleranceMeters &&
          (segment_target_projection_point - segment_target).norm() < kToleranceMeters) {
        // Source and target lie on ray. Determine, which point is closer.
        if ((segment_source_projection_point - ray_source).norm() <= (segment_target_projection_point - ray_source).norm()) {
          intersection->intersection_location = SegmentIntersectionType::kSource;
          intersection->intersection_point = segment.source();
          return true;
        } else {
          intersection->intersection_location = SegmentIntersectionType::kTarget;
          intersection->intersection_point = segment.target();
          return true;
        }
      } else if ((segment_source_projection_point - segment_source).norm() < kToleranceMeters) {
        // Only source lies on ray.
        intersection->intersection_location = SegmentIntersectionType::kSource;
        intersection->intersection_point = segment.source();
        return true;
      } else if ((segment_target_projection_point - segment_target).norm() < kToleranceMeters) {
        // Only target lies on ray.
        intersection->intersection_location = SegmentIntersectionType::kTarget;
        intersection->intersection_point = segment.target();
        return true;
      }
    } else if (segment_source_projection >= 0.0) {
      if ((segment_source_projection_point - segment_source).norm() < kToleranceMeters) {
        intersection->intersection_location = SegmentIntersectionType::kSource;
        intersection->intersection_point = segment.source();
        return true;
      }
    } else if (segment_target_projection >= 0.0) {
      if ((segment_target_projection_point - segment_target).norm() < kToleranceMeters) {
        intersection->intersection_location = SegmentIntersectionType::kTarget;
        intersection->intersection_point = segment.target();
        return true;
      }
    }
    return false;
  }

  }  // namespace convex_plane_extraction
