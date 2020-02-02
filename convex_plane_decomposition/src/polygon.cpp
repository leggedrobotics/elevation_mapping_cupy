#include "polygon.hpp"

namespace convex_plane_extraction {

  void performConvexDecomposition(const CgalPolygon2d& polygon, CgalPolygon2dListContainer* output_polygon_list){
    CHECK_GE(polygon.size(), 3);
    CHECK(polygon.is_simple());
    LOG(INFO) << "Started convex decomposition...";
    size_t old_list_size = output_polygon_list->size();
    CGAL::optimal_convex_partition_2(polygon.vertices_begin(),
                                     polygon.vertices_end(),
                                     std::back_inserter(*output_polygon_list));

    assert(CGAL::partition_is_valid_2(polygon.vertices_begin(),
                                      polygon.vertices_end(),
                                      polygon_list.begin(),
                                      polygon_list.end()));
//    *output_polygon_list = decomposeInnerApproximation(polygon);
    CHECK_GT(output_polygon_list->size(), old_list_size);
    LOG(INFO) << "done.";
  }

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

  // A dent is a vertex, which lies in a counter-clockwise oriented polygon on the left side of its neighbors.
  // Dents cause non-convexity.
  void detectDentLocations(std::map<double, int>* dent_locations, const CgalPolygon2d& polygon){
    CHECK_NOTNULL(dent_locations);
    CHECK(dent_locations->empty());
    CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
    for (auto source_it = polygon.vertices_begin(); source_it != polygon.vertices_end(); ++source_it){
      auto dent_it = next(source_it, polygon);
      auto destination_it = next(dent_it, polygon);
      Vector2d source_point = Vector2d(source_it->x(), source_it->y());
      Vector2d destination_point = Vector2d(destination_it->x(), destination_it->y());
      Vector2d direction_vector = destination_point - source_point;
      Vector2d test_point = Vector2d(dent_it->x(), dent_it->y());
      const double kAngleThresholdRad = 0.001; // 0.05 deg in rad.
      if(isPointOnLeftSide(source_point, direction_vector, test_point)){
        double angle = abs(computeAngleBetweenVectors(source_point - test_point, destination_point - test_point));
        // Ignore very shallow dents. Approximate convex decomposition.
        if ( angle > kAngleThresholdRad)
          dent_locations->insert(std::make_pair(angle, std::distance(polygon.vertices_begin(), dent_it)));
      }
    }
  }

  std::list<CgalPolygon2d> decomposeInnerApproximation(const CgalPolygon2d& polygon){
    CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
    std::list<CgalPolygon2d> return_list;
    std::map<double, int> dent_locations;
    LOG(INFO) << "Before dent detection";
    detectDentLocations(&dent_locations, polygon);
    LOG(INFO) << "Passed dent detection";
    if (dent_locations.empty()){
      // No dents detected, polygon must be convex.
      // CGAL convexity check might still fail, since very shallow dents are ignored.
      return_list.push_back(polygon);
      return return_list;
    }
    int dent_location = dent_locations.begin()->second;
    bool intersection_clockwise_flag = false;
    Intersection intersection_clockwise;
    bool intersection_counterclockwise_flag = false;
    Intersection intersection_counterclockwise;
    intersection_clockwise_flag = intersectPolygonWithRay(dent_location, CGAL::COUNTERCLOCKWISE,
        polygon, &intersection_counterclockwise);
    intersection_counterclockwise_flag = intersectPolygonWithRay(dent_location, CGAL::CLOCKWISE,
        polygon, &intersection_clockwise);
    if (!intersection_clockwise_flag || !intersection_counterclockwise_flag ){
      LOG(FATAL) << "At least one intersection of dent ray with polygon failed!";
    }
    // Generate resulting polygons from cut.
    // Resulting cut from counter clockwise ray intersection.
    CgalPolygon2d polygon_counterclockwise_1 = polygon;
    CgalPolygon2d polygon_counterclockwise_2;
    auto first_vertex_to_erase_it = polygon_counterclockwise_1.vertices_begin();
    std::advance(first_vertex_to_erase_it, dent_location);
    // Add dent to second polygon.
    polygon_counterclockwise_2.push_back(*first_vertex_to_erase_it);
    first_vertex_to_erase_it = next(first_vertex_to_erase_it, polygon_counterclockwise_1);
    auto last_vertex_to_erase_it = polygon_counterclockwise_1.vertices_begin();
    // Intersection somewhere must be at or after source vertex of intersection edge.
    std::advance(last_vertex_to_erase_it, intersection_counterclockwise.edge_source_location_);
    // Take next vertex due to exclusive upper limit logic.
    last_vertex_to_erase_it = next(last_vertex_to_erase_it, polygon_counterclockwise_1);
    // Copy vertices that will be deleted to second polygon.
    copyVertices(polygon_counterclockwise_1,first_vertex_to_erase_it,last_vertex_to_erase_it,
    &polygon_counterclockwise_2, polygon_counterclockwise_2.vertices_end());
    // Get last point that was erased.
    CgalPoint2d point_before_intersection = *(previous(last_vertex_to_erase_it, polygon_counterclockwise_2));
    // To avoid numerical issues and duplicate vertices, intersection point is only inserted if sufficiently
    // far away from exisiting vertex.
    constexpr double kSquaredLengthThreshold = 1e-6;
    if ((intersection_counterclockwise.intersection_point_ - point_before_intersection).squared_length() > kSquaredLengthThreshold) {
      polygon_counterclockwise_2.push_back(intersection_counterclockwise.intersection_point_);
    }
    LOG(INFO) << "Before erase!";
    CgalPolygon2dVertexIterator element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_counterclockwise_1);
    // Add intersection vertex to first polygon if existing vertex too far away.
    if ((intersection_counterclockwise.intersection_point_ - *element_behind_deleted_it).squared_length() > kSquaredLengthThreshold) {
      polygon_counterclockwise_1.insert(element_behind_deleted_it, intersection_counterclockwise.intersection_point_);
    }
    // Resulting cut from clockwise ray intersection.
    CgalPolygon2d polygon_clockwise_1 = polygon;
    CgalPolygon2d polygon_clockwise_2;

    first_vertex_to_erase_it = polygon_clockwise_1.vertices_begin();
    std::advance(first_vertex_to_erase_it, intersection_clockwise.edge_target_location_);
    if ((*first_vertex_to_erase_it - intersection_clockwise.intersection_point_).squared_length() > kSquaredLengthThreshold) {
      polygon_clockwise_2.push_back(intersection_clockwise.intersection_point_);
    }
    last_vertex_to_erase_it = polygon_clockwise_1.vertices_begin();
    std::advance(last_vertex_to_erase_it, dent_location);
    copyVertices(polygon_clockwise_1, first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_2,
                 polygon_clockwise_2.vertices_end());
    polygon_clockwise_2.push_back(*last_vertex_to_erase_it);
    element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_1);
    LOG(INFO) << "After erase!";
    if ((intersection_clockwise.intersection_point_ - *element_behind_deleted_it).squared_length() > kSquaredLengthThreshold) {
      polygon_clockwise_1.insert(element_behind_deleted_it, intersection_clockwise.intersection_point_);
    }
    printPolygon(polygon_counterclockwise_1);
    printPolygon(polygon_counterclockwise_2);
    printPolygon(polygon_clockwise_1);
    printPolygon(polygon_clockwise_2);
    // Take the cut with smallest min. area of resulting polygons.
    double area[4] = {polygon_counterclockwise_1.area(), polygon_counterclockwise_2.area(),
                      polygon_clockwise_1.area(), polygon_clockwise_2.area()};
    LOG(INFO) << "Passed area computation!";
    double* min_area_strategy = std::min_element(area, area+4);
    std::list<CgalPolygon2d> recursion_1;
    std::list<CgalPolygon2d> recursion_2;
    if ((min_area_strategy - area) < 2){
      // In this case the counter clockwise intersection leads to less loss in area.
      // Perform recursion with this split.
      if(polygon_counterclockwise_1.orientation() != CGAL::COUNTERCLOCKWISE){
        LOG(WARNING) << "Polygon orientation wrong! Printing polygon ...";
        printPolygon(polygon_counterclockwise_1);
        recursion_1 = std::list<CgalPolygon2d>();
      } else {
        recursion_1 = decomposeInnerApproximation(polygon_counterclockwise_1);
      }
      if(polygon_counterclockwise_2.orientation() != CGAL::COUNTERCLOCKWISE){
        LOG(WARNING) << "Polygon orientation wrong! Printing polygon ...";
        printPolygon(polygon_counterclockwise_2);
        recursion_2 = std::list<CgalPolygon2d>();
      } else{
        recursion_2 = decomposeInnerApproximation(polygon_counterclockwise_2);
      }
    } else {
      // In this case the clockwise intersection leads to less loss in area.
      if(polygon_clockwise_1.orientation() != CGAL::COUNTERCLOCKWISE){
        LOG(WARNING) << "Polygon orientation wrong! Printing polygon ...";
        printPolygon(polygon_clockwise_1);
        recursion_1 = std::list<CgalPolygon2d>();
      } else{
        recursion_1 = decomposeInnerApproximation(polygon_clockwise_1);
      }
      if(polygon_clockwise_2.orientation() != CGAL::COUNTERCLOCKWISE){
        LOG(WARNING) << "Polygon orientation wrong! Printing polygon ...";
        printPolygon(polygon_clockwise_2);
        recursion_2 = std::list<CgalPolygon2d>();
      } else{
        recursion_2 = decomposeInnerApproximation(polygon_clockwise_2);
      }
    }
    return_list.splice(return_list.end(), recursion_1);
    return_list.splice(return_list.end(), recursion_2);
    LOG(INFO) << "Reached bottom!";
    return return_list;
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

}
