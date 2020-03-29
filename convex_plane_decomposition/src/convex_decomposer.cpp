#include "convex_decomposer.hpp"

using namespace convex_plane_extraction;

ConvexDecomposer::ConvexDecomposer(const convex_plane_extraction::ConvexDecomposerParameters &parameters)
  :parameters_(parameters){}

CgalPolygon2dContainer ConvexDecomposer::performConvexDecomposition(const CgalPolygon2d& polygon) const {
  VLOG(1) << "Started convex decomposition...";
  CgalPolygon2dContainer convex_polygons;
  if (polygon.size() == 2) {
    convex_polygons.push_back(polygon);
    return convex_polygons;
  }
  CHECK(polygon.is_simple());
  if(polygon.is_convex()){
    LOG(INFO) << "Polygon already convex, no decompostion performed.";
    convex_polygons.push_back(polygon);
    return convex_polygons;
  }
  switch (parameters_.decomposition_type){
    case ConvexDecompositionType::kGreeneOptimalDecomposition :
      convex_polygons = performOptimalConvexDecomposition(polygon);
      break;
    case ConvexDecompositionType::kInnerConvexApproximation :
      convex_polygons = performInnerConvexApproximation(polygon);
      break;
  }
  VLOG(1) << "done.";
  return convex_polygons;
}

CgalPolygon2dContainer ConvexDecomposer::performOptimalConvexDecomposition(const CgalPolygon2d& polygon) const{
  std::vector<Traits::Polygon_2> polygon_buffer;
  size_t old_container_size = polygon_buffer.size();
  Traits::Polygon_2 input_polygon;
  for (const auto& vertex : polygon.container()){
    input_polygon.insert(input_polygon.vertices_end(),Traits::Point_2(vertex.x(), vertex.y()));
  }
  CGAL::optimal_convex_partition_2(polygon.vertices_begin(), polygon.vertices_end(),
      std::back_inserter(polygon_buffer));
//  assert(CGAL::partition_is_valid_2(polygon.vertices_begin(), polygon.vertices_end(), output_polygons.begin(),
//      output_polygons.end()));
  CHECK_GT(polygon_buffer.size(), old_container_size);
  CgalPolygon2dContainer output_polygons;
  for (const auto& buffer_polygon : polygon_buffer){
    CgalPolygon2d temp_polygon;
    for (const auto& vertex : buffer_polygon.container()){
      temp_polygon.insert(temp_polygon.vertices_end(), CgalPoint2d(vertex.x(), vertex.y()));
    }
    output_polygons.push_back(temp_polygon);
  }
  return output_polygons;
}

CgalPolygon2dContainer ConvexDecomposer::performInnerConvexApproximation(const CgalPolygon2d& polygon) const {
  CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
  VLOG(1) << "Polygon under decomposition: ";
  printPolygon(polygon);
  CgalPolygon2dContainer convex_polygons;
  if (polygon.size() == 3) {
    convex_polygons.push_back(polygon);
    return convex_polygons;
  }
  const std::multimap<double, int> dent_locations = detectDentLocations(polygon);
  if (dent_locations.empty()) {
    // No dents detected, polygon must be convex.
    // CGAL convexity check might still fail, since very shallow dents are ignored.
    convex_polygons.push_back(polygon);
    return convex_polygons;
  }
  int dent_location = dent_locations.begin()->second;
  bool intersection_clockwise_flag = false;
  Intersection intersection_clockwise;
  bool intersection_counterclockwise_flag = false;
  Intersection intersection_counterclockwise;
  intersection_clockwise_flag = intersectPolygonWithRay(dent_location, CGAL::COUNTERCLOCKWISE,
                                                        polygon, &intersection_counterclockwise);
  intersection_counterclockwise_flag = intersectPolygonWithRay(dent_location, CGAL::CLOCKWISE, polygon, &intersection_clockwise);
  if (!intersection_clockwise_flag || !intersection_counterclockwise_flag) {
    LOG(FATAL) << "At least one intersection of dent ray with polygon failed!";
  }

  // Generate resulting polygons from cut.
  // Resulting cut from counter clockwise ray intersection.
  CgalPolygon2d polygon_counterclockwise_1 = polygon;
  CgalPolygon2d polygon_counterclockwise_2;

  SegmentIntersectionType intersection_type = intersection_counterclockwise.intersection_type_;

  auto first_vertex_to_erase_it = polygon_counterclockwise_1.vertices_begin();
  std::advance(first_vertex_to_erase_it, dent_location);
  // Add dent to second polygon.
  polygon_counterclockwise_2.push_back(*first_vertex_to_erase_it);
  // First vertex to erase is the vertex following the dent.
  first_vertex_to_erase_it = next(first_vertex_to_erase_it, polygon_counterclockwise_1);
  // Last vertex to erase is the source vertex of the intersecting polygon edge.
  auto last_vertex_to_erase_it = polygon_counterclockwise_1.vertices_begin();
  std::advance(last_vertex_to_erase_it, intersection_counterclockwise.edge_source_location_);
  // Take next vertex due to exclusive upper limit logic.
  last_vertex_to_erase_it = next(last_vertex_to_erase_it, polygon_counterclockwise_1);
  // Copy vertices that will be deleted to second polygon.
  copyVertices(polygon_counterclockwise_1, first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_counterclockwise_2,
               polygon_counterclockwise_2.vertices_end());

  CgalPolygon2dVertexIterator element_behind_deleted_it;

  switch (intersection_type) {
    case SegmentIntersectionType::kInterior:
      polygon_counterclockwise_2.push_back(intersection_counterclockwise.intersection_point_);
      element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_counterclockwise_1);
      polygon_counterclockwise_1.insert(element_behind_deleted_it, intersection_counterclockwise.intersection_point_);
      break;
    case SegmentIntersectionType::kSource:
      element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_counterclockwise_1);
      // Add source vertex again to polygon, since it coincides with intersection point.
      polygon_counterclockwise_1.insert(element_behind_deleted_it, intersection_counterclockwise.intersection_point_);
      break;
    case SegmentIntersectionType::kTarget:
      // The intersection point coincides with the target of the intersecting polygon edge, and has to be added here.
      polygon_counterclockwise_2.push_back(intersection_counterclockwise.intersection_point_);
      element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_counterclockwise_1);
      break;
  }

  VLOG(1) << "Finished counter-clockwise cut.";

  // Resulting cut from clockwise ray intersection.
  CgalPolygon2d polygon_clockwise_1 = polygon;
  CgalPolygon2d polygon_clockwise_2;

  intersection_type = intersection_clockwise.intersection_type_;

  // First vertex, which is pruned off is the target of the intersecting polygon edge.
  first_vertex_to_erase_it = polygon_clockwise_1.vertices_begin();
  std::advance(first_vertex_to_erase_it, intersection_clockwise.edge_target_location_);
  // Last vertex, which is pruned off is the vertex before the dent. (Past-the-end-logic!)
  last_vertex_to_erase_it = polygon_clockwise_1.vertices_begin();
  std::advance(last_vertex_to_erase_it, dent_location);

  switch (intersection_type) {
    case SegmentIntersectionType::kInterior:
      // First vertex to be added to new polygon is the intersection point.
      polygon_clockwise_2.push_back(intersection_clockwise.intersection_point_);
      // Added vertices to be deleted to new polygon (Past-the-end-logic!)
      copyVertices(polygon_clockwise_1, first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_2,
                   polygon_clockwise_2.vertices_end());
      // Add dent vertex as well.
      polygon_clockwise_2.push_back(*last_vertex_to_erase_it);
      // Perform erase operation.
      element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_1);
      // Add intersection point to polygon.
      polygon_clockwise_1.insert(element_behind_deleted_it, intersection_clockwise.intersection_point_);
      break;
    case SegmentIntersectionType::kSource:
      // First vertex to be added to new polygon is source of intersecting polygon edge.
      polygon_clockwise_2.push_back(intersection_clockwise.intersection_point_);  // Intersection point is equivalent here with source.
      // Added vertices to be deleted to new polygon (Past-the-end-logic!)
      copyVertices(polygon_clockwise_1, first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_2,
                   polygon_clockwise_2.vertices_end());
      // Add dent vertex as well.
      polygon_clockwise_2.push_back(*last_vertex_to_erase_it);
      // Perform erase operation.
      element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_1);
      break;
    case SegmentIntersectionType::kTarget:
      // Added vertices to be deleted to new polygon (Past-the-end-logic!)
      copyVertices(polygon_clockwise_1, first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_2,
                   polygon_clockwise_2.vertices_end());
      // Add dent vertex as well.
      polygon_clockwise_2.push_back(*last_vertex_to_erase_it);
      // Perform erase operation.
      element_behind_deleted_it = erase(first_vertex_to_erase_it, last_vertex_to_erase_it, &polygon_clockwise_1);
      // Add intersection point to polygon, which is in this case the target vertex of the intersecting polygon edge.
      polygon_clockwise_1.insert(element_behind_deleted_it, intersection_clockwise.intersection_point_);
      break;
  }

  VLOG(1) << "Finished clockwise cut.";
  printPolygon(polygon_counterclockwise_1);
  printPolygon(polygon_counterclockwise_2);
  printPolygon(polygon_clockwise_1);
  printPolygon(polygon_clockwise_2);
  CHECK(polygon_counterclockwise_1.is_simple());
  CHECK(polygon_counterclockwise_2.is_simple());
  CHECK(polygon_clockwise_1.is_simple());
  CHECK(polygon_clockwise_2.is_simple());
  VLOG(1) << "Started cut area computation...";
  // Take the cut with smallest min. area of resulting polygons.
  double area[4] = {polygon_counterclockwise_1.area(), polygon_counterclockwise_2.area(), polygon_clockwise_1.area(),
                    polygon_clockwise_2.area()};
  VLOG(1) << "done.";
  double* min_area_strategy = std::min_element(area, area + 4);
  CgalPolygon2dContainer recursion_1;
  CgalPolygon2dContainer recursion_2;
  if ((min_area_strategy - area) < 2) {
    // In this case the counter clockwise intersection leads to less loss in area.
    // Perform recursion with this split.
    if(polygon_counterclockwise_1.orientation() != CGAL::COUNTERCLOCKWISE){
      LOG(WARNING) << "Polygon orientation wrong! Printing polygon ...";
      printPolygon(polygon_counterclockwise_1);
      recursion_1 = std::vector<CgalPolygon2d>();
    } else {
      recursion_1 = performInnerConvexApproximation(polygon_counterclockwise_1);
    }
    if(polygon_counterclockwise_2.orientation() != CGAL::COUNTERCLOCKWISE){
      LOG(WARNING) << "Polygon orientation wrong! Printing polygon ...";
      printPolygon(polygon_counterclockwise_2);
      recursion_2 = std::vector<CgalPolygon2d>();
    } else{
      recursion_2 = performInnerConvexApproximation(polygon_counterclockwise_2);
    }
  } else {
    // In this case the clockwise intersection leads to less loss in area.
    if(polygon_clockwise_1.orientation() != CGAL::COUNTERCLOCKWISE){
      LOG(WARNING) << "Polygon orientation wrong! Printing polygon ...";
      printPolygon(polygon_clockwise_1);
      recursion_1 = std::vector<CgalPolygon2d>();
    } else{
      recursion_1 = performInnerConvexApproximation(polygon_clockwise_1);
    }
    if(polygon_clockwise_2.orientation() != CGAL::COUNTERCLOCKWISE){
      LOG(WARNING) << "Polygon orientation wrong! Printing polygon ...";
      printPolygon(polygon_clockwise_2);
      recursion_2 = std::vector<CgalPolygon2d>();
    } else{
      recursion_2 = performInnerConvexApproximation(polygon_clockwise_2);
    }
  }
  std::move(recursion_1.begin(), recursion_1.end(), std::back_inserter(convex_polygons));
  std::move(recursion_2.begin(), recursion_2.end(), std::back_inserter(convex_polygons));
  VLOG(1) << "Reached bottom!";
  return convex_polygons;
}

// A dent is a vertex, which lies in a counter-clockwise oriented polygon on the left side of its neighbors.
// Dents cause non-convexity.
std::multimap<double, int> ConvexDecomposer::detectDentLocations(const CgalPolygon2d& polygon) const {
  VLOG(1) << "Starting dent detection...";
  CHECK(polygon.orientation() == CGAL::COUNTERCLOCKWISE);
  std::multimap<double, int> dent_locations;
  for (auto source_it = polygon.vertices_begin(); source_it != polygon.vertices_end(); ++source_it){
    auto dent_it = next(source_it, polygon);
    auto destination_it = next(dent_it, polygon);
    Vector2d source_point = Vector2d(source_it->x(), source_it->y());
    Vector2d destination_point = Vector2d(destination_it->x(), destination_it->y());
    Vector2d direction_vector = destination_point - source_point;
    Vector2d test_point = Vector2d(dent_it->x(), dent_it->y());
    if(isPointOnLeftSide(source_point, direction_vector, test_point)){
      double angle = abs(computeAngleBetweenVectors(source_point - test_point, destination_point - test_point));
      // Ignore very shallow dents. Approximate convex decomposition.
      if ( angle > parameters_.dent_angle_threshold_rad)
        dent_locations.insert(std::make_pair(angle, std::distance(polygon.vertices_begin(), dent_it)));
    }
  }
  VLOG(1) << "done.";
  return dent_locations;
}

//    *output_polygon_list = decomposeInnerApproximation(polygon);