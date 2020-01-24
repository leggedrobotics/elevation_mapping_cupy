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

}