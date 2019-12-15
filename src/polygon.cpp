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
      LOG(INFO) << "Source: " << test_segment.source() << " target: " << test_segment.target();
      if (do_intersect(test_segment, segment)){
        LOG(INFO) << "OK!";
        return true;
      }
      LOG(INFO) << "OK!";
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

}