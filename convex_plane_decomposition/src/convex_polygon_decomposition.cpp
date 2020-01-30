#include "convex_polygon_decomposition.hpp"

namespace convex_plane_extraction {
  void performConvexDecomposition(const CgalPolygon2d& polygon, CgalPolygon2dListContainer* output_polyong_list){
    CHECK_GE(polygon.size(), 3);
    CHECK(polygon.is_simple());
    LOG(INFO) << "Started convex decomposition...";
    CGAL::optimal_convex_partition_2(polygon.vertices_begin(),
                                     polygon.vertices_end(),
                                     std::back_inserter(*output_polyong_list));

    assert(CGAL::partition_is_valid_2(polygon.vertices_begin(),
                                      polygon.vertices_end(),
                                      polygon_list.begin(),
                                      polygon_list.end()));
    LOG(INFO) << "done.";
  }
}

