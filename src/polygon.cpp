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

}