#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_CONVEX_POLYGON_DECOMPOSITION_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_CONVEX_POLYGON_DECOMPOSITION_HPP_

#include <list>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <glog/logging.h>

#include "types.hpp"

namespace convex_plane_extraction {

  typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
  typedef CGAL::Partition_traits_2<K>                           Traits;
  typedef Traits::Point_2                                       CgalPoint2d;
  typedef Traits::Polygon_2                                     CgalPolygon2d;
  typedef std::list<CgalPolygon2d>                              CgalPolygon2dListContainer;

  template <typename Iter>
  CgalPolygon2d createCgalPolygonFromOpenCvPoints_impl(Iter begin, Iter end, std::bidirectional_iterator_tag){
    CgalPolygon2d polygon;
    for (auto it = begin; it < end; ++it) {
      polygon.push_back(CgalPoint2d((*it).x, (*it).y));
      std::cout << *it << std::endl;
    }
    return polygon;
  }

  template <typename Iter>
  CgalPolygon2d createCgalPolygonFromOpenCvPoints(Iter begin, Iter end){
    return createCgalPolygonFromOpenCvPoints_impl(begin, end,
                                typename std::iterator_traits<Iter>::iterator_category());
  }

  void performConvexDecomposition(const CgalPolygon2d& polygon, CgalPolygon2dListContainer* output_polyong_list);

}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_CONVEX_POLYGON_DECOMPOSITION_HPP_
