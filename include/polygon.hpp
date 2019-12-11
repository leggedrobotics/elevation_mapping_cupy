#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_

#include <iostream>
#include <list>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/partition_2.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Surface_sweep_2_algorithms.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <glog/logging.h>

#include "types.hpp"

namespace convex_plane_extraction{

  typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
  typedef CGAL::Partition_traits_2<K>                           Traits;
  typedef Traits::Point_2                                       CgalPoint2d;
  typedef Traits::Polygon_2                                     CgalPolygon2d;
  typedef std::list<CgalPolygon2d>                              CgalPolygon2dListContainer;

  typedef std::vector<Eigen::Vector3d>                          Polygon3d;
  typedef std::vector<Polygon3d>                                Polygon3dVectorContainer;


  template <typename Iter>
  bool isContourSimple_impl(Iter begin, Iter end, std::bidirectional_iterator_tag){
    CgalPolygon2d polygon;
    for (auto it = begin; it < end; ++it) {
      polygon.push_back(Point((*it).x, (*it).y));
      std::cout << *it << std::endl;
    }
    return polygon.is_simple();
  };

  template <typename Iter>
  bool isContourSimple(Iter begin, Iter end){
    return isContourSimple_impl(begin, end,
                  typename std::iterator_traits<Iter>::iterator_category());
  };

  template <typename Iter>
  CgalPolygon2d createCgalPolygonFromOpenCvPoints_impl(Iter begin, Iter end, double resolution, std::bidirectional_iterator_tag){
    CgalPolygon2d polygon;
    for (auto it = begin; it < end; ++it) {
      polygon.push_back(CgalPoint2d(static_cast<double>((*it).y * resolution), static_cast<double>((*it).x) * resolution));
    }
    return polygon;
  };

  template <typename Iter>
  CgalPolygon2d createCgalPolygonFromOpenCvPoints(Iter begin, Iter end, double resolution){
    return createCgalPolygonFromOpenCvPoints_impl(begin, end, resolution,
                                                  typename std::iterator_traits<Iter>::iterator_category());
  };

  void performConvexDecomposition(const CgalPolygon2d& polygon, CgalPolygon2dListContainer* output_polyong_list);

}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_
