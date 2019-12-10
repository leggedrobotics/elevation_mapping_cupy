#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_

#include <iostream>
#include <vector>
#include <list>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Surface_sweep_2_algorithms.h>

namespace convex_plane_extraction{

  typedef CGAL::Exact_predicates_inexact_constructions_kernel       K;
  typedef K::Point_2                                                Point;
  typedef CGAL::Polygon_2<K>                                        Polygon_2;
  typedef CGAL::Arr_segment_traits_2<K>                             Traits_2;
  typedef Traits_2::Curve_2                                         Segment_2;


  template <typename Iter>
  bool isContourSimple_impl(Iter begin, Iter end, std::bidirectional_iterator_tag){
    Polygon_2 polygon;
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

  class Polygon{
   public:
    Polygon();

    virtual ~Polygon();

   private:
  };
}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_
