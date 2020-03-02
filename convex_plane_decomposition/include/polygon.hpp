#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_

#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/intersections.h>
#include <CGAL/partition_2.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/Surface_sweep_2_algorithms.h>
#include <CGAL/Vector_2.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <glog/logging.h>

#include "grid_map_core/GridMap.hpp"
#include "geometry_utils.hpp"
#include "types.hpp"

namespace convex_plane_extraction{

  typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
  typedef CGAL::Partition_traits_2<K>                           Traits;
  typedef Traits::Point_2                                       CgalPoint2d;
  typedef Traits::Vector_2                                      CgalVector2d;
  typedef Traits::Polygon_2                                     CgalPolygon2d;
  typedef Traits::Segment_2                                     CgalSegment2d;
  typedef std::vector<CgalPolygon2d>                            CgalPolygon2dContainer;
  typedef CgalPolygon2dContainer::const_iterator                CgalPolygon2dConstIterator;
  typedef CGAL::Polygon_set_2<K, std::vector<CgalPoint2d>>      CgalPolygon2dSetContainer;
  typedef CgalPolygon2d::Vertex_const_iterator                  CgalPolygon2dVertexConstIterator;
  typedef CgalPolygon2d::Vertex_iterator                        CgalPolygon2dVertexIterator;

  typedef std::vector<Eigen::Vector3d>                          Polygon3d;
  typedef std::vector<Polygon3d>                                Polygon3dVectorContainer;
  typedef K::Intersect_2                                        Intersect_2;

  struct PolygonWithHoles{
    CgalPolygon2d outer_contour;
    CgalPolygon2dContainer holes;
  };


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
  CgalPolygon2d createCgalPolygonFromOpenCvPoints_impl(Iter begin, Iter end, const double& resolution, std::bidirectional_iterator_tag){
    CgalPolygon2d polygon;
    for (auto it = begin; it < end; ++it) {

      polygon.push_back(CgalPoint2d(static_cast<double>((*it).y * resolution), static_cast<double>((*it).x) * resolution));
    }
    return polygon;
  };

  template <typename Iter>
  CgalPolygon2d createCgalPolygonFromOpenCvPoints(Iter begin, Iter end, const double& resolution){
    return createCgalPolygonFromOpenCvPoints_impl(begin, end, resolution,
                                                  typename std::iterator_traits<Iter>::iterator_category());
  };

  void performConvexDecomposition(const CgalPolygon2d& polygon, CgalPolygon2dContainer* output_polyong_list);

  bool doPolygonAndSegmentIntersect(const CgalPolygon2d& polygon, const CgalSegment2d& segment, bool print_flag);

  int getClosestPolygonVertexPosition(const CgalPolygon2d& polygon, const CgalPoint2d& point);

  void getVertexPositionsInAscendingDistanceToPoint(const CgalPolygon2d& polygon, const CgalPoint2d& point,
                                                    std::multimap<double, int>* vertex_positions);

  void getSegmentNormalVector(const CgalSegment2d& segment, Eigen::Vector2d* normal_vector);

  double computeTriangleArea(double side_length_a, double side_length_b, double side_length_c);

  CgalPolygon2dVertexIterator next(const CgalPolygon2dVertexIterator& iterator, const CgalPolygon2d& polygon);

  void approximateContour(CgalPolygon2d* polygon);

  void upSampleLongEdges(CgalPolygon2d* polygon);

  double getEdgeLength(const CgalPolygon2dVertexIterator& source, const CgalPolygon2d& polygon);

  void detectDentLocations(std::map<double, int>* dent_locations, const CgalPolygon2d& polygon);

  void connectSecondPolygonToFirst(CgalPolygon2d& left_polygon, CgalPolygon2d& right_polygon);

  std::pair<int, int> getIndicesOfClosestVertexPair(CgalPolygon2d& left_polygon, CgalPolygon2d& right_polygon);

  struct Intersection{
    void setEdgeSourceLocation(const int location){
      edge_source_location_ = location;
    }
    void setEdgeTargetLocation(const int location){
      edge_target_location_ = location;
    }
    void setIntersectionPoint(const CgalPoint2d& intersection_point){
      intersection_point_ = intersection_point;
    }
    void setAllMembers(const int source_location, const int target_location, const CgalPoint2d& intersection_point){
      edge_source_location_ = source_location;
      edge_target_location_ = target_location;
      intersection_point_ = intersection_point;
    }

    int edge_source_location_;
    int edge_target_location_;
    CgalPoint2d intersection_point_;
  };

  bool intersectPolygonWithRay(int ray_target_location, CGAL::Orientation orientation, const CgalPolygon2d& polygon,
      Intersection* intersection);

  CgalPolygon2dVertexIterator erase(CgalPolygon2dVertexIterator first, CgalPolygon2dVertexIterator last,
                                    CgalPolygon2d* polygon);

  void copyVertices(const CgalPolygon2d& old_polygon, const CgalPolygon2dVertexIterator first, const CgalPolygon2dVertexIterator last,
                    CgalPolygon2d* new_polygon, const CgalPolygon2dVertexIterator insert_position);

  std::list<CgalPolygon2d> decomposeInnerApproximation(const CgalPolygon2d& polygon);

  void printPolygon(const CgalPolygon2d& polygon);
}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_
