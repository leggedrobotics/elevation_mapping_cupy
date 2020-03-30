#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_

#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <cmath>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/Ray_2.h>
#include <CGAL/Surface_sweep_2_algorithms.h>
#include <CGAL/Vector_2.h>
#include <CGAL/intersections.h>
#include <CGAL/partition_2.h>
#include <CGAL/squared_distance_2.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <glog/logging.h>

#include "grid_map_core/GridMap.hpp"
#include "geometry_utils.hpp"
#include "types.hpp"

namespace convex_plane_extraction{

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_exact_constructions_kernel EKernel;
typedef CGAL::Partition_traits_2<K> Traits;
typedef CGAL::Point_2<K>                                      CgalPoint2d;
  typedef CGAL::Vector_2<K>                                     CgalVector2d;
  typedef CGAL::Polygon_2<K>                                    CgalPolygon2d;
  typedef CGAL::Polygon_with_holes_2<K>                         CgalPolygonWithHoles2d;
  typedef Traits::Segment_2                                     CgalSegment2d;
  typedef std::vector<CgalPolygon2d> CgalPolygon2dContainer;
  typedef Traits::Ray_2 CgalRay2d;
  typedef Traits::Circle_2 CgalCircle2d;
  typedef CgalPolygon2dContainer::const_iterator CgalPolygon2dConstIterator;
  typedef CGAL::Polygon_set_2<K, std::vector<CgalPoint2d>>      CgalPolygon2dSetContainer;
  typedef CgalPolygon2d::Vertex_const_iterator                  CgalPolygon2dVertexConstIterator;
  typedef CgalPolygon2d::Vertex_iterator CgalPolygon2dVertexIterator;

  typedef std::vector<Eigen::Vector3d> Polygon3d;
  typedef std::vector<Polygon3d> Polygon3dVectorContainer;
  typedef K::Intersect_2 Intersect_2;

  struct PolygonWithHoles {
    CgalPolygon2d outer_contour;
    CgalPolygon2dContainer holes;
  };

  enum class SegmentIntersectionType : int { kSource, kTarget, kInterior };

  struct RaySegmentIntersection {
    SegmentIntersectionType intersection_location;
    CgalPoint2d intersection_point;
  };

  template <typename Iter>
  bool isContourSimple_impl(Iter begin, Iter end, std::bidirectional_iterator_tag) {
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

  bool next(const CgalPolygon2dVertexIterator& iterator, CgalPolygon2dVertexIterator& output_iterator, const CgalPolygon2d& polygon);

  CgalPolygon2dVertexIterator previous(const CgalPolygon2dVertexIterator& iterator, const CgalPolygon2d& polygon);

  void approximateContour(CgalPolygon2d* polygon, int max_number_of_iterations, double relative_local_area_threshold,
                          double absolute_local_area_threshold, double relative_total_area_threshold, double absolute_total_area_threshold);

  void upSampleLongEdges(CgalPolygon2d* polygon);

  double getEdgeLength(const CgalPolygon2dVertexIterator& source, const CgalPolygon2d& polygon);

  std::multimap<double, int> detectDentLocations(const CgalPolygon2d& polygon);

  void connectSecondPolygonToFirst(CgalPolygon2d& left_polygon, CgalPolygon2d& right_polygon);

  std::pair<int, int> getIndicesOfClosestVertexPair(CgalPolygon2d& first_polygon, CgalPolygon2d& second_polygon);

  std::multimap<double, std::pair<int, int>> getClosestVertexPairsOrdered(const CgalPolygon2d& first_polygon, const CgalPolygon2d& second_polygon);

  CgalPoint2d getPolygonVertexAtIndex(const CgalPolygon2d& polygon, int index);

  bool doPointAndPolygonIntersect(const CgalPolygon2d& polygon, const CgalPoint2d& point, int& segment_source_vertex_index);

  struct Intersection{
    void setEdgeSourceLocation(const int location){
      edge_source_location_ = location;
    }
    void setEdgeTargetLocation(const int location){
      edge_target_location_ = location;
    }
    void setIntersectionPoint(const CgalPoint2d& intersection_point) { intersection_point_ = intersection_point; }
    void setAllMembers(const int source_location, const int target_location, const CgalPoint2d& intersection_point,
                       const SegmentIntersectionType& intersection_type) {
      edge_source_location_ = source_location;
      edge_target_location_ = target_location;
      intersection_type_ = intersection_type;
      intersection_point_ = intersection_point;
    }

    int edge_source_location_;
    int edge_target_location_;
    SegmentIntersectionType intersection_type_;
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

  std::pair<int, int> getVertexPositionsWithHighestHoleSlConcavityMeasure(const CgalPolygon2d& polygon);

  std::pair<int, CgalPoint2d> getClosestPointAndSegmentOnPolygonToPoint(const CgalPoint2d& point, const CgalPolygon2d& polygon);

  std::vector<std::pair<int, int>> getCommonVertexPairIndices(const CgalPolygon2d& first_polygon, const CgalPolygon2d& second_polygon);

  std::vector<int> getVertexIndicesOfFirstPolygonContainedInSecondPolygonContour(const CgalPolygon2d& first_polygon,
                                                                                 const CgalPolygon2d& second_polygon);

  bool doRayAndSegmentIntersect(const CgalRay2d& ray, const CgalSegment2d& segment, RaySegmentIntersection* intersection);

  }     // namespace convex_plane_extraction
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_POLYGON_HPP_
