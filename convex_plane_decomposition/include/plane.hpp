#ifndef CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_
#define CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_

#include <list>

#include "geometry_utils.hpp"
#include "polygon.hpp"

namespace convex_plane_extraction {

  class Plane {

   public:

    Plane();

    Plane(CgalPolygon2d& outer_polygon, CgalPolygon2dListContainer& hole_polygon_list);

    virtual ~Plane();

    bool addOuterPolygon(const CgalPolygon2d& outer_polygon);

    bool addHolePolygon(const CgalPolygon2d& hole_polygon);

    bool setNormalAndSupportVector(const Eigen::Vector3d& normal_vector,const Eigen::Vector3d& support_vector);

    bool decomposePlaneInConvexPolygons();

    bool hasOuterContour() const;

    CgalPolygon2dVertexConstIterator outerPolygonVertexBegin() const;

    CgalPolygon2dVertexConstIterator outerPolygonVertexEnd() const;

    CgalPolygon2dListConstIterator holePolygonBegin() const;

    CgalPolygon2dListConstIterator holePolygonEnd() const;

    bool isValid() const;

    bool convertConvexPolygonsToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const;

    bool convertOuterPolygonToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const;

    bool convertHolePolygonsToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const;

    void convertPoint2dToWorldFrame(const CgalPoint2d& point, Vector2d* output_point, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const;

    void computePoint3dWorldFrame(const Vector2d& input_point, Vector3d* output_point) const;

    void resolveHoles();

    const CgalPolygon2dListContainer& getConvexPolygons() const;

   private:

    void extractSlConcavityPointsOfHole(const CgalPolygon2d& hole, std::vector<int>* concavity_positions);

    void slConcavityHoleVertexSorting(const CgalPolygon2d& hole, std::multimap<double, std::pair<int, int>>* concavity_positions);

    bool initialized_;

    CgalPolygon2d outer_polygon_;
    CgalPolygon2dListContainer hole_polygon_list_;
    CgalPolygon2dListContainer convex_polygon_list_;

    Vector3d normal_vector_;
    Vector3d support_vector_;
  };

  typedef std::list<Plane> PlaneListContainer;

}
#endif //CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_
