#ifndef CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_
#define CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_

#include <list>

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

    bool isValid();

    bool convertConvexPolygonsToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Vector2d& map_position) const;

   private:

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
