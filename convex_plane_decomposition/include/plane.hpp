#ifndef CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_
#define CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_

#include <list>

#include "geometry_utils.hpp"
#include "polygon.hpp"

namespace convex_plane_extraction {

  struct PlaneParameters{

    PlaneParameters(Eigen::Vector3d normal_vector, Eigen::Vector3d support_vector)
    : support_vector(support_vector),
      normal_vector(normal_vector){};

    Eigen::Vector3d support_vector;
    Eigen::Vector3d normal_vector;
  };

  class Plane {

   public:

    Plane(const CgalPolygon2d& plane_contour, const PlaneParameters parameters)
      : plane_contour_(plane_contour),
        normal_vector_(parameters.normal_vector),
        support_vector_(parameters.support_vector){}

    PlaneParameters getPlaneParameters() const {
      return PlaneParameters(normal_vector_, support_vector_);
    }

    bool setPlaneContour(const CgalPolygon2d& plane_contour){
      plane_contour_ = plane_contour;
    }

    const Eigen::Vector3d& getPlaneNormalVector() const{
      return normal_vector_;
    }

    Eigen::Vector3d& getPlaneNormalVectorMutable(){
      return normal_vector_;
    }

    bool setNormalAndSupportVector(const Eigen::Vector3d& normal_vector,const Eigen::Vector3d& support_vector);

    bool hasOuterContour() const;

    CgalPolygon2d& getOuterPolygonMutable(){
      return plane_contour_;
    }

    const CgalPolygon2d& getOuterPolygon() const{
      return plane_contour_;
    }


    CgalPolygon2dContainer& getConvexPolygonsMutable(){
      return convex_polygons_;
    }

    const CgalPolygon2dContainer& getConvexPolygons() const{
      return convex_polygons_;
    }

    void addConvexPolygon(const CgalPolygon2d& convex_polygon){
      convex_polygons_.push_back(convex_polygon);
    }

    void setConvexPolygons(CgalPolygon2dContainer& convex_polygons){
      convex_polygons_ = convex_polygons;
    }

   private:

    CgalPolygon2d plane_contour_;
    CgalPolygon2dContainer convex_polygons_;

    Vector3d normal_vector_;
    Vector3d support_vector_;
  };

}
#endif //CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_
