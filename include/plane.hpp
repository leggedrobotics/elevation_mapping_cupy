#ifndef CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_
#define CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_

#include <list>

#include "polygon.hpp"
namespace convex_plane_extraction {

  typedef std::list<Polygon> PolygonContainer;

  class Plane {

   public:
    Plane();

    virtual ~Plane();

   private:

    Polygon outer_boundary_;
    PolygonContainer hole_boundaries_;

  };

}
#endif //CONVEX_PLANE_EXTRACTION_SRC_PLANE_HPP_
