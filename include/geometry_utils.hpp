#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_

#include "types.hpp"

namespace convex_plane_extraction {

double scalarCrossProduct(const Vector2d &leftVector, const Vector2d &rightVector);

double distanceToLine(const Vector2d& lineSupportVector, const Vector2d& lineDirectionalVector, const Vector2d& testPoint);

}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_
