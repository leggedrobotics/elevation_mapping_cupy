#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_

#include "types.hpp"

namespace convex_plane_extraction {

double scalarCrossProduct(const Vector2d &leftVector, const Vector2d &rightVector);

double distanceToLine(const Vector2d& lineSupportVector, const Vector2d& lineDirectionalVector, const Vector2d& testPoint);

bool isPointOnRightSide(const Vector2d& line_support_vector, const Vector2d& line_direction_vector, const Vector2d& point);

double computeAngleBetweenVectors(const Vector2d& first_vector, const Vector2d& second_vector);

bool isPointOnLeftSide(const Vector2d& line_support_vector, const Vector2d& line_direction_vector, const Vector2d& point);

}
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_
