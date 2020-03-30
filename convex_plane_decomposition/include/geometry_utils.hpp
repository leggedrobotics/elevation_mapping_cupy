#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_

#include "types.hpp"

namespace convex_plane_extraction {

double scalarCrossProduct(const Vector2d& leftVector, const Vector2d& rightVector);

double computeAngleBetweenVectors(const Vector2d& first_vector, const Vector2d& second_vector);

double distanceBetweenPoints(const Vector2d& first, const Vector2d& second);

double distanceToLine(const Vector2d& lineSupportVector, const Vector2d& lineDirectionalVector, const Vector2d& testPoint);

std::pair<double, Vector2d> getDistanceAndClosestPointOnLineSegment(const Vector2d& point, const Vector2d& source, const Vector2d& target);

double getDistanceOfPointToLineSegment(const Vector2d& point, const Vector2d& source, Vector2d& target);

bool intersectLineSegmentWithLineSegment(const Vector2d& segment_1_source, const Vector2d& segment_1_target,
                                         const Vector2d& segment_2_source, const Vector2d& segment_2_target, Vector2d* intersection_point);

bool intersectRayWithLineSegment(const Vector2d& ray_source, const Vector2d& ray_direction, const Vector2d& segment_source,
                                 const Vector2d& segment_target, Vector2d* intersection_point);

bool isPointOnLeftSide(const Vector2d& line_support_vector, const Vector2d& line_direction_vector, const Vector2d& point);

bool isPointOnRightSideOfLine(const Vector2d& line_support_vector, const Vector2d& line_direction_vector, const Vector2d& point);

}  // namespace convex_plane_extraction
#endif  // CONVEX_PLANE_EXTRACTION_INCLUDE_GEOMETRY_UTILS_HPP_
