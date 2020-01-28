#include <glog/logging.h>
#include "geometry_utils.hpp"

namespace convex_plane_extraction {

  double scalarCrossProduct(const Vector2d &leftVector, const Vector2d &rightVector) {
    return leftVector.y() * rightVector.x() - leftVector.x() * rightVector.y();
  }

  // Return the non-normalized distance of a 2D-point to a line given by support vector and direction vector.
  // Negative distance corresponds to a point on the left of the direction vector
  double distanceToLine(const Vector2d &lineSupportVector,
                        const Vector2d &lineDirectionalVector,
                        const Vector2d &testPoint) {
    Vector2d secondPointOnLine = lineSupportVector + lineDirectionalVector;
    return scalarCrossProduct(lineDirectionalVector, testPoint)
        + scalarCrossProduct(lineSupportVector, secondPointOnLine);
  }

  bool isPointOnRightSide(const Vector2d& line_support_vector, const Vector2d& line_direction_vector, const Vector2d& point){
    Vector2d normal_vector(line_direction_vector.y(), (-1.0)*line_direction_vector.x());
    normal_vector.normalize();
    Vector2d point_vector = point - line_support_vector;
    return point_vector.transpose() * normal_vector > 0;
  }

  bool isPointOnLeftSide(const Vector2d& line_support_vector, const Vector2d& line_direction_vector, const Vector2d& point){
    Vector2d normal_vector(line_direction_vector.y(), (-1.0)*line_direction_vector.x());
    normal_vector.normalize();
    Vector2d point_vector = point - line_support_vector;
    return point_vector.transpose() * normal_vector < 0;
  }

  double computeAngleBetweenVectors(const Vector2d& first_vector, const Vector2d& second_vector){
    double scalar_product = first_vector.transpose() * second_vector;
    return acos( scalar_product / (first_vector.norm() * second_vector.norm()));
  }

  bool intersectRayWithLineSegment(const Vector2d& ray_source, const Vector2d& ray_direction,
      const Vector2d& segment_source, const Vector2d& segment_target, Vector2d* intersection_point){
    CHECK_NOTNULL(intersection_point);
    Vector2d segment_direction = segment_target - segment_source;
    Matrix2d A;
    A.col(0) = ray_direction;
    A.col(1) = - segment_direction;
    Vector2d b = segment_source - ray_source;
    Vector2d solution = A.fullPivHouseholderQr().solve(b);
    if (solution(0) <= 0 || solution(1) < 0 || solution(1) > 1){
      return false;
    }
    *intersection_point = segment_source + solution(1) * segment_direction;
    return true;
  }

  double distanceBetweenPoints(Vector2d first, Vector2d second){
    return (second - first).norm();
  }

}

