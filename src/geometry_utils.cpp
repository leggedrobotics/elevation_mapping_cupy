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

}

