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

}

