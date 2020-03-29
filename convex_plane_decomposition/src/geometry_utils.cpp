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

  bool isPointOnRightSideOfLine(const Vector2d& line_support_vector, const Vector2d& line_direction_vector, const Vector2d& point){
    Vector2d normal_vector(line_direction_vector.y(), (-1.0)*line_direction_vector.x());
    normal_vector.normalize();
    Vector2d point_vector = point - line_support_vector;
    return point_vector.transpose() * normal_vector > 0;
  }

  bool isPointOnLeftSide(const Vector2d& line_support_vector, const Vector2d& line_direction_vector, const Vector2d& point){
    Vector2d normal_vector(line_direction_vector.y(), (-1.0)*line_direction_vector.x());
    normal_vector.normalize();
    Vector2d point_vector = point - line_support_vector;
    return point_vector.dot(normal_vector) < 0;
  }

  double computeAngleBetweenVectors(const Vector2d& first_vector, const Vector2d& second_vector) {
    double scalar_product = first_vector.transpose() * second_vector;
    return acos(abs(scalar_product) / (first_vector.norm() * second_vector.norm()));
  }

  //  bool intersectRayWithLineSegment(const Vector2d& ray_source, const Vector2d& ray_direction,
  //      const Vector2d& segment_source, const Vector2d& segment_target, Vector2d* intersection_point){
  //    CHECK_NOTNULL(intersection_point);
  //    Vector2d segment_direction = segment_target - segment_source;
  //    Matrix2d A;
  //    A.col(0) = ray_direction;
  //    A.col(1) = - segment_direction;
  //    Vector2d b = segment_source - ray_source;
  //    auto householder_qr = A.fullPivHouseholderQr();
  //    if (householder_qr.rank() < 2) {
  //      // Check whether segment and ray overlap.
  //      const Vector2d p_ray_source_segment_source = segment_source - ray_source;
  //      const Vector2d p_ray_source_segment_target = segment_target - ray_source;
  //      if (p_ray_source_segment_source.norm() > p_ray_source_segment_target) {
  //      }
  //      Vector2d ray_parameter_solution =
  //          Vector2d(p_ray_source_segment_source.x() / ray_direction.x(), p_ray_source_segment_source.y() / ray_direction.y());
  //      constexpr double kSolutionDeviation = 0.0001;
  //      if (abs(ray_parameter_solution.x() - ray_parameter_solution.y()) < kSolutionDeviation) {
  //        double parameter_solution_tmp = ray_parameter_solution.mean();
  //        if (parameter_solution_tmp < 0) {
  //          return false;
  //        }
  //        CHECK_GT(parameter_solution_tmp, 0);
  //        Vector2d p_ray_source_segment_target = segment_target - ray_source;
  //        ray_parameter_solution =
  //            Vector2d(p_ray_source_segment_target.x() / ray_direction.x(), p_ray_source_segment_target.y() / ray_direction.y());
  //        // If ray is parallel (rank loss) and source point lies on ray, then target point has to as well.
  //        CHECK_LT(abs(ray_parameter_solution.x() - ray_parameter_solution.y()), kSolutionDeviation);
  //        if (ray_parameter_solution.mean() < 0) {
  //          return false;
  //        }
  //        if (ray_parameter_solution.mean() < parameter_solution_tmp) {
  //          *intersection_point = segment_target;
  //        } else {
  //          *intersection_point = segment_source;
  //        }
  //        return true;
  //      }
  //      return false;
  //    }
  //    Vector2d solution = householder_qr.solve(b);
  //    Vector2d ray_solution = ray_source + solution(0) * segment_direction;
  //    Vector2d segment_solution = segment_source + solution(1) * segment_direction;
  //    constexpr double kSegmentBorderToleranceMeters = 0.001;
  //    if (solution(0) <= 1) {
  //      return false;
  //    } else if (solution(1) < 0) {
  //      if ((segment_solution - segment_source).norm() > kSegmentBorderToleranceMeters) {
  //        return false;
  //      } else {
  //        *intersection_point = segment_source;
  //        return true;
  //      }
  //    } else if (solution(1) > 1) {
  //      if ((segment_solution - segment_target).norm() > kSegmentBorderToleranceMeters) {
  //        return false;
  //      } else {
  //        *intersection_point = segment_target;
  //      }
  //    } else {
  //      *intersection_point = segment_solution;
  //      return true;
  //    }
  //  }

  bool intersectLineSegmentWithLineSegment(const Vector2d& segment_1_source, const Vector2d& segment_1_target,
                                           const Vector2d& segment_2_source, const Vector2d& segment_2_target,
                                           Vector2d* intersection_point) {
    CHECK_NOTNULL(intersection_point);
    Vector2d segment_1_direction = segment_1_target - segment_1_source;
    Vector2d segment_2_direction = segment_2_target - segment_2_source;
    Matrix2d A;
    A.col(0) = segment_1_direction;
    A.col(1) = -segment_2_direction;
    Vector2d b = segment_2_source - segment_1_source;
    auto householder_qr = A.fullPivHouseholderQr();
    if (householder_qr.rank() < 2){
      // Check whether segment and ray overlap.
      Vector2d p_ray_source_segment_source = segment_2_source - segment_1_source;
      Vector2d ray_parameter_solution = Vector2d(p_ray_source_segment_source.x() / segment_1_direction.x(),
                                                 p_ray_source_segment_source.y() / segment_1_direction.y());
      constexpr double kSolutionDeviation = 0.0001;
      if (abs(ray_parameter_solution.x() - ray_parameter_solution.y()) < kSolutionDeviation){
        double parameter_solution_tmp = ray_parameter_solution.mean();
        if (parameter_solution_tmp < 0){
          return false;
        }
        CHECK_GT(parameter_solution_tmp, 0);
        Vector2d p_ray_source_segment_target = segment_2_target - segment_1_source;
        ray_parameter_solution = Vector2d(p_ray_source_segment_target.x() / segment_1_direction.x(),
                                          p_ray_source_segment_target.y() / segment_1_direction.y());
        // If ray is parallel (rank loss) and source point lies on ray, then target point has to as well.
        CHECK(abs(ray_parameter_solution.x() - ray_parameter_solution.y()) < kSolutionDeviation);
        if(ray_parameter_solution.mean() < 0){
          return false;
        }
        if (ray_parameter_solution.mean() < parameter_solution_tmp) {
          *intersection_point = segment_2_target;
        } else {
          *intersection_point = segment_2_source;
        }
        return true;
      }
      return false;
    }
    Vector2d solution = A.inverse() * b;  // householder_qr.solve(b);
    Vector2d ray_solution = segment_1_source + solution(0) * segment_2_direction;
    Vector2d segment_solution = segment_2_source + solution(1) * segment_2_direction;
    if (solution(0) < 0 || solution(0) > 1 || solution(1) < 0 || solution(1) > 1) {
      return false;
    }
    *intersection_point = segment_solution;
    return true;
  }

  double distanceBetweenPoints(const Vector2d& first, const Vector2d& second){
    return (second - first).norm();
  }

  double getDistanceOfPointToLineSegment(const Vector2d& point, const Vector2d& source, const Vector2d& target){
    return getDistanceAndClosestPointOnLineSegment(point, source, target).first;
  }

  std::pair<double, Vector2d> getDistanceAndClosestPointOnLineSegment(const Vector2d& point, const Vector2d& source, const Vector2d& target){
    const double segment_squared_length = (target - source).squaredNorm();
    if (segment_squared_length == 0.0){
      return std::make_pair(distanceBetweenPoints(point, source), source);
    }
    const double t = (point - source).dot(target - source) / segment_squared_length;
    if (t > 1.0){
      return std::make_pair((target - point).norm(), target);
    } else if (t < 0.0){
      return std::make_pair((source - point).norm(), source);
    } else {
      const Vector2d point_temp = abs(t)*(target - source);
      return std::make_pair((point_temp - point).norm(), point_temp);
    }
  }

}

