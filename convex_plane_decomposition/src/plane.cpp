#include "plane.hpp"

namespace convex_plane_decomposition{

  bool Plane::setNormalAndSupportVector(const Eigen::Vector3d& normal_vector, const Eigen::Vector3d& support_vector){
    normal_vector_ = normal_vector;
    support_vector_ = support_vector;
    return true;
  }

}
