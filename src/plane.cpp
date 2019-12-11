#include "plane.hpp"

namespace convex_plane_extraction{

  Plane::Plane()
    : initialized_(false){}

  Plane::Plane(CgalPolygon2d& outer_polygon, CgalPolygon2dListContainer& hole_polygon_list)
    : outer_polygon_(outer_polygon),
      hole_polygon_list_(hole_polygon_list),
      initialized_(false){};

  Plane::~Plane() = default;

  bool Plane::addHolePolygon(const convex_plane_extraction::CgalPolygon2d & hole_polygon) {
    if (initialized_){
      LOG(ERROR) << "Hole cannot be added to plane since already initialized!";
      return false;
    }
    CHECK(!hole_polygon.is_empty());
    hole_polygon_list_.push_back(hole_polygon);
    return true;
  }

  bool Plane::addOuterPolygon(const convex_plane_extraction::CgalPolygon2d & outer_polygon) {
    if (initialized_){
      LOG(ERROR) << "Outer polygon cannot be added to plane since already initialized!";
      return false;
    }
    CHECK(!outer_polygon.is_empty());
    outer_polygon_ = (outer_polygon);
    return true;
  }

  bool Plane::setNormalAndSupportVector(const Eigen::Vector3d& normal_vector, const Eigen::Vector3d& support_vector){
    if (initialized_){
      LOG(ERROR) << "Could not set normal and support vector of planbe since already initialized!";
      return false;
    }
    normal_vector_ = normal_vector;
    support_vector_ = support_vector;
    if (!outer_polygon_.is_empty()) {
      initialized_ = true;
      LOG(INFO) << "Initialized plane located around support vector " << support_vector << " !";
    }
    return true;
  }

  bool Plane::isValid(){
    return !outer_polygon_.is_empty() && initialized_;
  }

  bool Plane::decomposePlaneInConvexPolygons(){
    if (!initialized_){
      LOG(ERROR) << "Connot perform convex decomposition of plane, since not yet initialized.";
      return false;
    }
    performConvexDecomposition(outer_polygon_, &convex_polygon_list_);
    return true;
  }

  bool Plane::convertConvexPolygonsToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Vector2d& map_position) const{
    if (convex_polygon_list_.empty()){
      LOG(INFO) << "No convex polygons to convert!";
      return false;
    }
    for (const auto& polygon : convex_polygon_list_){
      if (polygon.is_empty()){
        continue;
      }
      Polygon3d polygon_temp;
      for (const auto& point : polygon){
        double x = -point.x() + 100*0.02 + map_position.x();
        LOG_IF(FATAL, !isfinite(x)) << "Not finite x value!";
        double y = -point.y() + 100*0.02 + map_position.y();
        LOG_IF(FATAL, !isfinite(y)) << "Not finite y value!";
        double z = (-(x - support_vector_.x())*normal_vector_.x() -
            (y - support_vector_.y())* normal_vector_.y())/normal_vector_(2) + support_vector_(2);
        LOG_IF(FATAL, !isfinite(z)) << "Not finite z value!";
        polygon_temp.push_back(Eigen::Vector3d(x, y, z));
      }
      output_container->push_back(polygon_temp);
    }
    return true;
  }

  bool Plane::hasOuterContour() const{
    return !outer_polygon_.is_empty();
  }
}
