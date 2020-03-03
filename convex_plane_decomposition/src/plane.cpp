#include "plane.hpp"

namespace convex_plane_extraction{

  Plane::Plane()
    : initialized_(false){}

  Plane::Plane(CgalPolygon2d& outer_polygon, CgalPolygon2dContainer& holes)
    : outer_polygon_(outer_polygon),
      hole_polygons_(holes),
      initialized_(false){}

  bool Plane::setNormalAndSupportVector(const Eigen::Vector3d& normal_vector, const Eigen::Vector3d& support_vector){
    CHECK(!initialized_) << "Could not set normal and support vector of plane since already initialized!";
    normal_vector_ = normal_vector;
    support_vector_ = support_vector;
    constexpr double inclinationThreshold = 0.35; // cos(70°)
    Eigen::Vector3d upwards(0,0,1);
    if (abs(normal_vector.transpose()*upwards < inclinationThreshold)){
      initialized_ = false;
      LOG(WARNING) << "Inclination to high, plane will be ignored!";
    } else if (!outer_polygon_.is_empty()) {
      initialized_ = true;
      LOG(INFO) << "Initialized plane located around support vector " << support_vector << " !";
    }
    return true;
  }

  bool Plane::isValid() const{
    return !outer_polygon_.is_empty() && initialized_;
  }

  bool Plane::decomposePlaneInConvexPolygons(){
    if (!initialized_){
      LOG(ERROR) << "Connot perform convex decomposition of plane, since not yet initialized.";
      return false;
    }
    performConvexDecomposition(outer_polygon_, &convex_polygons_);
    return true;
  }

  bool Plane::convertConvexPolygonsToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const{
    CHECK_NOTNULL(output_container);
    if (convex_polygons_.empty()){
      LOG(INFO) << "No convex polygons to convert!";
      return false;
    }
    for (const auto& polygon : convex_polygons_){
      if (polygon.is_empty()){
        continue;
      }
      Polygon3d polygon_temp;
      for (const auto& point : polygon){
        Vector2d point_2d_world_frame;
        convertPoint2dToWorldFrame(point, &point_2d_world_frame, transformation, map_position);
        Vector3d point_3d_world_frame;
        computePoint3dWorldFrame(point_2d_world_frame, &point_3d_world_frame);
        polygon_temp.push_back(point_3d_world_frame);
      }
      output_container->push_back(polygon_temp);
    }
    return true;
  }

  bool Plane::convertOuterPolygonToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const{
    CHECK_NOTNULL(output_container);
    if (outer_polygon_.is_empty()){
      LOG(INFO) << "No convex polygons to convert!";
      return false;
    }
    Polygon3d polygon_temp;
    for (const auto& point : outer_polygon_){
      Vector2d point_2d_world_frame;
      convertPoint2dToWorldFrame(point, &point_2d_world_frame, transformation, map_position);
      Vector3d point_3d_world_frame;
      computePoint3dWorldFrame(point_2d_world_frame, &point_3d_world_frame);
      polygon_temp.push_back(point_3d_world_frame);
    }
    output_container->push_back(polygon_temp);
    return true;
  }

  bool Plane::convertHolePolygonsToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const{
    CHECK_NOTNULL(output_container);
    if (hole_polygons_.empty()){
      LOG(INFO) << "No hole polygons to convert!";
      return false;
    }
    for (const auto& polygon : hole_polygons_){
      if (polygon.is_empty()){
        continue;
      }
      Polygon3d polygon_temp;
      for (const auto& point : polygon){
        Vector2d point_2d_world_frame;
        convertPoint2dToWorldFrame(point, &point_2d_world_frame, transformation, map_position);
        Vector3d point_3d_world_frame;
        computePoint3dWorldFrame(point_2d_world_frame, &point_3d_world_frame);
        polygon_temp.push_back(point_3d_world_frame);
      }
      output_container->push_back(polygon_temp);
    }
    return true;
  }

  void Plane::convertPoint2dToWorldFrame(const CgalPoint2d& point, Vector2d* output_point, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const{
    CHECK_NOTNULL(output_point);
    Eigen::Vector2d point_vector(point.x(), point.y());
    point_vector = transformation * point_vector;
    point_vector = point_vector + map_position;
    LOG_IF(FATAL, !std::isfinite(point_vector.x())) << "Not finite x value!";
    LOG_IF(FATAL, !std::isfinite(point_vector.y())) << "Not finite y value!";
    *output_point = point_vector;
  }

  void Plane::computePoint3dWorldFrame(const Vector2d& input_point, Vector3d* output_point) const{
    CHECK_NOTNULL(output_point);
    double z = (-(input_point.x() - support_vector_.x())*normal_vector_.x() -
        (input_point.y() - support_vector_.y())* normal_vector_.y())/normal_vector_(2) + support_vector_(2);
    LOG_IF(FATAL, !std::isfinite(z)) << "Not finite z value!";
    *output_point = Vector3d(input_point.x(), input_point.y(), z);
  }


}
