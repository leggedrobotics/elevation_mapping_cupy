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
      LOG(ERROR) << "Could not set normal and support vector of plane since already initialized!";
      return false;
    }
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
    performConvexDecomposition(outer_polygon_, &convex_polygon_list_);
    return true;
  }

  bool Plane::convertConvexPolygonsToWorldFrame(Polygon3dVectorContainer* output_container, const Eigen::Matrix2d& transformation, const Eigen::Vector2d& map_position) const{
    CHECK_NOTNULL(output_container);
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
    if (hole_polygon_list_.empty()){
      LOG(INFO) << "No convex polygons to convert!";
      return false;
    }
    for (const auto& polygon : hole_polygon_list_){
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
    LOG_IF(FATAL, !isfinite(point_vector.x())) << "Not finite x value!";
    LOG_IF(FATAL, !isfinite(point_vector.y())) << "Not finite y value!";
    *output_point = point_vector;
  }

  void Plane::computePoint3dWorldFrame(const Vector2d& input_point, Vector3d* output_point) const{
    CHECK_NOTNULL(output_point);
    double z = (-(input_point.x() - support_vector_.x())*normal_vector_.x() -
        (input_point.y() - support_vector_.y())* normal_vector_.y())/normal_vector_(2) + support_vector_(2);
    LOG_IF(FATAL, !isfinite(z)) << "Not finite z value!";
    *output_point = Vector3d(input_point.x(), input_point.y(), z);
  }

  bool Plane::hasOuterContour() const{
    return !outer_polygon_.is_empty();
  }

  CgalPolygon2dVertexConstIterator Plane::outerPolygonVertexBegin() const{
    CHECK(isValid());
    return outer_polygon_.vertices_begin();
  }

  CgalPolygon2dVertexConstIterator Plane::outerPolygonVertexEnd() const{
    CHECK(isValid());
    return outer_polygon_.vertices_end();
  }

  CgalPolygon2dListConstIterator Plane::holePolygonBegin() const{
    CHECK(isValid());
    return hole_polygon_list_.begin();
  }

  CgalPolygon2dListConstIterator Plane::holePolygonEnd() const{
    CHECK(isValid());
    return hole_polygon_list_.end();
  }
}
