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

  void Plane::resolveHoles() {
    if (hole_polygon_list_.empty()) {
      return;
    }
    while(!hole_polygon_list_.empty()) {
      // Compute distance to outer contour for each hole.
      std::vector<double> distance_to_outer_polygon;
      std::vector<Polygon::iterator> outer_contour_connection_vertices; // closest outer contour vertex for each hole
      std::vector<Polygon::iterator> hole_connection_vertices;
    }
  }

double scalarCrossProduct(const Vector2d& leftVector, const Vector2d& rightVector) {
  return leftVector.y() * rightVector.x() - leftVector.x() * rightVector.y();
}

// Return the non-normalized distance of a 2D-point to a line given by support vector and direction vector.
// Negative distance corresponds to a point on the left of the direction vector
double distanceToLine(const Vector2d& lineSupportVector, const Vector2d& lineDirectionalVector, const Vector2d& testPoint){
  Vector2d secondPointOnLine = lineSupportVector + lineDirectionalVector;
  return scalarCrossProduct(lineDirectionalVector, testPoint) + scalarCrossProduct(lineSupportVector, secondPointOnLine);
}

void Plane::extractSlConcavityPointsOfHole(const CgalPolygon2d& hole, std::vector<int>* concavity_positions){
  CHECK_NOTNULL(concavity_positions);
  if (hole.is_empty()) {
    return;
  } else if (hole.size() == 1){
    concavity_positions->push_back(0);
  } else if (hole.size() == 2){
    concavity_positions->push_back(0);
    concavity_positions->push_back(1);
  } else {
    MatrixXd data_matrix(hole.size(), 2);
    Eigen::Vector2d mean_position = Eigen::Vector2d::Zero(2,1);
    auto hole_vertex_it = hole.vertices_begin();
    for (int row = 0; row < hole.size(); ++row){
      data_matrix(row, 0) = (*hole_vertex_it).x();
      data_matrix(row, 1) = (*hole_vertex_it).y();
      mean_position(0) += (*hole_vertex_it).x();
      mean_position(1) += (*hole_vertex_it).y();
      ++hole_vertex_it;
    }
    mean_position *= (1.0 / static_cast<double>(hole.size()));
    for (int row = 0; row < data_matrix.rows(); ++row) {
      data_matrix.row(row) -= mean_position.transpose();
    }
    Eigen::BDCSVD<MatrixXd> svd = data_matrix.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector2d orthonormal_to_PC_axis = svd.matrixV().col(1);
    // Place vector to mean position of hole.
    Eigen::Vector2d second_point_axis = mean_position + orthonormal_to_PC_axis;
    double max_distance = 0;
    double min_distance = 0;
    int position_right_concavity_point;
    int position_left_concavity_point;
    hole_vertex_it = hole.vertices_begin();
    for (; hole_vertex_it != hole.vertices_end(); ++hole_vertex_it){
      Eigen::Vector2d vertex_point;
      vertex_point << (*hole_vertex_it).x(), (*hole_vertex_it).y();
      double current_distance = distanceToLine(mean_position, orthonormal_to_PC_axis, vertex_point);
      if (current_distance > max_distance) {
        max_distance = current_distance;
        position_right_concavity_point = std::distance(hole.begin(), hole_vertex_it);
      } else if (current_distance < min_distance){
        min_distance = current_distance;
        position_left_concavity_point = std::distance(hole.begin(), hole_vertex_it);
      }
    }
    Polygon::iterator it = hole.begin();
    std::advance(it, position_left_concavity_point);
    concavity_points.push_back(it);
    it = hole.begin();
    std::advance(it, position_right_concavity_point);
    concavity_points.push_back(it);
  }
  return;
}

}
