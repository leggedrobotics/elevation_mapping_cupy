#include "plane.hpp"

namespace convex_plane_extraction{

  bool Plane::setNormalAndSupportVector(const Eigen::Vector3d& normal_vector, const Eigen::Vector3d& support_vector){
    normal_vector_ = normal_vector;
    support_vector_ = support_vector;
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


}
