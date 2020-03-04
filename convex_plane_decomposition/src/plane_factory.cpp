#include "plane_factory.hpp"

using namespace convex_plane_extraction;

void PlaneFactory::computeMapTransformation(){
  Eigen::Vector2i map_size = map_.getSize();
  CHECK(map_.getPosition(Eigen::Vector2i::Zero(), map_offset_));
  Eigen::Vector2d lower_left_cell_position;
  CHECK(map_.getPosition(Eigen::Vector2i(map_size.x() - 1, 0), lower_left_cell_position));
  Eigen::Vector2d upper_right_cell_position;
  CHECK(map_.getPosition(Eigen::Vector2i(0, map_size.y() - 1), upper_right_cell_position));
  transformation_xy_to_world_frame_.col(0) = (lower_left_cell_position - map_offset_).normalized();
  transformation_xy_to_world_frame_.col(1) = (upper_right_cell_position - map_offset_).normalized();
}

void PlaneFactory::createPlanesFromLabeledImageAndPlaneParameters(const cv::Mat& labeled_image, const int number_of_labels,
          const std::map<int, convex_plane_extraction::PlaneParameters>& plane_parameters){
  CHECK_GT(number_of_labels, 0);
  CHECK_EQ(number_of_labels, plane_parameters.size());
  Polygonizer polygonizer(parameters_.polygonizer_parameters);
  for (int label = 1; label < number_of_labels; ++label){
    cv::Mat binary_image(labeled_image.size(), CV_8UC1);
    binary_image = labeled_image == label;
    const CgalPolygon2d plane_contour = polygonizer.runPolygonizationOnBinaryImage(binary_image);
    const auto plane_parameter_it = plane_parameters.find(label);
    CHECK(plane_parameter_it != plane_parameters.end()) << "Label not contained in plane parameter container!";
    if (isPlaneInclinationBelowThreshold(plane_parameter_it->second.normal_vector)){
      planes_.emplace_back(plane_contour, plane_parameter_it->second);
    } else {
      LOG(WARNING) << "Dropping plane due to exceeded inclination threshold!";
    }
  }
}

bool PlaneFactory::isPlaneInclinationBelowThreshold(const Eigen::Vector3d& plane_normal_vector) const {
  const Eigen::Vector3d z_axis(0,0,1);
  return std::atan2(1.0, z_axis.dot(plane_normal_vector)) < (parameters_.plane_inclination_threshold_degrees / 180.0 * M_PI);
}

void PlaneFactory::decomposePlanesInConvexPolygons(){
  ConvexDecomposer convex_decomposer(parameters_.convex_decomposer_parameters);
  for(Plane& plane : planes_){
    CgalPolygon2dContainer& convex_polygons = plane.getConvexPolygonsMutable();
    convex_polygons = convex_decomposer.performConvexDecomposition(plane.getOuterPolygon());
  }
}

Polygon3dVectorContainer PlaneFactory::getConvexPolygonsInWorldFrame() const{
  Polygon3dVectorContainer polygon_buffer;
  for(const Plane& plane : planes_){
    Polygon3dVectorContainer temp_polygon_buffer = convertPlanePolygonsToWorldFrame(plane.getConvexPolygons(), plane.getPlaneParameters());
    std::move(temp_polygon_buffer.begin(), temp_polygon_buffer.end(), std::back_inserter(polygon_buffer));
  }
  return polygon_buffer;
}

Polygon3dVectorContainer PlaneFactory::getPlaneContoursInWorldFrame() const{
  Polygon3dVectorContainer polygon_buffer;
  for(const Plane& plane : planes_){
    Polygon3dVectorContainer temp_polygon_buffer = convertPlanePolygonToWorldFrame(plane.getOuterPolygon(), plane.getPlaneParameters());
    std::move(temp_polygon_buffer.begin(), temp_polygon_buffer.end(), std::back_inserter(polygon_buffer));
  }
  return polygon_buffer;
}

Polygon3dVectorContainer PlaneFactory::convertPlanePolygonToWorldFrame(const CgalPolygon2d& polygon, const PlaneParameters& plane_parameters) const{
  Polygon3dVectorContainer output_polygons;
  CHECK(!polygon.is_empty());
  Polygon3d polygon_temp;
  for (const auto& point : polygon){
    polygon_temp.push_back(convertPlanePointToWorldFrame(point, plane_parameters));
  }
  output_polygons.push_back(polygon_temp);
  return output_polygons;
}

Polygon3dVectorContainer PlaneFactory::convertPlanePolygonsToWorldFrame(const CgalPolygon2dContainer& polygons, const PlaneParameters& plane_parameters) const{
  CHECK(!polygons.empty()) << "Input polygon container empty!";
  Polygon3dVectorContainer output_polygons;
  for (const auto& polygon : polygons){
    CHECK(!polygon.is_empty());
    Polygon3d polygon_temp;
    for (const auto& point : polygon){
      polygon_temp.push_back(convertPlanePointToWorldFrame(point, plane_parameters));
    }
    output_polygons.push_back(polygon_temp);
  }
  return output_polygons;
}

Eigen::Vector3d PlaneFactory::convertPlanePointToWorldFrame(const CgalPoint2d& point, const PlaneParameters& plane_parameters) const{
  Eigen::Vector2d temp_point(point.x(), point.y());
  temp_point = transformation_xy_to_world_frame_ * temp_point;
  temp_point = temp_point + map_offset_;
  LOG_IF(FATAL, !std::isfinite(temp_point.x())) << "Not finite x value!";
  LOG_IF(FATAL, !std::isfinite(temp_point.y())) << "Not finite y value!";
  const double z = (-(temp_point.x() - plane_parameters.support_vector.x())*plane_parameters.normal_vector.x() -
      (temp_point.y() - plane_parameters.support_vector.y())* plane_parameters.normal_vector.y()) /
          plane_parameters.normal_vector(2) + plane_parameters.support_vector(2);
  LOG_IF(FATAL, !std::isfinite(z)) << "Not finite z value!";
  return Vector3d(temp_point.x(), temp_point.y(), z);
}

