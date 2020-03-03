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
    CHECK_NE(plane_parameter_it, plane_parameters.end()) << "Label not contained in plane parameter container!";
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
