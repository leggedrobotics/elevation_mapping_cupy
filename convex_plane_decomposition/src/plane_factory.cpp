//
// Created by andrej on 3/1/20.
//

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

void createPlanesFromLabeledImageAndPlaneParameters(const cv::Mat& labeled_image, const int number_of_labels,
          const std::map<int, convex_plane_extraction::PlaneParameters>& parameters){
  CHECK_GT(number_of_labels, 0);
  CHECK_EQ(number_of_labels, parameters.size());
  for (int label = 1; label < number_of_labels; ++label){
    cv::Mat binary_image(labeled_image.size(), CV_8UC1);
    binary_image = labeled_image == label;
    extractPolygonsFromBinaryImage(binary_image)
  }
}




for (int label_it = 1; label_it <= number_of_extracted_planes_; ++label_it) {
auto polygonizer_start = std::chrono::system_clock::now();
convex_plane_extraction::Plane plane;
std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;
cv::Mat binary_image(labeled_image_.size(), CV_8UC1);
binary_image = labeled_image_ == label_it;
constexpr int kUpSamplingFactor = 3;
cv::Mat binary_image_upsampled;
cv::resize(binary_image, binary_image_upsampled, cv::Size(kUpSamplingFactor*binary_image.size().height, kUpSamplingFactor*binary_image.size().width));
findContours(binary_image_upsampled, contours, hierarchy,
    CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
std::list<std::vector<cv::Point>> approx_contours;
int hierachy_it = 0;
for (auto& contour : contours) {
++hierachy_it;
std::vector<cv::Point> approx_contour;
if (contour.size() <= 10){
approx_contour = contour;
} else {
cv::approxPolyDP(contour, approx_contour, 9, true);
}
if (approx_contour.size() <= 2) {
LOG_IF(WARNING, hierarchy[hierachy_it-1][3] < 0 && contour.size() > 4) << "Removing parental polygon since too few vertices!";
continue;
}
convex_plane_extraction::CgalPolygon2d polygon = convex_plane_extraction::createCgalPolygonFromOpenCvPoints(approx_contour.begin(), approx_contour.end(), resolution_ / kUpSamplingFactor);

if(!polygon.is_simple()) {
convex_plane_extraction::Vector2i index;
index << (*polygon.begin()).x(), (*polygon.begin()).y();
LOG(WARNING) << "Polygon starting at " << index << " is not simple, will be ignored!";
continue;
}
constexpr int kParentFlagIndex = 3;
if (hierarchy[hierachy_it-1][kParentFlagIndex] < 0) {
//convex_plane_extraction::upSampleLongEdges(&polygon);
//          convex_plane_extraction::approximateContour(&polygon);
CHECK(plane.addOuterPolygon(polygon));
} else {
CHECK(plane.addHolePolygon(polygon));
}
}
if(plane.hasOuterContour()) {
//LOG(WARNING) << "Dropping plane, no outer contour detected!";
computePlaneFrameFromLabeledImage(binary_image, &plane);
auto polygonizer_end = std::chrono::system_clock::now();
auto polygonizer_duration = std::chrono::duration_cast<std::chrono::microseconds>(polygonizer_end - polygonizer_start);
polygonizer_file << time_stamp.count() << ", " << polygonizer_duration.count() << "\n";
if(plane.isValid()) {
LOG(INFO) << "Starting resolving holes...";
plane.resolveHoles();
LOG(INFO) << "done.";
auto decomposer_start = std::chrono::system_clock::now();
CHECK(plane.decomposePlaneInConvexPolygons());
auto decomposer_end = std::chrono::system_clock::now();
auto decomposer_duration = std::chrono::duration_cast<std::chrono::microseconds>(decomposer_end - decomposer_start);
decomposer_file << time_stamp.count() << ", " << decomposer_duration.count() << "\n";
planes_.push_back(plane);
} else {
LOG(WARNING) << "Dropping plane, normal vector could not be inferred!";
}
}
}
