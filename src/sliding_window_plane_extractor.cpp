
#include "sliding_window_plane_extractor.hpp"

namespace sliding_window_plane_extractor{

  using namespace grid_map;

  SlidingWindowPlaneExtractor::SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution,
      const std::string& layer_height, const std::string& normal_layer_prefix, SlidingWindowParameters& parameters)
      : map_(map),
        resolution_(resolution),
        elevation_layer_(layer_height),
        normal_layer_prefix_(normal_layer_prefix),
        kernel_size_(parameters.kernel_size),
        plane_error_threshold_(parameters.plane_error_threshold){
    number_of_extracted_planes_ = 0;
    computeMapTransformation();
  }

  SlidingWindowPlaneExtractor::SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution,
                         const std::string& normal_layer_prefix, const std::string& layer_height)
      : map_(map),
        resolution_(resolution),
        elevation_layer_(layer_height),
        normal_layer_prefix_(normal_layer_prefix){
    kernel_size_ = 5;
    plane_error_threshold_ = 0.004;
    number_of_extracted_planes_ = 0;
    computeMapTransformation();
  }


  SlidingWindowPlaneExtractor::~SlidingWindowPlaneExtractor() = default;

  void SlidingWindowPlaneExtractor::computeMapTransformation(){
    Eigen::Vector2i map_size = map_.getSize();
    CHECK(map_.getPosition(Eigen::Vector2i::Zero(), map_offset_));
    Eigen::Vector2d lower_left_cell_position;
    CHECK(map_.getPosition(Eigen::Vector2i(map_size.x() - 1, 0), lower_left_cell_position));
    Eigen::Vector2d upper_right_cell_position;
    CHECK(map_.getPosition(Eigen::Vector2i(0, map_size.y() - 1), upper_right_cell_position));
    transformation_xy_to_world_frame_.col(0) = (lower_left_cell_position - map_offset_).normalized();
    transformation_xy_to_world_frame_.col(1) = (upper_right_cell_position - map_offset_).normalized();
  }

  void SlidingWindowPlaneExtractor::runDetection(){
    std::cout << "Starting detection!" << std::endl;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> binary_map =
        Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic>::Zero(map_.getSize().x(), map_.getSize().y());
    CHECK(map_.getSize().x() >= kernel_size_);
    CHECK(map_.getSize().y() >= kernel_size_);
    SlidingWindowIterator window_iterator(map_, elevation_layer_, SlidingWindowIterator::EdgeHandling::INSIDE,
        kernel_size_);
    for (; !window_iterator.isPastEnd(); ++window_iterator) {
      Eigen::MatrixXf window_data = window_iterator.getData();
      int instance_iterator = 0;
      std::vector<double> height_instances;
      std::vector<double> row_position;
      std::vector<double> col_position;
      for (int kernel_col = 0; kernel_col < kernel_size_; ++kernel_col) {
        for (int kernel_row = 0; kernel_row < kernel_size_; ++kernel_row) {
          if (!isfinite(window_data(kernel_row, kernel_col))) {
            continue;
          }
          height_instances.push_back(window_data(kernel_row, kernel_col));
          row_position.push_back(kernel_row - static_cast<int>(kernel_size_ / 2));
          col_position.push_back(kernel_col - static_cast<int>(kernel_size_ / 2));
        }
      }
      constexpr int kMinNumberOfDataPoints = 9;
      if (height_instances.size() < kMinNumberOfDataPoints) {
        continue;
      }
      CHECK(height_instances.size() == col_position.size());
      CHECK(height_instances.size() == row_position.size());
      // Center height around mean.
      double mean_height = std::accumulate(height_instances.begin(), height_instances.end(), 0.0);
      mean_height = mean_height / static_cast<double>(height_instances.size());
      for(double& element : height_instances) {
        element -= mean_height;
      }
      // Collect centered data points into matrix.
      Eigen::MatrixXd data_points(height_instances.size(), 3);
      data_points.col(0) = Eigen::Map<Eigen::VectorXd>(&row_position.front(), row_position.size()) * resolution_;
      data_points.col(1) = Eigen::Map<Eigen::VectorXd>(&col_position.front(), col_position.size()) * resolution_;
      data_points.col(2) = Eigen::Map<Eigen::VectorXd>(&height_instances.front(), height_instances.size());
      Eigen::BDCSVD<Eigen::MatrixXd> svd = data_points.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
      CHECK(svd.rank() >= 3);
      Eigen::Matrix3d V = svd.matrixV();
      Eigen::Vector3d n = (V.col(0)).cross(V.col(1));
      Index index = *window_iterator;
      double mean_error = ((data_points * n).cwiseAbs()).sum() / height_instances.size();
      if (mean_error < plane_error_threshold_) {
        binary_map((*window_iterator).x(), (*window_iterator).y()) = true;
      }
    }
    cv::Mat binary_image(binary_map.rows(), binary_map.cols(), CV_8U, binary_map.data());
    cv::eigen2cv(binary_map, binary_image);
    number_of_extracted_planes_ = cv::connectedComponents(binary_image, labeled_image_, 8, CV_32SC1);
  }

  void SlidingWindowPlaneExtractor::setParameters(const SlidingWindowParameters& parameters){
    kernel_size_ = parameters.kernel_size;
    plane_error_threshold_ = parameters.plane_error_threshold;
  }

  void SlidingWindowPlaneExtractor::slidingWindowPlaneVisualization(){
    Eigen::MatrixXf new_layer = Eigen::MatrixXf::Zero(map_.getSize().x(), map_.getSize().y());
    cv::cv2eigen(labeled_image_, new_layer);
    map_.add("sliding_window_planes", new_layer);
    std::cout << "Added ransac plane layer!" << std::endl;
  }

  void SlidingWindowPlaneExtractor::generatePlanes(){
    for (int label_it = 1; label_it <= number_of_extracted_planes_; ++label_it) {
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
          cv::approxPolyDP(contour, approx_contour, 6, true);
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
          CHECK(plane.addOuterPolygon(polygon));
        } else {
          CHECK(plane.addHolePolygon(polygon));
        }
      }
      if(plane.hasOuterContour()) {
        //LOG(WARNING) << "Dropping plane, no outer contour detected!";
        computePlaneFrameFromLabeledImage(binary_image, &plane);
        if(plane.isValid()) {
          CHECK(plane.decomposePlaneInConvexPolygons());
          planes_.push_back(plane);
        } else {
          LOG(WARNING) << "Dropping plane, normal vector could not be inferred!";
        }
      }
    }
  }

  void SlidingWindowPlaneExtractor::computePlaneFrameFromLabeledImage(const cv::Mat& binary_image,
      convex_plane_extraction::Plane* plane){
    CHECK_NOTNULL(plane);
    Eigen::Vector3d normal_vector = Eigen::Vector3d::Zero();
    Eigen::Vector3d support_vector = Eigen::Vector3d::Zero();
    int number_of_normal_instances = 0;
    int number_of_position_instances = 0;
    for (int row = 0; row < binary_image.rows; ++row){
      for (int col = 0; col < binary_image.cols; ++col){
        if (binary_image.at<bool>(row, col)) {
          Eigen::Vector2i index;
          index << row, col;
          Eigen::Vector3d normal_vector_temp;
          Eigen::Vector3d support_vector_temp;
          if(map_.getVector(normal_layer_prefix_, index, normal_vector_temp)){
            normal_vector += normal_vector_temp;
            ++number_of_normal_instances;
          }
          if (map_.getPosition3(elevation_layer_, index, support_vector_temp)) {
            support_vector += support_vector_temp;
            ++number_of_position_instances;
          }
        }
      }
    }
    if (number_of_normal_instances == 0 || number_of_position_instances == 0){
      return;
    }
    normal_vector = normal_vector / static_cast<double>(number_of_normal_instances);
    support_vector = support_vector / static_cast<double>(number_of_position_instances);
    CHECK(plane->setNormalAndSupportVector(normal_vector, support_vector));
  }

  void SlidingWindowPlaneExtractor::visualizeConvexDecomposition(jsk_recognition_msgs::PolygonArray* ros_polygon_array){
    CHECK_NOTNULL(ros_polygon_array);
    if (planes_.empty()){
      LOG(INFO) << "No convex polygons to visualize!";
      return;
    }
    for (const auto& plane : planes_){
      convex_plane_extraction::Polygon3dVectorContainer polygon_container;
      if(!plane.convertConvexPolygonsToWorldFrame(&polygon_container, transformation_xy_to_world_frame_, map_offset_)){
        continue;
      }
      convex_plane_extraction::addRosPolygons(polygon_container, ros_polygon_array);
    }
  }
}