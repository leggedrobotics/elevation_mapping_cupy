
#include "sliding_window_plane_extractor.hpp"

namespace sliding_window_plane_extractor{

  using namespace grid_map;

  SlidingWindowPlaneExtractor::SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution,
      const std::string& layer_height, SlidingWindowParameters& parameters)
      : map_(map),
        resolution_(resolution),
        elevation_layer_(layer_height),
        kernel_size_(parameters.kernel_size),
        plane_error_threshold_(parameters.plane_error_threshold){
    number_of_extracted_planes_ = 0;
  }

  SlidingWindowPlaneExtractor::SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution,
                                                           const std::string& layer_height)
      : map_(map),
        resolution_(resolution),
        elevation_layer_(layer_height) {
    kernel_size_ = 5;
    plane_error_threshold_ = 0.004;
    number_of_extracted_planes_ = 0;
  }


  SlidingWindowPlaneExtractor::~SlidingWindowPlaneExtractor() = default;

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
    for (int label_it = 1; label_it < number_of_extracted_planes_; ++label_it) {
      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::Mat binary_image(labeled_image_.size(), CV_8UC1);
      binary_image = labeled_image_ == label_it;
      findContours(binary_image, contours, hierarchy,
                   CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
      std::list<std::vector<cv::Point>> approx_contours;
      for (auto& contour : contours) {
        std::vector<cv::Point> approx_contour;
        cv::approxPolyDP(contour, approx_contour, 2, true);
        if(convex_plane_extraction::isContourSimple<std::vector<cv::Point>::reverse_iterator>(approx_contour.rbegin(), approx_contour.rend())) {
          ROS_ERROR("Polygon not simple!");
          // for (auto point : approx_contour)
            // std::cout << point << std::endl;
        }
        //CHECK(convex_plane_extraction::isContourSimple<std::vector<cv::Point>::iterator>(approx_contour.begin(), approx_contour.end()));
        approx_contours.push_back(approx_contour);
      }
    }
  }
}