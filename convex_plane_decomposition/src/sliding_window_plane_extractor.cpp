
#include "sliding_window_plane_extractor.hpp"

namespace sliding_window_plane_extractor{

  using namespace grid_map;

  SlidingWindowPlaneExtractor::SlidingWindowPlaneExtractor(grid_map::GridMap &map, double resolution,
      const std::string& layer_height, const std::string& normal_layer_prefix,
      const SlidingWindowPlaneExtractorParameters& parameters,
      const ransac_plane_extractor::RansacPlaneExtractorParameters& ransac_parameters)
      : map_(map),
        resolution_(resolution),
        elevation_layer_(layer_height),
        normal_layer_prefix_(normal_layer_prefix),
        parameters_(parameters),
        ransac_parameters_(parameters.ransac_parameters){
    number_of_extracted_planes_ = -1;
    const grid_map::Size map_size = map.getSize();
    binary_image_patch_ = cv::Mat(map_size(0), map_size(1), CV_8U, false);
    binary_image_angle_ = cv::Mat(map_size(0), map_size(1), CV_8U, false);
  }

  void SlidingWindowPlaneExtractor::runSlidingWindowDetector(){
    const grid_map::Size map_size = map_.getSize();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> normal_x(map_size(0), map_size(1));
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> normal_y(map_size(0), map_size(1));
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> normal_z(map_size(0), map_size(1));
    CHECK(map_.getSize().x() >= parameters_.kernel_size);
    CHECK(map_.getSize().y() >= parameters_.kernel_size);
    SlidingWindowIterator window_iterator(map_, elevation_layer_, SlidingWindowIterator::EdgeHandling::INSIDE,
        parameters_.kernel_size);
    for (; !window_iterator.isPastEnd(); ++window_iterator) {
      Eigen::MatrixXf window_data = window_iterator.getData();
      int instance_iterator = 0;
      std::vector<double> height_instances;
      std::vector<double> row_position;
      std::vector<double> col_position;
      for (int kernel_col = 0; kernel_col < parameters_.kernel_size; ++kernel_col) {
        for (int kernel_row = 0; kernel_row < parameters_.kernel_size; ++kernel_row) {
          if (!isfinite(window_data(kernel_row, kernel_col))) {
            continue;
          }
          height_instances.push_back(window_data(kernel_row, kernel_col));
          row_position.push_back(kernel_row - static_cast<int>(parameters_.kernel_size / 2));
          col_position.push_back(kernel_col - static_cast<int>(parameters_.kernel_size / 2));
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
      const Eigen::Matrix3d covarianceMatrix(data_points.transpose() * data_points);
      Vector3 eigenvalues = Vector3::Ones();
      Eigen::Matrix3d eigenvectors = Eigen::Matrix3d::Identity();
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covarianceMatrix);
      eigenvalues = solver.eigenvalues().real();
      eigenvectors = solver.eigenvectors().real();
      int smallestId(0);
      double smallestValue(std::numeric_limits<double>::max());
      for (int j = 0; j < eigenvectors.cols(); j++) {
        if (eigenvalues(j) < smallestValue) {
          smallestId = j;
          smallestValue = eigenvalues(j);
        }
      }
      Vector3 eigenvector = eigenvectors.col(smallestId);
      const Eigen::Vector3d normalVectorPositiveAxis_(0,0,1);
      if (eigenvector.dot(normalVectorPositiveAxis_) < 0.0) eigenvector = -eigenvector;
      Eigen::Vector3d n = eigenvector.normalized();
      Index index = *window_iterator;
      double mean_error = ((data_points * n).cwiseAbs()).sum() / height_instances.size();
      Eigen::Vector3d upwards(0,0,1);
      constexpr double kInclinationThreshold = 0.35;
      if (mean_error < parameters_.plane_patch_error_threshold && abs(n.transpose()*upwards) > kInclinationThreshold) {
        binary_image_patch_.at<bool>((*window_iterator).x(), (*window_iterator).y()) = true;
        normal_x(index.x(), index.y()) = n.x();
        normal_y(index.x(), index.y()) = n.y();
        normal_z(index.x(), index.y()) = n.z();
      }
    }
    map_.add("normals_x", normal_x);
    map_.add("normals_y", normal_y);
    map_.add("normals_z", normal_z);
  }

  void SlidingWindowPlaneExtractor::runSurfaceNormalCurvatureDetection(){
    CHECK(map_.exists("normals_x"));
    CHECK(map_.exists("normals_y"));
    CHECK(map_.exists("normals_z"));
    const int surface_normal_map_boundary_offset = parameters_.kernel_size / 2;
    CHECK_GT(surface_normal_map_boundary_offset, 0);
    const grid_map::Size map_rows_cols = map_.getSize();
    for( int cols = surface_normal_map_boundary_offset; cols < map_rows_cols(1) - surface_normal_map_boundary_offset - 1; ++cols){
      for (int rows = surface_normal_map_boundary_offset; rows < map_rows_cols(0) - surface_normal_map_boundary_offset - 1; ++rows){
        const Eigen::Vector2f normal_vector_center(map_.at("normals_x", Index(rows, cols)), map_.at("normals_y", Index(rows, cols)),
            map_.at("normals_z", Index(rows, cols)));
        const Eigen::Vector2f normal_vector_next_row(map_.at("normals_x", Index(rows+1, cols)), map_.at("normals_y", Index(rows+1, cols)),
            map_.at("normals_z", Index(rows+1, cols)));
        const Eigen::Vector2f normal_vector_next_col(map_.at("normals_x", Index(rows, cols+1)), map_.at("normals_y", Index(rows, cols+1)),
            map_.at("normals_z", Index(rows, cols+1)));
        const float angle_in_col_direction_radians = std::atan2(1.0, normal_vector_center.dot(normal_vector_next_col));
        const float angle_in_row_direction_radians = std::atan2(1.0, normal_vector_center.dot(normal_vector_next_row));
        const double gradient_magnitude_normalized = sqrt((angle_in_col_direction_radians*angle_in_col_direction_radians) +
            (angle_in_row_direction_radians * angle_in_row_direction_radians)) / (sqrt(2.0)*M_PI);
        binary_image_angle_.at<bool>(rows, cols) = gradient_magnitude_normalized <= parameters_.surface_normal_angle_threshold_degrees;
      }
    }
  }

  // Label cells according to which cell they belong to using connected component labeling.
  void SlidingWindowPlaneExtractor::runSegmentation() {
    CHECK_EQ(binary_image_patch_.type(), CV_8U);
    const grid_map::Size map_size = map_.getSize();
    CHECK_EQ(binary_image_patch_.rows, map_size(0));
    CHECK_EQ(binary_image_patch_.cols, map_size(1));
    if (parameters_.include_curvature_detection){
      CHECK_EQ(binary_image_patch_.type(), binary_image_angle_.type());
      CHECK_EQ(binary_image_patch_.size, binary_image_angle_.size);
      number_of_extracted_planes_ = cv::connectedComponents(binary_image_patch_ & binary_image_angle_, labeled_image_, 8, CV_32SC1);
    } else {
      number_of_extracted_planes_ = cv::connectedComponents(binary_image_patch_, labeled_image_, 8, CV_32SC1);
    }
  }

  // Refine connected component using RANSAC. Input vector is modified by function!
  const auto&  SlidingWindowPlaneExtractor::runRansacRefinement(std::vector<ransac_plane_extractor::PointWithNormal>& points_with_normal) const {
    CHECK(!points_with_normal.empty());
    ransac_plane_extractor::RansacPlaneExtractor ransac_plane_extractor(points_with_normal, ransac_parameters_);
    ransac_plane_extractor.runDetection();
    return ransac_plane_extractor.getDetectedPlanes();
  }

  void SlidingWindowPlaneExtractor::setParameters(const SlidingWindowPlaneExtractorParameters& parameters){
    parameters_.kernel_size = parameters.kernel_size;
    parameters_.plane_patch_error_threshold = parameters.plane_patch_error_threshold;
  }

  /*
  void SlidingWindowPlaneExtractor::slidingWindowPlaneVisualization(){
    Eigen::MatrixXf new_layer = Eigen::MatrixXf::Zero(map_.getSize().x(), map_.getSize().y());
    cv::cv2eigen(labeled_image_, new_layer);
    map_.add("sliding_window_planes", new_layer);
    std::cout << "Added ransac plane layer!" << std::endl;
  }
*/
  void SlidingWindowPlaneExtractor::extractPlaneParametersFromLabeledImage(){
    if (number_of_extracted_planes_ < 1){
      LOG(WARNING) << "No planes detected by Sliding Window Plane Extractor!";
      return;
    }
    const int number_of_extracted_planes_without_refinement = number_of_extracted_planes_;
    for (int label = 1; label <= number_of_extracted_planes_without_refinement; ++label) {
      computePlaneParametersForLabel(label);
    }
  }

  void SlidingWindowPlaneExtractor::computePlaneParametersForLabel(int label){
    CHECK(map_.exists(elevation_layer_));
    CHECK(map_.exists("normals_x"));
    CHECK(map_.exists("normals_y"));
    CHECK(map_.exists("normals_z"));
    Eigen::Vector3d normal_vector = Eigen::Vector3d::Zero();
    Eigen::Vector3d support_vector = Eigen::Vector3d::Zero();
    std::vector<ransac_plane_extractor::PointWithNormal> points_with_normal;
    cv::Mat binary_image(labeled_image_.size(), CV_8UC1);
    binary_image = labeled_image_ == label;
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
          ransac_plane_extractor::PointWithNormal point_with_normal = std::make_pair<ransac_plane_extractor::Point3D,
          ransac_plane_extractor::Vector3D>(ransac_plane_extractor::Point3D(support_vector_temp.x(), support_vector_temp.y(), support_vector_temp.z()),
              ransac_plane_extractor::Vector3D(normal_vector_temp.x(), normal_vector_temp.y(), normal_vector_temp.z()));
          points_with_normal.push_back(point_with_normal);
        }
      }
    }
    if (number_of_normal_instances == 0 || number_of_position_instances == 0){
      LOG(WARNING) << "Label empty, NO plane parameters are created!";
      return;
    }
    normal_vector = normal_vector / static_cast<double>(number_of_normal_instances);
    support_vector = support_vector / static_cast<double>(number_of_position_instances);
    bool refinement_performed = false;
    if (parameters_.include_ransac_refinement) {
      // Compute error to fitted plane.
      if (computeAverageErrorToPlane(normal_vector, support_vector, points_with_normal)
          > parameters_.global_plane_fit_error_threshold){
        const auto& planes = runRansacRefinement(points_with_normal);
        CHECK(!planes.empty()) << "No planes detected by RANSAC in as planar classified region.";
        int label_counter = 0;
        for (const auto& plane : planes){
          const std::vector<std::size_t>& plane_point_indices = (*plane.get()).indices_of_assigned_points();
          CHECK(!plane_point_indices.empty());
          support_vector = Eigen::Vector3d::Zero();
          normal_vector = Eigen::Vector3d::Zero();
          for (const auto index : plane_point_indices) {
            const auto &point = points_with_normal.at(index).first;
            const auto &normal = points_with_normal.at(index).second;
            support_vector += Eigen::Vector3d(point.x(), point.y(), point.z());
            normal_vector += Eigen::Vector3d(normal.x(), normal.y(), normal.z());
            Eigen::Array2i map_indices;
            map_.getIndex(Eigen::Vector2d(point.x(), point.y()), map_indices);
            if (label_counter == 0){
              labeled_image_.at<int>(map_indices(0), map_indices(1)) = label;
            } else {
              labeled_image_.at<int>(map_indices(0), map_indices(1)) = number_of_extracted_planes_ + label_counter;
            }
          }
          support_vector /= static_cast<double>(plane_point_indices.size());
          normal_vector /= static_cast<double>(plane_point_indices.size());
          const convex_plane_extraction::PlaneParameters temp_plane_parameters(normal_vector, support_vector);
          if (label_counter == 0) {
            label_plane_parameters_map_.emplace(label, temp_plane_parameters);
          } else {
            label_plane_parameters_map_.emplace(number_of_extracted_planes_ + label_counter, temp_plane_parameters);
          }
          ++label_counter;
        }
        CHECK_EQ(label_counter + 1, planes.size());
        number_of_extracted_planes_ += label_counter;
        refinement_performed = true;
      }
    }
    if (refinement_performed){
      const convex_plane_extraction::PlaneParameters temp_plane_parameters(normal_vector, support_vector);
      label_plane_parameters_map_.emplace(label, temp_plane_parameters);
    }
  }

  double SlidingWindowPlaneExtractor::computeAverageErrorToPlane(const Eigen::Vector3d& normal_vector, const Eigen::Vector3d& support_vector,
      const std::vector<ransac_plane_extractor::PointWithNormal>& points_with_normal) const {
    CHECK(!points_with_normal.empty());
    double average_error = 0.0;
    for (const auto& point_with_normal : points_with_normal){
      Eigen::Vector3d p_S_P(point_with_normal.first.x() - support_vector.x(), point_with_normal.first.y() - support_vector.y(),
          point_with_normal.first.z() - support_vector.z());
      average_error += abs(normal_vector.dot(p_S_P));
    }
    average_error /= static_cast<double>(points_with_normal.size());
    return average_error;
  }

  void SlidingWindowPlaneExtractor::runExtraction(){
    VLOG(1) << "Started sliding window plane detector ...";
    runSlidingWindowDetector();
    VLOG(1) << "done.";
    if (parameters_.include_curvature_detection){
      VLOG(1) << "Starting surface normal curvature detection ...";
      runSurfaceNormalCurvatureDetection();
      VLOG(1) << "done.";
    }
    VLOG(1) << "Starting segmentation ...";
    runSegmentation();
    VLOG(1) << "done.";
    VLOG(1) << "Extracting plane parameters from labeled image ...";
    extractPlaneParametersFromLabeledImage();
    VLOG(1) << "done.";
  }

  /*
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

  void SlidingWindowPlaneExtractor::visualizePlaneContours(jsk_recognition_msgs::PolygonArray* outer_polygons, jsk_recognition_msgs::PolygonArray* hole_poylgons) const {
    CHECK_NOTNULL(outer_polygons);
    CHECK_NOTNULL(hole_poylgons);
    if (planes_.empty()){
      LOG(INFO) << "No convex polygons to visualize!";
      return;
    }
    for (const auto& plane : planes_){
      convex_plane_extraction::Polygon3dVectorContainer outer_contour;
      if(plane.convertOuterPolygonToWorldFrame(&outer_contour, transformation_xy_to_world_frame_, map_offset_)){
        convex_plane_extraction::addRosPolygons(outer_contour, outer_polygons);
      }
      convex_plane_extraction::Polygon3dVectorContainer hole_contours;
      if (plane.convertHolePolygonsToWorldFrame(&hole_contours, transformation_xy_to_world_frame_, map_offset_)){
        convex_plane_extraction::addRosPolygons(hole_contours, hole_poylgons);
      }
    }
  }
*/
}

