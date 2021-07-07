#include "convex_plane_decomposition/sliding_window_plane_extraction/SlidingWindowPlaneExtractor.h"

#include <chrono>
#include <cmath>
#include <iostream>

#include <Eigen/Eigenvalues>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <grid_map_core/grid_map_core.hpp>

namespace convex_plane_decomposition {
namespace sliding_window_plane_extractor {

double angleBetweenVectorsInDegrees(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
  double cos_rad = v1.normalized().dot(v2.normalized());
  if (cos_rad < -1.0) {
    cos_rad = -1.0;
  } else if (cos_rad > 1.0) {
    cos_rad = 1.0;
  }
  return std::abs(std::acos(cos_rad) * 180.0 / M_PI);
}

SlidingWindowPlaneExtractor::SlidingWindowPlaneExtractor(const SlidingWindowPlaneExtractorParameters& parameters,
                                                         const ransac_plane_extractor::RansacPlaneExtractorParameters& ransacParameters)
    : parameters_(parameters), ransacParameters_(ransacParameters) {}

void SlidingWindowPlaneExtractor::runExtraction(const grid_map::GridMap& map, const std::string& layer_height) {
  // Extract basic map information
  map_ = &map;
  elevationLayer_ = layer_height;
  segmentedPlanesMap_.resolution = map_->getResolution();
  map_->getPosition(Eigen::Vector2i(0, 0), segmentedPlanesMap_.mapOrigin);

  // Initialize based on map size.
  segmentedPlanesMap_.highestLabel = -1;
  segmentedPlanesMap_.labelPlaneParameters.clear();
  const auto& mapSize = map_->getSize();
  binaryImagePatch_ = cv::Mat(mapSize(0), mapSize(1), CV_8U, 0.0);  // Zero initialize to set untouched pixels to not planar;
  // Need a buffer of at least the linear size of the image. But no need to shrink if the buffer is already bigger.
  const int linearMapSize = mapSize(0) * mapSize(1);
  if (surfaceNormals_.size() < linearMapSize) {
    surfaceNormals_.reserve(linearMapSize);
    std::fill_n(surfaceNormals_.begin(), linearMapSize, Eigen::Vector3d::Zero());
  }

  // Run
  runSlidingWindowDetector();
  runSegmentation();
  extractPlaneParametersFromLabeledImage();
}

std::pair<Eigen::Vector3d, double> SlidingWindowPlaneExtractor::computeNormalAndErrorForWindow(const Eigen::MatrixXf& windowData) const {
  // Gather surrounding data.
  size_t nPoints = 0;
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  Eigen::Matrix3d sumSquared = Eigen::Matrix3d::Zero();
  for (int kernel_col = 0; kernel_col < parameters_.kernel_size; ++kernel_col) {
    for (int kernel_row = 0; kernel_row < parameters_.kernel_size; ++kernel_row) {
      float height = windowData(kernel_row, kernel_col);
      if (!std::isfinite(height)) {
        continue;
      }
      // No need to account for map offset. Will substract the mean anyway.
      Eigen::Vector3d point{-kernel_row * segmentedPlanesMap_.resolution, -kernel_col * segmentedPlanesMap_.resolution, height};
      nPoints++;
      sum += point;
      sumSquared.noalias() += point * point.transpose();
    }
  }

  if (nPoints < 3) {
    // Not enough points to establish normal direction
    return {Eigen::Vector3d::UnitZ(), 1e30};
  } else {
    const Eigen::Vector3d mean = sum / nPoints;
    const Eigen::Matrix3d covarianceMatrix = sumSquared / nPoints - mean * mean.transpose();

    // Compute Eigenvectors.
    // Eigenvalues are ordered small to large.
    // Worst case bound for zero eigenvalue from : https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    solver.computeDirect(covarianceMatrix, Eigen::DecompositionOptions::ComputeEigenvectors);
    if (solver.eigenvalues()(1) > 1e-8) {
      Eigen::Vector3d unitaryNormalVector = solver.eigenvectors().col(0);

      // Check direction of the normal vector and flip the sign upwards
      if (unitaryNormalVector.z() < 0.0) {
        unitaryNormalVector = -unitaryNormalVector;
      }
      // The first eigenvalue might become slightly negative due to numerics.
      double rmsError = (solver.eigenvalues()(0) > 0.0) ? std::sqrt(solver.eigenvalues()(0)) : 0.0;
      return {unitaryNormalVector, rmsError};
    } else {  // If second eigenvalue is zero, the normal is not defined.
      return {Eigen::Vector3d::UnitZ(), 1e30};
    }
  }
}

bool SlidingWindowPlaneExtractor::isLocallyPlanar(const Eigen::Vector3d& localNormal, double meanError) const {
  return (meanError < parameters_.plane_patch_error_threshold &&
          angleBetweenVectorsInDegrees(localNormal, Eigen::Vector3d::UnitZ()) < parameters_.plane_inclination_threshold_degrees);
}

void SlidingWindowPlaneExtractor::runSlidingWindowDetector() {
  grid_map::SlidingWindowIterator window_iterator(*map_, elevationLayer_, grid_map::SlidingWindowIterator::EdgeHandling::INSIDE,
                                                  parameters_.kernel_size);
  const int kernelMiddle = (parameters_.kernel_size - 1) / 2;

  for (; !window_iterator.isPastEnd(); ++window_iterator) {
    grid_map::Index index = *window_iterator;
    Eigen::MatrixXf window_data = window_iterator.getData();
    const auto middleValue = window_data(kernelMiddle, kernelMiddle);

    if (!std::isfinite(middleValue)) {
      binaryImagePatch_.at<bool>(index.x(), index.y()) = false;
    } else {
      Eigen::Vector3d n;
      double mean_error;
      std::tie(n, mean_error) = computeNormalAndErrorForWindow(window_data);

      surfaceNormals_[getLinearIndex(index.x(), index.y())] = n;
      binaryImagePatch_.at<bool>(index.x(), index.y()) = isLocallyPlanar(n, mean_error);
    }
  }

  // erode
  if (parameters_.planarity_erosion > 0) {
    const int erosionSize = 2 * parameters_.planarity_erosion + 1;
    const int erosionType = cv::MORPH_CROSS;
    const auto erosionKernel_ = cv::getStructuringElement(erosionType, cv::Size(erosionSize, erosionSize));
    cv::erode(binaryImagePatch_, binaryImagePatch_, erosionKernel_);
  }
}

// Label cells according to which cell they belong to using connected component labeling.
void SlidingWindowPlaneExtractor::runSegmentation() {
  int numberOfLabel = cv::connectedComponents(binaryImagePatch_, segmentedPlanesMap_.labeledImage, parameters_.connectivity, CV_32S);
  segmentedPlanesMap_.highestLabel = numberOfLabel - 1;  // Labels are [0, N-1]
}

void SlidingWindowPlaneExtractor::extractPlaneParametersFromLabeledImage() {
  const int numberOfExtractedPlanesWithoutRefinement =
      segmentedPlanesMap_.highestLabel;  // Make local copy. The highestLabel is incremented inside the loop

  // Skip label 0. This is the background, i.e. non-planar region.
  for (int label = 1; label <= numberOfExtractedPlanesWithoutRefinement; ++label) {
    computePlaneParametersForLabel(label);
  }
}

void SlidingWindowPlaneExtractor::computePlaneParametersForLabel(int label) {
  const auto& elevationData = (*map_)[elevationLayer_];

  std::vector<ransac_plane_extractor::PointWithNormal> points_with_normal;
  int approximate_num_points =
      segmentedPlanesMap_.labeledImage.rows * segmentedPlanesMap_.labeledImage.cols / segmentedPlanesMap_.highestLabel;
  points_with_normal.reserve(approximate_num_points);

  int number_of_normal_instances = 0;
  int number_of_position_instances = 0;
  Eigen::Vector3d normal_vector_sum = Eigen::Vector3d::Zero();
  Eigen::Vector3d support_vector_sum = Eigen::Vector3d::Zero();
  for (int row = 0; row < segmentedPlanesMap_.labeledImage.rows; ++row) {
    for (int col = 0; col < segmentedPlanesMap_.labeledImage.cols; ++col) {
      if (segmentedPlanesMap_.labeledImage.at<int>(row, col) == label) {
        double height = elevationData(row, col);
        if (std::isfinite(height)) {
          Eigen::Vector2i index{row, col};

          Eigen::Vector3d normal_vector_temp = surfaceNormals_[getLinearIndex(index.x(), index.y())];
          normal_vector_sum += normal_vector_temp;
          ++number_of_normal_instances;

          Eigen::Vector3d point3d{segmentedPlanesMap_.mapOrigin.x() - row * segmentedPlanesMap_.resolution,
                                  segmentedPlanesMap_.mapOrigin.y() - col * segmentedPlanesMap_.resolution, height};
          support_vector_sum += point3d;
          ++number_of_position_instances;

          points_with_normal.emplace_back(
              ransac_plane_extractor::Point3D(point3d.x(), point3d.y(), point3d.z()),
              ransac_plane_extractor::Vector3D(normal_vector_temp.x(), normal_vector_temp.y(), normal_vector_temp.z()));
        }
      }
    }
  }
  if (number_of_position_instances < parameters_.min_number_points_per_label) {
    // Label has too little points, no plane parameters are created
    return;
  }
  Eigen::Vector3d normal_vector_avg = normal_vector_sum / number_of_normal_instances;
  normal_vector_avg.normalize();  // Averaging does not preserve norm
  Eigen::Vector3d support_vector_avg = support_vector_sum / number_of_position_instances;

  // Compute error to fitted plane.
  if (parameters_.include_ransac_refinement && !isGloballyPlanar(normal_vector_avg, support_vector_avg, points_with_normal)) {
    // Fix the seed for each label to get deterministic behaviour
    CGAL::get_default_random() = CGAL::Random(0);

    // Run ransac
    ransac_plane_extractor::RansacPlaneExtractor ransac_plane_extractor(ransacParameters_);
    ransac_plane_extractor.detectPlanes(points_with_normal);
    const auto& planes = ransac_plane_extractor.getDetectedPlanes();

    bool reuseLabel = true;
    for (const auto& plane : planes) {
      // Bookkeeping of labels : reuse old label for the first plane
      int newLabel = (reuseLabel) ? label : ++segmentedPlanesMap_.highestLabel;
      reuseLabel = false;

      // Compute average plane parameters for refined segmentation
      const std::vector<std::size_t>& plane_point_indices = plane->indices_of_assigned_points();
      Eigen::Vector3d support_vector_refined_sum = Eigen::Vector3d::Zero();
      Eigen::Vector3d normal_vector_refined_sum = Eigen::Vector3d::Zero();
      for (const auto index : plane_point_indices) {
        const auto& point = points_with_normal[index].first;
        const auto& normal = points_with_normal[index].second;
        support_vector_refined_sum += Eigen::Vector3d(point.x(), point.y(), point.z());
        normal_vector_refined_sum += Eigen::Vector3d(normal.x(), normal.y(), normal.z());

        // relabel if required
        if (newLabel != label) {
          // Need to lookup indices in map, because RANSAC has reordered the points
          Eigen::Array2i map_indices;
          map_->getIndex(Eigen::Vector2d(point.x(), point.y()), map_indices);
          segmentedPlanesMap_.labeledImage.at<int>(map_indices(0), map_indices(1)) = newLabel;
        }
      }

      Eigen::Vector3d normal_vector_refined_avg = normal_vector_refined_sum / plane_point_indices.size();
      normal_vector_refined_avg.normalize();  // Averaging does not preserve norm
      Eigen::Vector3d support_vector_refined_avg = support_vector_refined_sum / plane_point_indices.size();

      if (angleBetweenVectorsInDegrees(normal_vector_refined_avg, Eigen::Vector3d::UnitZ()) <
          parameters_.plane_inclination_threshold_degrees) {
        const TerrainPlane temp_plane_parameters(
            support_vector_refined_avg, switched_model::orientationWorldToTerrainFromSurfaceNormalInWorld(normal_vector_refined_avg));
        segmentedPlanesMap_.labelPlaneParameters.emplace_back(newLabel, temp_plane_parameters);
      }
    }

    const auto& unassigned_points = ransac_plane_extractor.getUnassignedPointIndices();
    for (const auto index : unassigned_points) {
      const auto& point = points_with_normal[index].first;

      // Need to lookup indices in map, because RANSAC has reordered the points
      Eigen::Array2i map_indices;
      map_->getIndex(Eigen::Vector2d(point.x(), point.y()), map_indices);
      segmentedPlanesMap_.labeledImage.at<int>(map_indices(0), map_indices(1)) = 0;
    }
  } else {
    if (angleBetweenVectorsInDegrees(normal_vector_avg, Eigen::Vector3d::UnitZ()) < parameters_.plane_inclination_threshold_degrees) {
      const TerrainPlane temp_plane_parameters(support_vector_avg,
                                               switched_model::orientationWorldToTerrainFromSurfaceNormalInWorld(normal_vector_avg));
      segmentedPlanesMap_.labelPlaneParameters.emplace_back(label, temp_plane_parameters);
    }
  }
}

bool SlidingWindowPlaneExtractor::isGloballyPlanar(const Eigen::Vector3d& normalVectorPlane, const Eigen::Vector3d& supportVectorPlane,
                                                   const std::vector<ransac_plane_extractor::PointWithNormal>& points_with_normal) const {
  for (const auto& point_with_normal : points_with_normal) {
    Eigen::Vector3d p_S_P(point_with_normal.first.x() - supportVectorPlane.x(), point_with_normal.first.y() - supportVectorPlane.y(),
                          point_with_normal.first.z() - supportVectorPlane.z());
    double distanceError = std::abs(normalVectorPlane.dot(p_S_P));
    if (distanceError > parameters_.global_plane_fit_distance_error_threshold) {
      return false;
    } else {
      Eigen::Vector3d pointNormal{point_with_normal.second.x(), point_with_normal.second.y(), point_with_normal.second.z()};
      double angleError = angleBetweenVectorsInDegrees(pointNormal, normalVectorPlane);
      if (angleError > parameters_.global_plane_fit_angle_error_threshold_degrees) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace sliding_window_plane_extractor
}  // namespace convex_plane_decomposition
