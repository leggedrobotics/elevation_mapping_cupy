/**
 * @file        smoothing.cpp
 * @authors     Fabian Jenelten
 * @date        18.05, 2021
 * @affiliation ETH RSL
 * @brief       Smoothing and outlier rejection filters.
 */

// grid map filters rsl.
#include <grid_map_filters_rsl/smoothing.hpp>

// open cv.
#include <grid_map_cv/GridMapCvConverter.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/xphoto/bm3d_image_denoising.hpp>

// message logger.
#include <message_logger/message_logger.hpp>

namespace tamols_mapping {
namespace smoothing {

void pca(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, double sigma) {
  // https://en.wikipedia.org/wiki/Principal_component_analysis

  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Normalize columns (to have zero mean).
  const grid_map::Matrix& H_in = map.get(layerIn);
  grid_map::Matrix colMean;
  colMean.resize(H_in.cols(), H_in.rows());

  for (auto colId = 0; colId < H_in.cols(); ++colId) {
    colMean.col(colId).setConstant(H_in.col(colId).sum() / float(H_in.rows()));
  }

  // Singular value decomposition: H = U*S*V'
  Eigen::JacobiSVD<grid_map::Matrix> svd(H_in - colMean, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXf& s = svd.singularValues();

  // Determine number of relevant entries.
  const double thresholdEnergy = sigma * s.squaredNorm();
  auto numSingValues = 0U;
  double addedEnergy = 0.0;
  for (auto id = 0U; id < s.size(); ++id) {
    if (std::isnan(s(id))) {
      MELO_WARN_STREAM("Detected NaN singular value.")
      return;
    }

    // Add singular value.
    addedEnergy += s(id) * s(id);
    ++numSingValues;

    // Check if we have summed up enough energy.
    if (addedEnergy >= thresholdEnergy) {
      break;
    }
  }

  // Reject singular values smaller than threshold and compose.
  const grid_map::Matrix& U = svd.matrixU();
  const grid_map::Matrix& V = svd.matrixV();
  map.get(layerOut) = U.leftCols(numSingValues) * s.head(numSingValues).asDiagonal() * V.leftCols(numSingValues).transpose() + colMean;
}

void median(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, int deltaKernelSize,
            int numberOfRepeats) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  if (kernelSize + deltaKernelSize * (numberOfRepeats - 1) <= 5) {
    // Convert to image.
    cv::Mat elevationImage;
    cv::eigen2cv(map.get(layerIn), elevationImage);

    for (auto iter = 0; iter < numberOfRepeats; ++iter) {
      cv::medianBlur(elevationImage, elevationImage, kernelSize);
      kernelSize += deltaKernelSize;
    }

    // Set output layer.
    cv::cv2eigen(elevationImage, map.get(layerOut));
  }

  // Larger kernel sizes require a specific format.
  else {
    // Reference to in map.
    const grid_map::Matrix& H_in = map.get(layerIn);

    // Convert grid map to image.
    cv::Mat elevationImage;
    const float minValue = H_in.minCoeffOfFinites();
    const float maxValue = H_in.maxCoeffOfFinites();
    grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, layerIn, CV_8UC1, minValue, maxValue, elevationImage);

    for (auto iter = 0; iter < numberOfRepeats; ++iter) {
      cv::medianBlur(elevationImage, elevationImage, kernelSize);
      kernelSize += deltaKernelSize;
    }

    // Get image as float.
    cv::Mat elevationImageFloat;
    constexpr float maxUCharValue = 255.F;
    elevationImage.convertTo(elevationImageFloat, CV_32F, (maxValue - minValue) / maxUCharValue, minValue);

    // Convert back to grid map.
    cv::cv2eigen(elevationImageFloat, map.get(layerOut));
  }
}

void boxBlur(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, int numberOfRepeats) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Convert to image.
  cv::Mat elevationImage;
  cv::eigen2cv(map.get(layerIn), elevationImage);

  // Box blur.
  cv::Size kernelSize2D(kernelSize, kernelSize);
  for (auto iter = 0; iter < numberOfRepeats; ++iter) {
    cv::blur(elevationImage, elevationImage, kernelSize2D);
  }

  // Set output layer.
  cv::cv2eigen(elevationImage, map.get(layerOut));
}

void gaussianBlur(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, double sigma) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Convert to image.
  cv::Mat elevationImage;
  cv::eigen2cv(map.get(layerIn), elevationImage);

  // Box blur.
  cv::Size kernelSize2D(kernelSize, kernelSize);
  cv::GaussianBlur(elevationImage, elevationImage, kernelSize2D, sigma);

  // Set output layer.
  cv::cv2eigen(elevationImage, map.get(layerOut));
}

void nlm(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, int searchWindow, float w) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Get input layer.
  const grid_map::Matrix& H_in = map.get(layerIn);

  // Convert grid map to image.
  cv::Mat elevationImageIn;
  const float minValue = H_in.minCoeffOfFinites();
  const float maxValue = H_in.maxCoeffOfFinites();
  grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, layerIn, CV_8UC1, minValue, maxValue, elevationImageIn);

  // Filter.
  cv::Mat elevationImageOut;
  cv::fastNlMeansDenoising(elevationImageIn, elevationImageOut, w, kernelSize, searchWindow);

  // Get as float.
  cv::Mat elevationImageFloat;
  constexpr float maxUCharValue = 255.F;
  elevationImageOut.convertTo(elevationImageFloat, CV_32F, (maxValue - minValue) / maxUCharValue, minValue);

  // Set output layer.
  cv::cv2eigen(elevationImageFloat, map.get(layerOut));
}

void bm3d(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, int searchWindow, float w) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Get input layer.
  const grid_map::Matrix& H_in = map.get(layerIn);

  // Convert grid map to image.
  cv::Mat elevationImageIn;
  const float minValue = H_in.minCoeffOfFinites();
  const float maxValue = H_in.maxCoeffOfFinites();
  grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, layerIn, CV_8UC1, minValue, maxValue, elevationImageIn);

  // Filter.
  cv::Mat elevationImageOut;
  cv::xphoto::bm3dDenoising(elevationImageIn, elevationImageOut, w, kernelSize, searchWindow, 2500, 400, 8, 1, 0.0f, cv::NORM_L1,
                            cv::xphoto::BM3D_STEP1);

  // Get as float.
  cv::Mat elevationImageFloat;
  constexpr float maxUCharValue = 255.F;
  elevationImageOut.convertTo(elevationImageFloat, CV_32F, (maxValue - minValue) / maxUCharValue, minValue);

  // Set output layer.
  cv::cv2eigen(elevationImageFloat, map.get(layerOut));
}

void bilateralFilter(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, int kernelSize, double w) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Convert to image.
  cv::Mat elevationImageIn;
  cv::eigen2cv(map.get(layerIn), elevationImageIn);

  // Filter.
  cv::Mat elevationImageOut;
  cv::bilateralFilter(elevationImageIn, elevationImageOut, kernelSize, w, w);

  // Set output layer.
  cv::cv2eigen(elevationImageOut, map.get(layerOut));
}

void tvL1(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, double lambda, int n) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Get input layer.
  const grid_map::Matrix& H_in = map.get(layerIn);

  // Convert grid map to image.
  cv::Mat elevationImageIn;
  const float minValue = H_in.minCoeffOfFinites();
  const float maxValue = H_in.maxCoeffOfFinites();
  grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, layerIn, CV_8UC1, minValue, maxValue, elevationImageIn);

  // Filter.
  cv::Mat elevationImageOut;
  cv::denoise_TVL1({elevationImageIn}, elevationImageOut, lambda, n);

  // Get as float.
  cv::Mat elevationImageFloat;
  constexpr float maxUCharValue = 255.F;
  elevationImageOut.convertTo(elevationImageFloat, CV_32F, (maxValue - minValue) / maxUCharValue, minValue);

  // Set output layer.
  cv::cv2eigen(elevationImageFloat, map.get(layerOut));
}
}  // namespace smoothing
}  // namespace tamols_mapping
