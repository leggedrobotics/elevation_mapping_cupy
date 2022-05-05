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

namespace grid_map {
namespace smoothing {

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
}  // namespace grid_map
