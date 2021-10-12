#include "convex_plane_decomposition/Postprocessing.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <grid_map_filters_rsl/processing.hpp>

namespace convex_plane_decomposition {

Postprocessing::Postprocessing(const PostprocessingParameters& parameters) : parameters_(parameters) {}

void Postprocessing::postprocess(PlanarTerrain& planarTerrain, const std::string& elevationLayer,
                                 const std::string& planeSegmentationLayer) const {
  auto& elevationData = planarTerrain.gridMap.get(elevationLayer);
  const auto& planarityMask = planarTerrain.gridMap.get(planeSegmentationLayer);

  // post process planar regions
  addHeightOffset(planarTerrain.planarRegions);

  // Add smooth layer for base reference
  addSmoothLayer(planarTerrain.gridMap, elevationData, planarityMask);

  // post process elevation map
  dilationInNonplanarRegions(elevationData, planarityMask);
  addHeightOffset(elevationData, planarityMask);
}

void Postprocessing::dilationInNonplanarRegions(Eigen::MatrixXf& elevationData, const Eigen::MatrixXf& planarityMask) const {
  if (parameters_.nonplanar_horizontal_offset > 0) {
    // Convert to opencv image
    cv::Mat elevationImage;
    cv::eigen2cv(elevationData, elevationImage);  // creates CV_32F image

    // dilate
    const int dilationSize = 2 * parameters_.nonplanar_horizontal_offset + 1;  //
    const int dilationType = cv::MORPH_ELLIPSE;                                // ellipse inscribed in the square of size dilationSize
    const auto dilationKernel_ = cv::getStructuringElement(dilationType, cv::Size(dilationSize, dilationSize));
    cv::dilate(elevationImage, elevationImage, dilationKernel_);

    // convert back
    Eigen::MatrixXf elevationDilated;
    cv::cv2eigen(elevationImage, elevationDilated);

    // merge: original elevation for planar regions (mask = 1.0), dilated elevation for non-planar (mask = 0.0)
    elevationData = planarityMask.array() * elevationData.array() + (1.0 - planarityMask.array()) * elevationDilated.array();
  }
}

void Postprocessing::addHeightOffset(Eigen::MatrixXf& elevationData, const Eigen::MatrixXf& planarityMask) const {
  // lift elevation layer. For untraversable offset we first add the offset everywhere and substract it again in traversable regions.
  if (parameters_.extracted_planes_height_offset != 0.0 || parameters_.nonplanar_height_offset != 0.0) {
    elevationData.array() += (parameters_.extracted_planes_height_offset + parameters_.nonplanar_height_offset);

    if (parameters_.nonplanar_height_offset != 0.0) {
      elevationData.noalias() -= parameters_.nonplanar_height_offset * planarityMask;
    }
  }
}

void Postprocessing::addHeightOffset(std::vector<PlanarRegion>& planarRegions) const {
  if (parameters_.extracted_planes_height_offset != 0.0) {
    for (auto& planarRegion : planarRegions) {
      planarRegion.planeParameters.positionInWorld.z() += parameters_.extracted_planes_height_offset;
    }
  }
}

void Postprocessing::addSmoothLayer(grid_map::GridMap& gridMap, const Eigen::MatrixXf& elevationData,
                                    const Eigen::MatrixXf& planarityMask) const {
  const int dilationSize = 2 * std::round(parameters_.smoothing_dilation_size / gridMap.getResolution()) + 1;
  const int kernel = 2 * std::round(parameters_.smoothing_box_kernel_size / gridMap.getResolution()) + 1;
  const int kernelGauss = 2 * std::round(parameters_.smoothing_gauss_kernel_size / gridMap.getResolution()) + 1;

  // Set nonplanar regions to "NaN"
  const auto lowestFloat = std::numeric_limits<float>::lowest(); // Take lowest to not interfere with Dilation, using true NaN doesn't work with opencv dilation.
  Eigen::MatrixXf elevationWithNaN =
      (planarityMask.array() == 1.0).select(elevationData, Eigen::MatrixXf::Constant(elevationData.rows(), elevationData.cols(), lowestFloat));

  // Convert to openCV
  cv::Mat elevationWithNaNImage;
  cv::eigen2cv(elevationWithNaN, elevationWithNaNImage);  // creates CV_32F image

  // Dilate
  const int dilationType = cv::MORPH_ELLIPSE;  // ellipse inscribed in the square of size dilationSize
  const auto dilationKernel_ = cv::getStructuringElement(dilationType, cv::Size(dilationSize, dilationSize));
  cv::dilate(elevationWithNaNImage, elevationWithNaNImage, dilationKernel_);

  // Take complement image where elevation data was set to NaN
  cv::Mat indicatorImage = cv::Mat::ones(elevationWithNaNImage.rows, elevationWithNaNImage.cols, CV_32F);
  indicatorImage.setTo(0.0, elevationWithNaNImage == lowestFloat);

  // Set NaN's to 0.0
  elevationWithNaNImage.setTo(0.0, elevationWithNaNImage == lowestFloat);

  // Filter definition
  auto smoothingFilter = [kernel, kernelGauss](const cv::Mat& imageIn) {
    cv::Mat imageOut;
    cv::boxFilter(imageIn, imageOut, -1, {kernel, kernel}, cv::Point(-1, -1), true, cv::BorderTypes::BORDER_CONSTANT);
    cv::GaussianBlur(imageOut, imageOut, {kernelGauss, kernelGauss}, 0, 0, cv::BorderTypes::BORDER_CONSTANT);
    return imageOut;
  };

  // Helper to convert to Eigen
  auto toEigen = [](const cv::Mat& image) {
    Eigen::MatrixXf data;
    cv::cv2eigen(image, data);
    return data;
  };

  // Filter trick for data with NaN from: https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
  auto planarOnly = smoothingFilter(elevationWithNaNImage);
  auto complement = smoothingFilter(indicatorImage);
  complement += 1e-6;  // Prevent division by zero
  cv::Mat result;
  cv::divide(planarOnly, complement, result);

  // Add layer to map.
  gridMap.add("smooth_planar", toEigen(result));
}

}  // namespace convex_plane_decomposition
