#include "convex_plane_decomposition/GridMapPreprocessing.h"

#include <grid_map_cv/GridMapCvConverter.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "convex_plane_decomposition/Nan.h"

namespace convex_plane_decomposition {

GridMapPreprocessing::GridMapPreprocessing(const PreprocessingParameters& parameters) : parameters_(parameters) {}

void GridMapPreprocessing::preprocess(grid_map::GridMap& gridMap, const std::string& layer) const {
  const float minValue = gridMap.get(layer).minCoeffOfFinites();
  const float maxValue = gridMap.get(layer).maxCoeffOfFinites();

  patchNans(gridMap.get(layer));
  denoise(gridMap, layer);

  if (parameters_.inpaintRadius > 0) {
    inpaint(gridMap, layer, minValue, maxValue);
    changeResolution(gridMap, layer);
  }
}

void GridMapPreprocessing::denoise(grid_map::GridMap& gridMap, const std::string& layer) const {
  Eigen::MatrixXf& elevation_map = gridMap.get(layer);

  cv::Mat elevationImage;
  cv::eigen2cv(elevation_map, elevationImage); // creates CV_32F image

  int kernelSize = parameters_.kernelSize;

  for (int i = 0; i < parameters_.numberOfRepeats; ++i) {
    kernelSize = std::max(3, std::min(kernelSize, 5)); // must be 3 or 5 for current image type, see doc of cv::medianBlur
    cv::medianBlur(elevationImage, elevationImage, kernelSize);
    if (parameters_.increasing) { // TODO (rgrandia) : remove this option or enable kernels of other size than 3 / 5
      kernelSize += 2;
    }
  }

  cv::cv2eigen(elevationImage, elevation_map);
}

void GridMapPreprocessing::changeResolution(grid_map::GridMap& gridMap, const std::string& layer) const {
  if (parameters_.resolution > 0.0 && gridMap.getResolution() != parameters_.resolution) {
    Eigen::MatrixXf elevation_map = std::move(gridMap.get(layer));

    cv::Mat elevationImage;
    cv::eigen2cv(elevation_map, elevationImage);

    double scaling = gridMap.getResolution() / parameters_.resolution;
    int width = int(elevationImage.size[1] * scaling);
    int height = int(elevationImage.size[0] * scaling);
    cv::Size dim{width, height};

    cv::Mat resizedImage;
    cv::resize(elevationImage, resizedImage, dim, 0, 0, cv::INTER_LINEAR);

    cv::cv2eigen(resizedImage, elevation_map);

    const auto oldPosition = gridMap.getPosition();
    gridMap.setGeometry({elevation_map.rows() * parameters_.resolution, elevation_map.cols() * parameters_.resolution},
                        parameters_.resolution, oldPosition);
    gridMap.get(layer) = std::move(elevation_map);
  }
}

void GridMapPreprocessing::inpaint(grid_map::GridMap& gridMap, const std::string& layer, float minValue, float maxValue) const {
  Eigen::MatrixXf& elevation_map = gridMap.get(layer);

  Eigen::Matrix<uchar, -1, -1> mask = elevation_map.unaryExpr([](float val) { return (isNan(val)) ? uchar(1) : uchar(0); });

  cv::Mat maskImage;
  cv::eigen2cv(mask, maskImage);

  cv::Mat elevationImage;
  grid_map::GridMapCvConverter::toImage<unsigned char, 1>(gridMap, layer, CV_8UC1, minValue, maxValue, elevationImage);

  // Inpainting
  cv::Mat filledImage;
  const double radiusInPixels = parameters_.inpaintRadius / gridMap.getResolution();
  cv::inpaint(elevationImage, maskImage, filledImage, radiusInPixels, cv::INPAINT_NS);

  // Get inpainting as float
  cv::Mat filledImageFloat;
  const float maxUCharValue = 255.F;
  filledImage.convertTo(filledImageFloat, CV_32F, (maxValue - minValue) / maxUCharValue, minValue);

  // Copy inpainted values back to elevation map
  cv::Mat elevationImageFloat;
  cv::eigen2cv(elevation_map, elevationImageFloat);
  filledImageFloat.copyTo(elevationImageFloat, maskImage);

  cv::cv2eigen(elevationImageFloat, elevation_map);
}

}  // namespace convex_plane_decomposition
