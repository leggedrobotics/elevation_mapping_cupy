/**
 * @file        inpainting.cpp
 * @authors     Fabian Jenelten
 * @date        18.05, 2021
 * @affiliation ETH RSL
 * @brief      Inpainting filter (extrapolate nan values from surrounding data).
 */

// grid map filters rsl.
#include <grid_map_filters_rsl/inpainting.hpp>

// open cv.
#include <grid_map_cv/GridMapCvConverter.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

// stl.
#include <limits>

namespace grid_map {
namespace inpainting {

void minValues(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut, map.get(layerIn));
  }

  // Reference to in and out maps.
  const grid_map::Matrix& H_in = map.get(layerIn);
  grid_map::Matrix& H_out = map.get(layerOut);
  bool success = true;
  constexpr auto infinity = static_cast<float>(std::numeric_limits<double>::max());

  for (auto colId = 0; colId < H_in.cols(); ++colId) {
    for (auto rowId = 0; rowId < H_in.rows(); ++rowId) {
      if (std::isnan(H_in(rowId, colId))) {
        auto minValue = infinity;

        // Search in negative direction.
        for (auto id = rowId - 1; id >= 0; --id) {
          auto newValue = H_in(id, colId);
          if (!std::isnan(newValue)) {
            minValue = std::fmin(minValue, newValue);
            break;
          }
        }

        for (auto id = colId - 1; id >= 0; --id) {
          auto newValue = H_in(rowId, id);
          if (!std::isnan(newValue)) {
            minValue = std::fmin(minValue, newValue);
            break;
          }
        }

        // Search in positive direction.
        for (auto id = rowId + 1; id < H_in.rows(); ++id) {
          auto newValue = H_in(id, colId);
          if (!std::isnan(newValue)) {
            minValue = std::fmin(minValue, newValue);
            break;
          }
        }

        for (auto id = colId + 1; id < H_in.cols(); ++id) {
          auto newValue = H_in(rowId, id);
          if (!std::isnan(newValue)) {
            minValue = std::fmin(minValue, newValue);
            break;
          }
        }

        // Replace.
        if (minValue < infinity) {
          H_out(rowId, colId) = minValue;
        } else {
          success = false;
        }
      }
    }
  }

  // If failed, use a more fancy method.
  if (!success) {
    return nonlinearInterpolation(map, layerIn, layerOut);
  }
}

void biLinearInterpolation(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut, map.get(layerIn));
  }

  // Helper variables.
  std::array<Eigen::Vector2i, 4> indices;
  std::array<float, 4> values;
  Eigen::Matrix4f A;
  Eigen::Vector4f b;
  A.setOnes();
  Eigen::Vector4f weights;
  bool success = true;
  constexpr auto infinity = static_cast<float>(std::numeric_limits<double>::max());

  // Init.
  std::fill(values.begin(), values.end(), NAN);
  std::fill(indices.begin(), indices.end(), Eigen::Vector2i(0, 0));

  // Reference to in and out maps.
  const grid_map::Matrix& H_in = map.get(layerIn);
  grid_map::Matrix& H_out = map.get(layerOut);

  for (auto colId = 0; colId < H_in.cols(); ++colId) {
    for (auto rowId = 0; rowId < H_in.rows(); ++rowId) {
      if (std::isnan(H_in(rowId, colId))) {
        // Note: if we don't find a valid neighbour, we use the previous index-value pair.
        auto minValue = infinity;
        const Eigen::Vector2i index0(rowId, colId);

        // Search in negative direction.
        for (auto id = rowId - 1; id >= 0; --id) {
          auto newValue = H_in(id, colId);
          if (!std::isnan(newValue)) {
            indices[0] = Eigen::Vector2i(id, colId);
            values[0] = newValue;
            minValue = std::fmin(minValue, newValue);
            break;
          }
        }

        for (auto id = colId - 1; id >= 0; --id) {
          auto newValue = H_in(rowId, id);
          if (!std::isnan(newValue)) {
            indices[1] = Eigen::Vector2i(rowId, id);
            values[1] = newValue;
            minValue = std::fmin(minValue, newValue);
            break;
          }
        }

        // Search in positive direction.
        for (auto id = rowId + 1; id < H_in.rows(); ++id) {
          auto newValue = H_in(id, colId);
          if (!std::isnan(newValue)) {
            indices[2] = Eigen::Vector2i(id, colId);
            values[2] = newValue;
            minValue = std::fmin(minValue, newValue);
            break;
          }
        }

        for (auto id = colId + 1; id < H_in.cols(); ++id) {
          auto newValue = H_in(rowId, id);
          if (!std::isnan(newValue)) {
            indices[3] = Eigen::Vector2i(rowId, id);
            values[3] = newValue;
            minValue = std::fmin(minValue, newValue);
            break;
          }
        }

        // Cannot interpolate if there are not 4 corner points.
        if (std::any_of(values.begin(), values.end(), [](float value) { return std::isnan(value); })) {
          if (minValue < infinity) {
            H_out(rowId, colId) = minValue;
          } else {
            success = false;
          }
          continue;
        }

        // Interpolation weights (https://en.wikipedia.org/wiki/Bilinear_interpolation).
        for (auto id = 0U; id < 4U; ++id) {
          A(id, 1U) = static_cast<float>(indices[id].x());
          A(id, 2U) = static_cast<float>(indices[id].y());
          A(id, 3U) = static_cast<float>(indices[id].x() * indices[id].y());
          b(id) = values[id];
        }
        weights = A.colPivHouseholderQr().solve(b);

        // Value according to bi-linear interpolation.
        H_out(rowId, colId) = weights.dot(Eigen::Vector4f(1.0, static_cast<float>(index0.x()), static_cast<float>(index0.y()),
                                                          static_cast<float>(index0.x() * index0.y())));
      }
    }
  }

  // If failed, use a more fancy method.
  if (!success) {
    return nonlinearInterpolation(map, layerIn, layerOut);
  }
}

void nonlinearInterpolation(grid_map::GridMap& map, const std::string& layerIn, const std::string& layerOut, double inpaintRadius) {
  // Create new layer if missing.
  if (!map.exists(layerOut)) {
    map.add(layerOut);
  }

  // Reference to in map.
  const grid_map::Matrix& H_in = map.get(layerIn);

  // Create mask.
  Eigen::Matrix<uchar, -1, -1> mask = H_in.unaryExpr([](float val) { return (std::isnan(val)) ? uchar(1) : uchar(0); });
  cv::Mat maskImage;
  cv::eigen2cv(mask, maskImage);

  // Convert grid map to image.
  cv::Mat elevationImageIn;
  const float minValue = H_in.minCoeffOfFinites();
  const float maxValue = H_in.maxCoeffOfFinites();
  grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, layerIn, CV_8UC1, minValue, maxValue, elevationImageIn);

  // Inpainting.
  cv::Mat elevationImageOut;
  const double radiusInPixels = inpaintRadius / map.getResolution();
  cv::inpaint(elevationImageIn, maskImage, elevationImageOut, radiusInPixels, cv::INPAINT_NS);

  // Get inpainting as float.
  cv::Mat filledImageFloat;
  constexpr float maxUCharValue = 255.F;
  elevationImageOut.convertTo(filledImageFloat, CV_32F, (maxValue - minValue) / maxUCharValue, minValue);

  // Copy inpainted values back to elevation map.
  cv::Mat elevationImageFloat;
  cv::eigen2cv(H_in, elevationImageFloat);
  filledImageFloat.copyTo(elevationImageFloat, maskImage);

  // Set new output layer.
  cv::cv2eigen(elevationImageFloat, map.get(layerOut));
}
}  // namespace inpainting
}  // namespace grid_map
