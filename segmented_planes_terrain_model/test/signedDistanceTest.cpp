//
// Created by rgrandia on 09.07.20.
//

#include <chrono>

#include <segmented_planes_terrain_model/distance_transform/dt.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_cv/grid_map_cv.hpp>

#include <signed_distance_field/SignedDistance2d.h>
#include <signed_distance_field/SignedDistanceField.h>
#include <grid_map_sdf/SignedDistanceField.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace distance_transform;

constexpr float myINF = 1e20F;

grid_map::GridMap loadElevationMapFromFile(const std::string& filePath, double resolution, double scale) {
  // Read the file
  cv::Mat image;
  image = cv::imread(filePath, cv::ImreadModes::IMREAD_GRAYSCALE);

  // Check for invalid input
  if (!image.data) {
    throw std::runtime_error("Could not open or find the image");
  }

  // Min max values
  double minValue, maxValue;
  cv::minMaxLoc(image, &minValue, &maxValue);

  grid_map::GridMap mapOut({"elevation"});
  mapOut.setFrameId("odom");
  grid_map::GridMapCvConverter::initializeFromImage(image, resolution, mapOut, grid_map::Position(0.0, 0.0));
  grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 1>(image, std::string("elevation"), mapOut, float(0.0), float(scale), 0.5);
  return mapOut;
}

grid_map::Matrix getPlanarSignedDistanceField(const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& data) {
  image<uchar>* input = new image<uchar>(data.rows(), data.cols(), true);

  for (int y = 0; y < input->height(); y++) {
    for (int x = 0; x < input->width(); x++) {
      imRef(input, x, y) = data(x, y);
    }
  }

  // Compute dt.
  image<float>* out = dt(input);

  grid_map::Matrix result(data.rows(), data.cols());

  // Take square roots.
  for (int y = 0; y < out->height(); y++) {
    for (int x = 0; x < out->width(); x++) {
      result(x, y) = sqrt(imRef(out, x, y));
    }
  }
  return result;
}

void squaredDistanceTransform_1d_inplace(Eigen::Ref<Eigen::VectorXf> squareDistance1d, Eigen::Ref<Eigen::VectorXf> d,
                                         Eigen::Ref<Eigen::VectorXf> z, std::vector<int>& v) {
  const int n = squareDistance1d.size();
  const auto& f = squareDistance1d;
  assert(d.size() == n);
  assert(z.size() == n + 1);
  assert(v.size() == n);

  // Initialize
  int k = 0;
  v[0] = 0;
  z[0] = -myINF;
  z[1] = myINF;

  // Compute bounds
  for (int q = 1; q < n; q++) {
    auto factor1 = static_cast<float>(q - v[k]);
    auto factor2 = static_cast<float>(q + v[k]);
    float s = 0.5F * ( (f[q] - f[v[k]]) / factor1  + factor2);
    while (s <= z[k]) {
      k--;
      factor1 = static_cast<float>(q - v[k]);
      factor2 = static_cast<float>(q + v[k]);
      s = 0.5F * ( (f[q] - f[v[k]]) / factor1  + factor2);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = myINF;
  }

  // Collect results
  k = 0;
  for (int q = 0; q < n; q++) {
    while (z[k + 1] < static_cast<float>(q)) {
      k++;
    }
    d[q] = square(q - v[k]) + f[v[k]];
  }

  // Write distance result back in place
  squareDistance1d = d;
}

void squaredDistanceTransform_2d_columnwiseInplace(grid_map::Matrix& squareDistance, bool skipCheck) {
  const size_t n = squareDistance.cols();
  Eigen::VectorXf workvector(2 * n + 1);
  std::vector<int> intWorkvector(n);

  for (size_t i = 0; i < n; i++) {
    // Only when there is a mix of elements and free space, we need to update the initialized distance.
    bool needsComputation = skipCheck || ((squareDistance.col(i).array() == 0.0).any() && (squareDistance.col(i).array() == myINF).any());
    if (needsComputation) {
      squaredDistanceTransform_1d_inplace(squareDistance.col(i), workvector.segment(0, n), workvector.segment(n, n+1),
                                          intWorkvector);
    }
  }
}

void squaredDistanceTransform_2d(grid_map::Matrix& squareDistance) {
  // TODO (check which one we want to do first)

  // Process columns
  squaredDistanceTransform_2d_columnwiseInplace(squareDistance, false);

  // Process rows
  squareDistance.transposeInPlace();
  squaredDistanceTransform_2d_columnwiseInplace(squareDistance, true);
  squareDistance.transposeInPlace();
}

void getPlanarSignedDistanceField_2(grid_map::Matrix& squareDistance) {
  squaredDistanceTransform_2d(squareDistance);
  squareDistance = squareDistance.cwiseSqrt();
}

grid_map::Matrix getPlanarSignedDistanceField_3(const grid_map::Matrix& map, float threshold, float resolution) {
  bool hasObstacles = (map.array() >= threshold).any();
  if (hasObstacles) {
    bool hasFreeSpace = (map.array() < threshold).any();
    if (hasFreeSpace) {
      // Both obstacles and free space -> compute planar signed distance
      // Compute pixel distance to obstacles
      grid_map::Matrix sdfObstacle = map.unaryExpr([=](float val) {return (val >= threshold) ? 0.0F : myINF;});
      getPlanarSignedDistanceField_2(sdfObstacle);

      // Compute pixel distance to obstacle free space
      grid_map::Matrix sdfObstacleFree = map.unaryExpr([=](float val) {return (val < threshold) ? 0.0F : myINF;});
      getPlanarSignedDistanceField_2(sdfObstacleFree);

      grid_map::Matrix sdf2d = resolution * (sdfObstacle - sdfObstacleFree);
      return sdf2d;
    } else {
     // Only obstacles -> distance is zero everywhere
     return grid_map::Matrix::Zero(map.rows(), map.cols());
    }
  } else {
    // No obstacles -> planar distance is infinite
    return grid_map::Matrix::Constant(map.rows(), map.cols(), myINF);
  }
}

int main(int argc, char** argv) {
  std::string folder = "/home/rgrandia/git/anymal/convex_terrain_representation/convex_plane_decomposition_ros/data/";
  //  std::string file = "elevationMap_8_139cm.png";
  //  double heightScale = 1.39;
  std::string file = "demo_map.png";
  double heightScale = 1.25;
  //  std::string file = "realStairs_125cm.png"; double heightScale = 1.25;
  //    std::string file = "terrain.png"; double heightScale = 1.25;
  //    std::string file = "holes.png"; double heightScale = 1.0;
  //    std::string file = "slope_1m_1m_20cm.png"; double heightScale = 0.2;
  //    std::string file = "straightStairs_1m_1m_60cm.png"; double heightScale = 0.6;
  double resolution = 0.02;

  auto messageMap = loadElevationMapFromFile(folder + file, resolution, heightScale);

  grid_map::SignedDistanceField signedDistanceField;

  double width = 4.0;
  bool success;
  grid_map::GridMap localMap = messageMap.getSubmap(messageMap.getPosition(), Eigen::Array2d(width, width), success);

  //  signedDistanceField.calculateSignedDistanceField(messageMap, "elevation", heightClearance);
  const grid_map::GridMap& gridMap = localMap;
  const std::string& layer = "elevation";
  const double heightClearance = 0.1;

  // Members
  float resolution_;
  grid_map::Size size_;
  grid_map::Position position_;
  std::vector<grid_map::Matrix> data_;
  float zIndexStartHeight_;
  float maxDistance_;
  const float lowestHeight_ = -1e5;
  using grid_map::Matrix;

  // initialization
  resolution_ = gridMap.getResolution();
  position_ = gridMap.getPosition();
  size_ = gridMap.getSize();
  Matrix map = gridMap.get(layer);  // Copy!

  float minHeight = map.minCoeffOfFinites();
  if (!std::isfinite(minHeight)) minHeight = lowestHeight_;
  float maxHeight = map.maxCoeffOfFinites();
  if (!std::isfinite(maxHeight)) maxHeight = lowestHeight_;

  const float valueForEmptyCells = lowestHeight_;  // maxHeight, minHeight (TODO Make this an option).
  for (size_t i = 0; i < map.size(); ++i) {
    if (std::isnan(map(i))) map(i) = valueForEmptyCells;
  }

  // Height range of the signed distance field is higher than the max height.
  maxHeight += heightClearance;

  auto t2 = std::chrono::high_resolution_clock::now();
  const int N = 50;
  for (int rep = 0; rep < N; rep++) {
//    // =========================================================
//    data_.clear();
//
//    // Calculate signed distance field from bottom.
//    for (float h = minHeight; h < maxHeight; h += resolution_) {
//      data_.emplace_back(signed_distance_field::signedDistanceAtHeight(map, h, resolution));
//    }
    auto sdf = signed_distance_field::SignedDistanceField(gridMap, layer, minHeight, maxHeight);

    // =========================================================
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  std::cout << "Sdf computation took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / N << " [ms]\n";
}