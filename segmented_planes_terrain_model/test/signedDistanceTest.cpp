//
// Created by rgrandia on 09.07.20.
//

#include <chrono>


#include <grid_map_core/GridMap.hpp>
#include <grid_map_cv/grid_map_cv.hpp>

#include <signed_distance_field/SignedDistance2d.h>
#include <grid_map_sdf/SignedDistanceField.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "segmented_planes_terrain_model/SegmentedPlanesSignedDistanceField.h"

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
    switched_model::SegmentedPlanesSignedDistanceField sdf(gridMap, layer, minHeight, maxHeight);

    // =========================================================
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  std::cout << "Sdf computation took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / N << " [ms]\n";
}