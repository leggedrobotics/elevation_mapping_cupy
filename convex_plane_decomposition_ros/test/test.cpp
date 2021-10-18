//
// Created by rgrandia on 04.06.20.
//

#include <grid_map_core/GridMap.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include <algorithm>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <convex_plane_decomposition/Draw.h>
#include <convex_plane_decomposition/GridMapPreprocessing.h>
#include <convex_plane_decomposition/contour_extraction/ContourExtraction.h>
#include <convex_plane_decomposition/sliding_window_plane_extraction/SlidingWindowPlaneExtractor.h>

using namespace convex_plane_decomposition;

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

void plotSlidingWindow(const sliding_window_plane_extractor::SlidingWindowPlaneExtractor& planeExtractor) {
  cv::namedWindow("Sliding Window binarymap", cv::WindowFlags::WINDOW_NORMAL);
  cv::Mat binaryImage = planeExtractor.getBinaryLabeledImage();
  binaryImage.convertTo(binaryImage, CV_8UC1, 255);
  cv::imshow("Sliding Window binarymap", binaryImage);

  cv::namedWindow("Sliding Window labels", cv::WindowFlags::WINDOW_NORMAL);
  const auto& labelledImage = planeExtractor.getSegmentedPlanesMap().labeledImage;
  const auto& labelsAndPlaneParameters = planeExtractor.getSegmentedPlanesMap().labelPlaneParameters;
  std::vector<cv::Vec3b> colors(planeExtractor.getSegmentedPlanesMap().highestLabel + 1);

  colors[0] = cv::Vec3b(0, 0, 0);  // background in white
  for (int label = 1; label <= planeExtractor.getSegmentedPlanesMap().highestLabel; ++label) {
    const auto labelIt = std::find_if(labelsAndPlaneParameters.begin(), labelsAndPlaneParameters.end(),
                                      [=](const std::pair<int, TerrainPlane>& x) { return x.first == label; });
    if (labelIt != labelsAndPlaneParameters.end()) {
      colors[label] = randomColor();
    } else {
      colors[label] = colors[0];
    }
  }
  cv::Mat colorImg(labelledImage.size(), CV_8UC3);
  for (int r = 0; r < colorImg.rows; ++r) {
    for (int c = 0; c < colorImg.cols; ++c) {
      int label = labelledImage.at<int>(r, c);
      auto& pixel = colorImg.at<cv::Vec3b>(r, c);
      pixel = colors[label];
    }
  }
  cv::imshow("Sliding Window labels", colorImg);
}

int main(int argc, char** argv) {
  std::string folder = "/home/rgrandia/git/anymal/convex_terrain_representation/convex_plane_decomposition_ros/data/";
  std::string file = "elevationMap_8_139cm.png";
  double heightScale = 1.39;
  //    std::string file = "demo_map.png"; double heightScale = 1.25;
  //  std::string file = "realStairs_125cm.png"; double heightScale = 1.25;
  //    std::string file = "terrain.png"; double heightScale = 1.25;
  //    std::string file = "holes.png"; double heightScale = 1.0;
  //    std::string file = "slope_1m_1m_20cm.png"; double heightScale = 0.2;
  //    std::string file = "straightStairs_1m_1m_60cm.png"; double heightScale = 0.6;
  double resolution = 0.02;

  auto elevationMap = loadElevationMapFromFile(folder + file, resolution, heightScale);

  cv::Mat image;
  grid_map::GridMapCvConverter::toImage<unsigned char, 1>(elevationMap, "elevation", CV_8UC1, 0.0, heightScale, image);
  cv::namedWindow("Elevation Map", cv::WindowFlags::WINDOW_NORMAL);
  cv::imshow("Elevation Map", image);

  // ================== Filter ======================== //
  PreprocessingParameters preprocessingParameters;
  preprocessingParameters.kernelSize = 5;
  preprocessingParameters.numberOfRepeats = 1;

  GridMapPreprocessing preprocessing(preprocessingParameters);
  preprocessing.preprocess(elevationMap, "elevation");

  grid_map::GridMapCvConverter::toImage<unsigned char, 1>(elevationMap, "elevation", CV_8UC1, 0.0, heightScale, image);
  cv::namedWindow("Filtered Map", cv::WindowFlags::WINDOW_NORMAL);
  cv::imshow("Filtered Map", image);

  // ============== Sliding Window =================== //
  sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters swParams;
  swParams.kernel_size = 3;
  swParams.plane_inclination_threshold = std::cos(45.0 * M_PI / 180);
  swParams.plane_patch_error_threshold = 0.005;
  swParams.min_number_points_per_label = 10;
  swParams.include_ransac_refinement = true;
  swParams.global_plane_fit_distance_error_threshold = 0.04;
  swParams.global_plane_fit_angle_error_threshold_degrees = 5.0;

  ransac_plane_extractor::RansacPlaneExtractorParameters ransacParameters;
  ransacParameters.probability = 0.01;
  ransacParameters.min_points = 25;
  ransacParameters.epsilon = 0.02;
  ransacParameters.cluster_epsilon = 0.05;
  ransacParameters.normal_threshold = 0.98;

  sliding_window_plane_extractor::SlidingWindowPlaneExtractor planeExtractor(swParams, ransacParameters);
  planeExtractor.runExtraction(elevationMap, "elevation");
  plotSlidingWindow(planeExtractor);

  // ============== Polygons =================== //
  const auto& labeled_image = planeExtractor.getSegmentedPlanesMap().labeledImage;
  cv::Mat binary_image(labeled_image.size(), CV_8UC1);

  for (const auto& label_plane : planeExtractor.getSegmentedPlanesMap().labelPlaneParameters) {
    const int label = label_plane.first;
    const auto& plane_parameters = label_plane.second;

    binary_image = labeled_image == label;

    // Plot binary image
    cv::Mat binaryImagePlot = binary_image;
    binaryImagePlot.convertTo(binaryImagePlot, CV_8UC1, 255);
    cv::namedWindow("Polygon binary image label", cv::WindowFlags::WINDOW_NORMAL);
    cv::imshow("Polygon binary image label", binaryImagePlot);

    // Extract
    int erosion_size = 2;                // single sided length of the kernel
    int erosion_type = cv::MORPH_CROSS;  // cv::MORPH_ELLIPSE, cv::MORPH_RECT
    cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));
    const auto boundariesAndInsets = contour_extraction::extractBoundaryAndInset(binary_image, element);

    // Plot eroded image
    cv::Mat erodedImagePlot = binary_image;
    erodedImagePlot.convertTo(erodedImagePlot, CV_8UC1, 255);
    cv::namedWindow("Polygon binary eroded label", cv::WindowFlags::WINDOW_NORMAL);
    cv::imshow("Polygon binary eroded label", erodedImagePlot);

    // Plot contours
    int scale = 5;
    cv::Size scaledSize = {scale * binary_image.size().width, scale * binary_image.size().height};  // transposed
    cv::Mat realPolygonImage(scaledSize, CV_8UC3, cv::Vec3b(0, 0, 0));
    for (const auto& boundaryAndInset : boundariesAndInsets) {
      drawContour(realPolygonImage, scaleShape(boundaryAndInset.boundary, scale));
      for (const auto& inset : boundaryAndInset.insets) {
        drawContour(realPolygonImage, scaleShape(inset, scale));
      }
    }

    cv::namedWindow("Polygon real", cv::WindowFlags::WINDOW_NORMAL);
    cv::imshow("Polygon real", realPolygonImage);
    cv::waitKey(0);  // Wait for a keystroke in the window
  }

  // ============== planar regions extraction ================
  contour_extraction::ContourExtractionParameters polyParams;
  polyParams.offsetSize = 2;

  contour_extraction::ContourExtraction contourExtraction(polyParams);
  contourExtraction.extractPlanarRegions(planeExtractor.getSegmentedPlanesMap());

  return 0;
}