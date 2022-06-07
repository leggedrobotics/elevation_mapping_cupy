//
// Created by rgrandia on 08.06.20.
//

#include <convex_plane_decomposition/ConvexRegionGrowing.h>
#include <convex_plane_decomposition/Draw.h>
#include <convex_plane_decomposition/GeometryUtils.h>

#include <chrono>

using namespace convex_plane_decomposition;

int main(int argc, char** argv) {
  CgalPolygonWithHoles2d parentShape;
  parentShape.outer_boundary().push_back({0.0, 0.0});
  parentShape.outer_boundary().push_back({0.0, 1000.0});
  parentShape.outer_boundary().push_back({400.0, 800.0});
  parentShape.outer_boundary().push_back({1000.0, 1000.0});
  parentShape.outer_boundary().push_back({800.0, 400.0});
  parentShape.outer_boundary().push_back({1000.0, 0.0});
  parentShape.add_hole(createRegularPolygon(CgalPoint2d(200.0, 200.0), 50, 4));
  parentShape.add_hole(createRegularPolygon(CgalPoint2d(600.0, 700.0), 100, 100));

  // bounded_side_2 -> is inside

  int numberOfVertices = 16;  // Multiple of 4 is nice.
  double growthFactor = 1.05;
  CgalPoint2d center{700.0, 300.0};

  cv::Size imgSize(1000, 1000);

  const auto parentColor = randomColor();
  const auto fitColor = randomColor();

  int N_test = 10;
  for (int i = 0; i < N_test; i++) {
    center = CgalPoint2d(rand()%1000, rand()%1000);
    cv::Mat polygonImage(imgSize, CV_8UC3, cv::Vec3b(0, 0, 0));
    drawContour(polygonImage, parentShape, &parentColor);
    drawContour(polygonImage, center, 2, &fitColor);
    if (isInside(center, parentShape )) {
      const auto fittedPolygon2d = growConvexPolygonInsideShape(parentShape, center, numberOfVertices, growthFactor);
      drawContour(polygonImage, fittedPolygon2d, &fitColor);
    }

    cv::namedWindow("Polygon", cv::WindowFlags::WINDOW_NORMAL);
    cv::imshow("Polygon", polygonImage);
    cv::waitKey(0);  // Wait for a keystroke in the window
  }

  return 0;
}
