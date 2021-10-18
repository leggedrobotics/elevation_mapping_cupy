//
// Created by rgrandia on 04.06.20.
//

#include "convex_plane_decomposition/contour_extraction/ContourExtraction.h"

#include <convex_plane_decomposition/GeometryUtils.h>
#include <opencv2/imgproc.hpp>

namespace convex_plane_decomposition {
namespace contour_extraction {

ContourExtraction::ContourExtraction(const ContourExtractionParameters& parameters)
    : parameters_(parameters), binaryImage_(cv::Size(0, 0), CV_8UC1) {
  int erosionSize = parameters_.offsetSize;  // single sided length of the kernel
  int erosionType = cv::MORPH_CROSS;         // cv::MORPH_ELLIPSE, cv::MORPH_RECT
  erosionKernel_ = cv::getStructuringElement(erosionType, cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1));
}

std::vector<PlanarRegion> ContourExtraction::extractPlanarRegions(const SegmentedPlanesMap& segmentedPlanesMap) {
  std::vector<PlanarRegion> planarRegions;
  for (const auto& label_plane : segmentedPlanesMap.labelPlaneParameters) {
    const int label = label_plane.first;
    const auto& plane_parameters = label_plane.second;

    binaryImage_ = segmentedPlanesMap.labeledImage == label;

    auto boundariesAndInsets = contour_extraction::extractBoundaryAndInset(binaryImage_, erosionKernel_);

    for (auto& boundaryAndInset : boundariesAndInsets) {
      // Transform points from pixel space to local terrain frame
      transformInPlace(boundaryAndInset, [&](CgalPoint2d& point) {
        auto pointInWorld = pixelToWorldFrame(point, segmentedPlanesMap.resolution, segmentedPlanesMap.mapOrigin);
        point = worldFrameToTerrainFrame(pointInWorld, plane_parameters);
      });

      PlanarRegion planarRegion;
      planarRegion.boundaryWithInset = std::move(boundaryAndInset);
      planarRegion.planeParameters = plane_parameters;
      planarRegion.bbox2d = planarRegion.boundaryWithInset.boundary.outer_boundary().bbox();
      planarRegions.push_back(std::move(planarRegion));
    }
  }
  return planarRegions;
}

std::vector<BoundaryWithInset> extractBoundaryAndInset(cv::Mat& binary_image, const cv::Mat& erosionKernel) {
  // Get boundary
  std::vector<CgalPolygonWithHoles2d> boundaries = extractPolygonsFromBinaryImage(binary_image);

  // Erode
  cv::erode(binary_image, binary_image, erosionKernel, cv::Point(-1,-1), 1, cv::BORDER_REPLICATE);

  // Get insets
  std::vector<CgalPolygonWithHoles2d> insets = extractPolygonsFromBinaryImage(binary_image);

  // Associate boundaries with insets
  std::vector<BoundaryWithInset> boundariesWithInsets;
  for (const auto& boundary : boundaries) {
    std::vector<CgalPolygonWithHoles2d> assignedInsets;
    for (const auto& inset : insets) {
      if (isInside(inset.outer_boundary().vertex(0), boundary)) {
        assignedInsets.push_back(inset);
      }
    }

    if (!assignedInsets.empty()) {
      BoundaryWithInset boundaryWithInset;
      boundaryWithInset.boundary = boundary;
      boundaryWithInset.insets = assignedInsets;
      boundariesWithInsets.push_back(std::move(boundaryWithInset));
    }
  }
  return boundariesWithInsets;
}

std::vector<CgalPolygonWithHoles2d> extractPolygonsFromBinaryImage(const cv::Mat& binary_image) {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;  // [Next, Previous, First_Child, Parent]
  auto isOuterContour = [](const cv::Vec4i& hierarchyVector) {
    return hierarchyVector[3] < 0;  // no parent
  };

  cv::findContours(binary_image, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

  std::vector<CgalPolygonWithHoles2d> plane_polygons;
  for (int i = 0; i < contours.size(); i++) {
    if (isOuterContour(hierarchy[i]) && contours[i].size() > 1) {
      CgalPolygonWithHoles2d polygon;
      polygon.outer_boundary() = cgalPolygonFromOpenCv(contours[i]);

      // Add children as holes
      int childIndex = hierarchy[i][2];  // First child
      while (childIndex > 0) {
        polygon.add_hole(cgalPolygonFromOpenCv(contours[childIndex]));
        childIndex = hierarchy[childIndex][0];  // Next child
      }
      plane_polygons.push_back(std::move(polygon));
    }
  }
  return plane_polygons;
}

CgalPolygon2d cgalPolygonFromOpenCv(const std::vector<cv::Point>& openCvPolygon) {
  CgalPolygon2d polygon;
  polygon.container().reserve(openCvPolygon.size());
  for (const auto& point : openCvPolygon) {
    polygon.container().emplace_back(point.x, point.y);
  }
  return polygon;
}

CgalPoint2d worldFrameToTerrainFrame(const CgalPoint2d& worldFrameXY, const TerrainPlane& plane) {
  const auto projectedPositionInWorld = projectPositionInWorldOntoPlaneAlongGravity({worldFrameXY.x(), worldFrameXY.y(), 0.0}, plane);
  const auto positionInTerrainFrame = positionInTerrainFrameFromPositionInWorld(projectedPositionInWorld, plane);
  return {positionInTerrainFrame.x(), positionInTerrainFrame.y()};
}

CgalPoint2d pixelToWorldFrame(const CgalPoint2d& pixelspaceCgalPoint2d, double resolution, const Eigen::Vector2d& mapOffset) {
  // Notice the transpose of x and y!
  return {mapOffset.x() - resolution * pixelspaceCgalPoint2d.y(), mapOffset.y() - resolution * pixelspaceCgalPoint2d.x()};
}

}  // namespace contour_extraction
}  // namespace convex_plane_decomposition
