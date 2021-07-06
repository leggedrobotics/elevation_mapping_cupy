#include "convex_plane_decomposition/Postprocessing.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace convex_plane_decomposition {

Postprocessing::Postprocessing(const PostprocessingParameters& parameters) : parameters_(parameters) {}

void Postprocessing::postprocess(PlanarTerrain& planarTerrain, const std::string& elevationLayer,
                                 const std::string& planeSegmentationLayer) const {
  auto& elevationData = planarTerrain.gridMap.get(elevationLayer);
  const auto& planarityMask = planarTerrain.gridMap.get(planeSegmentationLayer);

  // post process planar regions
  addHeightOffset(planarTerrain.planarRegions);

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

}  // namespace convex_plane_decomposition
