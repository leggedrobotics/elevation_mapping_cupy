//
// Created by rgrandia on 10.07.20.
//

#include "signed_distance_field/SignedDistanceField.h"

#include "signed_distance_field/DistanceDerivatives.h"
#include "signed_distance_field/SignedDistance2d.h"

namespace signed_distance_field {

SignedDistanceField::SignedDistanceField(const grid_map::GridMap& gridMap, const std::string& elevationLayer, double minHeight, double maxHeight) :
      resolution_(static_cast<float>(gridMap.getResolution())) {
  assert(maxHeight >= minHeight);
  grid_map::Position mapOriginXY;
  gridMap.getPosition(Eigen::Vector2i(0, 0), mapOriginXY);
  gridOrigin_ = {mapOriginXY.x(), mapOriginXY.y(), minHeight};

  // Take minimum of two layers to enable finite difference in Z direction
  const auto numZLayers = static_cast<size_t>(std::max(std::ceil((maxHeight - minHeight) / gridMap.getResolution() ), 2.0));
  const size_t numXrows = gridMap.getSize().x();
  const size_t numYrows = gridMap.getSize().y();
  gridsize_ = {numXrows, numYrows, numZLayers};

  data_.reserve(numXrows * numYrows * numZLayers);
  computeSignedDistance(gridMap.get(elevationLayer));
}
Eigen::Vector3d SignedDistanceField::nodePosition(size_t rowX, size_t colY, size_t layerZ) const {
  return {gridOrigin_.x() - rowX * resolution_, gridOrigin_.y() - colY * resolution_, gridOrigin_.z() + layerZ * resolution_};
}

size_t SignedDistanceField::linearIndex(size_t rowX, size_t colY, size_t layerZ) const {
  return layerZ * gridsize_[1] * gridsize_[0] + colY * gridsize_[0] + rowX;
}

void SignedDistanceField::computeSignedDistance(const grid_map::Matrix& elevation) {
  // First layer: forward difference in z
  grid_map::Matrix currentLayer = signed_distance_field::signedDistanceAtHeight(elevation, gridOrigin_.z(), resolution_);
  grid_map::Matrix nextLayer = signed_distance_field::signedDistanceAtHeight(elevation, gridOrigin_.z() + resolution_, resolution_);
  grid_map::Matrix dz = signed_distance_field::layerFiniteDifference(currentLayer, nextLayer, resolution_); // dz / layer = +resolution
  grid_map::Matrix dy = signed_distance_field::columnwiseCentralDifference(currentLayer, -resolution_); // dy / dcol = -resolution
  grid_map::Matrix dx = signed_distance_field::rowwiseCentralDifference(currentLayer, -resolution_); // dx / drow = -resolution
  emplacebackLayerData(currentLayer, dx, dy, dz);

  // Middle layers: central difference in z
  for (size_t layerZ = 1; layerZ + 1 < gridsize_[2]; ++layerZ) {
    grid_map::Matrix previousLayer = std::move(currentLayer);
    currentLayer = std::move(nextLayer);
    nextLayer = signed_distance_field::signedDistanceAtHeight(elevation, gridOrigin_.z() + (layerZ + 1) * resolution_, resolution_);

    dz = signed_distance_field::layerCentralDifference(previousLayer, nextLayer, resolution_);
    dy = signed_distance_field::columnwiseCentralDifference(currentLayer, -resolution_);
    dx = signed_distance_field::rowwiseCentralDifference(currentLayer, -resolution_);
    emplacebackLayerData(currentLayer, dx, dy, dz);
  }

  // Last layer: backward difference in z
  grid_map::Matrix previousLayer = std::move(currentLayer);
  currentLayer = std::move(nextLayer);
  dz = signed_distance_field::layerCentralDifference(previousLayer, currentLayer, resolution_);
  dy = signed_distance_field::columnwiseCentralDifference(currentLayer, -resolution_);
  dx = signed_distance_field::rowwiseCentralDifference(currentLayer, -resolution_);
  emplacebackLayerData(currentLayer, dx, dy, dz);
}

void SignedDistanceField::emplacebackLayerData(const grid_map::Matrix& signedDistance, const grid_map::Matrix& dx,
                                        const grid_map::Matrix& dy, const grid_map::Matrix& dz) {
  for (size_t colY=0; colY<gridsize_[1]; ++colY) {
    for (size_t rowX=0; rowX<gridsize_[0]; ++rowX) {
      data_.emplace_back(node_data_t{signedDistance(rowX, colY), dx(rowX, colY), dy(rowX, colY), dz(rowX, colY)});
    }
  }
}

pcl::PointCloud<pcl::PointXYZI> SignedDistanceField::asPointCloud() const {
  pcl::PointCloud<pcl::PointXYZI> points;
  points.reserve(gridsize_[0]*gridsize_[1]*gridsize_[2]);
  for (size_t layerZ=0; layerZ<gridsize_[2]; ++layerZ) {
    for (size_t colY = 0; colY < gridsize_[1]; ++colY) {
      for (size_t rowX = 0; rowX < gridsize_[0]; ++rowX) {
        const auto index = linearIndex(rowX, colY, layerZ);
        const auto p = nodePosition(rowX, colY, layerZ);
        pcl::PointXYZI point;
        point.x = p.x();
        point.y = p.y();
        point.z = p.z();
        point.intensity = data_[index][0];
        points.push_back(point);
      }
    }
  }
  return points;
}

}  // namespace signed_distance_field
