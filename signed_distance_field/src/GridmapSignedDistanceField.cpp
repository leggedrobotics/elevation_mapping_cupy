/*
 * GridmapSignedDistanceField.cpp
 *
 *  Created on: Jul 10, 2020
 *      Author: Ruben Grandia
 *   Institute: ETH Zurich
 */

#include "signed_distance_field/GridmapSignedDistanceField.h"

#include "signed_distance_field/DistanceDerivatives.h"
#include "signed_distance_field/SignedDistance2d.h"

namespace grid_map {

// Import from the signed_distance_field namespace
using signed_distance_field::columnwiseCentralDifference;
using signed_distance_field::Gridmap3dLookup;
using signed_distance_field::layerCentralDifference;
using signed_distance_field::layerFiniteDifference;
using signed_distance_field::signedDistanceAtHeightTranspose;

GridmapSignedDistanceField::GridmapSignedDistanceField(const GridMap& gridMap, const std::string& elevationLayer, double minHeight,
                                                       double maxHeight)
    : frameId_(gridMap.getFrameId()), timestamp_(gridMap.getTimestamp()) {
  assert(maxHeight >= minHeight);

  // Determine origin of the 3D grid
  Position mapOriginXY;
  gridMap.getPosition(Eigen::Vector2i(0, 0), mapOriginXY);
  Position3 gridOrigin(mapOriginXY.x(), mapOriginXY.y(), minHeight);

  // Round up the Z-discretization. We need a minimum of two layers to enable finite difference in Z direction
  const auto numZLayers = static_cast<size_t>(std::max(std::ceil((maxHeight - minHeight) / gridMap.getResolution()), 2.0));
  const size_t numXrows = gridMap.getSize().x();
  const size_t numYrows = gridMap.getSize().y();
  Gridmap3dLookup::size_t_3d gridsize = {numXrows, numYrows, numZLayers};

  // Initialize 3D lookup
  gridmap3DLookup_ = Gridmap3dLookup(gridsize, gridOrigin, gridMap.getResolution());

  // Allocate the internal data structure
  data_.reserve(gridmap3DLookup_.linearSize());

  // Check for NaN
  const auto& elevationData = gridMap.get(elevationLayer);
  if (elevationData.hasNaN()) {
    std::cerr << "[GridmapSignedDistanceField] elevation data contains NaN. The generated SDF will be invalid! Apply inpainting first"
              << std::endl;
  }

  // Compute the SDF
  computeSignedDistance(elevationData);
}

double GridmapSignedDistanceField::value(const Position3& position) const noexcept {
  const auto nodeIndex = gridmap3DLookup_.nearestNode(position);
  const auto nodePosition = gridmap3DLookup_.nodePosition(nodeIndex);
  const auto nodeData = data_[gridmap3DLookup_.linearIndex(nodeIndex)];
  const auto jacobian = derivative(nodeData);
  return distance(nodeData) + jacobian.dot(position - nodePosition);
}

GridmapSignedDistanceField::Derivative3 GridmapSignedDistanceField::derivative(const Position3& position) const noexcept {
  const auto nodeIndex = gridmap3DLookup_.nearestNode(position);
  const auto nodeData = data_[gridmap3DLookup_.linearIndex(nodeIndex)];
  return derivative(nodeData);
}

std::pair<double, GridmapSignedDistanceField::Derivative3> GridmapSignedDistanceField::valueAndDerivative(
    const Position3& position) const noexcept {
  const auto nodeIndex = gridmap3DLookup_.nearestNode(position);
  const auto nodePosition = gridmap3DLookup_.nodePosition(nodeIndex);
  const auto nodeData = data_[gridmap3DLookup_.linearIndex(nodeIndex)];
  const auto jacobian = derivative(nodeData);
  return {distance(nodeData) + jacobian.dot(position - nodePosition), jacobian};
}

void GridmapSignedDistanceField::computeSignedDistance(const Matrix& elevation) {
  const auto gridOriginZ = static_cast<float>(gridmap3DLookup_.gridOrigin_.z());
  const auto resolution = static_cast<float>(gridmap3DLookup_.resolution_);
  const auto minHeight = elevation.minCoeff();
  const auto maxHeight = elevation.maxCoeff();

  /*
   * General strategy to reduce the amount of transposing:
   *    - SDF at a height is in transposed form after computing it.
   *    - Take the finite difference in dx, now that this data is continuous in memory.
   *    - Transpose the SDF
   *    - Take other finite differences. Now dy is efficient.
   *    - When writing to the 3D structure, keep in mind that dx is still transposed.
   */

  // Memory needed to compute the SDF at a layer
  Matrix tmp;           // allocated on first use
  Matrix tmpTranspose;  // allocated on first use
  Matrix sdfTranspose;  // allocated on first use

  // Memory needed to keep a buffer of layers. We need 3 due to the central difference
  Matrix currentLayer;   // allocated on first use
  Matrix nextLayer;      // allocated on first use
  Matrix previousLayer;  // allocated on first use

  // Memory needed to compute finite differences
  Matrix dxTranspose = Matrix::Zero(elevation.cols(), elevation.rows());
  Matrix dxNextTranspose = Matrix::Zero(elevation.cols(), elevation.rows());
  Matrix dy = Matrix::Zero(elevation.rows(), elevation.cols());
  Matrix dz = Matrix::Zero(elevation.rows(), elevation.cols());

  // Compute SDF of first layer
  computeLayerSdfandDeltaX(elevation, currentLayer, dxTranspose, sdfTranspose, tmp, tmpTranspose, gridOriginZ, resolution, minHeight,
                           maxHeight);

  // Compute SDF of second layer
  computeLayerSdfandDeltaX(elevation, nextLayer, dxNextTranspose, sdfTranspose, tmp, tmpTranspose, gridOriginZ + resolution, resolution,
                           minHeight, maxHeight);

  // First layer: forward difference in z
  layerFiniteDifference(currentLayer, nextLayer, dz, resolution);  // dz / layer = +resolution
  columnwiseCentralDifference(currentLayer, dy, -resolution);      // dy / dcol = -resolution

  emplacebackLayerData(currentLayer, dxTranspose, dy, dz);

  // Middle layers: central difference in z
  for (size_t layerZ = 1; layerZ + 1 < gridmap3DLookup_.gridsize_.z; ++layerZ) {
    // Circulate layer buffers
    previousLayer.swap(currentLayer);
    currentLayer.swap(nextLayer);
    dxTranspose.swap(dxNextTranspose);

    // Compute SDF of next layer
    computeLayerSdfandDeltaX(elevation, nextLayer, dxNextTranspose, sdfTranspose, tmp, tmpTranspose,
                             gridOriginZ + (layerZ + 1) * resolution, resolution, minHeight, maxHeight);

    // Compute other finite differences
    layerCentralDifference(previousLayer, nextLayer, dz, resolution);
    columnwiseCentralDifference(currentLayer, dy, -resolution);

    // Add the data to the 3D structure
    emplacebackLayerData(currentLayer, dxTranspose, dy, dz);
  }

  // Circulate layer buffers one last time
  previousLayer.swap(currentLayer);
  currentLayer.swap(nextLayer);
  dxTranspose.swap(dxNextTranspose);

  // Last layer: backward difference in z
  layerFiniteDifference(previousLayer, currentLayer, dz, resolution);
  columnwiseCentralDifference(currentLayer, dy, -resolution);

  // Add the data to the 3D structure
  emplacebackLayerData(currentLayer, dxTranspose, dy, dz);
}

void GridmapSignedDistanceField::computeLayerSdfandDeltaX(const Matrix& elevation, Matrix& currentLayer, Matrix& dxTranspose,
                                                          Matrix& sdfTranspose, Matrix& tmp, Matrix& tmpTranspose, float height,
                                                          float resolution, float minHeight, float maxHeight) const {
  // Compute SDF + dx of layer: compute sdfTranspose -> take dxTranspose -> transpose to get sdf
  signedDistanceAtHeightTranspose(elevation, sdfTranspose, tmp, tmpTranspose, height, resolution, minHeight, maxHeight);
  columnwiseCentralDifference(sdfTranspose, dxTranspose, -resolution);  // dx / drow = -resolution
  currentLayer = sdfTranspose.transpose();
}

void GridmapSignedDistanceField::emplacebackLayerData(const Matrix& signedDistance, const Matrix& dxdxTranspose, const Matrix& dy,
                                                      const Matrix& dz) {
  for (size_t colY = 0; colY < gridmap3DLookup_.gridsize_.y; ++colY) {
    for (size_t rowX = 0; rowX < gridmap3DLookup_.gridsize_.x; ++rowX) {
      data_.emplace_back(node_data_t{signedDistance(rowX, colY), dxdxTranspose(colY, rowX), dy(rowX, colY), dz(rowX, colY)});
    }
  }
}

pcl::PointCloud<pcl::PointXYZI> GridmapSignedDistanceField::asPointCloud(size_t decimation,
                                                                         const std::function<bool(float)>& condition) const {
  pcl::PointCloud<pcl::PointXYZI> points;
  points.header.stamp = timestamp_;
  points.header.frame_id = frameId_;

  points.reserve(gridmap3DLookup_.linearSize());
  for (size_t layerZ = 0; layerZ < gridmap3DLookup_.gridsize_.z; layerZ += decimation) {
    for (size_t colY = 0; colY < gridmap3DLookup_.gridsize_.y; colY += decimation) {
      for (size_t rowX = 0; rowX < gridmap3DLookup_.gridsize_.x; rowX += decimation) {
        const Gridmap3dLookup::size_t_3d index3d = {rowX, colY, layerZ};
        const auto index = gridmap3DLookup_.linearIndex(index3d);
        const auto signeddistance = distanceFloat(data_[index]);
        if (condition(signeddistance)) {
          const auto p = gridmap3DLookup_.nodePosition(index3d);
          pcl::PointXYZI point;
          point.x = p.x();
          point.y = p.y();
          point.z = p.z();
          point.intensity = signeddistance;
          points.push_back(point);
        }
      }
    }
  }
  return points;
}

pcl::PointCloud<pcl::PointXYZI> GridmapSignedDistanceField::freeSpacePointCloud(size_t decimation) const {
  return asPointCloud(decimation, [](float signedDistance) { return signedDistance >= 0.0F; });
}

pcl::PointCloud<pcl::PointXYZI> GridmapSignedDistanceField::obstaclePointCloud(size_t decimation) const {
  return asPointCloud(decimation, [](float signedDistance) { return signedDistance <= 0.0F; });
}

}  // namespace grid_map
