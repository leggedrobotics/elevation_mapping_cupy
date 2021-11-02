//
// Created by rgrandia on 10.07.20.
//

#include "signed_distance_field/GridmapSignedDistanceField.h"

#include "signed_distance_field/DistanceDerivatives.h"
#include "signed_distance_field/SignedDistance2d.h"

namespace signed_distance_field {

GridmapSignedDistanceField::GridmapSignedDistanceField(const grid_map::GridMap& gridMap, const std::string& elevationLayer,
                                                       double minHeight, double maxHeight) {
  assert(maxHeight >= minHeight);
  grid_map::Position mapOriginXY;
  gridMap.getPosition(Eigen::Vector2i(0, 0), mapOriginXY);
  Eigen::Vector3d gridOrigin(mapOriginXY.x(), mapOriginXY.y(), minHeight);

  // Take minimum of two layers to enable finite difference in Z direction
  const auto numZLayers = static_cast<size_t>(std::max(std::ceil((maxHeight - minHeight) / gridMap.getResolution()), 2.0));
  const size_t numXrows = gridMap.getSize().x();
  const size_t numYrows = gridMap.getSize().y();
  Gridmap3dLookup::size_t_3d gridsize = {numXrows, numYrows, numZLayers};

  gridmap3DLookup_ = Gridmap3dLookup(gridsize, gridOrigin, gridMap.getResolution());

  data_.reserve(gridmap3DLookup_.linearSize());
  const auto& elevationData = gridMap.get(elevationLayer);
  if (elevationData.hasNaN()) {
    std::cerr << "[GridmapSignedDistanceField] elevation data contains NaN" << std::endl;
  }
  computeSignedDistance(elevationData);
}

GridmapSignedDistanceField::GridmapSignedDistanceField(const GridmapSignedDistanceField& other)
    : switched_model::SignedDistanceField(), gridmap3DLookup_(other.gridmap3DLookup_), data_(other.data_) {}

GridmapSignedDistanceField* GridmapSignedDistanceField::clone() const {
  return new GridmapSignedDistanceField(*this);
}

switched_model::scalar_t GridmapSignedDistanceField::value(const switched_model::vector3_t& position) const {
  const auto nodeIndex = gridmap3DLookup_.nearestNode(position);
  const Eigen::Vector3d nodePosition = gridmap3DLookup_.nodePosition(nodeIndex);
  const node_data_t nodeData = data_[gridmap3DLookup_.linearIndex(nodeIndex)];
  const Eigen::Vector3d jacobian = derivative(nodeData);
  return distance(nodeData) + jacobian.dot(position - nodePosition);
}

switched_model::vector3_t GridmapSignedDistanceField::derivative(const switched_model::vector3_t& position) const {
  const auto nodeIndex = gridmap3DLookup_.nearestNode(position);
  const node_data_t nodeData = data_[gridmap3DLookup_.linearIndex(nodeIndex)];
  return derivative(nodeData);
}

std::pair<switched_model::scalar_t, switched_model::vector3_t> GridmapSignedDistanceField::valueAndDerivative(
    const switched_model::vector3_t& position) const {
  const auto nodeIndex = gridmap3DLookup_.nearestNode(position);
  const Eigen::Vector3d nodePosition = gridmap3DLookup_.nodePosition(nodeIndex);
  const node_data_t nodeData = data_[gridmap3DLookup_.linearIndex(nodeIndex)];
  const Eigen::Vector3d jacobian = derivative(nodeData);
  return {distance(nodeData) + jacobian.dot(position - nodePosition), jacobian};
}

void GridmapSignedDistanceField::computeSignedDistance(const grid_map::Matrix& elevation) {
  using signed_distance_field::columnwiseCentralDifference;
  using signed_distance_field::layerCentralDifference;
  using signed_distance_field::layerFiniteDifference;
  using signed_distance_field::signedDistanceAtHeightTranspose;

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
  grid_map::Matrix tmp; // allocated on first use
  grid_map::Matrix tmpTranspose; // allocated on first use
  grid_map::Matrix sdfTranspose; // allocated on first use

  // Memory needed to keep a buffer of layers. We need 3 due to the central difference
  grid_map::Matrix currentLayer; // allocated on first use
  grid_map::Matrix nextLayer; // allocated on first use
  grid_map::Matrix previousLayer; // allocated on first use

  // Memory needed to compute finite differences
  grid_map::Matrix dxTranspose = grid_map::Matrix::Zero(elevation.cols(), elevation.rows());
  grid_map::Matrix dxNextTranspose = grid_map::Matrix::Zero(elevation.cols(), elevation.rows());
  grid_map::Matrix dy = grid_map::Matrix::Zero(elevation.rows(), elevation.cols());
  grid_map::Matrix dz = grid_map::Matrix::Zero(elevation.rows(), elevation.cols());

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

  // Circulate layer buffers on last time
  previousLayer.swap(currentLayer);
  currentLayer.swap(nextLayer);
  dxTranspose.swap(dxNextTranspose);

  // Last layer: backward difference in z
  layerFiniteDifference(previousLayer, currentLayer, dz, resolution);
  columnwiseCentralDifference(currentLayer, dy, -resolution);

  // Add the data to the 3D structure
  emplacebackLayerData(currentLayer, dxTranspose, dy, dz);
}
void GridmapSignedDistanceField::computeLayerSdfandDeltaX(const grid_map::Matrix& elevation, grid_map::Matrix& currentLayer,
                                                          grid_map::Matrix& dxTranspose, grid_map::Matrix& sdfTranspose,
                                                          grid_map::Matrix& tmp, grid_map::Matrix& tmpTranspose, const float gridOriginZ,
                                                          const float resolution, const float minHeight, const float maxHeight) const {
  // Compute SDF + dx of layer: compute sdfTranspose -> take dxTranspose -> transpose to get sdf
  signedDistanceAtHeightTranspose(elevation, sdfTranspose, tmp, tmpTranspose, gridOriginZ, resolution, minHeight, maxHeight);
  columnwiseCentralDifference(sdfTranspose, dxTranspose, -resolution);  // dx / drow = -resolution
  currentLayer = sdfTranspose.transpose();
}

void GridmapSignedDistanceField::emplacebackLayerData(const grid_map::Matrix& signedDistance, const grid_map::Matrix& dxdxTranspose,
                                                      const grid_map::Matrix& dy, const grid_map::Matrix& dz) {
  for (size_t colY = 0; colY < gridmap3DLookup_.gridsize_.y; ++colY) {
    for (size_t rowX = 0; rowX < gridmap3DLookup_.gridsize_.x; ++rowX) {
      data_.emplace_back(node_data_t{signedDistance(rowX, colY), dxdxTranspose(colY, rowX), dy(rowX, colY), dz(rowX, colY)});
    }
  }
}

pcl::PointCloud<pcl::PointXYZI> GridmapSignedDistanceField::asPointCloud(size_t decimation,
                                                                         const std::function<bool(float)>& condition) const {
  pcl::PointCloud<pcl::PointXYZI> points;
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

}  // namespace signed_distance_field
