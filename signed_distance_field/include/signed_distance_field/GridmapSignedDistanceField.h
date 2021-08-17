//
// Created by rgrandia on 10.07.20.
//

#pragma once

#include <vector>

#include <Eigen/Dense>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/TypeDefs.hpp>

#include <pcl/conversions.h>
#include <pcl/point_types.h>

#include <ocs2_switched_model_interface/terrain/SignedDistanceField.h>

#include "Gridmap3dLookup.h"

namespace signed_distance_field {

class GridmapSignedDistanceField : public switched_model::SignedDistanceField {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GridmapSignedDistanceField(const grid_map::GridMap& gridMap, const std::string& elevationLayer, double minHeight, double maxHeight);
  ~GridmapSignedDistanceField() override = default;
  GridmapSignedDistanceField* clone() const override;

  switched_model::scalar_t value(const switched_model::vector3_t& position) const override;
  switched_model::vector3_t derivative(const switched_model::vector3_t& position) const override;
  std::pair<switched_model::scalar_t, switched_model::vector3_t> valueAndDerivative(
      const switched_model::vector3_t& position) const override;

  /**
   * Return the signed distance field as a pointcloud. The signed distance is assigned to the point's intensity.
   * @param decimation : specifies how many points are returned. 1: all points, 2: every second point, etc.
   * @param condition : specifies the condition on the distance value to add it to the pointcloud (default = any distance is added)
   */
  pcl::PointCloud<pcl::PointXYZI> asPointCloud(
      size_t decimation = 1, const std::function<bool(float)>& condition = [](float) { return true; }) const;

  /**
   * Return the signed distance field as a pointcloud where the distance is positive.
   * @param decimation : specifies how many points are returned. 1: all points, 2: every second point, etc.
   */
  pcl::PointCloud<pcl::PointXYZI> freeSpacePointCloud(size_t decimation = 1) const;

  /**
   * Return the signed distance field as a pointcloud where the distance is negative.
   * @param decimation : specifies how many points are returned. 1: all points, 2: every second point, etc.
   */
  pcl::PointCloud<pcl::PointXYZI> obstaclePointCloud(size_t decimation = 1) const;

 private:
  GridmapSignedDistanceField(const GridmapSignedDistanceField& other);
  void computeSignedDistance(const grid_map::Matrix& elevation);
  void emplacebackLayerData(const grid_map::Matrix& signedDistance, const grid_map::Matrix& dx, const grid_map::Matrix& dy,
                            const grid_map::Matrix& dz);

  using node_data_t = std::array<float, 4>;
  static double distance(const node_data_t& nodeData) noexcept { return nodeData[0]; }
  static float distanceFloat(const node_data_t& nodeData) noexcept { return nodeData[0]; }
  static Eigen::Vector3d derivative(const node_data_t& nodeData) noexcept { return {nodeData[1], nodeData[2], nodeData[3]}; }

  Gridmap3dLookup gridmap3DLookup_;
  std::vector<node_data_t> data_;
};

}  // namespace signed_distance_field
