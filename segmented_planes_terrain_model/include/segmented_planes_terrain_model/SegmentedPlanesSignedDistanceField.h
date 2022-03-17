//
// Created by rgrandia on 17.03.22.
//

#pragma once

#include <ocs2_switched_model_interface/terrain/SignedDistanceField.h>

#include <signed_distance_field/GridmapSignedDistanceField.h>

namespace switched_model {

/**
 * Simple wrapper class to implement the switched_model::SignedDistanceField interface.
 * See the forwarded function for documentation.
 */
class SegmentedPlanesSignedDistanceField : public SignedDistanceField {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SegmentedPlanesSignedDistanceField(const grid_map::GridMap& gridMap, const std::string& elevationLayer, double minHeight,
                                     double maxHeight)
      : sdf_(gridMap, elevationLayer, minHeight, maxHeight) {}

  ~SegmentedPlanesSignedDistanceField() override = default;
  SegmentedPlanesSignedDistanceField* clone() const override { return new SegmentedPlanesSignedDistanceField(*this); };

  switched_model::scalar_t value(const switched_model::vector3_t& position) const override { return sdf_.value(position); }

  switched_model::vector3_t derivative(const switched_model::vector3_t& position) const override { return sdf_.derivative(position); }

  std::pair<switched_model::scalar_t, switched_model::vector3_t> valueAndDerivative(
      const switched_model::vector3_t& position) const override {
    return sdf_.valueAndDerivative(position);
  }

  pcl::PointCloud<pcl::PointXYZI> asPointCloud(
      size_t decimation = 1, const std::function<bool(float)>& condition = [](float) { return true; }) const {
    return sdf_.asPointCloud(decimation, condition);
  }

  pcl::PointCloud<pcl::PointXYZI> freeSpacePointCloud(size_t decimation = 1) const { return sdf_.freeSpacePointCloud(decimation); }

  pcl::PointCloud<pcl::PointXYZI> obstaclePointCloud(size_t decimation = 1) const { return sdf_.obstaclePointCloud(decimation); }

 protected:
  SegmentedPlanesSignedDistanceField(const SegmentedPlanesSignedDistanceField& other) : sdf_(other.sdf_){};

 private:
  grid_map::GridmapSignedDistanceField sdf_;
};

}  // namespace switched_model