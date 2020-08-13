//
// Created by rgrandia on 13.08.20.
//

#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

namespace signed_distance_field {

/**
 * Stores 3 dimensional grid information and provides methods to convert between position - 3d Index - linear index.
 *
 * As with grid map, the X-Y position is opposite to the row-col-index: lowest at (0,0) and highest at (n, m).
 * The z-position is increasing with the layer-index.
 */
struct Gridmap3dLookup {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// size_t in 3 dimensions
  struct size_t_3d {
    size_t x;
    size_t y;
    size_t z;
  };

  size_t_3d gridsize_;
  Eigen::Vector3d gridOrigin_;
  double resolution_;

  /**
   * Constructor
   * @param gridsize : x, y, z size of the grid
   * @param gridOrigin : position at x=y=z=0
   * @param resolution : (>0.0) size of 1 voxel
   */
  Gridmap3dLookup(const size_t_3d& gridsize, const Eigen::Vector3d& gridOrigin, double resolution) :
      gridsize_(gridsize), gridOrigin_(gridOrigin), resolution_(resolution) {
            assert(resolution_ > 0.0);
        };

  /** Default constructor: creates an empty grid */
  Gridmap3dLookup() : Gridmap3dLookup({0, 0, 0}, {0.0,0.0,0.0}, 1.0) {}

  /** Returns the 3d index of the grid node closest to the query position */
  size_t_3d nearestNode(const Eigen::Vector3d& position) const noexcept {
    auto round = [](double val) {
      return static_cast<size_t>(std::round(val));
    };
    Eigen::Vector3d subpixelVector = (position - gridOrigin_) / resolution_;
    return {round(subpixelVector.x()), round(subpixelVector.y()), round(subpixelVector.z())};
  }

  /** Returns the 3d node position from a 3d index */
  Eigen::Vector3d nodePosition(const size_t_3d& index) const noexcept {
    return {gridOrigin_.x() - index.x * resolution_, gridOrigin_.y() - index.y * resolution_, gridOrigin_.z() + index.z * resolution_};
  }

  /** Returns the linear node index from a 3d node index */
  size_t linearIndex(const size_t_3d& index) const noexcept { return index.z * gridsize_.y * gridsize_.x + index.y * gridsize_.x + index.x; }

  /** Linear size */
  size_t linearSize() const noexcept { return gridsize_.x * gridsize_.y * gridsize_.z; }
};

}  // namespace signed_distance_field
