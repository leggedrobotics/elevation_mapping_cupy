//
// Created by rgrandia on 13.08.20.
//

#pragma once

#include <grid_map_core/TypeDefs.hpp>

namespace grid_map {
namespace signed_distance_field {

/**
 * Stores 3 dimensional grid information and provides methods to convert between position - 3d Index - linear index.
 *
 * As with grid map, the X-Y position is opposite to the row-col-index: (X,Y) is highest at (0,0) and lowest at (n, m).
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
  Position3 gridOrigin_;
  double resolution_;

  /**
   * Constructor
   * @param gridsize : x, y, z size of the grid
   * @param gridOrigin : position at x=y=z=0
   * @param resolution : (>0.0) size of 1 voxel
   */
  Gridmap3dLookup(const size_t_3d& gridsize, const Position3& gridOrigin, double resolution)
      : gridsize_(gridsize), gridOrigin_(gridOrigin), resolution_(resolution) {
    assert(resolution_ > 0.0);
    assert(gridsize_.x > 0);
    assert(gridsize_.y > 0);
    assert(gridsize_.z > 0);
  };

  /** Default constructor: creates an empty grid */
  Gridmap3dLookup() : Gridmap3dLookup({1, 1, 1}, {0.0, 0.0, 0.0}, 1.0) {}

  /** Returns the 3d index of the grid node closest to the query position */
  size_t_3d nearestNode(const Position3& position) const noexcept {
    Position3 subpixelVector{(gridOrigin_.x() - position.x()) / resolution_, (gridOrigin_.y() - position.y()) / resolution_,
                             (position.z() - gridOrigin_.z()) / resolution_};
    return {nearestPositiveInteger(subpixelVector.x(), gridsize_.x - 1), nearestPositiveInteger(subpixelVector.y(), gridsize_.y - 1),
            nearestPositiveInteger(subpixelVector.z(), gridsize_.z - 1)};
  }

  /** Returns the 3d node position from a 3d index */
  Position3 nodePosition(const size_t_3d& index) const noexcept {
    return {gridOrigin_.x() - index.x * resolution_, gridOrigin_.y() - index.y * resolution_, gridOrigin_.z() + index.z * resolution_};
  }

  /** Returns the linear node index from a 3d node index */
  size_t linearIndex(const size_t_3d& index) const noexcept { return (index.z * gridsize_.y + index.y) * gridsize_.x + index.x; }

  /** Linear size */
  size_t linearSize() const noexcept { return gridsize_.x * gridsize_.y * gridsize_.z; }

  /** rounds subindex value and clamps it to [0, max] */
  static size_t nearestPositiveInteger(double val, size_t max) noexcept {
    // Comparing bounds as double prevents underflow/overflow
    if (val > 0.0) {
      return static_cast<size_t>(std::min(std::round(val), static_cast<double>(max)));
    } else {
      return 0;
    }
  }
};

}  // namespace signed_distance_field
}  // namespace grid_map